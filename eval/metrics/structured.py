"""Structured-output reliability instrumentation (PROJECT_SPEC.md §4A.1 /
§7 Phase 2). Four call sites in src/agent.py depend on
llm_model.with_structured_output(): query_router (RouteDecision),
rewrite_query (RewrittenQuery), and both graders inside
hallucinations_and_relevance_router (HallucinationScore, RelevanceScore).
LangChain's with_structured_output() gives no built-in signal about how
often it actually succeeds -- this module wraps it so the eval harness can
measure what a hosted tool-calling call site never surfaces on its own:

  - structured_output_validity_rate -- % of calls that parsed into the
    Pydantic schema on the first attempt
  - structured_output_retry_rate -- % that needed a measurement-only retry
    (same prompt, re-invoked once) to parse at all
  - silent_coercion_rate -- best-effort only (see _detect_silent_coercion):
    % of valid parses where a raw tool-call argument's JSON type differed
    from what Pydantic ended up storing, i.e. Pydantic silently repaired
    something a naive caller would never notice. None (not 0.0) whenever
    the raw tool-call args aren't in an inspectable shape -- "no coercion
    detected" and "couldn't check" must never be conflated.
  - malformed_to_misroute_rate -- needs both this module's per-call
    validity signal for query_router AND eval.metrics.router's route
    comparison; only computable when eval.harness passes
    `misroute_context` into compute_structured_metrics(), None otherwise.

Usage: build an InstrumentingLLM(real_llm, recorder) and pass it as the
`llm_model` argument to src.agent.build_agent_graph() -- that function
already takes llm_model as a constructor parameter, so no change to
src/agent.py's node logic is needed. Every with_structured_output(Schema)
call made while building the graph's prompt|llm chains gets wrapped
transparently; plain-text chains (`prompt | llm_model | StrOutputParser()`,
used by generate/chitchat) pass straight through, uninstrumented, via
InstrumentingLLM.__call__/.invoke().

InstrumentedStructuredRunnable does a genuine measurement-only retry (up to
`max_attempts`, default 2) and, on final failure, raises
StructuredOutputValidationError -- deliberately mirroring what the real
with_structured_output(Schema) (include_raw=False, the default
src/agent.py uses directly) already raises on a parse failure, so
src/agent.py's existing try/except fallback paths behave identically
whether or not this instrumentation is in use. The retry is new
API-call cost that only exists inside this eval-only wrapper; it is never
part of the deployed pipeline, same as the vLLM ablation backend
(PROJECT_SPEC.md §4A) -- a measurement instrument, not a feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.agent import HallucinationScore, RelevanceScore, RewrittenQuery, RouteDecision

SCHEMA_TO_NODE = {
    RouteDecision: "query_router",
    RewrittenQuery: "rewrite_query",
    HallucinationScore: "hallucination_check",
    RelevanceScore: "relevance_check",
}


class StructuredOutputValidationError(Exception):
    """Raised by InstrumentedStructuredRunnable when every attempt
    (first try + measurement-only retries) fails to parse -- mirrors the
    exception the real with_structured_output(Schema) call would raise,
    so callers' existing try/except fallback logic is unaffected."""


@dataclass
class StructuredCallEvent:
    node: str
    valid: bool
    retried: bool
    coerced: Optional[bool]  # None when undetermined -- see module docstring
    error: Optional[str]


class StructuredOutputRecorder:
    """Shared sink InstrumentedStructuredRunnable logs to. One instance per
    eval run (or per item, if the harness wants per-item breakdowns) --
    the harness owns its lifetime, this module never resets it itself."""

    def __init__(self):
        self.events: list[StructuredCallEvent] = []

    def record(self, event: StructuredCallEvent) -> None:
        self.events.append(event)


def _detect_silent_coercion(raw_message, parsed) -> Optional[bool]:
    """Best-effort only. Compares each field's raw tool-call JSON argument
    type against the type Pydantic actually stored after validation -- a
    str->bool or str->int mismatch here means the model's raw output
    wasn't already schema-native and Pydantic quietly repaired it, which a
    caller using the un-instrumented with_structured_output() would never
    see. Returns None (not False) whenever the raw AIMessage's tool_calls
    aren't in the expected shape (provider differences, no tool call at
    all, etc.) -- "no coercion detected" and "couldn't check" are
    different claims and must not be conflated (this is this module's one
    significant limitation, published per invariant 16 rather than
    papered over with a fake precise number)."""
    try:
        tool_calls = getattr(raw_message, "tool_calls", None) or []
        if not tool_calls:
            return None
        raw_args = tool_calls[0].get("args", {})
        if not isinstance(raw_args, dict):
            return None
        parsed_dict = parsed.model_dump()
        for field_name, raw_value in raw_args.items():
            if field_name not in parsed_dict:
                continue
            parsed_value = parsed_dict[field_name]
            if type(raw_value) is not type(parsed_value):
                return True
        return False
    except Exception:
        return None


class InstrumentedStructuredRunnable:
    """Drop-in replacement for `llm_model.with_structured_output(Schema)`'s
    return value -- exposes __call__ so LangChain's `prompt | wrapper`
    composition auto-coerces it into a RunnableLambda (langchain_core's
    coerce_to_runnable checks isinstance(Runnable) first, then callable();
    a plain object with only .invoke() would not qualify)."""

    def __init__(
        self,
        raw_runnable,
        *,
        schema: type,
        node_label: str,
        recorder: StructuredOutputRecorder,
        max_attempts: int = 2,
    ):
        self._raw_runnable = raw_runnable  # real_llm.with_structured_output(schema, include_raw=True)
        self._schema = schema
        self._node_label = node_label
        self._recorder = recorder
        self._max_attempts = max(1, max_attempts)

    def __call__(self, inputs):
        return self.invoke(inputs)

    def invoke(self, inputs, **kwargs):
        last_error = None
        for attempt in range(self._max_attempts):
            result = self._raw_runnable.invoke(inputs, **kwargs)
            parsed = result.get("parsed")
            parsing_error = result.get("parsing_error")
            if parsed is not None and parsing_error is None:
                coerced = _detect_silent_coercion(result.get("raw"), parsed)
                self._recorder.record(
                    StructuredCallEvent(node=self._node_label, valid=True, retried=attempt > 0, coerced=coerced, error=None)
                )
                return parsed
            last_error = parsing_error

        self._recorder.record(
            StructuredCallEvent(
                node=self._node_label, valid=False, retried=self._max_attempts > 1, coerced=None, error=str(last_error)
            )
        )
        raise StructuredOutputValidationError(
            f"{self._node_label}: schema {self._schema.__name__} failed to parse after "
            f"{self._max_attempts} attempt(s): {last_error}"
        )


class InstrumentingLLM:
    """Wraps a chat model client so its with_structured_output(Schema) call
    sites get instrumented, while everything else (plain
    `prompt | llm_model | StrOutputParser()` piping used by generate/
    chitchat, and any other attribute access) passes straight through.
    Substitute for `llm_model` in build_agent_graph(llm_model=...) --
    src/agent.py itself needs no changes."""

    def __init__(self, real_llm, recorder: StructuredOutputRecorder, max_attempts: int = 2):
        self._real = real_llm
        self._recorder = recorder
        self._max_attempts = max_attempts

    def __call__(self, *args, **kwargs):
        return self._real.invoke(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        return self._real.invoke(*args, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        node_label = SCHEMA_TO_NODE.get(schema, getattr(schema, "__name__", str(schema)))
        raw_runnable = self._real.with_structured_output(schema, include_raw=True, **kwargs)
        return InstrumentedStructuredRunnable(
            raw_runnable, schema=schema, node_label=node_label, recorder=self._recorder, max_attempts=self._max_attempts
        )

    def __getattr__(self, name):
        return getattr(self._real, name)


def compute_structured_metrics(events: Sequence[StructuredCallEvent], *, misroute_context: Optional[Sequence[dict]] = None) -> dict:
    """Aggregate StructuredOutputRecorder.events into per-node and overall
    rates. `misroute_context`, when given, is one
    {"router_valid": bool, "predicted_route": str, "expected_route": str}
    dict per golden-set item whose first node was query_router -- needed
    only for malformed_to_misroute_rate, since that metric requires
    cross-referencing this module's validity signal with router.py's route
    comparison (eval.harness wires the two together). Without it,
    malformed_to_misroute_rate is None, not 0.0 -- "not measured" and
    "zero misroutes from malformed output" are different claims.
    """
    by_node: dict[str, list[StructuredCallEvent]] = {}
    for e in events:
        by_node.setdefault(e.node, []).append(e)

    def _rates(node_events: Sequence[StructuredCallEvent]) -> dict:
        n = len(node_events)
        n_valid = sum(1 for e in node_events if e.valid)
        n_retried = sum(1 for e in node_events if e.retried)
        coercion_checked = [e for e in node_events if e.coerced is not None]
        n_coerced = sum(1 for e in coercion_checked if e.coerced)
        return {
            "n_calls": n,
            "validity_rate": (n_valid / n) if n else None,
            "retry_rate": (n_retried / n) if n else None,
            "silent_coercion_rate": (n_coerced / len(coercion_checked)) if coercion_checked else None,
        }

    per_node = {node: _rates(node_events) for node, node_events in by_node.items()}
    aggregate = _rates(events)

    malformed_to_misroute_rate = None
    if misroute_context:
        malformed_router_items = [c for c in misroute_context if c.get("router_valid") is False]
        if malformed_router_items:
            n_misrouted = sum(1 for c in malformed_router_items if c.get("predicted_route") != c.get("expected_route"))
            malformed_to_misroute_rate = n_misrouted / len(malformed_router_items)

    return {
        "per_node": per_node,
        "aggregate": aggregate,
        "malformed_to_misroute_rate": malformed_to_misroute_rate,
    }
