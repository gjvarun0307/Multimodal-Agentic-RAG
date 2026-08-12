"""Tests for eval/metrics/structured.py's instrumentation wrapper and
aggregation. No real LLM calls -- fake `real_llm`/`raw_runnable` objects
stand in for langchain_core Runnables, verified against real
ChatPromptTemplate composition (the `prompt | wrapper` mechanism this
module's __call__ methods depend on) so the fakes don't drift from actual
LangChain behavior.
"""

from pydantic import BaseModel

from eval.metrics.structured import (
    InstrumentedStructuredRunnable,
    InstrumentingLLM,
    StructuredCallEvent,
    StructuredOutputRecorder,
    StructuredOutputValidationError,
    _detect_silent_coercion,
    compute_structured_metrics,
)


class _FakeSchema(BaseModel):
    value: bool


class _FakeRawRunnable:
    """Stands in for real_llm.with_structured_output(schema, include_raw=True)."""

    def __init__(self, results: list[dict]):
        self._results = list(results)
        self.n_calls = 0

    def invoke(self, inputs, **kwargs):
        result = self._results[self.n_calls] if self.n_calls < len(self._results) else self._results[-1]
        self.n_calls += 1
        return result


class _FakeRawMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


# --------------------------------------------------------------------------
# InstrumentedStructuredRunnable
# --------------------------------------------------------------------------


def test_valid_first_attempt_records_valid_not_retried():
    recorder = StructuredOutputRecorder()
    parsed = _FakeSchema(value=True)
    raw = _FakeRawRunnable([{"parsed": parsed, "parsing_error": None, "raw": _FakeRawMessage([])}])
    runnable = InstrumentedStructuredRunnable(raw, schema=_FakeSchema, node_label="test_node", recorder=recorder)

    result = runnable.invoke({"question": "x"})

    assert result is parsed
    assert raw.n_calls == 1
    assert len(recorder.events) == 1
    assert recorder.events[0] == StructuredCallEvent(node="test_node", valid=True, retried=False, coerced=None, error=None)


def test_fails_then_succeeds_on_retry_records_retried_true():
    recorder = StructuredOutputRecorder()
    parsed = _FakeSchema(value=True)
    raw = _FakeRawRunnable(
        [
            {"parsed": None, "parsing_error": "bad json", "raw": _FakeRawMessage([])},
            {"parsed": parsed, "parsing_error": None, "raw": _FakeRawMessage([])},
        ]
    )
    runnable = InstrumentedStructuredRunnable(raw, schema=_FakeSchema, node_label="test_node", recorder=recorder)

    result = runnable.invoke({"question": "x"})

    assert result is parsed
    assert raw.n_calls == 2
    assert recorder.events[0].valid is True
    assert recorder.events[0].retried is True


def test_fails_all_attempts_raises_and_records_invalid():
    recorder = StructuredOutputRecorder()
    raw = _FakeRawRunnable([{"parsed": None, "parsing_error": "still bad", "raw": _FakeRawMessage([])}])
    runnable = InstrumentedStructuredRunnable(raw, schema=_FakeSchema, node_label="test_node", recorder=recorder, max_attempts=2)

    try:
        runnable.invoke({"question": "x"})
        assert False, "expected StructuredOutputValidationError"
    except StructuredOutputValidationError:
        pass

    assert raw.n_calls == 2
    assert recorder.events[0].valid is False
    assert recorder.events[0].retried is True


def test_callable_via_prompt_pipe(monkeypatch):
    from langchain_core.prompts import ChatPromptTemplate

    recorder = StructuredOutputRecorder()
    parsed = _FakeSchema(value=True)
    raw = _FakeRawRunnable([{"parsed": parsed, "parsing_error": None, "raw": _FakeRawMessage([])}])
    runnable = InstrumentedStructuredRunnable(raw, schema=_FakeSchema, node_label="test_node", recorder=recorder)

    prompt = ChatPromptTemplate([("human", "Q: {question}")], input_variables=["question"])
    chain = prompt | runnable
    result = chain.invoke({"question": "hi"})
    assert result is parsed


# --------------------------------------------------------------------------
# _detect_silent_coercion
# --------------------------------------------------------------------------


def test_detect_silent_coercion_type_mismatch_flags_true():
    parsed = _FakeSchema(value=True)
    raw = _FakeRawMessage([{"args": {"value": "true"}}])  # raw was a string, parsed is bool
    assert _detect_silent_coercion(raw, parsed) is True


def test_detect_silent_coercion_matching_types_flags_false():
    parsed = _FakeSchema(value=True)
    raw = _FakeRawMessage([{"args": {"value": True}}])
    assert _detect_silent_coercion(raw, parsed) is False


def test_detect_silent_coercion_no_tool_calls_returns_none():
    parsed = _FakeSchema(value=True)
    raw = _FakeRawMessage([])
    assert _detect_silent_coercion(raw, parsed) is None


# --------------------------------------------------------------------------
# InstrumentingLLM
# --------------------------------------------------------------------------


class _FakeRealLLM:
    def __init__(self):
        self.structured_calls = []
        self.plain_calls = []

    def with_structured_output(self, schema, **kwargs):
        self.structured_calls.append((schema, kwargs))
        return _FakeRawRunnable([{"parsed": _FakeSchema(value=True), "parsing_error": None, "raw": _FakeRawMessage([])}])

    def invoke(self, *args, **kwargs):
        self.plain_calls.append(args)
        return "plain response"


def test_instrumenting_llm_with_structured_output_forces_include_raw():
    real = _FakeRealLLM()
    recorder = StructuredOutputRecorder()
    wrapped = InstrumentingLLM(real, recorder)

    result_runnable = wrapped.with_structured_output(_FakeSchema)
    assert isinstance(result_runnable, InstrumentedStructuredRunnable)
    assert real.structured_calls[0][1] == {"include_raw": True}


def test_instrumenting_llm_plain_invoke_passes_through_unwrapped():
    real = _FakeRealLLM()
    recorder = StructuredOutputRecorder()
    wrapped = InstrumentingLLM(real, recorder)

    result = wrapped.invoke("some prompt value")
    assert result == "plain response"
    assert len(recorder.events) == 0


def test_instrumenting_llm_getattr_delegates():
    class _Real:
        temperature = 0.7

        def with_structured_output(self, schema, **kwargs):
            raise NotImplementedError

        def invoke(self, *a, **k):
            raise NotImplementedError

    wrapped = InstrumentingLLM(_Real(), StructuredOutputRecorder())
    assert wrapped.temperature == 0.7


# --------------------------------------------------------------------------
# compute_structured_metrics
# --------------------------------------------------------------------------


def test_compute_structured_metrics_per_node_and_aggregate():
    events = [
        StructuredCallEvent(node="query_router", valid=True, retried=False, coerced=False, error=None),
        StructuredCallEvent(node="query_router", valid=False, retried=True, coerced=None, error="bad"),
        StructuredCallEvent(node="rewrite_query", valid=True, retried=False, coerced=True, error=None),
    ]
    result = compute_structured_metrics(events)

    assert result["per_node"]["query_router"]["n_calls"] == 2
    assert result["per_node"]["query_router"]["validity_rate"] == 0.5
    assert result["per_node"]["query_router"]["retry_rate"] == 0.5
    assert result["per_node"]["rewrite_query"]["validity_rate"] == 1.0
    assert result["per_node"]["rewrite_query"]["silent_coercion_rate"] == 1.0

    assert result["aggregate"]["n_calls"] == 3
    assert result["aggregate"]["validity_rate"] == 2 / 3
    assert result["malformed_to_misroute_rate"] is None


def test_compute_structured_metrics_malformed_to_misroute_rate():
    events = [StructuredCallEvent(node="query_router", valid=False, retried=True, coerced=None, error="bad")]
    misroute_context = [
        {"router_valid": False, "predicted_route": "websearch", "expected_route": "vectorstore"},
        {"router_valid": False, "predicted_route": "vectorstore", "expected_route": "vectorstore"},
        {"router_valid": True, "predicted_route": "chitchat", "expected_route": "chitchat"},
    ]
    result = compute_structured_metrics(events, misroute_context=misroute_context)
    # Only router_valid=False rows count; 1 of those 2 was an actual misroute.
    assert result["malformed_to_misroute_rate"] == 0.5


def test_compute_structured_metrics_silent_coercion_rate_none_when_all_undetermined():
    events = [StructuredCallEvent(node="query_router", valid=True, retried=False, coerced=None, error=None)]
    result = compute_structured_metrics(events)
    assert result["per_node"]["query_router"]["silent_coercion_rate"] is None
