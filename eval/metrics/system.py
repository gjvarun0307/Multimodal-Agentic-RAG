"""System-level metrics: latency percentiles (warm-up excluded per
PROJECT_SPEC.md §4B.3), per-stage latency, tokens, cost, error rate by
stage, and the trace-derivable half of Self-RAG correction (fire_rate +
mean_retries -- correction_improve_rate/degrade_rate need a judge-scored
correctness signal from eval/metrics/generation.py and are only computable
once eval.harness passes that in, not here).

Nothing in this module makes an LLM or network call; every function is a
pure aggregation over per-item records the harness already collected from
agent.run_query_with_state()'s trace_info. Callers are responsible for
excluding warm-up queries before calling anything here -- this module has
no concept of "warm-up", it only ever sees what it's given (spec §4B.3:
"warm-up queries are excluded from all latency statistics").

Known granularity gap (Phase 2 -> Phase 5): trace_info["stage_latencies_ms"]
is keyed by LangGraph node name (retrieve_and_rerank, generate,
rewrite_query, web_search, chitchat), not the finer embed/retrieve/rerank/
route/generate/correct breakdown the spec's per-stage OTLP tracing targets
(§5, "One span per stage"). retrieve_and_rerank in particular bundles
embed+retrieve+rerank into one timing; query_router's own decision time is
folded into whichever node runs first, since it's a conditional edge, not
a node. per_stage_mean_ms below reports exactly what's available today --
Phase 5's OTLP spans are where the finer split gets built, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class SystemItem:
    id: str
    total_latency_ms: Optional[float] = None
    stage_latencies_ms: dict = field(default_factory=dict)  # node_name -> ms, from trace_info
    node_sequence: list = field(default_factory=list)  # trace_info["node_sequence"]
    fallback_events: list = field(default_factory=list)  # trace_info["fallback_events"]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Linear-interpolation percentile (numpy's default 'linear' method),
    p in [0, 100]. None for an empty input -- never 0.0, which would be a
    real (and wrong) value."""
    if not values:
        return None
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    rank = (p / 100) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def compute_latency_metrics(items: Sequence[SystemItem]) -> dict:
    totals = [it.total_latency_ms for it in items if it.total_latency_ms is not None]
    per_stage: dict[str, list[float]] = {}
    for it in items:
        for stage, ms in it.stage_latencies_ms.items():
            per_stage.setdefault(stage, []).append(ms)

    return {
        "n_items": len(items),
        "warm_p50_ms": percentile(totals, 50),
        "warm_p95_ms": percentile(totals, 95),
        "warm_p99_ms": percentile(totals, 99),
        "per_stage_mean_ms": {stage: _mean(values) for stage, values in per_stage.items()},
    }


def compute_token_and_cost_metrics(items: Sequence[SystemItem]) -> dict:
    prompt_tokens = [it.prompt_tokens for it in items if it.prompt_tokens is not None]
    completion_tokens = [it.completion_tokens for it in items if it.completion_tokens is not None]
    costs = [it.cost_usd for it in items if it.cost_usd is not None]

    mean_cost = _mean(costs)
    return {
        "mean_prompt_tokens": _mean(prompt_tokens),
        "mean_completion_tokens": _mean(completion_tokens),
        "cost_per_query_usd": mean_cost,
        "cost_per_1k_queries_usd": (mean_cost * 1000) if mean_cost is not None else None,
    }


def compute_error_rate_by_stage(items: Sequence[SystemItem]) -> dict:
    """A "stage error" is any fallback_events tag recorded for that item --
    invariant 15's loud-fallback tags (rerank_fallback, rewrite_error,
    web_search_error, chitchat_error, graph_execution_error) are exactly
    this signal already surfaced in trace_info, reused here rather than
    re-derived."""
    n = len(items)
    tag_counts: dict[str, int] = {}
    for it in items:
        for tag in it.fallback_events:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return {
        "n_items": n,
        "error_rate_by_tag": ({tag: count / n for tag, count in tag_counts.items()} if n else {}),
    }


def compute_correction_metrics(
    items: Sequence[SystemItem], *, correctness_deltas: Optional[Sequence[Optional[float]]] = None
) -> dict:
    """correction_fire_rate and mean_retries are trace-derivable alone:
    fires whenever rewrite_query appears in node_sequence or generate runs
    more than once (same definition src/api.py's correction_fired already
    uses). correction_improve_rate/degrade_rate need a judge-scored
    correctness delta per item (positive = improved, negative = degraded,
    0 = unchanged) -- pass `correctness_deltas` (same length and order as
    `items`; None entries for items with no delta available, e.g.
    correction didn't fire) once eval.harness has judge output. Without
    it, both rates are None, not 0.0 -- "not measured" and "correction
    never helped" are different claims.
    """
    fired = [("rewrite_query" in it.node_sequence) or (it.node_sequence.count("generate") > 1) for it in items]
    n = len(items)
    n_fired = sum(fired)
    retries = [it.node_sequence.count("rewrite_query") for it in items]

    improve_rate = None
    degrade_rate = None
    if correctness_deltas is not None:
        if len(correctness_deltas) != len(items):
            raise ValueError("correctness_deltas must be the same length as items")
        fired_deltas = [d for d, f in zip(correctness_deltas, fired) if f and d is not None]
        if fired_deltas:
            improve_rate = sum(1 for d in fired_deltas if d > 0) / len(fired_deltas)
            degrade_rate = sum(1 for d in fired_deltas if d < 0) / len(fired_deltas)

    return {
        "n_items": n,
        "fire_rate": (n_fired / n) if n else None,
        "mean_retries": _mean(retries),
        "improve_rate": improve_rate,
        "degrade_rate": degrade_rate,
    }
