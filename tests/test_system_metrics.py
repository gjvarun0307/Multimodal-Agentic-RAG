"""Unit tests for eval/metrics/system.py's latency/token/cost/error-rate/
correction aggregation. Pure functions over SystemItem records -- no graph
execution, no LLM.
"""

from eval.metrics.system import (
    SystemItem,
    compute_correction_metrics,
    compute_error_rate_by_stage,
    compute_latency_metrics,
    compute_token_and_cost_metrics,
    percentile,
)


def test_percentile_empty_is_none():
    assert percentile([], 50) is None


def test_percentile_single_value():
    assert percentile([42.0], 95) == 42.0


def test_percentile_matches_known_linear_interpolation():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 0) == 10
    assert percentile(values, 100) == 50
    assert percentile(values, 50) == 30


def test_percentile_rejects_out_of_range():
    try:
        percentile([1, 2], 150)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_compute_latency_metrics_percentiles_and_per_stage():
    items = [
        SystemItem(id="1", total_latency_ms=1000, stage_latencies_ms={"retrieve_and_rerank": 400, "generate": 600}),
        SystemItem(id="2", total_latency_ms=2000, stage_latencies_ms={"retrieve_and_rerank": 800, "generate": 1200}),
    ]
    result = compute_latency_metrics(items)
    assert result["n_items"] == 2
    assert result["warm_p50_ms"] == 1500.0
    assert result["per_stage_mean_ms"]["retrieve_and_rerank"] == 600.0
    assert result["per_stage_mean_ms"]["generate"] == 900.0


def test_compute_latency_metrics_missing_totals_excluded_not_zeroed():
    items = [SystemItem(id="1", total_latency_ms=None), SystemItem(id="2", total_latency_ms=500)]
    result = compute_latency_metrics(items)
    assert result["warm_p50_ms"] == 500.0


def test_compute_token_and_cost_metrics():
    items = [
        SystemItem(id="1", prompt_tokens=100, completion_tokens=50, cost_usd=0.002),
        SystemItem(id="2", prompt_tokens=200, completion_tokens=100, cost_usd=0.004),
    ]
    result = compute_token_and_cost_metrics(items)
    assert result["mean_prompt_tokens"] == 150.0
    assert result["mean_completion_tokens"] == 75.0
    assert result["cost_per_query_usd"] == 0.003
    assert abs(result["cost_per_1k_queries_usd"] - 3.0) < 1e-9


def test_compute_error_rate_by_stage():
    items = [
        SystemItem(id="1", fallback_events=["rerank_fallback"]),
        SystemItem(id="2", fallback_events=[]),
        SystemItem(id="3", fallback_events=["rerank_fallback", "rewrite_error"]),
        SystemItem(id="4", fallback_events=[]),
    ]
    result = compute_error_rate_by_stage(items)
    assert result["n_items"] == 4
    assert result["error_rate_by_tag"]["rerank_fallback"] == 0.5
    assert result["error_rate_by_tag"]["rewrite_error"] == 0.25


def test_compute_error_rate_by_stage_empty_items():
    assert compute_error_rate_by_stage([]) == {"n_items": 0, "error_rate_by_tag": {}}


def test_compute_correction_metrics_fire_rate_and_mean_retries():
    items = [
        SystemItem(id="1", node_sequence=["retrieve_and_rerank", "generate"]),  # no correction
        SystemItem(id="2", node_sequence=["retrieve_and_rerank", "generate", "rewrite_query", "retrieve_and_rerank", "generate"]),
        SystemItem(id="3", node_sequence=["chitchat"]),
    ]
    result = compute_correction_metrics(items)
    assert result["n_items"] == 3
    assert result["fire_rate"] == 1 / 3
    assert result["mean_retries"] == 1 / 3  # only item 2 has one rewrite_query
    assert result["improve_rate"] is None
    assert result["degrade_rate"] is None


def test_compute_correction_metrics_with_deltas():
    items = [
        SystemItem(id="1", node_sequence=["retrieve_and_rerank", "generate", "rewrite_query", "generate"]),
        SystemItem(id="2", node_sequence=["retrieve_and_rerank", "generate", "rewrite_query", "generate"]),
        SystemItem(id="3", node_sequence=["retrieve_and_rerank", "generate"]),  # no correction fired
    ]
    # item 1 improved, item 2 degraded, item 3's delta must be ignored (didn't fire)
    deltas = [0.3, -0.2, 0.5]
    result = compute_correction_metrics(items, correctness_deltas=deltas)
    assert result["improve_rate"] == 0.5
    assert result["degrade_rate"] == 0.5


def test_compute_correction_metrics_deltas_length_mismatch_raises():
    items = [SystemItem(id="1", node_sequence=["retrieve_and_rerank", "generate"])]
    try:
        compute_correction_metrics(items, correctness_deltas=[])
        assert False, "expected ValueError"
    except ValueError:
        pass
