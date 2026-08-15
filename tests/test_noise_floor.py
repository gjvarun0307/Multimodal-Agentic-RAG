"""Unit tests for eval/noise_floor.py's pure aggregation logic
(flatten_numeric_metrics, compute_noise_floor_stats). run_noise_floor()
itself (which calls eval.harness.run_eval() N times, real LLM/judge calls)
is exercised live, not here.
"""

import math

from eval.noise_floor import compute_noise_floor_stats, flatten_numeric_metrics


def test_flatten_numeric_metrics_nested_dicts():
    metrics = {
        "retrieval": {"stage1": {"recall@10": 0.5, "recall@20": 0.6}, "rerank_lift": -0.02},
        "router": {"accuracy": 0.84},
    }
    flat = flatten_numeric_metrics(metrics)
    assert flat["retrieval.stage1.recall@10"] == 0.5
    assert flat["retrieval.stage1.recall@20"] == 0.6
    assert flat["retrieval.rerank_lift"] == -0.02
    assert flat["router.accuracy"] == 0.84


def test_flatten_numeric_metrics_keeps_none_skips_non_numeric():
    metrics = {
        "generation": {"faithfulness": {"rate": None}},
        "backend": "byok-hosted",  # string -- should be dropped
        "per_item_ids": ["gs_0001"],  # list -- should be dropped
        "warmup_excluded": True,  # bool -- should be dropped
    }
    flat = flatten_numeric_metrics(metrics)
    assert flat["generation.faithfulness.rate"] is None
    assert "backend" not in flat
    assert "per_item_ids" not in flat
    assert "warmup_excluded" not in flat


def test_compute_noise_floor_stats_basic():
    runs = [
        {"router.accuracy": 0.80},
        {"router.accuracy": 0.85},
        {"router.accuracy": 0.90},
    ]
    stats = compute_noise_floor_stats(runs)
    s = stats["router.accuracy"]
    assert s["n_runs_with_data"] == 3
    assert math.isclose(s["mean"], 0.85)
    assert s["min"] == 0.80
    assert s["max"] == 0.90
    assert math.isclose(s["range"], 0.10)
    assert s["stdev"] > 0
    assert math.isclose(s["threshold_2sigma"], 2 * s["stdev"])


def test_compute_noise_floor_stats_excludes_none_not_zero():
    runs = [
        {"generation.faithfulness.rate": 1.0},
        {"generation.faithfulness.rate": None},  # e.g. judge rate-limited this run
        {"generation.faithfulness.rate": 0.9},
    ]
    stats = compute_noise_floor_stats(runs)
    s = stats["generation.faithfulness.rate"]
    assert s["n_runs_with_data"] == 2
    assert math.isclose(s["mean"], 0.95)  # (1.0 + 0.9) / 2, not /3


def test_compute_noise_floor_stats_all_none_reports_zero_coverage():
    runs = [{"x": None}, {"x": None}]
    stats = compute_noise_floor_stats(runs)
    assert stats["x"]["n_runs_with_data"] == 0
    assert stats["x"]["mean"] is None
    assert stats["x"]["threshold_2sigma"] is None


def test_compute_noise_floor_stats_single_value_zero_stdev():
    runs = [{"x": 0.5}]
    stats = compute_noise_floor_stats(runs)
    assert stats["x"]["n_runs_with_data"] == 1
    assert stats["x"]["stdev"] == 0.0
    assert stats["x"]["range"] == 0.0


def test_compute_noise_floor_stats_union_of_keys_across_runs():
    # A metric present in only some runs (e.g. a category with zero items
    # in one run's sample) still gets a stats entry, scoped to the runs
    # that actually had it.
    runs = [{"a": 1.0, "b": 2.0}, {"a": 1.5}]
    stats = compute_noise_floor_stats(runs)
    assert stats["a"]["n_runs_with_data"] == 2
    assert stats["b"]["n_runs_with_data"] == 1
