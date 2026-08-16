"""Unit tests for eval/noise_floor.py's pure aggregation logic
(flatten_numeric_metrics, compute_noise_floor_stats). run_noise_floor()
itself (which calls eval.harness.run_eval() N times, real LLM/judge calls)
is exercised live, not here -- except for its --existing-results resume
path, which is pure file I/O + aggregation and is worth a mocked test
below (2026-08-15/16: this is the path that recovers a 5-run pass after
the background process running it was killed mid-run-3, an unexplained
but recurring interruption -- losing runs 1-2 to a from-scratch retry
would waste both wall time and already-spent Groq judge quota).
"""

import json
import math

from eval.noise_floor import compute_noise_floor_stats, flatten_numeric_metrics, run_noise_floor


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


def test_run_noise_floor_resumes_from_existing_results(monkeypatch, tmp_path):
    """--existing-results reuses already-completed run JSONs (from a pass
    interrupted mid-run) instead of re-running them, and only calls
    run_eval() for the remaining runs to reach --runs total."""
    existing_paths = []
    for i, run_id in enumerate(["run1", "run2"], start=1):
        p = tmp_path / f"existing_{i}.json"
        p.write_text(json.dumps({"run_id": run_id, "metrics": {"router": {"accuracy": 0.8 + i * 0.01}}}), encoding="utf-8")
        existing_paths.append(p)

    calls = []

    def fake_run_eval(*, config_path, split, retrieval_only, results_dir):
        calls.append(split)
        run_id = f"live{len(calls)}"
        results = {"run_id": run_id, "metrics": {"router": {"accuracy": 0.9}}}
        out_path = results_dir / f"{run_id}.json"
        return results, out_path

    monkeypatch.setattr("eval.noise_floor.run_eval", fake_run_eval)

    summary, out_path = run_noise_floor(
        config_path=tmp_path / "config.yaml",
        split="fast",
        n_runs=5,
        results_dir=tmp_path,
        out_dir=tmp_path / "noise_floor",
        existing_result_paths=existing_paths,
    )

    assert len(calls) == 3  # only the remaining 3 runs, not all 5
    assert summary["run_ids"] == ["run1", "run2", "live1", "live2", "live3"]
    assert summary["stats"]["router.accuracy"]["n_runs_with_data"] == 5
    assert out_path.exists()
