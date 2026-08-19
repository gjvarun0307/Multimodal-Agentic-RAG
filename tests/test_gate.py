"""Unit tests for eval/gate.py's pure diff/verdict logic (CLAUDE.md Phase 4
"Gate-comparison script" checklist item). No CLI/subprocess invocation --
evaluate_gate/overall_verdict/render_pr_comment are called directly against
synthetic results dicts shaped like real eval.harness output."""

from eval.gate import (
    BLOCK,
    INFO,
    INSUFFICIENT_DATA,
    NO_BASELINE,
    NOT_RUN,
    PASS,
    RATE_LIMITED,
    WARN,
    evaluate_gate,
    overall_verdict,
    render_pr_comment,
)


def _run(metrics: dict) -> dict:
    return {"run_id": "test_run", "git_sha": "abc123", "metrics": metrics}


def test_recall_drop_within_noise_floor_passes():
    baseline = _run({"retrieval": {"stage1": {"recall@10": 0.50}}})
    current = _run({"retrieval": {"stage1": {"recall@10": 0.48}}})  # -0.02, under 0.045
    results = evaluate_gate(current, baseline)
    recall = next(r for r in results if r.label == "recall@10")
    assert recall.status == PASS


def test_recall_drop_beyond_threshold_blocks():
    baseline = _run({"retrieval": {"stage1": {"recall@10": 0.50}}})
    current = _run({"retrieval": {"stage1": {"recall@10": 0.40}}})  # -0.10, over 0.045
    results = evaluate_gate(current, baseline)
    recall = next(r for r in results if r.label == "recall@10")
    assert recall.status == BLOCK
    assert overall_verdict(results) == BLOCK


def test_recall_improvement_passes():
    baseline = _run({"retrieval": {"stage1": {"recall@10": 0.50}}})
    current = _run({"retrieval": {"stage1": {"recall@10": 0.65}}})
    results = evaluate_gate(current, baseline)
    recall = next(r for r in results if r.label == "recall@10")
    assert recall.status == PASS


def test_router_accuracy_drop_warns_not_blocks():
    baseline = _run({"retrieval": {"stage1": {"recall@10": 0.50}}, "router": {"accuracy": 0.80}})
    current = _run({"retrieval": {"stage1": {"recall@10": 0.50}}, "router": {"accuracy": 0.70}})  # -0.10
    results = evaluate_gate(current, baseline)
    router = next(r for r in results if r.label == "router.accuracy")
    assert router.status == WARN
    assert overall_verdict(results) == WARN


def test_structured_validity_any_drop_warns():
    baseline = _run({"structured": {"aggregate": {"validity_rate": 1.0}}})
    current = _run({"structured": {"aggregate": {"validity_rate": 0.99}}})
    results = evaluate_gate(current, baseline)
    validity = next(r for r in results if r.label == "structured.validity_rate")
    assert validity.status == WARN


def test_warm_p95_latency_increase_beyond_20pct_warns():
    baseline = _run({"system": {"warm_p95_ms": 100_000}})
    current = _run({"system": {"warm_p95_ms": 130_000}})  # +30%
    results = evaluate_gate(current, baseline)
    latency = next(r for r in results if r.label == "system.warm_p95_ms")
    assert latency.status == WARN


def test_warm_p95_latency_increase_within_20pct_passes():
    baseline = _run({"system": {"warm_p95_ms": 100_000}})
    current = _run({"system": {"warm_p95_ms": 110_000}})  # +10%
    results = evaluate_gate(current, baseline)
    latency = next(r for r in results if r.label == "system.warm_p95_ms")
    assert latency.status == PASS


def test_metric_missing_from_current_run_is_not_run_not_a_fabricated_pass():
    # Fast-tier (--retrieval-only) runs have no router/structured/system metrics at all.
    baseline = _run({"retrieval": {"stage1": {"recall@10": 0.50}}, "router": {"accuracy": 0.80}})
    current = _run({"retrieval": {"stage1": {"recall@10": 0.50}}})  # router metrics absent
    results = evaluate_gate(current, baseline)
    router = next(r for r in results if r.label == "router.accuracy")
    assert router.status == NOT_RUN
    assert overall_verdict(results) == PASS  # not_run never blocks or warns


def test_metric_missing_from_baseline_is_no_baseline():
    baseline = _run({})  # eval/baselines/main.json doesn't exist yet, or lacks this metric
    current = _run({"retrieval": {"stage1": {"recall@10": 0.50}}})
    results = evaluate_gate(current, baseline)
    recall = next(r for r in results if r.label == "recall@10")
    assert recall.status == NO_BASELINE
    assert overall_verdict(results) == PASS


def test_info_metrics_never_gate_even_on_large_drop():
    baseline = _run(
        {
            "generation": {
                "answer_correctness": {"rate": 0.90, "n_scored": 10},
            }
        }
    )
    current = _run(
        {
            "generation": {
                "answer_correctness": {"rate": 0.10, "n_scored": 10},
            }
        }
    )
    results = evaluate_gate(current, baseline)
    correctness = next(r for r in results if r.label == "answer_correctness")
    assert correctness.status == INFO
    assert overall_verdict(results) == PASS


def test_info_metric_with_zero_n_scored_is_insufficient_data_not_a_bogus_delta():
    baseline = _run({"generation": {"faithfulness": {"rate": 0.96, "n_scored": 1}}})
    current = _run({"generation": {"faithfulness": {"rate": None, "n_scored": 0}}})
    results = evaluate_gate(current, baseline)
    faithfulness = next(r for r in results if r.label == "faithfulness")
    assert faithfulness.status == INSUFFICIENT_DATA
    assert faithfulness.delta is None


def test_chunk_id_determinism_row_added_only_when_flag_passed():
    baseline = _run({})
    current = _run({})

    no_flag = evaluate_gate(current, baseline, determinism_passed=None)
    assert not any(r.label == "chunk_id_determinism" for r in no_flag)

    passed = evaluate_gate(current, baseline, determinism_passed=True)
    det = next(r for r in passed if r.label == "chunk_id_determinism")
    assert det.status == PASS

    failed = evaluate_gate(current, baseline, determinism_passed=False)
    det = next(r for r in failed if r.label == "chunk_id_determinism")
    assert det.status == BLOCK
    assert overall_verdict(failed) == BLOCK


def test_block_outranks_warn_in_overall_verdict():
    baseline = _run({"retrieval": {"stage1": {"recall@10": 0.50}}, "router": {"accuracy": 0.80}})
    current = _run({"retrieval": {"stage1": {"recall@10": 0.30}}, "router": {"accuracy": 0.70}})
    results = evaluate_gate(current, baseline)
    assert overall_verdict(results) == BLOCK


def test_render_pr_comment_includes_block_header_and_status_emoji():
    baseline = _run({"retrieval": {"stage1": {"recall@10": 0.50}}})
    current = _run({"retrieval": {"stage1": {"recall@10": 0.30}}})
    results = evaluate_gate(current, baseline)
    verdict = overall_verdict(results)
    comment = render_pr_comment(results, verdict)
    assert "BLOCKED" in comment
    assert "recall@10" in comment
    assert "❌ FAIL" in comment


def test_router_accuracy_drop_is_rate_limited_not_warn_when_run_was_quota_constrained():
    # Same -0.10 drop as test_router_accuracy_drop_warns_not_blocks, but this
    # run's error_rate_by_tag shows real rate-limiting -- the numbers are
    # unreliable, not a proven regression, so it must not render as WARN.
    baseline = _run({"retrieval": {"stage1": {"recall@10": 0.50}}, "router": {"accuracy": 0.80}})
    current = _run(
        {
            "retrieval": {"stage1": {"recall@10": 0.50}},
            "router": {"accuracy": 0.70},
            "system": {"error_rate_by_stage": {"error_rate_by_tag": {"rate_limited": 0.15}}},
        }
    )
    results = evaluate_gate(current, baseline)
    router = next(r for r in results if r.label == "router.accuracy")
    assert router.status == RATE_LIMITED
    assert overall_verdict(results) == PASS  # a quota blip must never render as a red FAIL


def test_metric_not_rate_limit_sensitive_still_warns_during_a_rate_limited_run():
    # system.warm_p95_ms isn't LLM-driven (rate_limit_sensitive=False) --
    # a rate-limited run elsewhere shouldn't mask a real latency regression.
    baseline = _run({"system": {"warm_p95_ms": 100_000}})
    current = _run(
        {
            "system": {
                "warm_p95_ms": 130_000,  # +30%
                "error_rate_by_stage": {"error_rate_by_tag": {"rate_limited": 0.15}},
            }
        }
    )
    results = evaluate_gate(current, baseline)
    latency = next(r for r in results if r.label == "system.warm_p95_ms")
    assert latency.status == WARN


def test_router_accuracy_drop_still_warns_when_run_was_not_rate_limited():
    baseline = _run({"router": {"accuracy": 0.80}})
    current = _run(
        {"router": {"accuracy": 0.70}, "system": {"error_rate_by_stage": {"error_rate_by_tag": {}}}}
    )
    results = evaluate_gate(current, baseline)
    router = next(r for r in results if r.label == "router.accuracy")
    assert router.status == WARN


def test_info_metric_low_n_scored_from_rate_limiting_reports_rate_limited_not_insufficient_data():
    baseline = _run({"generation": {"faithfulness": {"rate": 0.96, "n_scored": 10, "n_rate_limited": 0}}})
    current = _run({"generation": {"faithfulness": {"rate": 0.50, "n_scored": 2, "n_rate_limited": 8}}})
    results = evaluate_gate(current, baseline)
    faithfulness = next(r for r in results if r.label == "faithfulness")
    assert faithfulness.status == RATE_LIMITED


def test_info_metric_low_n_scored_without_rate_limiting_stays_insufficient_data():
    baseline = _run({"generation": {"faithfulness": {"rate": 0.96, "n_scored": 1}}})
    current = _run({"generation": {"faithfulness": {"rate": None, "n_scored": 0, "n_rate_limited": 0}}})
    results = evaluate_gate(current, baseline)
    faithfulness = next(r for r in results if r.label == "faithfulness")
    assert faithfulness.status == INSUFFICIENT_DATA
