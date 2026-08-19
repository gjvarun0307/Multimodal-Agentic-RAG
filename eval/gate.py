"""python -m eval.gate --run eval/results/<run_id>.json --baseline eval/baselines/main.json
    [--determinism-passed true|false] [--out gate_report.json]

Diffs an eval.harness results JSON against the committed CI baseline
(eval/baselines/main.json) per the gate table in CLAUDE.md's "Phase 4 --
Eval-as-CI" section (thresholds sourced from docs/NOISE_FLOOR.md, fast
split, n=4 -- these deliberately override PROJECT_SPEC.md §7's literal gate
table, which blocks on faithfulness/refusal_accuracy that Phase 3 found
unusable at n=1 or unusable sigma).

Emits a per-metric pass/warn/block verdict, renders the PR-comment diff
table (PROJECT_SPEC.md §7 Phase 4 example), and exits non-zero iff any
metric actually hit its block threshold.

Metrics absent from the *current* run are reported "not run this tier",
never treated as a silent 0 -- a --retrieval-only fast-tier run has no
router/structured/system/generation metrics at all (eval.harness's own
docstring), so most of the gate table simply doesn't apply there yet; those
rows only activate on full-tier (nightly) runs. Metrics absent from the
baseline are reported "no baseline yet", which is the expected state until
eval/baselines/main.json is first committed from a real run (separate,
not-yet-done checklist item as of 2026-08-18) -- invariant 15, no silent
fallback to a fabricated pass.

chunk_id_determinism is enforced by CI as its own hard pytest step
(tests/test_chunk_id_determinism.py), not computed by eval.harness, so it
isn't in any results JSON. --determinism-passed lets a caller (the CI
workflow, from that step's exit code) fold it into this script's combined
report/PR-comment table; omit the flag to leave that row out entirely
rather than guessing.

Judge-scored generation metrics (faithfulness, citation_precision,
refusal_accuracy, answer_correctness) are always informational-only, per
docs/NOISE_FLOOR.md's finding that their noise floor isn't measurable yet
-- shown in the PR comment for visibility, never gated. n_scored is checked
per informational metric so a Groq-quota-starved run (small n_scored) reads
as "insufficient data" by default; when eval.harness's own
n_rate_limited count (src.helper.is_rate_limit_error-detected 429s, see
eval/judge.py's JudgeRateLimitError) shows that quota, not some other
cause, is why n_scored is low, this reports "rate-limited" instead --
a more specific reason, not a different meaning.

Rate-limit awareness also covers the *gated* full-tier metrics that are
themselves LLM-driven (structured.validity_rate, correction.fire_rate,
router.accuracy): src.agent.run_query_with_state tags a rate-limited item
with the "rate_limited" fallback_events tag distinctly from the generic
"graph_execution_error" catch-all (src/agent.py), which flows into
system.error_rate_by_stage.error_rate_by_tag.rate_limited automatically
(eval/metrics/system.py's compute_error_rate_by_stage, no gate-specific
plumbing needed). When that rate is nonzero and a metric would otherwise
warn/block, this script reports "rate-limited" instead -- a quota blip
must never render as a red ❌ FAIL (PROJECT_SPEC.md §7 Phase 4 acceptance
criterion).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from eval.noise_floor import flatten_numeric_metrics

DEFAULT_BASELINE_PATH = Path("eval/baselines/main.json")

PASS, WARN, BLOCK, NOT_RUN, NO_BASELINE, INFO, INSUFFICIENT_DATA, RATE_LIMITED = (
    "pass",
    "warn",
    "block",
    "not_run",
    "no_baseline",
    "info",
    "insufficient_data",
    "rate_limited",
)

STATUS_EMOJI = {
    PASS: "✅",
    WARN: "⚠️ WARN",
    BLOCK: "❌ FAIL",
    NOT_RUN: "⏭️ not run this tier",
    NO_BASELINE: "\U0001f195 no baseline",
    INFO: "ℹ️ info",
    INSUFFICIENT_DATA: "ℹ️ insufficient data",
    RATE_LIMITED: "⏳ rate-limited (not a regression)",
}

# system.error_rate_by_stage.error_rate_by_tag.rate_limited -- the fraction
# of full-tier items whose LLM call(s) hit a provider 429 (see module
# docstring). Any nonzero value is enough to suppress a warn/block verdict
# on a rate_limit_sensitive metric: a partial-quota run already means that
# metric's numbers are unreliable, not that a lower-but-nonzero rate should
# still be trusted as a real signal.
RATE_LIMITED_TAG_KEY = "system.error_rate_by_stage.error_rate_by_tag.rate_limited"


@dataclass(frozen=True)
class GateMetric:
    key: str  # dot path under the run JSON's "metrics" dict, e.g. "retrieval.stage1.recall@10"
    label: str  # short display name for the PR comment
    direction: str  # "higher_better" or "lower_better"
    mode: str  # "abs_drop" or "pct_change"
    threshold: float  # magnitude that trips the action (never negative)
    action: str  # "block" or "warn" -- what a breach becomes
    tier_note: Optional[str] = None
    n_scored_key: Optional[str] = None  # companion sample-size key, for insufficient-data checks
    n_rate_limited_key: Optional[str] = None  # companion rate-limited count, for rate-limited checks
    rate_limit_sensitive: bool = False  # True if this metric's own LLM call can be rate-limited


GATE_METRICS: list[GateMetric] = [
    GateMetric(
        key="retrieval.stage1.recall@10",
        label="recall@10",
        direction="higher_better",
        mode="abs_drop",
        threshold=0.045,
        action=BLOCK,
    ),
    GateMetric(
        key="router.accuracy",
        label="router.accuracy",
        direction="higher_better",
        mode="abs_drop",
        threshold=0.03,
        action=WARN,
        tier_note="full tier only",
        rate_limit_sensitive=True,
    ),
    GateMetric(
        key="structured.aggregate.validity_rate",
        label="structured.validity_rate",
        direction="higher_better",
        mode="abs_drop",
        threshold=0.0,  # "any drop from 1.000"
        action=WARN,
        tier_note="full tier only",
        rate_limit_sensitive=True,
    ),
    GateMetric(
        key="correction.fire_rate",
        label="correction.fire_rate",
        direction="higher_better",
        mode="abs_drop",
        threshold=0.07,
        action=WARN,
        tier_note="full tier only",
        rate_limit_sensitive=True,
    ),
    GateMetric(
        key="system.warm_p95_ms",
        label="system.warm_p95_ms",
        direction="lower_better",
        mode="pct_change",
        threshold=0.20,
        action=WARN,
        tier_note="full tier only",
    ),
]

# label -> (metric key, companion n_scored key). Always informational, never gated.
INFO_METRICS: list[GateMetric] = [
    GateMetric(
        key="generation.faithfulness.rate",
        label="faithfulness",
        direction="higher_better",
        mode="abs_drop",
        threshold=0.0,
        action=INFO,
        n_scored_key="generation.faithfulness.n_scored",
        n_rate_limited_key="generation.faithfulness.n_rate_limited",
    ),
    GateMetric(
        key="generation.citation_precision.mean",
        label="citation_precision",
        direction="higher_better",
        mode="abs_drop",
        threshold=0.0,
        action=INFO,
        n_scored_key="generation.citation_precision.n_scored",
        n_rate_limited_key="generation.citation_precision.n_rate_limited",
    ),
    GateMetric(
        key="generation.refusal_accuracy.rate",
        label="refusal_accuracy",
        direction="higher_better",
        mode="abs_drop",
        threshold=0.0,
        action=INFO,
        n_scored_key="generation.refusal_accuracy.n_scored",
        n_rate_limited_key="generation.refusal_accuracy.n_rate_limited",
    ),
    GateMetric(
        key="generation.answer_correctness.rate",
        label="answer_correctness",
        direction="higher_better",
        mode="abs_drop",
        threshold=0.0,
        action=INFO,
        n_scored_key="generation.answer_correctness.n_scored",
        n_rate_limited_key="generation.answer_correctness.n_rate_limited",
    ),
]


@dataclass(frozen=True)
class GateResult:
    label: str
    baseline: Optional[float]
    current: Optional[float]
    delta: Optional[float]
    status: str
    tier_note: Optional[str] = None


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _evaluate_one(
    metric: GateMetric, current_flat: dict[str, Any], baseline_flat: dict[str, Any], *, rate_limited_frac: float = 0.0
) -> GateResult:
    current = current_flat.get(metric.key)
    baseline = baseline_flat.get(metric.key)

    # Checked before the n_scored/insufficient-data path: a nonzero
    # n_rate_limited on the current run means low n_scored (or a skewed
    # rate/mean) is explained by quota, not some other cause -- a more
    # specific status than "insufficient data" (module docstring).
    if metric.n_rate_limited_key is not None and (current_flat.get(metric.n_rate_limited_key) or 0):
        return GateResult(metric.label, baseline, current, None, RATE_LIMITED, metric.tier_note)

    if metric.n_scored_key is not None:
        current_n = current_flat.get(metric.n_scored_key) or 0
        baseline_n = baseline_flat.get(metric.n_scored_key) or 0
        if not current_n or not baseline_n:
            return GateResult(metric.label, baseline, current, None, INSUFFICIENT_DATA, metric.tier_note)

    if current is None:
        return GateResult(metric.label, baseline, current, None, NOT_RUN, metric.tier_note)
    if metric.key not in baseline_flat or baseline is None:
        return GateResult(metric.label, baseline, current, None, NO_BASELINE, metric.tier_note)

    delta = current - baseline

    if metric.action == INFO:
        return GateResult(metric.label, baseline, current, delta, INFO, metric.tier_note)

    if metric.mode == "abs_drop":
        drop = -delta if metric.direction == "higher_better" else delta
        breached = drop > metric.threshold
    elif metric.mode == "pct_change":
        if baseline == 0:
            breached = current != baseline
        else:
            pct_change = (current - baseline) / abs(baseline)
            # lower_better (e.g. latency): a regression is an increase.
            # higher_better: a regression is a decrease.
            breached = (
                pct_change > metric.threshold if metric.direction == "lower_better" else -pct_change > metric.threshold
            )
    else:
        raise ValueError(f"unknown gate mode: {metric.mode}")

    # A breach on a rate_limit_sensitive metric during a run that shows any
    # rate-limited items is unreliable, not a proven regression -- the
    # metric's own numbers may be undercounted by the same quota blip
    # (module docstring). Report it distinctly rather than as WARN/BLOCK.
    if breached and metric.rate_limit_sensitive and rate_limited_frac > 0:
        return GateResult(metric.label, baseline, current, delta, RATE_LIMITED, metric.tier_note)

    status = metric.action if breached else PASS
    return GateResult(metric.label, baseline, current, delta, status, metric.tier_note)


def evaluate_gate(
    current_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    determinism_passed: Optional[bool] = None,
) -> list[GateResult]:
    """current_metrics/baseline_metrics: each is a full results-JSON dict
    (with a top-level "metrics" key) OR an already-flattened metrics dict --
    flattening is idempotent-safe here since we only ever read via .get()."""
    current_flat = flatten_numeric_metrics(current_metrics.get("metrics", current_metrics))
    baseline_flat = flatten_numeric_metrics(baseline_metrics.get("metrics", baseline_metrics))
    rate_limited_frac = current_flat.get(RATE_LIMITED_TAG_KEY) or 0.0

    results: list[GateResult] = []
    if determinism_passed is not None:
        status = PASS if determinism_passed else BLOCK
        results.append(GateResult("chunk_id_determinism", None, None, None, status))

    for metric in GATE_METRICS:
        results.append(_evaluate_one(metric, current_flat, baseline_flat, rate_limited_frac=rate_limited_frac))
    for metric in INFO_METRICS:
        results.append(_evaluate_one(metric, current_flat, baseline_flat, rate_limited_frac=rate_limited_frac))

    return results


def overall_verdict(results: list[GateResult]) -> str:
    if any(r.status == BLOCK for r in results):
        return BLOCK
    if any(r.status == WARN for r in results):
        return WARN
    return PASS


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}" if abs(value) < 1000 else f"{value:.1f}"


def render_pr_comment(results: list[GateResult], verdict: str) -> str:
    lines = [
        "| Metric | Baseline | This PR | Δ | Status |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        label = f"{r.label} ({r.tier_note})" if r.tier_note else r.label
        lines.append(
            f"| {label} | {_fmt(r.baseline)} | {_fmt(r.current)} | {_fmt(r.delta)} | {STATUS_EMOJI[r.status]} |"
        )

    header = {
        BLOCK: "### ❌ Eval gate: BLOCKED",
        WARN: "### ⚠️ Eval gate: passed with warnings",
        PASS: "### ✅ Eval gate: passed",
    }[verdict]
    return "\n".join([header, "", *lines])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="results JSON from eval.harness to evaluate")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument(
        "--determinism-passed",
        type=str,
        default=None,
        choices=["true", "false"],
        help="pytest exit status of tests/test_chunk_id_determinism.py, if run in this CI job",
    )
    parser.add_argument("--out", type=Path, default=None, help="optional path to write the JSON gate report")
    args = parser.parse_args()

    current = load_results(args.run)

    if args.baseline.exists():
        baseline = load_results(args.baseline)
    else:
        print(
            f"WARNING: baseline file {args.baseline} does not exist yet -- "
            "every metric will report 'no baseline' and the gate cannot block. "
            "This is expected until eval/baselines/main.json is first committed.",
            file=sys.stderr,
        )
        baseline = {"metrics": {}}

    determinism_passed = None
    if args.determinism_passed is not None:
        determinism_passed = args.determinism_passed == "true"

    results = evaluate_gate(current, baseline, determinism_passed)
    verdict = overall_verdict(results)
    comment = render_pr_comment(results, verdict)
    print(comment)

    if args.out is not None:
        report = {
            "verdict": verdict,
            "run_id": current.get("run_id"),
            "git_sha": current.get("git_sha"),
            "baseline_run_id": baseline.get("run_id"),
            "baseline_git_sha": baseline.get("git_sha"),
            "results": [r.__dict__ for r in results],
        }
        args.out.write_text(json.dumps(report, indent=2))

    sys.exit(1 if verdict == BLOCK else 0)


if __name__ == "__main__":
    main()
