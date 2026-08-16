"""python -m eval.noise_floor --config configs/default.yaml --split fast --runs 5

Phase 3 (PROJECT_SPEC.md §7): "know your measurement precision before
gating on it." Runs the identical commit against the same split N times
(spec's literal procedure says the full 145-item golden set; this project
deliberately scopes noise-floor runs to the fast split instead -- Groq's
free-tier daily token budget (100k TPD) doesn't fit 5 full-set judge-graded
runs in one day, confirmed by hitting 429s twice in one session during
Phase 2. Documented as a deliberate scope choice, not silently substituted).

For each run, calls eval.harness.run_eval() exactly as `python -m
eval.harness` would -- same config, same split, same temperature=0 pin --
producing N independent results JSON files under eval/results/ (each one
already a complete, standalone record per invariant 7). This module's own
job is purely the cross-run statistics: flatten every numeric metric out
of each run's `metrics` dict, then compute mean / sample stdev / min / max
/ range across the N runs, per metric.

A metric missing from a given run (None -- e.g. a judge grading call that
hit a rate limit and was skipped, per eval/judge.py's JudgeGradingError
resilience) is excluded from that metric's statistics for that run, not
treated as 0 -- `n_runs_with_data` in the output records exactly how many
of the N runs actually contributed a value, so a metric with thin
coverage is visibly thin, not silently averaged as if it were complete.

Acceptance criteria this targets (PROJECT_SPEC.md §7 Phase 3):
  - 5 identical runs recorded with full results JSON each (via run_eval())
  - Per-metric sigma and range documented (this module's output)
  - Proposed CI thresholds >= 2*sigma -- this module proposes a
    threshold_2sigma value per metric; the actual gate table (with written
    justification per metric) is Phase 4's job, using this module's numbers
    as input, not decided here.

Resuming after an interrupted pass (2026-08-15/16: this process has been
killed mid-run by something outside this codebase -- background-job
termination with no traceback, same unexplained pattern hit twice before
during Tavily fixture discovery -- three times now, root cause never
found on this side). Each run's results JSON is already a complete,
standalone record (invariant 7) written to disk the moment that run
finishes, so a kill mid-run-3-of-5 doesn't lose runs 1-2 -- only the
final aggregated summary, which this module can rebuild from those JSON
files without re-running them. --existing-results (or
existing_result_paths= in code) takes those paths, seeds all_run_metrics
from them, and only runs however many more are needed to reach --runs
total.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from eval.harness import DEFAULT_CONFIG, DEFAULT_RESULTS_DIR, run_eval

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOISE_FLOOR_DIR = REPO_ROOT / "eval" / "results" / "noise_floor"

DEFAULT_N_RUNS = 5


def flatten_numeric_metrics(d: dict, prefix: str = "") -> dict[str, Optional[float]]:
    """Recursively flattens a nested metrics dict into {"a.b.c": value}
    pairs, keeping every numeric leaf (including None -- "not computed
    this run" is itself informative) and skipping non-numeric leaves
    (strings, lists) since those aren't meaningfully averaged across runs."""
    flat: dict[str, Optional[float]] = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_numeric_metrics(value, path))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)) or value is None:
            flat[path] = value
    return flat


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_stdev(values: list[float], mean: float) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def compute_noise_floor_stats(all_run_metrics: list[dict[str, Optional[float]]]) -> dict[str, dict[str, Any]]:
    """all_run_metrics: one flattened metrics dict per run (same order as
    the runs were executed, though order doesn't affect the stats). Returns
    {metric_path: {n_runs_with_data, mean, stdev, min, max, range,
    threshold_2sigma}}, sorted by metric_path for stable output."""
    all_keys: set[str] = set()
    for run_metrics in all_run_metrics:
        all_keys.update(run_metrics.keys())

    stats: dict[str, dict[str, Any]] = {}
    for key in sorted(all_keys):
        values = [run_metrics[key] for run_metrics in all_run_metrics if run_metrics.get(key) is not None]
        n = len(values)
        if n == 0:
            stats[key] = {
                "n_runs_with_data": 0,
                "mean": None,
                "stdev": None,
                "min": None,
                "max": None,
                "range": None,
                "threshold_2sigma": None,
            }
            continue
        mean = _mean(values)
        stdev = _sample_stdev(values, mean)
        stats[key] = {
            "n_runs_with_data": n,
            "mean": mean,
            "stdev": stdev,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "threshold_2sigma": 2 * stdev,
        }
    return stats


def run_noise_floor(
    *,
    config_path: Path = DEFAULT_CONFIG,
    split: str = "fast",
    n_runs: int = DEFAULT_N_RUNS,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    out_dir: Path = DEFAULT_NOISE_FLOOR_DIR,
    existing_result_paths: Optional[list[Path]] = None,
) -> tuple[dict, Path]:
    all_run_metrics: list[dict[str, Optional[float]]] = []
    run_ids: list[str] = []
    result_paths: list[str] = []

    for path in existing_result_paths or []:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"=== Reusing existing run: {path} ===", file=sys.stderr)
        all_run_metrics.append(flatten_numeric_metrics(results["metrics"]))
        run_ids.append(results["run_id"])
        result_paths.append(str(path))

    if len(result_paths) > n_runs:
        raise ValueError(f"Got {len(result_paths)} existing result paths but only --runs {n_runs} were requested.")

    for i in range(len(result_paths) + 1, n_runs + 1):
        print(f"=== Noise floor run {i}/{n_runs} (split={split!r}) ===", file=sys.stderr)
        results, out_path = run_eval(
            config_path=config_path, split=split, retrieval_only=False, results_dir=results_dir
        )
        all_run_metrics.append(flatten_numeric_metrics(results["metrics"]))
        run_ids.append(results["run_id"])
        result_paths.append(str(out_path))

    stats = compute_noise_floor_stats(all_run_metrics)

    summary = {
        "n_runs": n_runs,
        "split": split,
        "config_path": str(config_path),
        "run_ids": run_ids,
        "result_paths": result_paths,
        "stats": stats,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"noise_floor_{split}_{n_runs}runs_{run_ids[0]}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    print(f"Wrote noise-floor summary to {out_path}", file=sys.stderr)
    return summary, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=["full", "fast", "dev"], default="fast")
    parser.add_argument("--runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_NOISE_FLOOR_DIR)
    parser.add_argument(
        "--existing-results",
        type=str,
        default=None,
        help="Comma-separated paths to already-completed run result JSONs (e.g. from a prior pass "
        "that was interrupted) -- reused instead of re-run, counted toward --runs.",
    )
    args = parser.parse_args()

    existing_result_paths = (
        [Path(p.strip()) for p in args.existing_results.split(",") if p.strip()] if args.existing_results else None
    )

    summary, out_path = run_noise_floor(
        config_path=args.config,
        split=args.split,
        n_runs=args.runs,
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        existing_result_paths=existing_result_paths,
    )
    print(f"Wrote {out_path}", file=sys.stderr)
    print(json.dumps(summary["stats"], indent=2, default=str))


if __name__ == "__main__":
    main()
