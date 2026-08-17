# Noise Floor (Phase 3)

PROJECT_SPEC.md §"PHASE 3 — Noise floor": know your measurement precision
before gating on it. This document records the observed run-to-run variance
for every metric the harness computes, so Phase 4's CI thresholds can be set
above that variance instead of guessed.

## Scope deviation from the spec's literal procedure (documented, not silent)

The spec's literal procedure is "run the identical commit against the full
eval set 5 times." That didn't fit inside a single day's Groq free-tier
budget (100k tokens/day, shared across every judge-graded dimension) — see
`CLAUDE.md`'s Phase 2/3 notes for the two occasions this was hit mid-run.
Two further deviations, both forced by the same background-process
environment killing the harness mid-run with no traceback (an
infrastructure issue outside this codebase, hit three times across
2026-08-15/16 — see `eval/noise_floor.py`'s module docstring):

- **Split: `fast` (39 items), not `full` (145 items).** Same reasoning as
  the rest of Phase 2/3: the full set doesn't fit the daily judge budget
  even once, let alone 5 times.
- **Runs: 4, not 5.** The 5th run was killed 4 items in (of 39) by the
  same unexplained process-termination issue. Rather than keep re-attempting
  a clean 5th run against a structurally-limited daily budget, this
  document reports what 4 completed runs actually show. `eval/noise_floor.py`
  supports `--existing-results` specifically so a killed pass doesn't lose
  already-completed runs — this dataset is the result of two resumed
  passes, not one clean one.

Temperature was pinned to 0 (`configs/default.yaml`), Tavily ran in replay
mode against the frozen fixture, and all 4 runs used the same commit
(`37377a7` for runs 1–2, `be1ef92` for runs 3–4 — the only diff between
those two commits is `eval/noise_floor.py`'s resume support itself, which
doesn't touch the eval pipeline being measured).

Source data: `eval/results/noise_floor/noise_floor_fast_4runs_20260815T175416Z_37377a7.json`
(gitignored, like all `eval/results/` output — this document is the
committed, durable record per invariant 7).

## Headline metrics — full n=4 coverage

These never touch the Groq judge, so all 4 runs contributed a value with no
quota exposure. This is the primary noise-floor evidence for Phase 4 gating.

| Metric | mean | σ | range | proposed CI threshold (≥2σ) |
|---|---|---|---|---|
| `retrieval.stage1.recall@10` | 0.5217 | 0.0177 | 0.0435 | 0.045 (≈4.5 pts) |
| `retrieval.stage1.recall@20` | 0.6033 | 0.0208 | 0.0435 | 0.045 |
| `retrieval.stage1.recall@50` | 0.7446 | 0.0208 | 0.0435 | 0.045 |
| `retrieval.stage2.recall@1` | 0.2228 | 0.0109 | 0.0217 | 0.025 |
| `retrieval.stage2.recall@3` | 0.4348 | 0.0355 | 0.0870 | 0.075 |
| `retrieval.stage2.recall@5` | 0.4620 | 0.0326 | 0.0652 | 0.070 |
| `retrieval.rerank_lift` | 0.0489 | 0.0572 | 0.1087 | 0.12 |
| `retrieval.threshold_loss.rate` | 0.0732 | 0.0283 | 0.0524 | 0.06 |
| `router.accuracy` | 0.7885 | 0.0128 | 0.0256 | 0.03 (≈3 pts) |
| `router.n_misrouted` | 8.25 | 0.50 | 1 | 1.2 items |
| `structured.aggregate.validity_rate` | 1.0000 | 0.0000 | 0.0000 | any drop from 1.0 is a real regression, not noise |
| `structured.aggregate.retry_rate` | 0.0000 | 0.0000 | 0.0000 | same — 0 variance observed |
| `correction.fire_rate` | 0.3974 | 0.0331 | 0.0769 | 0.07 |
| `correction.mean_retries` | 0.9103 | 0.1156 | 0.2821 | 0.25 |
| `system.warm_p50_ms` | 40,625 | 10,201 | 23,481 | 20,400 ms |
| `system.warm_p95_ms` | 177,173 | 46,165 | 96,295 | 92,330 ms |

`system.warm_p99_ms` is excluded from the headline table on purpose: mean
568,146ms, σ=696,637ms — the stdev exceeds the mean, driven by a single
`retrieve_and_rerank` outlier in one run (system.per_stage_mean_ms.
retrieve_and_rerank ranges 75,103ms across the 4 runs). p99 on a 39-item
sample is 1 item — not a stable enough statistic to gate on yet. Flagged as
a known limitation, not silently included as if it were reliable.

## Judge-scored generation metrics — thin coverage, not gated on yet

Every dimension that needs the Groq judge lost coverage to the same daily
budget across these 4 runs. Per invariant 16 (negative results get
published), this is reported honestly rather than papered over with a
partial number presented as if it were a full n=4 measurement:

| Metric | n_runs_with_data | mean (of runs with data) | σ | Verdict |
|---|---|---|---|---|
| `generation.answer_correctness.rate` | 4 | 0.4938 | 0.1743 | Usable but noisy — average of only ~10/39 items scored per run (judge quota ran out mid-run every time). σ=0.17 is large relative to the mean; do not gate below a ~0.35-point threshold. |
| `generation.refusal_accuracy.rate` | 4 | 0.7500 | 0.5000 | **Not usable.** Only ~2 refusal items scored per run on average — σ=0.50 on a metric bounded in [0,1] means this is noise, not signal. |
| `generation.faithfulness.rate` | **1** | 0.9600 | — | **Not usable.** Only one of 4 runs scored any faithfulness items before hitting the rate limit. No variance can be computed from n=1. |
| `generation.citation_precision.mean` | **1** | 0.9114 | — | **Not usable.** Same cause as faithfulness — one run got lucky before quota ran out. |

**Implication for Phase 4:** CI gating can launch on the headline
(non-judge) metrics now. Faithfulness, citation precision, and refusal
accuracy need a dedicated judge-only noise-floor pass — smaller item count
per call, spread across multiple days if needed to get 4-5 independent
values per dimension — before they can carry a real threshold. Until then,
report them in results JSON as informational, not gating.

## README line (draft, per spec §7 Phase 3 acceptance criteria)

> Run-to-run variance ±4.35 pts recall@10 and ±2.56 pts router accuracy over
> 4 identical runs (fast split, temperature 0, Tavily replay); regression
> thresholds set at 4.5 pts and 3.0 pts respectively. Judge-scored metrics
> (faithfulness, citation precision) await a dedicated low-volume noise-floor
> pass — Groq's free-tier daily budget doesn't support 4-5 full judge-graded
> passes in a single day.

## Acceptance criteria status (PROJECT_SPEC.md §7 Phase 3)

- [x] Identical-commit runs recorded with full results JSON each — 4 of the
      planned 5 (deviation documented above), each a standalone JSON under
      `eval/results/`.
- [x] Per-metric σ and range documented — headline table above; judge-scored
      table documents thin coverage instead of a fabricated σ.
- [x] Proposed CI thresholds all ≥ 2σ, justified in writing — headline table;
      judge-scored metrics deliberately excluded from gating until their own
      noise floor exists.
- [x] README line drafted — above.
