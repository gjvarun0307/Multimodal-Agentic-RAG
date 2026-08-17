# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

**Source of truth:** `PROJECT_SPEC.md` (frozen). This file is the *living* companion —
distilled working instructions, current repo state, and phase progress. When this file
and `PROJECT_SPEC.md` disagree, `PROJECT_SPEC.md` wins; fix this file to match, not the
other way around. Update this file at the end of every phase (Section 7 of the spec).

## What this project is now

A working single-shot LangGraph agentic RAG demo (adaptive routing + Self-RAG
correction, hybrid Milvus retrieval, BYOK LLM) is becoming a **measured, deployed,
CI-gated evaluation project**. Full architecture, phase plan, and rationale live in
`PROJECT_SPEC.md` — read it before starting any phase below, it is long and dense by
design and this file does not restate it in full.

**Zero new user-facing features.** Every change must serve measurement, reliability,
deployment, or reproducibility (`PROJECT_SPEC.md` §2.1). Off-topic ideas go in
`docs/BACKLOG.md` (not yet created), not into the codebase.

## Phase state

**Phases 0–3 are complete (Unblock, golden evaluation set, metrics harness,
noise floor). Phase 4 (Eval-as-CI) is starting now.** Full detail in
`PROJECT_SPEC.md` §7 Phase 4; this section is the working checklist, not a
restatement.

Full phase list: 0 Unblock (done) → 1 Golden set (done) → 2 Metrics harness
(done) → 3 Noise floor (done) → **4 Eval-as-CI (in progress)** → 5 Deploy &
observability → 6 Ablations & failure taxonomy → 7 README rewrite.

### Phase 4 — prep notes (scoped with user 2026-08-16, before build starts)

Spec §7 Phase 4's own text has two internal conflicts that would have caused
real build mistakes if not resolved first. Both are now decided, not open:

1. **CI has no corpus/index today — the real Phase 4 blocker.** The corpus
   (`data/raw_pdfs/*.pdf`, `artifacts/parsed_md/*.md`) is deliberately not
   committed (`artifacts/SOURCES.md` — arXiv's default license doesn't
   authorize redistribution; the Milvus paper carries an ACM posting
   restriction). A fresh Actions runner starts with neither the corpus nor a
   built `milvus.db`, and `deploy/fetch_corpus.py` /
   `deploy/build_ingest_artifacts.py` (the spec's intended fix) are Phase 5
   deliverables that don't exist yet — fast-tier CI cannot run retrieval
   without solving this first. **Decision: private Hugging Face Dataset.**
   Upload the local `milvus.db` + `artifacts/parsed_md/` bundle once to a
   *private* HF Dataset; CI fetches it at job start via an `HF_TOKEN` secret
   (small corpus, seconds not minutes). Keeps full text out of public
   git/Action logs — distinct from Phase 5's separate, later decision about
   what the public-facing Space itself bakes in (§4.1 already sanctions "HF
   Dataset fetched at boot" as an alternative to Git LFS; this reuses that
   idea privately, scoped to CI only). Re-upload the bundle whenever the
   corpus, chunking config, or embedding model changes — it is a build
   artifact, not source of truth; `artifacts/metadata.jsonl` /
   `corpus_seed.csv` stay the source of truth as always.
2. **Fast tier is literally zero-LLM-calls, not "retrieval + structured."**
   Spec's own tier table says fast = "retrieval + structured, no judge," but
   its acceptance criteria say fast = "< 3 min, zero LLM calls" — structured-
   output metrics necessarily call the *generation* LLM, which is not zero
   calls and risks real per-push quota/cost on a public repo with
   unlimited-trigger Actions (the same Groq 100k-TPD ceiling already hit
   twice, Phase 2 + Phase 3). **Decision: fast tier = `eval.harness --split
   fast --retrieval-only` + `tests/test_chunk_id_determinism.py` only.**
   True zero LLM calls, zero secrets needed beyond the HF Dataset token,
   matches the acceptance criterion literally. Structured-output metrics move
   into the nightly/labeled full tier alongside the judge, where cost is
   already budgeted and bounded to once a day.

**Gate design — thresholds available now vs. deferred** (from
`docs/NOISE_FLOOR.md`, fast split, n=4, non-judge metrics only — see Phase 3
outcome below for why judge metrics aren't gateable yet):

| Metric | Threshold | Action | Source |
|---|---|---|---|
| `chunk_id_determinism` | any failure | **block** | structural (invariant 3), not statistical |
| `retrieval.stage1.recall@10` | drop > 0.045 (4.5 pts) | **block** | noise floor 2σ |
| `router.accuracy` | drop > 0.03 (3 pts) | warn | noise floor 2σ |
| `structured.aggregate.validity_rate` | any drop from 1.000 | warn | noise floor (σ=0, full tier only per decision 2) |
| `correction.fire_rate` | drop > 0.07 | warn | noise floor 2σ (full tier only) |
| `system.warm_p95_ms` | increase > 20% | warn | spec default (noise-floor figure is a raw ms σ, not a %) |
| `faithfulness`, `citation_precision`, `refusal_accuracy`, `answer_correctness` | — | **informational only, no gate** | Phase 3 found n=1 or unusable σ (see below) — do not block on these until a dedicated low-volume judge-only noise-floor pass exists |

This deliberately overrides spec's literal gate table, which blocks on
`faithfulness` and `refusal_accuracy` — a documented, justified deviation
(Phase 3's own conclusion), not a silent one.

**Already usable, no new build needed:**
- `eval/harness.py --split fast --retrieval-only` — zero-LLM-call path,
  live-verified in Phase 2 — is the fast-tier core as-is.
- `TAVILY_MODE=replay` needs no live API key (reads the committed
  `eval/fixtures/tavily_v1.json` directly) — moot for the retrieval-only fast
  tier, matters once structured/full-tier CI exercises `web_search`.
- `tests/test_chunk_id_determinism.py` (Phase 0) already asserts the exact
  invariant the determinism gate needs — CI should invoke it as a real pytest
  job, not reimplement the check.

**Not yet built, first-time Phase 4 work:**
- One-time: upload `milvus.db` + `artifacts/parsed_md/` to a private HF
  Dataset; wire the fetch step into both workflows below.
- `.github/workflows/fast-eval.yml` (every push/PR, retrieval-only +
  determinism, budget < 3 min, $0)
- `.github/workflows/full-eval.yml` (nightly + `run-full-eval` label, full
  metric set including judge, budget < 25 min)
- `.github/workflows/lint.yml` (`ruff check .` + `pytest`)
- `.github/workflows/keep-warm.yml` — **defer.** No Space exists until Phase
  5; there is nothing to ping yet. Do not stub a workflow with no target.
- A gate-comparison step/script: diff current run vs. `eval/baselines/main.json`
  against the threshold table above, emit pass/warn/block, render the PR
  comment diff table (spec §7 Phase 4 example table).
- `eval/baselines/main.json` — first commit needs a real `eval.harness` run's
  results JSON (git SHA, config, golden-set/judge/Tavily-fixture versions),
  never hand-written (invariant 7). Baseline updates after this are PR-only
  with written justification, never automatic (spec §7 Phase 4).
- Rate-limit/quota errors must be distinguishable from real regressions in
  both the gate logic and the PR comment — a Groq 429 must never render as a
  ❌ FAIL row.
- A deliberately-broken retrieval config, run through the fast workflow, to
  prove the gate actually blocks a test PR (acceptance criterion).
- Repo must be public for unlimited free Actions minutes (spec §7 Phase 4
  cost-control note) — confirm this before relying on nightly + on-label runs
  not hitting a minutes cap.

### Phase 2 — what's already in place (verified against code, 2026-08-12)

Phase 0 plumbing that Phase 2 builds directly on top of, confirmed present:

- `src/agent.py:run_query_with_state(app_graph, question, chat_history)` →
  `(answer, final_state, trace_info)`. `trace_info["node_sequence"][0]` is the
  route decision; `trace_info["stage_latencies_ms"]` is per-node wall time;
  `trace_info["fallback_events"]` carries loud-fallback tags (invariant 15).
  `final_state["retrieved_chunk_ids"]` / `["reranked_chunk_ids"]` are the two
  ID lists Stage 1/Stage 2 retrieval metrics need (spec §5.7, §Phase 2 table).
- `src/runtime.py:get_runtime(config)` — single construction point; the harness
  should call this, never build models itself.
- `src/configuration.py:config_rag(overrides={...})` — every tunable Phase 2
  needs is already a config field: `search_limit`, `reranker_top_k`,
  `sparse_weight`, `dense_weight`, `reranker_score_threshold`, `chunk_size`,
  `max_gen_retries`, `vllm_base_url`. No config-promotion work left to do.
- Four structured-output nodes confirmed in `src/agent.py`: `RouteDecision`
  (router), `RewrittenQuery` (rewrite_query), `HallucinationScore` and
  `RelevanceScore` (hallucinations_and_relevance_router) — all via
  `llm_model.with_structured_output(...)`. This is the complete node set
  `eval/metrics/structured.py` must instrument (spec §4A.1 / §Phase 2).
  `with_structured_output` gives no built-in validity/retry signal — the
  harness has to wrap these calls itself to observe schema violations,
  retries, and silent coercion; nothing upstream provides this for free.
- `TavilySearch` (via `langchain_tavily`) is the live web-search call inside
  `web_search` node — `eval/tavily_cache.py` needs to intercept at this call
  site (`TAVILY_API_KEY` config key already resolves through `config_rag`).
- `LLM_PROVIDERS["groq"]` already has a `base_url` + `default_model` entry —
  the judge (invariant 11) can reuse this provider entry as-is; still needs a
  pinned model + prompt version choice, not just "groq" generically.

Not yet present, first-time builds for Phase 2: `eval/harness.py`,
`eval/metrics/` (all 5 files), `eval/judge.py`, `eval/tavily_cache.py`,
`configs/default.yaml`, `eval/results/` (gitignored), `eval/baselines/main.json`
(actually a Phase 4 artifact, not Phase 2 — don't build it early).

### Phase 2 — build progress (updated incrementally, not a phase close)

Landed and wired together: `configs/default.yaml`, all 5
`eval/metrics/*.py`, `eval/tavily_cache.py`, `eval/judge.py`,
`eval/harness.py` (`run_eval()` + `python -m eval.harness`), and
`eval/judge_calibration.py`. Judge calibration is done (2026-08-14/15,
30-item stratified sample, `llama-3.3-70b-versatile` judge, see below).
A full non-retrieval-only `eval.harness --split fast` run is
**live-verified** (2026-08-15, `eval/results/20260815T170351Z_fb38c61.json`,
gitignored) — all 6 metric categories produced real numbers, nothing
crashed. Phase 2 is functionally done; what's left (noise floor, gated
baselines) is explicitly Phase 3/4 scope, not a Phase 2 gap.

**Known, accepted limitation — Tavily fixture coverage.** The frozen
fixture only ever covers literal query text; the `web_search` fallback
(retrieval exhausts `rewrite_query` attempts → falls through to web
search with an LLM-rewritten query, not the original question) generates
query text that varies slightly run-to-run even at `temperature: 0`
(sampling/hardware-level variance in hosted inference), so some cache
misses are structurally unavoidable without fuzzy-matching the fixture
(out of scope). `eval.tavily_cache.record_from_harness_run()`
(`python -m eval.tavily_cache --record --split fast --config
configs/default.yaml`) discovers and records whatever queries a real run
actually needs, generation-only (no judge, so it doesn't touch Groq
quota) — cut the `web_search_error` rate on the fast split from 25.6%
(10/39) to 10.3% (4/39) in one real pass. Re-running it (idempotent,
merges into the existing fixture) narrows this further but won't reach
zero; that residual is expected, not a bug to keep chasing.

**Groq daily token budget (free tier, 100k TPD) is a real, already-hit
constraint**, not a hypothetical. Each `eval.harness` full-mode run makes
several judge calls per item; a handful of consecutive fast-split runs in
one day is enough to exhaust it, and `JudgeGradingError`'s graceful-skip
means later items in a run lose more grading coverage than earlier ones
once the budget runs out mid-run — visible as smaller `n_scored` counts,
not a quality regression. **This directly constrains Phase 3's
noise-floor procedure** (5 identical full-set runs) — spec's literal "run
the full eval set 5 times" is not achievable in a single day on the free
tier; scope this explicitly with the user before starting (fast split
instead of full, spread across days, and/or measure judge-free metrics'
noise floor separately from judge-dependent ones) rather than assuming
it fits.

**`temperature` promoted to a first-class `config_rag()` field**
(`src/configuration.py`, `TEMPERATURE` env var) — `None` by default
(provider default applies; production/`app.py`/`src/api.py` unaffected),
pinned to `0` in `configs/default.yaml`'s `overrides` (eval-only). Serves
two purposes at once: PROJECT_SPEC.md's own Phase 3 noise-floor procedure
("Temperature 0. Fix every available seed.") requires it, and it's what
makes the Tavily fixture-discovery pass above worth running at all —
without it, `query_router`/`rewrite_query` output varies enough between
runs that a recorded fixture entry frequently doesn't match on replay.

**Judge calibration result** (`eval/judge_calibration/calibration_v1_labeled.csv`,
committed): 30-item stratified sample of `golden_set.jsonl` (never
`dev_split.jsonl`), 46 labeled (item, dimension) rows, 0 left blank, one row
lost to a graceful `JudgeGradingError` skip (`gs_0123`/refusal — the
resilience path working as designed, not a labeling gap).
- `correctness`: n=24, raw_agreement=0.958, **κ=0.917** — the statistically
  meaningful number here (real variance on both sides: judge 50% true,
  human 54% true). Both disagreements had the judge grading `FALSE`
  where the human graded `TRUE` — the judge being the stricter side, not
  randomly wrong.
- `faithfulness`: n=18, raw_agreement=0.944, **κ=0.000** — do not quote
  the κ number alone, it's misleading here. The human labeled all 18 rows
  `TRUE` (zero variance); Cohen's κ is mathematically degenerate whenever
  one rater's marginal rate is exactly 100% (the chance-agreement term
  collapses to equal the raw agreement, forcing κ=0 regardless of how few
  disagreements exist — here just 1/18). Report raw agreement for this
  dimension, with this caveat, not κ in isolation.
- `refusal`: n=4, raw_agreement=1.000, κ=1.000 — sample too small (one
  `unanswerable_refuse` item's grading call failed) to carry much weight
  alone; corroborates rather than proves.
- `citation_precision` is not calibrated (by design — a continuous ratio,
  not suited to Cohen's κ; spot-check it by eye if needed).

Two real live-run findings surfaced *during* calibration, both fixed and
committed before this result was produced:
- `openai/gpt-oss-20b` (the original judge placeholder) hard-400'd on
  Groq — it emitted chain-of-thought text instead of a clean tool call.
  Separately (and more fundamentally): Groq's `json_schema` structured-
  output mode is *only* supported by the gpt-oss family — every other
  model 400s outright unless `with_structured_output(..., method=
  "function_calling")` is passed explicitly. All 4 `eval/judge.py` grade_*
  functions now pass that. `DEFAULT_JUDGE_MODEL` moved to
  `llama-3.3-70b-versatile`.
- `eval/judge_calibration.py`'s `run_sample()` built the graph without a
  `web_search_tool=` override, so any sampled item that fell through to
  `web_search` made a **live, uncontrolled Tavily call** instead of
  replaying from the frozen fixture — caught because a calibration row
  for an `unanswerable_refuse` item came back with scraped-web-blog
  content instead of corpus text. Fixed to wire in
  `eval.tavily_cache.build_tavily_tool()`, matching `eval/harness.py`.

Ten upstream additions made along the way, all additive (no existing
caller broke, full suite + ruff verified clean after each):

- **`GraphState["retrieved_chunk_scores"]`** (`src/agent.py`) — reranker
  score per `retrieved_chunk_ids` entry, pre-threshold, aligned by position;
  `[]` whenever no reranker ran (no reranker configured, or the
  `rerank_fallback` path fired). Without this, `threshold_loss` (spec §7
  Phase 2 retrieval metric) can't be computed without re-running the
  reranker inside eval — the pipeline already discards per-candidate scores
  once it filters to `final_documents`. Threaded through
  `run_query_with_state()`'s stateless init and `/query`'s `QueryResponse`
  in `src/api.py` too, matching a field the spec's own Tier 1 demo trace
  JSON (§4C) already expected (`"reranker_scores": [...]`).
- **`build_agent_graph(..., web_search_tool=None)`** (`src/agent.py`) — optional
  override for the web-search tool, same constructor-injection pattern
  already used for `database`/`embedding_model`/`rerank_model`/`llm_model`.
  `eval/tavily_cache.py`'s `build_tavily_tool(config)` supplies a
  replay-or-record wrapper here; `web_search()` itself is unchanged. Real
  `TavilySearch` still gets built exactly as before when the override is
  omitted (`app.py`/`src/api.py`'s existing calls need no changes).
- `eval/tavily_cache.py`: `TAVILY_MODE` env var (`replay` default / `live`),
  fixture at `eval/fixtures/tavily_v1.json`, keyed by sha256 of
  `(query, max_results, topic)`. Replay miss raises `TavilyCacheMissError`
  loudly (invariant 15) — never falls through to a live call. Recorded
  for real (2026-08-13, committed at `eval/fixtures/tavily_v1.json`) —
  all 4 `unanswerable_websearch` golden items covered.
- `eval/metrics/router.py`: `expected_route: "refuse"` (unanswerable_refuse
  items) has no router edge to compare against — `query_router` only ever
  picks vectorstore/websearch/chitchat. Resolved by treating "refuse"
  items' correct router target as `vectorstore` for `router_accuracy`
  purposes only (`EXPECTED_ROUTE_FOR_ROUTER` in that module); actual
  refusal behavior is `eval/metrics/generation.py`'s `refusal_accuracy`,
  not router's job. This is the one place that mapping lives.
- `eval/metrics/structured.py`: `with_structured_output()` gives no signal
  about how often it actually succeeds. Solved with `InstrumentingLLM`, a
  wrapper substituted for `llm_model` in `build_agent_graph(llm_model=...)`
  (no `src/agent.py` change needed — `llm_model` was already a constructor
  parameter). It forces `include_raw=True` on the 4 structured-output call
  sites so parse failures surface as data instead of exceptions, does one
  measurement-only retry to get a real `retry_rate`, then re-raises to
  match what the real (uninstrumented) call would have raised — every
  existing try/except fallback path in `src/agent.py` behaves identically
  either way. `silent_coercion_rate` is honestly a best-effort heuristic
  (raw tool-call arg type vs. parsed field type) and returns `None`, not a
  fake precise number, whenever it can't be checked — flagged as a known
  measurement limitation in the module docstring per invariant 16, not
  something to silently over-claim later.
- `eval/metrics/system.py`: per-stage latency is only as granular as
  `trace_info["stage_latencies_ms"]` already is today — per LangGraph node
  (`retrieve_and_rerank`, `generate`, ...), not the finer embed/retrieve/
  rerank/route/generate/correct split the spec's Phase 5 OTLP tracing
  targets (§5). `retrieve_and_rerank` bundles embed+retrieve+rerank into
  one number; `query_router`'s own decision time folds into whichever node
  runs first, since it's a conditional edge, not a node. Documented as a
  known Phase 2 → Phase 5 gap in the module docstring, not fixed here —
  fixing it means adding real per-stage spans, which is explicitly Phase 5
  scope. Also: `correction_fire_rate`/`mean_retries` are trace-only
  (computable now); `correction_improve_rate`/`degrade_rate` need a
  judge-scored correctness delta per item and return `None` until
  `eval.harness` wires `eval/metrics/generation.py`'s judge output in.
- **`config_rag()` gained `judge_api_key`** (`src/configuration.py`,
  `JUDGE_API_KEY` env var) — deliberately separate from `llm_api_key` so
  the judge (`eval/judge.py`, pinned to Groq, invariant 11) never
  accidentally shares credentials with the generation model under test,
  even when that model is also Groq-hosted. Set in `api_keys.json` as
  `"judge_api_key": "..."` — set locally now; `eval.judge.build_judge_llm()`
  raises `JudgeConfigError` loudly if it's ever missing.
- `eval/judge.py` grades one item at a time (faithfulness, correctness,
  refusal, citation precision) via 4 separate structured-output judge
  calls; `eval/metrics/generation.py` only aggregates already-graded
  results into rates, it never calls the judge itself — same
  grade/aggregate split as `structured.py`+`router.py`. All 4 judge
  prompts carry `JUDGE_VERSION` (`"v1"`, kept in sync with
  `configs/default.yaml`'s `judge_version`) — bump both together on any
  prompt wording change, never separately (invariant 11). Live-verified
  via judge calibration (2026-08-14/15, see above) — `llama-3.3-70b-versatile`
  pinned as `DEFAULT_JUDGE_MODEL` after `openai/gpt-oss-20b` failed live;
  every `with_structured_output()` call needs `method="function_calling"`
  since Groq's `json_schema` mode only supports the gpt-oss family.
- **`retrieve_and_rerank_core`** (`src/agent.py`, module-level, extracted
  from the `retrieve_and_rerank` node closure) — the hybrid-search +
  rerank + threshold-filter logic, with no GraphState/graph dependency.
  Exists so `eval/harness.py --retrieval-only` exercises the *exact*
  retrieval code path the real graph uses instead of a separately
  maintained reimplementation that could drift. The node closure is now a
  thin wrapper around it; behavior is unchanged (full suite, including the
  real `test_smoke.py` end-to-end run, passes identically before/after).
- **`eval/harness.py`** is built and real-verified: `python -m eval.harness
  --config configs/default.yaml --split fast --retrieval-only` ran for real
  against the live corpus (2026-08-12, no LLM key needed — retrieval-only
  never constructs one), producing genuine numbers (recall@10=0.63,
  recall@50=0.85, rerank_lift=-0.022 on that small fast-split sample —
  not a baseline, just proof the pipeline is real). Results land in
  `eval/results/<run_id>.json` (gitignored); `config` in that file has
  `llm_api_key`/`judge_api_key`/`tavilly_api_key` redacted before writing,
  even though the directory is gitignored — results get pasted/shared in
  ways a repo file doesn't.
  **Not yet live-verified:** the non-`--retrieval-only` full-graph path
  (needs a real `llm_provider`/`llm_api_key` and `judge_api_key`, neither
  configured locally yet) and Tavily replay (needs
  `python -m eval.tavily_cache --record` run once against a real
  `TAVILY_API_KEY` — the 4 `unanswerable_websearch` golden items will hit
  `TavilyCacheMissError` until then). `--split fast`/`--split dev` are
  implemented (deterministic proportional-per-category sampling, seed
  `20260812`, never sampling `dev_split.jsonl` into `fast`/`full` per
  invariant 6) but not yet the thing gating CI — that wiring is Phase 4.
  **Known gap, not yet fixed:** `correction_improve_rate`/`degrade_rate`
  always report `None` — they'd need the pre-correction generation text,
  which `GraphState` currently overwrites rather than preserves across a
  `rewrite_query` loop iteration. `correction_fire_rate`/`mean_retries`
  work today (trace-derivable alone). Fixing this is a separate, not-yet-
  scoped `GraphState` change.

### Phase 3 outcome (closed 2026-08-16)

`eval/noise_floor.py` built and run for real; full write-up in
`docs/NOISE_FLOOR.md` (committed). **Deviated from the spec's literal "5
runs against the full set" on two axes, both documented in that file, not
silent:** fast split (39 items) instead of full (145) — same Groq
100k-TPD-budget reasoning as Phase 2 — and **4 completed runs, not 5**. The
5th run was cut short 4 items in by the same unexplained background-process
kill (no traceback, hit three separate times across 2026-08-15/16 trying to
run this pass — root cause never found on the agent-tooling side, not a
bug in this codebase). Rather than keep re-attempting a clean 5th run
against a same-day quota ceiling that wouldn't be any less exhausted on a
6th try, `eval/noise_floor.py` gained an `--existing-results` resume flag
(seeds the aggregate from already-completed run JSONs, only executes
however many more are needed) so a kill mid-pass doesn't throw away
already-spent quota and wall time — this is how 2 runs survived the first
kill and 4 survived the second.

**Headline metrics (retrieval, router, structured-output, system/latency,
correction) have full, clean n=4 coverage** — none of these touch the Groq
judge, so they were never at risk from the quota ceiling. Selected
results (`docs/NOISE_FLOOR.md` has the full table + proposed
`threshold_2sigma` CI gates for each): `recall@10`=0.522±0.018,
`router.accuracy`=0.789±0.013, `structured.aggregate.validity_rate`=
1.000±0.000, `correction.fire_rate`=0.397±0.033, `warm_p50`=40.6s±10.2s.
`warm_p99` was excluded from gating — on a 39-item sample p99 is
effectively one item, and one run's `retrieve_and_rerank` outlier alone
made σ (696s) exceed the mean (568s); not a stable statistic yet.

**Judge-scored generation metrics are honestly thin, per invariant 16 —
not gated on yet, not papered over with a fake number either.**
`answer_correctness.rate` got n=4 but noisy (only ~10/39 items scored per
run before quota ran out: 0.494±0.174 — usable with a wide gate).
`refusal_accuracy.rate` got n=4 but off ~2 scored items/run (σ=0.50 on a
[0,1] metric — not usable). `faithfulness.rate` and
`citation_precision.mean` got **n=1** — only one of the 4 runs scored any
items for those dimensions before hitting the daily limit; no variance is
computable from a single value. **This is the concrete argument for why
Phase 4 should launch CI gating on the non-judge headline metrics now and
treat faithfulness/citation-precision/refusal as informational-only until
a dedicated low-volume judge-only noise-floor pass exists** (fewer items
per call, likely spread across more than one day).

README line drafted in `docs/NOISE_FLOOR.md`: *"Run-to-run variance ±4.35
pts recall@10 and ±2.56 pts router accuracy over 4 identical runs (fast
split, temperature 0, Tavily replay); regression thresholds set at 4.5 pts
and 3.0 pts respectively."*

### Phase 1 outcome (closed 2026-08-11)

`eval/golden/golden_set.jsonl` (145 items) + `eval/golden/dev_split.jsonl` (30 items),
175 total, all `verified_by: "human"`, all `version: 1`. Composition within ±1pp of
every target share in spec §7. Dev split stratified proportionally (largest-remainder
apportionment) from the five largest buckets only — single_hop/multi_hop/table_figure/
unanswerable_refuse/ambiguous — with adversarial/unanswerable_websearch/chitchat (≤9
items each) excluded entirely and kept fully in `golden_set.jsonl`. Fixed seed `20260811`,
carve script not committed (one-off; rerunning would need re-verification against the
now-frozen split, not a repeatable pipeline step). `python -m eval.validate_golden
--require-verified` is fully clean: 14 PASS, 0 FAIL, 0 WARN, 0 SKIP, including passage
resolution at chunk_size 512/1024/2048 (169/169 gold passages resolve; ~1.6-2.2%
unresolved-chunk canary, unchanged since Checkpoint 1, all non-prose junk — VLM "blank
image" caption repeats and empty table cells, not gold-passage-adjacent).

**Known finding for Phase 6 (not a Phase 1 blocker):** most VLM figure captions in this
corpus are low quality (repeated "Blank Image" captions, mislabeled diagrams). Only 2
`table_figure` items use genuine VLM captions; the rest lean on real embedded data
tables. Worth a negative-results writeup under invariant 16.

### Phase 1 — mechanics that carry forward (spec §5.2.1, still load-bearing)

- `gold_passages` is the only ground truth (verbatim text + `doc_id`); chunk IDs are
  *never* stored in the golden set, only derived at eval time (invariant 4).
- A `passage_text` that isn't unique in its source `.md` is a hard error — extend the
  passage, don't pick the first match.
- `gold_answer` is empty string for `unanswerable_refuse` items.
- `gold_doc_ids` is diagnostics only, never a gating metric (near-ceiling at 15 docs).
- Cache resolved chunk IDs per config in `eval/golden/resolved/<config_hash>.json`.
- Post-freeze edits to either file are a `version` bump, not a silent edit (invariant 2).

## Non-goals (do not implement, park in `docs/BACKLOG.md`)

New retrieval strategies (HyDE/RAPTOR/GraphRAG/ColBERT), multi-document scoping,
MCP integration, streaming tokens, conversation persistence, Docling migration,
image-input queries, UI redesign, any new agent hop/tool. Three exceptions are in
scope: defect fixes, the vLLM backend as a *measurement instrument* (§4A, not a
deployed feature), and CPU-viable reranking (deploy target has no GPU).

## Global invariants (`PROJECT_SPEC.md` §8)

1. No new features (see Non-goals above).
2. Corpus is frozen — any change is an eval-set version bump + re-verification.
3. Chunk IDs are deterministic; rebuilds from identical sources produce identical IDs, enforced in CI.
4. `gold_passages` is ground truth; chunk IDs are always derived, never hand-edited.
5. `doc_id` is immutable once assigned.
6. Never tune against the main golden set — dev split only.
7. Every published number traces to a committed results JSON (git SHA, config, golden-set version, judge version, Tavily fixture version, backend).
8. Retrieval and generation metrics are always reported separately.
9. Warm and cold latency are always reported separately, never blended.
10. Ablations are config-only — a code-edit ablation means config is incomplete.
11. Judge is pinned (Groq) and independent of the generation model; prompt changes force a re-baseline.
12. Tavily replays from frozen fixtures in every eval/CI run — never live.
13. Precached demo output (Tier 1) is always labeled as precached and recorded from real runs, never hand-written.
14. Streamlit and FastAPI both call `get_runtime()` in-process — no self-HTTP hop.
15. Silent fallbacks are forbidden — disabled reranker, skipped model, rate limit, cache miss: all loud, logged, surfaced in results.
16. Negative results get published — a component that doesn't earn its cost is a finding, not an embarrassment.

## Repo state going into Phase 1

- **Git history:** `main` is a reseeded orphan history (old history on local branch
  `archive/pre-reseed-2026-08-11`, not on `main`). `origin` still has the pre-reseed
  history; nothing has been pushed since — local `main` and `origin/main` share no
  common ancestry. Force-pushing needs separate explicit confirmation before it happens.
- **Corpus is not committed to git** (`data/raw_pdfs/*.pdf`, `artifacts/parsed_md/*.md`)
  — several papers' licenses don't clearly permit redistributing full text (per-paper
  detail in `artifacts/SOURCES.md`, not needed here). `artifacts/metadata.jsonl` and
  `corpus_seed.csv` (metadata + content hashes, no substantial text) stay committed.
  **Practical effect:** the corpus must exist locally (`data/raw_pdfs/*.pdf` +
  `artifacts/parsed_md/*.md`, today built by re-running `parse.py`'s pipeline; a
  `deploy/fetch_corpus.py` that re-downloads from arXiv is planned but not built yet)
  before `pytest` runs fully, before annotating golden-set passages, or before rebuilding
  the index. Corpus-dependent tests skip cleanly (not fail) when it's absent.
- **Module layout:** core modules live under `src/` (`runtime.py`, `api.py`, `agent.py`,
  `hybrid_database.py`, `parse.py`, `configuration.py`, `helper.py`, `logging_utils.py`);
  `app.py`/`pages/1_Setup.py` stay at repo root.
- **Uploads are rejected** — corpus is frozen, so `app.py` has no upload UI.
  `agent.py`'s active-document-scoping fields/logic (never set by any caller)
  were removed outright rather than left as dead code.

## Implementation notes for later phases

Decisions made during Phase 0 that aren't spelled out in `PROJECT_SPEC.md` but matter
going forward:

- **Config-only overrides:** `config_rag(overrides={...})` applies a dict on top of the
  resolved config (`defaults → api_keys.json → env → overrides`). This is the mechanism
  the eval harness (Phase 2) and ablation runner (Phase 6) should use to construct a run
  from a config dict alone — reach for this, not env vars.
- **Route decision isn't a `GraphState` field.** `query_router` is a conditional-edge
  selector, not a node, so its choice is never written into state. Use
  `agent.run_query_with_state()` — returns `(answer, final_state, trace_info)`, and
  `trace_info["node_sequence"][0]` is the route decision (needed for Phase 2's
  `router_accuracy`).
- **Milvus collection name is `arag_project_v2`** (bumped when chunk IDs went
  deterministic; an old `arag_project` collection may still exist locally alongside it).
- **Reranker default is `bge-reranker-v2-m3`** (CPU-viable) via
  `configuration.build_reranker()`; `bge-reranker-v2-gemma` stays available as a config
  value for the GPU ablation row.
- **vLLM ablation backend:** set `llm_provider=vllm` and `vllm_base_url` (a config
  field, not part of the static `LLM_PROVIDERS` entry) to point at a tunneled vLLM server.

## Commands

```bash
# setup
pip install -r requirements.lock      # exact reproducibility (or requirements.txt for direct deps only)

# dev
pytest
ruff check . && ruff format .

# corpus / metadata (requires data/raw_pdfs/*.pdf + artifacts/parsed_md/*.md locally)
python deploy/build_metadata.py --seed artifacts/corpus_seed.csv --out artifacts/metadata.jsonl
python -m src.hybrid_database          # (re)build ./milvus.db -- DESTRUCTIVE, drops + recreates arag_project_v2

# serve
uvicorn src.api:app --reload --port 8000
streamlit run app.py

# eval (golden set; requires corpus locally, see below)
python -m eval.resolve_passages         # writes eval/golden/resolved/<config_hash>.json
python -m eval.validate_golden --require-verified
python -m eval.harness --config configs/default.yaml --split fast   # results -> eval/results/<run_id>.json
python -m eval.noise_floor --config configs/default.yaml --split fast --runs 5  # add --existing-results a.json,b.json to resume
```

Not yet available (later phases — see `PROJECT_SPEC.md` §9 for the full target list):
`deploy/fetch_corpus.py`, `deploy/build_ingest_artifacts.py`, `deploy/record_demo_traces.py`
(all Phase 5), CI workflow wiring the harness as a gate (Phase 4, starting now).

## Working agreements for this upgrade

- **Every phase ends with a commit and a `CLAUDE.md` update** (spec §7) — update Phase
  state as part of that phase's final commit, not as an afterthought later. Keep this
  file distilled: facts and conventions future phases need, not a changelog of what was
  done — that's what git history is for.
- **Config-only ablations, no exceptions** (invariant 10) — if an ablation row requires
  editing code rather than a config value, config promotion is incomplete; fix that
  instead of hand-editing for one run.
- **Say when something is unverified, and verify for real when you can** — run it, don't
  just reason about it, whenever the environment allows (model downloads, live LLM
  calls, a real server launch, a fresh-clone check), per the spec's "provably works" goal.
- Never commit `api_keys.json`, `milvus.db`, `logs/`, `fail_logs.txt`, corpus text
  (`data/`, `artifacts/parsed_md/`), or any real key/secret.
- Prefer `git add <path>` over `git add -A`; check `git status` before staging.
- Message format: short imperative subject (`feat:`/`fix:`/`refactor:`/`docs:`/`chore:`), body explaining why when not obvious from the diff.
- **Commit messages describe the change, not the internal planning mechanics used to get
  there** — no "checkpoint N" language, no references to an in-session plan's own
  checkpoint numbering. Squash/merge intermediate in-progress commits into one commit
  per logical unit of work when the user asks for it (confirmed 2026-08-11: multiple
  golden-set authoring commits were reset and re-committed as a single
  `eval/golden/golden_set.jsonl` commit once fully human-verified).
- Pause for user confirmation between phase milestones before proceeding to the next one.
