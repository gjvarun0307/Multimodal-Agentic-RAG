# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

**Source of truth:** `PROJECT_SPEC.md` (frozen). This file is the *living* companion —
distilled working instructions, current repo state, and the current phase's checklist.
When this file and `PROJECT_SPEC.md` disagree, `PROJECT_SPEC.md` wins; fix this file to
match, not the other way around. Update this file at phase boundaries (spec §7) — track
the current phase's steps here and check them off as they land; don't restate finished
phases in prose, git history and the committed docs already have that.

## What this project is now

A working single-shot LangGraph agentic RAG demo (adaptive routing + Self-RAG
correction, hybrid Milvus retrieval, BYOK LLM) is becoming a **measured, deployed,
CI-gated evaluation project**. Full architecture, phase plan, and rationale live in
`PROJECT_SPEC.md` — read it before starting any phase, it is long and dense by design
and this file does not restate it.

**Zero new user-facing features.** Every change must serve measurement, reliability,
deployment, or reproducibility (spec §2.1). Off-topic ideas go in `docs/BACKLOG.md`
(not yet created), not into the codebase.

## Phase state

0 Unblock (done) → 1 Golden set (done) → 2 Metrics harness (done) → 3 Noise floor
(done) → **4 Eval-as-CI (in progress)** → 5 Deploy & observability → 6 Ablations &
failure taxonomy → 7 README rewrite.

Full detail per phase in `PROJECT_SPEC.md` §7. Completed-phase facts worth carrying
forward live in "Facts carried forward" below and in the committed docs they
reference — not restated here as narrative.

## Phase 4 — Eval-as-CI (current)

Spec §7 Phase 4's own text has two internal conflicts, resolved with the user
2026-08-16 before build started — both decisions, not open questions:

1. **Corpus delivery to CI: private Hugging Face Dataset**
   (`gjvarun0307/arag-eval-corpus`) — corpus is deliberately not committed
   to git (`artifacts/SOURCES.md`, license restrictions), so CI fetches
   `milvus.db` + `artifacts/parsed_md/` via a read-only `HF_TOKEN` secret
   instead of building it. Re-upload whenever corpus/chunking/embedding
   model changes; `deploy/fetch_corpus.py` (spec's intended long-term fix)
   is a Phase 5 deliverable that doesn't exist yet.
2. **Fast tier = zero LLM calls, literally** — `eval.harness --split fast
   --retrieval-only` + `test_chunk_id_determinism.py` only. Structured-
   output metrics (which need the generation LLM) moved to the labeled
   full tier instead, to avoid Groq-quota cost on every push.

**Gate thresholds** (from `docs/NOISE_FLOOR.md`, fast split, n=4 —
deliberately overrides spec's literal gate table, which blocks on
`faithfulness`/`refusal_accuracy`; Phase 3 found those judge metrics n=1 or
unusable σ, so they're informational-only until a dedicated judge-only
noise-floor pass exists):

| Metric | Threshold | Action |
|---|---|---|
| `chunk_id_determinism` | any failure | **block** |
| `retrieval.stage1.recall@10` | drop > 0.045 | **block** |
| `router.accuracy` | drop > 0.03 | warn |
| `structured.aggregate.validity_rate` | any drop from 1.000 | warn |
| `correction.fire_rate` | drop > 0.07 | warn (full tier only) |
| `system.warm_p95_ms` | increase > 20% | warn |
| `faithfulness`, `citation_precision`, `refusal_accuracy`, `answer_correctness` | — | informational only, no gate |

**Real timing finding, still overrides the fast-tier spec's "< 3 min"
budget:** a `--retrieval-only` run against the real 39-item split takes
~12-15 min wall clock on CPU (BGE-M3 + `bge-reranker-v2-m3`, no GPU either
locally or on GitHub-hosted runners) — accepted as the real budget rather
than shrinking the item count, since the 39-item split is also what Phase
3's noise-floor thresholds were computed against. Still $0 (public repo,
unlimited Actions minutes).

**Status: every build-checklist item is done except final verification of
`full-eval.yml`.** All workflows/scripts below are live-verified in real CI
unless noted; git history has full build detail per item, not restated here.

- [x] Corpus uploaded to the private HF Dataset.
- [x] `fast-eval.yml` — every push/PR, retrieval-only + determinism +
      `eval.gate`. Fully green end-to-end (`timeout-minutes: 30`); the
      gate's block path was proven for real against a deliberately-broken
      config (`test/gate-block-proof`, closed without merging) — that pass
      also caught and fixed a real bug (`tee` swallowing `eval.gate`'s
      exit code without `set -o pipefail`, fixed `9357eec`). Not yet
      posting an actual PR comment via the GitHub API (`pull-requests:
      write` + `actions/github-script` — a step up in blast radius,
      deliberately held off pending explicit sign-off); the diff table
      goes to `$GITHUB_STEP_SUMMARY` instead.
- [x] `lint.yml` — `ruff check .` + `pytest`, fully green.
- [x] `eval/gate.py` (`python -m eval.gate --run <results.json> --baseline
      eval/baselines/main.json [--determinism-passed true|false] [--out
      report.json]`) — diffs a run against the baseline per the gate table,
      renders the PR-comment diff table, reports `rate_limited` instead of
      a false WARN/BLOCK when a run's judge/generation calls hit a
      provider 429 (see rate-limit item below). 21 unit tests
      (`tests/test_gate.py`), live-verified against real results JSONs.
- [x] `eval/baselines/main.json` — a verbatim copy of a real fast/
      retrieval-only run (`run_id=20260818T184206Z_cf44a44`,
      `recall@10=0.6304`), never hand-written (invariant 7). Updates are
      PR-only with written justification, never automatic.
- [x] Rate-limit/quota errors rendered distinctly from real regressions
      (a Groq 429 must never show as ❌ FAIL). Every LLM call in this
      project routes through `langchain_openai.ChatOpenAI`, so a 429
      always surfaces as `openai.RateLimitError` regardless of provider —
      `src/helper.is_rate_limit_error()` detects it through the exception
      chain. `src/agent.py` tags a rate-limited item `"rate_limited"`
      instead of the generic `"graph_execution_error"`;
      `eval/judge.py.JudgeRateLimitError(JudgeGradingError)` does the same
      for judge calls, counted per-dimension into each metric's
      `n_rate_limited` (`eval/harness.py`). `eval/gate.py` overrides a
      would-be WARN/BLOCK with `rate_limited` on the three LLM-driven
      full-tier gated metrics (`router.accuracy`, `structured.
      validity_rate`, `correction.fire_rate`) when the run shows any
      rate-limiting, and overrides `insufficient_data` with the same on
      judge-scored info metrics. 7 tests in `tests/test_gate.py`/
      `tests/test_judge.py` prove the identical numeric drop renders WARN
      clean vs. `rate_limited` when quota-constrained.
- [x] Screenshot of a real blocked-merge run for the Phase 7 README —
      `imgs/ci_blocked_merge_run_32180138001.jpg`. Not yet placed into an
      actual README (Phase 7's job).
- [ ] `full-eval.yml` — **in progress. Judge model is fixed and confirmed
      working live in CI; the remaining blocker is `timeout-minutes: 45`
      is way too low — see the second resume point below.**

**`full-eval.yml` resume point (2026-08-19):** written, `run-full-eval`
label-trigger only (not nightly — `schedule:` cron written but commented
out; still pending one successful labeled run as a track record). Generation
LLM is `nvidia_nim` (`meta/llama-3.1-8b-instruct`), switched off Groq
2026-08-18 to stop competing with the judge for quota. PR #2 ("First-full-
eval", branch `ci/verify-full-eval-run`, Actions run `32213994859`) was
**cancelled by the user mid-run 2026-08-19** (was on step 9, "Full eval (all
metrics incl. judge)," ~22 min in against `timeout-minutes: 45`) once the
judge-model defect below surfaced live in its logs — not a real
success/failure/timeout signal, so wall-clock time and whether the Groq TPD
budget would have held for ~283 judge calls are both still unverified.
PR #2 closed without merging, `ci/verify-full-eval-run` deleted (remote +
local), same pattern as `test/gate-block-proof`.

**Defect found and fixed this session: the judge model was dead on Groq.**
`DEFAULT_JUDGE_MODEL` (`eval/judge.py`) and `configs/default.yaml`'s
`judge_model` were both pinned to `llama-3.3-70b-versatile`, which Groq
deprecated for free/developer-tier usage — every real call 404'd
(`model_not_found`), confirmed live mid-run in `full-eval.yml`'s logs. A
plain 404 doesn't go through `is_rate_limit_error()` / `JudgeRateLimitError`
(that path only catches 429s), so it fell into the generic
`JudgeGradingError` skip path — loud and logged (invariant 15 held), but
**every** judge call failed, not a quota-driven subset, so
`faithfulness`/`citation_precision`/`correctness` landed at `n_scored≈0`
on every run since the deprecation (informational-only in the gate table,
so nothing was silently blocked, but the judge was non-functional).

**Fixed:** live-queried Groq's `/v1/models` with the real `judge_api_key`
(`llama-3.3-70b-versatile` confirmed absent from the list entirely) and
live-calibrated `openai/gpt-oss-120b` — Groq's suggested replacement —
against all four real grading functions (`grade_faithfulness`,
`grade_correctness`, `grade_refusal`, `grade_citation_precision`,
`method="function_calling"`) on real golden-set content: 6/6 calls
returned clean structured output with substantively correct judgments
(it correctly flagged a deliberately-flawed answer as `is_correct=False`
for the right reason). This is a different model from `gpt-oss-20b`,
which failed the original 2026-08-13 calibration with a hard `400
output_parse_failed` — 120b was verified separately, not assumed to
inherit the fix. `DEFAULT_JUDGE_MODEL` and `configs/default.yaml`'s
`judge_model` both updated to `openai/gpt-oss-120b`, `JUDGE_VERSION` and
`configs/default.yaml`'s `judge_version` both bumped `v1` → `v2` (model
change invalidates comparability, same as a prompt change). 43 tests in
`tests/test_judge.py`/`tests/test_harness.py`/`tests/test_gate.py` still
pass unchanged (none hardcode the old model string or judge version).
**Not yet done:** no real re-baseline run against the new judge exists
yet — `eval/baselines/main.json` is a fast/retrieval-only run with
`judge_version: null`, so it's unaffected by this change, but a future
full-tier baseline update will need the new judge_version.

**Second `full-eval.yml` attempt (2026-08-19, same session): judge model
confirmed working live in CI, but the run timed out on wall clock, not on
the judge.** PR #3 ("Verify full-eval.yml with fixed judge model", branch
`ci/verify-full-eval-run-2`, Actions run `32217728998`) hit the hard
`timeout-minutes: 45` kill mid-run — step 9 ("Full eval, all metrics incl.
judge") ran exactly 42m28s (05:01:57Z→05:44:25Z) and got through 60/145
items before being cancelled. Closed/deleted same pattern as before
(PR #3, branch `ci/verify-full-eval-run-2`, remote + local).

Good news confirmed from the partial log (downloaded via `gh run download`
before cleanup): the judge fix is real — 59 real generations completed, and
judge grading calls succeeded for the overwhelming majority of them (only 4
failures out of the run's judge calls, all the same distinct new failure
mode: `400 tool_use_failed` / "Tool choice is required, but model did not
call a tool" on `citation_precision` — a real but occasional
`openai/gpt-oss-120b` function-calling miss, correctly caught by the
generic `JudgeGradingError` path same as any other grading failure,
invariant 15 held, not a new code defect to fix). No `model_not_found` 404s
recurred anywhere in this run's log — the v2 judge model is solid.

**The real blocker is budget, not the judge:** 60/145 items in 42m28s
extrapolates to **~103 minutes** for the full split — `timeout-minutes: 45`
was set from a guess (`< 25 min` per spec, already flagged as optimistic)
that undershot by roughly 4x. Setup (steps 1-8) only takes ~2.5 min, so
total realistic wall clock is ~105-110 min. GitHub-hosted `ubuntu-latest`
jobs allow up to 6 hours, and this is a public repo with unlimited Actions
minutes, so raising `timeout-minutes` has no real cost.

**`timeout-minutes` bumped to 150, committed directly to `main`
(2026-08-19, commit `01d4913`)** — done, per the plan below. Unlike the
judge-model fix, this change was accidentally made only on a throwaway
verification-PR branch the first time; that branch was closed/deleted
without merging (same pattern as always) before noticing the fix would be
lost with it, so it was reapplied and committed straight to `main`
instead. Lesson: durable code changes go on `main` directly; only the
trivial trigger commit belongs on the throwaway PR branch.

**Third `full-eval.yml` attempt (2026-08-19, same session): cancelled by
the user almost immediately — real Groq TPD rate-limiting confirmed, not
a guess.** PR #4 ("Verify full-eval.yml with corrected 150min timeout",
branch `ci/verify-full-eval-run-3`, Actions run `32251235113`) was
triggered right after the timeout bump, then cancelled by the user within
minutes once rate-limiting was visible in the live log. Confirmed from
the log before cancellation: Groq's `openai/gpt-oss-120b` TPD limit is
actually **200,000**, not the 100k this file has assumed elsewhere (see
"Groq free-tier 100k TPD..." below — that number needs correcting, not
yet done) — and it was already at ~199,000-199,900 used by item
`gs_0032` (32/145 items in), i.e. essentially exhausted before this
attempt even started, from the day's two earlier labeled runs. Every
judge call from `gs_0032` onward was hitting 429s and skipping gracefully
(`JudgeRateLimitError`, invariant 15 held, logged not silent) — correct
behavior, but pointless to let run to completion since the
judge-scored info metrics for the remaining ~113 items would all land as
`rate_limited`. PR #4 closed without merging, `ci/verify-full-eval-run-3`
deleted (remote + local), same pattern as before.

**Deferred to 2026-08-21 (day after tomorrow), per explicit user
instruction — not tomorrow, to give the Groq TPD budget more room to
actually reset and stay clear:**
1. Trigger a fresh attempt the same way as before: new branch off `main`,
   trivial commit only (no substantive fix bundled in — see lesson
   above), PR, `gh pr edit <N> --add-label run-full-eval`.
2. If it completes clean, close the trigger-vehicle PR without merging
   (same pattern each time), record real wall-clock time here, and only
   then mark this checklist item `[x]` done.
3. Worth fixing in passing: correct "100k TPD" to "200k TPD" wherever
   this file states the Groq free-tier limit (see below) — confirmed
   wrong from a real 429 response body this session.

**Deferred, not this phase:** `.github/workflows/keep-warm.yml` — no Space
exists until Phase 5, nothing to ping yet.

**Known, unfixed defects (not blocking Phase 4, flagged so they're not a
surprise later):**
- **`app.py` (Streamlit) cannot be imported** — still imports deleted
  top-level modules (`config.py`/`hybrid_database.py`/`parse.py`, removed
  in `381c51d`) and reimplements its own runtime instead of calling
  `src.runtime.get_runtime()` (contradicts invariant 14). Genuinely
  undefined names further down (`ruff` F821, confirmed real bugs).
  Excluded from `ruff` (`pyproject.toml`) rather than rewritten — that's
  separate follow-up work. `src/api.py` (FastAPI) is unaffected. Also
  still has a full upload UI despite "Repo facts" below claiming otherwise
  — that claim is stale until this file is fixed.
- **`src/parse.py` has the same stale-import disease**
  (`from config import ...` / `from helper import ...` instead of
  `.configuration`/`.helper`) — dormant (ingest-only, nothing in `tests/`
  imports it), not caught by `ruff` or `pytest`. Phase 5's
  `deploy/fetch_corpus.py` work is the likely trigger for this to matter.

**Incident (2026-08-18, remediated same day): the corpus was briefly
committed to git and public on `origin/main` for ~1 week** — `e461362`'s
`artifacts/parsed_md/*.md` files predated the `.gitignore` rule and were
never actually untracked by it. Full-history scrub via `git filter-repo`
+ force-push; current `main` is clean (verified via GitHub's tree API).
Backup bundle taken first (`~/Hecker/backups/
multimodal-agentic-rag_full_repo_backup_2026-08-18.bundle`). **Known,
accepted residual exposure:** the pre-scrub SHA is still fetchable until
GitHub's GC runs, and anyone who cloned/forked before the scrub keeps the
old content. `data/raw_pdfs/*.pdf` was confirmed never committed — scoped
to `parsed_md/` only.

## Facts carried forward from earlier phases

Non-obvious conventions later phases depend on. Full build logs for finished
phases are git history + `docs/NOISE_FLOOR.md` + `eval/judge_calibration/`,
not restated here.

**Golden set** (`eval/golden/golden_set.jsonl` 145 items + `dev_split.jsonl`
30 items, Phase 1, closed 2026-08-11): `gold_passages` (verbatim text +
`doc_id`) is the only ground truth — chunk IDs are always derived at eval
time, never stored (invariant 4). Resolved chunk IDs cache per config at
`eval/golden/resolved/<config_hash>.json`. Post-freeze edits are a `version`
bump, not a silent edit (invariant 2). Never tune against `golden_set.jsonl`
— `dev_split.jsonl` only (invariant 6). Most VLM figure captions in this
corpus are low quality (repeated "Blank Image" captions) — a Phase 6
negative-results finding, not a Phase 1 blocker.

**Eval harness** (`eval/harness.py`, `eval/metrics/*.py`, Phase 2, closed
2026-08-15/16): `eval/metrics/router.py` maps `expected_route: "refuse"`
items to `vectorstore` for router-accuracy purposes only
(`EXPECTED_ROUTE_FOR_ROUTER`) — actual refusal behavior is
`generation.py`'s `refusal_accuracy`. `eval/metrics/structured.py`'s
`InstrumentingLLM` wraps `llm_model` to surface structured-output parse
failures/retries as data; `silent_coercion_rate` is a best-effort heuristic
that returns `None` (not a fake number) when it can't be checked.
`retrieve_and_rerank_core` (`src/agent.py`, module-level) lets
`--retrieval-only` exercise the exact retrieval path the real graph uses.
`correction_improve_rate`/`degrade_rate` always return `None` — needs a
`GraphState` change to preserve pre-correction generation text, not yet
scoped. Judge is pinned to `llama-3.3-70b-versatile` on Groq
(`DEFAULT_JUDGE_MODEL`, `eval/judge.py`) — every `with_structured_output()`
call needs `method="function_calling"` (Groq's `json_schema` mode only
supports the gpt-oss family). `JUDGE_VERSION` (`"v1"`) must move in
lockstep with `configs/default.yaml`'s `judge_version` on any prompt change
(invariant 11). `TAVILY_MODE=replay` (default) reads
`eval/fixtures/tavily_v1.json`, keyed by sha256 of `(query, max_results,
topic)`; a residual ~10% miss rate on the fast split's `web_search`
fallback is expected, not a bug (LLM-rewritten query text varies slightly
run-to-run even at `temperature: 0`). `temperature` is a first-class
`config_rag()` field (`TEMPERATURE` env var), `None` by default (prod
unaffected), pinned to `0` in `configs/default.yaml` (eval-only).

**Groq free-tier TPD token budget is a real, already-hit constraint**
(hit repeatedly: Phase 2, Phase 3, and twice more in one Phase 4 session
on 2026-08-19) — a handful of consecutive full-mode `eval.harness` runs
in one day exhausts it; `JudgeGradingError`/`JudgeRateLimitError` skips
gracefully rather than crashing, visible as smaller `n_scored`
(`n_rate_limited` specifically), not a quality regression. **Correction
(2026-08-19): the actual limit is 200,000 TPD, not 100k** — this file
previously said "100k," which was never verified against a real error
body; a real Groq 429 during the third `full-eval.yml` attempt quoted
`Limit 200000` directly for `openai/gpt-oss-120b`. As of 2026-08-18,
generation moved off Groq to `nvidia_nim` locally and in CI
(`full-eval.yml`) — this budget is now the judge's alone (invariant 11
keeps it pinned to Groq regardless), not shared with generation anymore.

**Noise floor** (`eval/noise_floor.py`, Phase 3, closed 2026-08-16, full
write-up `docs/NOISE_FLOOR.md`): fast split (39 items), n=4 not 5 — same
Groq-budget reasoning, documented deviation. `--existing-results` resume
flag seeds the aggregate from already-completed run JSONs so a killed pass
doesn't waste spent quota. Headline non-judge metrics have clean n=4
coverage (feeds the Phase 4 gate table above); judge-scored metrics
(`faithfulness`, `citation_precision`) got n=1 — no variance computable,
hence informational-only in the gate table, not a papered-over number
(invariant 16).

## Non-goals (do not implement, park in `docs/BACKLOG.md`)

New retrieval strategies (HyDE/RAPTOR/GraphRAG/ColBERT), multi-document scoping,
MCP integration, streaming tokens, conversation persistence, Docling migration,
image-input queries, UI redesign, any new agent hop/tool. Three exceptions are in
scope: defect fixes, the vLLM backend as a *measurement instrument* (spec §4A, not
a deployed feature), and CPU-viable reranking (deploy target has no GPU).

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

## Repo facts

- **Corpus is not committed to git** (`data/raw_pdfs/*.pdf`,
  `artifacts/parsed_md/*.md`) — license restrictions per paper
  (`artifacts/SOURCES.md`). `artifacts/metadata.jsonl` and `corpus_seed.csv`
  stay committed. The corpus must exist locally (built via `parse.py`'s
  pipeline; `deploy/fetch_corpus.py` is planned, not built) before `pytest`
  runs fully or the index rebuilds. Corpus-dependent tests skip cleanly, not
  fail, when it's absent.
- **Module layout:** core modules under `src/` (`runtime.py`, `api.py`,
  `agent.py`, `hybrid_database.py`, `parse.py`, `configuration.py`,
  `helper.py`, `logging_utils.py`); `app.py`/`pages/1_Setup.py` at repo root.
- **Uploads are rejected** — corpus is frozen, `app.py` is *intended* to have
  no upload UI. **Stale as of 2026-08-18: `app.py` still has a full upload
  flow** (see Phase 4's `app.py` known-defect note) — this line describes
  the intended end state, not current code, until that file is fixed.

## Implementation notes for later phases

- **Config-only overrides:** `config_rag(overrides={...})` applies a dict on top of the
  resolved config (`defaults → api_keys.json → env → overrides`). This is the mechanism
  the eval harness and ablation runner (Phase 6) construct a run from — reach for this,
  not env vars.
- **Route decision isn't a `GraphState` field.** `query_router` is a conditional-edge
  selector, so its choice is never written into state. Use
  `agent.run_query_with_state()` — returns `(answer, final_state, trace_info)`, and
  `trace_info["node_sequence"][0]` is the route decision.
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
(all Phase 5), CI workflow wiring the harness as a gate (Phase 4 — see checklist above).

## Working agreements for this upgrade

- **Track the current phase's steps in this file as checkboxes; check them off as
  they land.** Once a phase closes, collapse its detail into "Facts carried forward"
  (only what later phases actually need) and remove the checklist — git history and
  the committed docs (`docs/NOISE_FLOOR.md` etc.) are the changelog, not this file.
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
  checkpoint numbering.
- Pause for user confirmation between phase milestones before proceeding to the next one.
