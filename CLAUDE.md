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

1. **Corpus delivery to CI: private Hugging Face Dataset.** CI has no
   corpus/index today and the corpus is deliberately not committed
   (`artifacts/SOURCES.md` — license restrictions per paper).
   `deploy/fetch_corpus.py` / `deploy/build_ingest_artifacts.py` (spec's
   intended long-term fix) are Phase 5 deliverables that don't exist yet.
   Upload `milvus.db` + `artifacts/parsed_md/` once to a *private* HF
   Dataset; CI fetches it via an `HF_TOKEN` secret. Re-upload whenever
   corpus/chunking/embedding model changes — it's a build artifact, not
   source of truth (`artifacts/metadata.jsonl` / `corpus_seed.csv` stay
   canonical).
2. **Fast tier = zero LLM calls, literally.** Spec's tier table says fast =
   "retrieval + structured, no judge" but its acceptance criteria say "zero
   LLM calls" — structured-output metrics call the generation LLM, which
   risks real Groq-quota cost on every push. Fast tier =
   `eval.harness --split fast --retrieval-only` +
   `test_chunk_id_determinism.py` only. Structured-output metrics move into
   the nightly/labeled full tier with the judge, where cost is already
   budgeted.

**Gate thresholds** (from `docs/NOISE_FLOOR.md`, fast split, n=4 —
deliberately overrides spec's literal gate table, which blocks on
`faithfulness` and `refusal_accuracy`; Phase 3 found those judge metrics n=1
or unusable σ, so they're informational-only until a dedicated judge-only
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

**Already usable, no new build needed:** `eval.harness --split fast
--retrieval-only` (live-verified Phase 2), `TAVILY_MODE=replay` (no live key
needed), `tests/test_chunk_id_determinism.py` (Phase 0) — CI should invoke
it as a real pytest job, not reimplement the check.

**Real timing finding (2026-08-18), overrides the fast-tier "< 3 min"
budget:** a live `eval.harness --split fast --retrieval-only` run against
the real 39-item split, on CPU (no GPU), took **~12 minutes wall clock** end
to end — zero LLM calls as designed, but real BGE-M3 dense+sparse query
embedding and `bge-reranker-v2-m3` cross-encoder scoring per item is
genuinely slow on CPU. GitHub-hosted Actions runners have no GPU either, so
this isn't a local-machine-only artifact. **Decided with the user
2026-08-18: accept ~12–15 min as the real fast-tier budget, keep the full
39-item split (preserves Phase 3's noise-floor thresholds, which were
computed against exactly this split), document "< 3 min" as an unrealistic
spec target rather than shrinking the item count.** Cost is still $0 (public
repo, unlimited Actions minutes) — only PR feedback latency changes, not
correctness or noise-floor validity. `.github/workflows/fast-eval.yml`'s job
`timeout-minutes: 30` reflects this (bumped from an initial 15 after a real
Actions run hit ~15m15s total and got killed by the timeout, not a real
failure — every step through the harness run itself succeeded).

**Build checklist** (check off as each lands; repo confirmed public
2026-08-18 — unlimited free Actions minutes, no action needed):

- [x] Upload `milvus.db` + `artifacts/parsed_md/` to a private HF Dataset —
      done 2026-08-18, `gjvarun0307/arag-eval-corpus` (private). CI needs a
      separate **read-only** fine-grained `HF_TOKEN` scoped to this repo
      (not the write-scoped `upload-bundle-token` used to create it).
- [x] `.github/workflows/fast-eval.yml` — every push/PR, retrieval-only +
      determinism. Budget ~13–15 min (see timing finding above), not spec's
      literal < 3 min; $0 either way. **Fully green end-to-end in a real
      Actions run, 2026-08-18** (run 32119217309, `timeout-minutes: 30`).
- [ ] `.github/workflows/full-eval.yml` — nightly + `run-full-eval` label,
      full metric set incl. judge, < 25 min
- [x] `.github/workflows/lint.yml` — `ruff check .` + `pytest`. **Fully
      green in a real Actions run, 2026-08-18** (run 32119217301).

**Known defect found while wiring lint.yml (2026-08-18), not fixed —
`app.py` (Streamlit) cannot be imported.** It still imports the deleted
top-level `config.py`/`hybrid_database.py`/`parse.py` (removed in `381c51d`,
"move core modules into src/, delete config.py shim") and reimplements its
own `load_runtime()`/`build_graph()`/`run_query()` instead of calling
`src.runtime.get_runtime()` — directly contradicting invariant 14 and
`runtime.py`'s own docstring. It's a partial migration, not just stale: the
top ~450 lines are the pre-refactor implementation, but further down
(`main()`) references `runtime.database`/`runtime.embedding_model` and
`MAX_CHAT_TURNS` as if the migration had already happened — both are
genuinely undefined names (`ruff` F821, confirmed real bugs not style).
`src/api.py` (FastAPI) is unaffected and confirmed working (`test_api.py`,
`test_smoke.py` both pass against the real `get_runtime()` path). Separately,
this file still has a full upload UI (`st.file_uploader`, `ingest_uploaded_pdf`)
despite CLAUDE.md's "Repo facts" claiming uploads were rejected and no
upload UI exists — that claim is stale and needs reconciling once this file
is actually fixed. **Decided with the user 2026-08-18: excluded from `ruff`
(`pyproject.toml` `[tool.ruff] exclude`) rather than rewritten now — the
rewrite (get_runtime()-based, plus resolving the upload-UI question) is
separate follow-up work, not folded into Phase 4 CI setup.**

**Same disease, different file: `src/parse.py` also imports deleted
top-level modules** (`from config import config_parse`, `from helper import
clean_json_text` — should be `.configuration`/`.helper`, matching every
other file under `src/`). Not caught by `ruff` (no import-resolution rule
selected) or by the full pytest run (nothing in `tests/` currently imports
`src.parse` — it's ingest-only, never on the query path per its own module
docstring, so this has been silently dormant). Not fixed or excluded here
since it isn't blocking anything today; flagged so it isn't a surprise
whenever ingest/re-parsing is next touched (Phase 5's `deploy/fetch_corpus.py`
work is the likely trigger).
- [ ] Gate-comparison script — diff a run vs. `eval/baselines/main.json`
      against the table above, emit pass/warn/block, render the PR comment
      diff table (spec §7 example)
- [ ] `eval/baselines/main.json` — first commit from a real `eval.harness`
      run's results JSON (git SHA, config, golden-set/judge/Tavily-fixture
      versions — invariant 7), never hand-written. Baseline updates after
      this are PR-only with written justification, never automatic.
- [ ] Rate-limit/quota errors rendered distinctly from real regressions, in
      both gate logic and PR comment (a Groq 429 must never show as ❌ FAIL)
- [ ] A deliberately-broken retrieval config, run through the fast workflow,
      proving the gate actually blocks a test PR (acceptance criterion)
- [ ] Screenshot a real blocked-merge run for the Phase 7 README

**Deferred, not this phase:** `.github/workflows/keep-warm.yml` — no Space
exists until Phase 5, nothing to ping yet.

**Incident (2026-08-18): the corpus was actually committed to git and public
on `origin/main` for ~1 week, contradicting the "corpus is not committed"
claim this whole plan (and the private-HF-Dataset decision above) rests on.**
All 15 `artifacts/parsed_md/*.md` files were added in `e461362` (part of the
2026-08-11 reseed, not caused by any Phase 4 work) and were never actually
gitignored-in-practice — `.gitignore` listing a path doesn't untrack files
already committed before the rule existed. Found because `lint.yml`'s pytest
run didn't skip `test_harness_live.py`/`test_resolve_passages.py`'s
real-corpus tests on a fresh CI checkout, which should have been the tell
that the corpus was actually present there. **Remediated the same day:**
full-history scrub via `git filter-repo --path artifacts/parsed_md
--invert-paths --force` + force-push to `origin/main` (backup bundle taken
first, `~/Hecker/backups/multimodal-agentic-rag_full_repo_backup_2026-08-18.bundle`,
covers `main` + the local-only `archive/pre-reseed-2026-08-11` branch, which
was never on origin and wasn't otherwise touched). Current `main` tree and
all reachable history are clean (verified via GitHub's tree API). **Known,
accepted residual exposure, not further fixable from here:** the pre-scrub
commit SHA is still directly fetchable from GitHub until their scheduled GC
runs (force-push doesn't instantly purge unreachable objects), and anyone
who already cloned/forked before today keeps the old content regardless.
`data/raw_pdfs/*.pdf` was checked and confirmed never committed — this
incident was scoped to `parsed_md/` only.

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

**Groq free-tier 100k TPD token budget is a real, already-hit constraint**
(hit twice: Phase 2, Phase 3) — a handful of consecutive full-mode
`eval.harness` runs in one day exhausts it; `JudgeGradingError` skips
gracefully rather than crashing, visible as smaller `n_scored`, not a
quality regression.

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
