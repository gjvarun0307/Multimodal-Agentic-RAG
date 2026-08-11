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

**Current phase: PHASE 0 — Unblock. Not started.** No task below is checked off yet;
this repo is exactly the pre-upgrade state the spec's audit describes (verified by
re-reading the code on 2026-08-11, see "Audit confirmed" below).

Phases are dependency-ordered — do not start Phase 1 before Phase 0's acceptance
criteria are met (`PROJECT_SPEC.md` §7, Phase 0 section). Full phase list: 0 Unblock →
1 Golden set → 2 Metrics harness → 3 Noise floor → 4 Eval-as-CI → 5 Deploy &
observability → 6 Ablations & failure taxonomy → 7 README rewrite.

### Git history — resolved: reseeded

`PROJECT_SPEC.md` Phase 0 task 1 ("a clean first commit beats preserved history") was
followed literally: this commit is a fresh orphan history for `main` — one clean
initial commit containing the full pre-upgrade working tree (code, docs, this file),
old commit-by-commit history discarded from `main` going forward. Approved explicitly
by the user, destructive-and-fine.

- The old 28-commits-ahead-of-`origin/main` history is **not deleted from disk** —
  it's preserved under the local branch `archive/pre-reseed-2026-08-11` (pointing at
  the old tip, `cfa8773`) as a recovery net, but `main` no longer contains it and no
  future work should merge from it.
- `origin` (`github.com/gjvarun0307/Multimodal-Agentic-RAG.git`) still has the **old**
  history — reseeding was local-only. Pushing this new history to `origin/main`
  requires a force-push, which was **not done** and needs separate explicit
  confirmation before it happens (force-pushing a public remote is its own
  destructive action, distinct from the local reseed).

## Audit confirmed (2026-08-11)

Re-read every file `PROJECT_SPEC.md` §5 cites. All findings check out against the code
as it exists right now — this is not stale from when the spec was written:

- Line counts match spec's module map exactly: `app.py` 242, `agent.py` 705,
  `hybrid_database.py` 291, `parse.py` 218, `configuration.py` 305.
- Corpus: 15 PDFs in `data/raw_pdfs/`, 15 `.md` in `data/raw_pdfs_parsed/`, 15 lines in
  `metadata.jsonl` — counts align today, but §5.1's point stands: alignment is by
  **mtime-sort zipped against line order**, not a real key, so it is one re-parse away
  from silent misalignment.
- `metadata.jsonl` confirmed schema-less: each line is only
  `{paper_name|title, authors, year, topic}`, no `doc_id`, no source filename, no hash.
- `hybrid_database.py:29` (`prepare_input_data`) confirmed: `pdf_files.sort(key=os.path.getmtime)`
  then positional zip against `metadata_list[idx]`.
- `hybrid_database.py:116`: `auto_id=True` confirmed — chunk IDs are not deterministic.
- `logging_utils.py`: parent logger `multimodal_rag` set to `INFO` in `setup_logger()`;
  `get_logger()` returns children with no explicit level (`NOTSET`), which inherit the
  effective `INFO` — every `logger.debug(...)` call across the codebase is confirmed
  silently dropped.
- `agent.py:694` `create_agent()` confirmed unreferenced anywhere else in the repo —
  dead code, per spec §5.4 slated for deletion (not deprecation).
- `configuration.py:292` `config_rag()` confirmed to return only 10 keys
  (`device`, `tavilly_api_key`, `database_path`, `input_folder_path`, `chunk_size`,
  `overlap_size`, `domain_topics`, `llm_provider`, `llm_model`, `llm_api_key`); every
  other tunable named in spec §5.5 (`search_limit`, `reranker_top_k`, `sparse_weight`,
  `dense_weight`, score threshold, `max_gen_retries`, etc.) is hardcoded at call sites.
- No `tests/`, `pyproject.toml`, `.pre-commit-config.yaml`, `src/`, `eval/`,
  `.github/workflows/`, or `docs/` directories exist yet — Phase 0 creates all of them.
- `requirements.txt` has exactly two pinned deps (`transformers==4.51.3`,
  `flagembedding==1.3.5`); everything else unpinned, no `requirements.lock`.
- Repo hygiene already partly done by prior commits: no stray `=1.3.3` file, no
  `README_old.md`, no `graph_rag.ipynb`/`collab_run/` — these were removed in
  `cfa8773` (chore: remove obsolete docs/notebook and stray repo clutter), so spec
  §Known-issue-6 / Phase-0-task-1's cleanup portion is already satisfied.
- `api_keys.json` and `milvus.db/` are both present locally but correctly **not**
  tracked by git (confirmed via `git ls-files`) and both covered by `.gitignore`.
  `PROJECT_SPEC.md`'s untracked-file warning does not currently apply — keep verifying
  this on every commit, since Phase 0 touches `.gitignore`-adjacent config a lot.

## Non-goals (do not implement, park in `docs/BACKLOG.md`)

New retrieval strategies (HyDE/RAPTOR/GraphRAG/ColBERT), multi-document scoping,
MCP integration, streaming tokens, conversation persistence, Docling migration,
image-input queries, UI redesign, any new agent hop/tool. Three exceptions are in
scope: the §5 defect fixes, the vLLM backend as a *measurement instrument* (§4A, not a
deployed feature), and CPU-viable reranking (deploy target has no GPU).

## Global invariants (apply from Phase 0 onward — `PROJECT_SPEC.md` §8)

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

## Phase 0 — Unblock: task checklist

Full detail and rationale in `PROJECT_SPEC.md` §5 and §7 (Phase 0). Copied here as the
actionable list; check items off in this file as they land, one commit per item where
practical.

- [ ] **Decide the git-history question above** before anything else touches `.git`
- [ ] Repo hygiene: audit imports vs `requirements.txt`, strip unused deps, verify `.gitignore`
- [ ] Fix `logging_utils.py` level mismatch; add `LOG_LEVEL` env var (default `INFO`)
- [ ] Regenerate `metadata.jsonl` from scratch (§5.1): hand-assign 15 `doc_id` slugs,
      verify PDF↔MD correspondence by hand, hand-correct VLM-generated title/author/year,
      add `content_sha256`. Author `artifacts/corpus_seed.csv`, write `deploy/build_metadata.py`.
- [ ] Delete the mtime-sort/positional-zip in `hybrid_database.py:29-37`; replace with a `doc_id`-keyed dict join; raise on any unmatched entry
- [ ] Migrate Milvus schema to `auto_id=False` + deterministic VARCHAR chunk IDs
      (`{doc_id}::{chunk_index:04d}::{content_sha8}`, §5.2); new collection name `arag_project_v2`
- [ ] Create `src/runtime.py` → `get_runtime()`; migrate `app.py`/`agent.py` onto it; delete `create_agent()` (`agent.py:694`) outright
- [ ] Promote every hardcoded tunable into `Config`/`config_rag()` with documented precedence (defaults → `api_keys.json` → env → explicit override)
- [ ] Plumb `retrieved_chunk_ids` (pre-rerank) and `reranked_chunk_ids` (post-rerank, post-threshold) through `GraphState`
- [ ] Set chunk-level `doc_id` metadata; keep `source_file` one release for the active-document filter, then plan its removal
- [ ] Verify stateless headless graph invocation — no `st.session_state` dependency, no cross-query residue (§5.6)
- [ ] Add a CPU-viable reranker option (`bge-reranker-v2-m3` or `bge-reranker-base`); CPU fallback must be loud/logged, never silent (currently hard-skipped — §5.4.2 in spec's §4.2 item 2)
- [ ] Add `ruff`, `pytest`, `pre-commit`, `pyproject.toml`
- [ ] Pin `requirements.txt` exactly; generate `requirements.lock`; record Python version
- [ ] Add vLLM as a config-only `LLM_PROVIDERS` entry (§4A) — no runtime coupling
- [ ] Create `src/api.py` (FastAPI): `/query` (full trace), `/health`, `/metrics`, `/trace/{id}` — both adapters call `get_runtime()` in-process
- [ ] Check per-paper licenses; write `artifacts/SOURCES.md`; decide committed-vs-fetched per `.md`

**Acceptance criteria for closing Phase 0** are the checklist in `PROJECT_SPEC.md` §7
(Phase 0 section) — `pytest` passes with a headless end-to-end smoke query, `ruff check .`
clean, metadata validator clean with full bijection, double-build chunk-ID determinism
test passes, config-only runtime construction, fresh-clone reproducibility from
`requirements.lock`.

## Target directory structure (post-Phase-0)

See `PROJECT_SPEC.md` §6 for the complete tree (`src/`, `eval/`, `deploy/`, `artifacts/`,
`observability/`, `tests/`, `docs/`, `.github/workflows/`). Do not partially adopt it —
e.g. don't create `src/runtime.py` while leaving `agent.py`/`hybrid_database.py` at repo
root; the module map in §3.1 vs. §6 is the "as-is" vs. "target" pair to reconcile in one
pass per file, not piecemeal.

## Current architecture (as-is, pre-Phase-0)

Condensed from `PROJECT_SPEC.md` §3 — read that section for the full node inventory and
stack table.

- **Modules (repo root, not yet under `src/`):** `app.py` (Streamlit + `load_runtime()`),
  `agent.py` (`GraphState` + graph nodes + `build_agent_graph()`, plus dead
  `create_agent()`), `hybrid_database.py` (Milvus schema/search/rebuild),
  `parse.py` (LlamaParse + Qwen2.5-VL captioning), `configuration.py` (`Config`
  singleton, `LLM_PROVIDERS`, `build_llm_client()`), `config.py` (back-compat shim),
  `helper.py`, `logging_utils.py`, `pages/1_Setup.py` (BYOK dashboard).
- **Graph:** `query_router` → `chitchat_node` / `web_search` / `retrieve_and_rerank`;
  no-docs path → `rewrite_query` (capped at 3) → `web_search` (or `generate` directly
  if scoped to an active uploaded document); `generate` → hallucination/relevance
  grading → `END` / regenerate / `rewrite_query`.
- **Known structural gap (§5.4):** `agent.py:get_models()` and `app.py:load_runtime()`
  independently construct `database`/`embedding_model`/`rerank_model` — only the LLM
  client is unified via `build_llm_client()`. This already caused one fix (CPU reranker
  skip) to land in one path and silently miss the other. `src/runtime.py` in Phase 0
  closes this permanently — don't fix one side and call it done.
- **Corpus:** 15 papers (LLM serving/architecture, RAG methodology, ConvNeXt), frozen
  for the duration of the upgrade (invariant 2 above).

## Commands

Current (pre-Phase-0, still root-level):
```bash
pip install -r requirements.txt
python parse.py              # data/test_pdfs/*.pdf -> data/test_pdf_parsed/*.md + metadata.jsonl
python hybrid_database.py    # (re)build ./milvus.db — DESTRUCTIVE, drops + recreates the collection
streamlit run app.py
```

Target (post-Phase-0, see `PROJECT_SPEC.md` §9 for the full list):
```bash
pytest
ruff check . && ruff format .
python -m eval.harness --config configs/default.yaml --split full
python -m eval.harness --config configs/default.yaml --split fast --retrieval-only
python -m eval.noise_floor --runs 5
python deploy/build_metadata.py --seed artifacts/corpus_seed.csv --out artifacts/metadata.jsonl
uvicorn src.api:app --reload --port 8000
streamlit run app.py
```

## Working agreements for this upgrade

- **Every phase ends with a commit and a `CLAUDE.md` update** (spec §7) — update the
  Phase state section above and check off completed checklist items as part of that
  phase's final commit, not as an afterthought later.
- **Config-only ablations, no exceptions** (invariant 10) — if implementing an ablation
  row requires editing code rather than a config value, the config promotion work
  (Phase 0 task) is incomplete; fix that instead of hand-editing for one run.
- **Say when something is unverified.** This repo has had a running pattern (see prior
  commits' "Not run end-to-end" notes) of fixes landing without live execution because
  the working environment lacks deps/GPU/API keys. Keep doing that — state plainly what
  was read/reasoned vs. what was actually run, per the spec's "provably works" goal.
- Never commit `api_keys.json`, `milvus.db`, `logs/`, `fail_logs.txt`, or any real key/secret.
- Prefer `git add <path>` over `git add -A`; check `git status` before staging.
- Message format: short imperative subject (`feat:`/`fix:`/`refactor:`/`docs:`/`chore:`), body explaining why when not obvious from the diff.
