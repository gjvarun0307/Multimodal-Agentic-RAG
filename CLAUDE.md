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

**Phase 0 — Unblock: COMPLETE (2026-08-11).** All 16 checklist items landed, one commit
each, in dependency order (M1 pre-flight through M13 acceptance pass). Every acceptance
criterion in `PROJECT_SPEC.md` §7 (Phase 0 section) was verified for real — not just
reasoned through — including a genuine fresh-clone-and-`pip install -r requirements.lock`
reproducibility check in a scratch directory.

**Phase 1 — Golden evaluation set is next.** Read `PROJECT_SPEC.md` §7 (Phase 1 section)
before starting; do not begin until re-confirming the acceptance criteria below still
hold (re-run `pytest` and `ruff check .` fresh if any time has passed).

Full phase list: 0 Unblock (done) → 1 Golden set → 2 Metrics harness → 3 Noise floor →
4 Eval-as-CI → 5 Deploy & observability → 6 Ablations & failure taxonomy → 7 README rewrite.

### Git history

- Phase 0 task 1 ("a clean first commit beats preserved history") was followed
  literally before Phase 0 work began: `main`'s history was reseeded to a fresh orphan
  commit. The old 28-commits-ahead-of-`origin/main` history is preserved on the local
  branch `archive/pre-reseed-2026-08-11` (tip `cfa8773`) as a recovery net; `main`
  doesn't contain it.
- During Phase 0 (licensing review, see below), `artifacts/parsed_md/*.md` — committed
  in the metadata-regeneration commit — was removed from `main`'s history entirely via
  `git filter-repo --refs main` (not just untracked going forward), since several
  papers' licenses don't clearly permit redistributing full text. Scoped to `main` only
  (`--refs` implies `--partial`, which skips `origin` remote changes and other refs);
  `archive/pre-reseed-2026-08-11` and `origin` were verified untouched. A temporary
  safety branch was created before the rewrite, verified, then deleted with
  `git gc --prune=now` after confirming the rewrite succeeded.
- `origin` (`github.com/gjvarun0307/Multimodal-Agentic-RAG.git`) still has the **old**
  (pre-reseed) history — none of the above has been pushed. Local `main` and
  `origin/main` share zero common ancestry (`git merge-base` returns nothing). Pushing
  requires a force-push, which needs separate explicit confirmation before it happens.

### Corpus text is not committed

Neither `data/raw_pdfs/*.pdf` nor `artifacts/parsed_md/*.md` is tracked by git — see
`artifacts/SOURCES.md` for the full per-paper license audit (7 of 15 papers are CC BY
4.0 / CC0 and would have been fine to commit; the other 8 either only carry arXiv's
default non-exclusive-to-arXiv license or an ACM notice requiring permission for
server-posting, so all 15 were excluded uniformly rather than split by license).
`artifacts/metadata.jsonl` and `artifacts/corpus_seed.csv` (bibliographic metadata +
content hashes, no substantial text) stay committed.

**Practical effect:** a fresh clone has no corpus on disk. `tests/test_metadata_alignment.py`,
`tests/test_chunk_id_determinism.py`, and `tests/test_smoke.py` detect this and `skip`
(not fail) with a clear reason — verified directly by temporarily moving
`artifacts/parsed_md/` aside and confirming all 13 corpus-dependent tests report
`SKIPPED`. To run them, the corpus must exist locally first: `data/raw_pdfs/*.pdf` +
`artifacts/parsed_md/*.md` (today: re-run `parse.py`'s pipeline against locally-held
PDFs; a `deploy/fetch_corpus.py` that re-downloads from arXiv is listed in the target
tree but not yet built — Phase 5 territory).

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

## Phase 0 — Unblock: task checklist (COMPLETE)

Full detail and rationale in `PROJECT_SPEC.md` §5 and §7 (Phase 0). One commit per item,
in this order:

- [x] **Decide the git-history question** — reseeded (see Git history above)
- [x] Repo hygiene: audit imports vs `requirements.txt`, strip unused deps, verify `.gitignore`
- [x] Fix `logging_utils.py` level mismatch; add `LOG_LEVEL` env var (default `INFO`) — verified live: default drops `.debug()`, `LOG_LEVEL=DEBUG` shows it
- [x] Regenerate `metadata.jsonl` from scratch (§5.1): 15 hand-assigned `doc_id` slugs,
      PDF↔MD correspondence verified (filenames already 1:1 by stem), titles/authors/
      years read directly from each PDF (not the old VLM captions — corrected 2/15
      placeholder rows), `content_sha256` added. `artifacts/corpus_seed.csv` +
      `deploy/build_metadata.py` written.
- [x] Delete the mtime-sort/positional-zip in `hybrid_database.py`; replaced with a `doc_id`-keyed dict join on `metadata.jsonl`'s `source_md` field; raises on any unmatched entry
- [x] Migrate Milvus schema to `auto_id=False` + deterministic VARCHAR chunk IDs
      (`{doc_id}::{chunk_index:04d}::{content_sha8}`); new collection `arag_project_v2` — rebuilt for real against all 15 papers, 2,694 entities (matches spec's own audited count)
- [x] Create `src/runtime.py` → `get_runtime()`; migrated `app.py`/`agent.py` onto it; deleted `get_models()`/`create_agent()` outright (confirmed dead)
- [x] Promote every hardcoded tunable into `Config`/`config_rag()` — 10 keys → 22, precedence `defaults → api_keys.json → env → explicit override` (the last via `config_rag(overrides=...)`, new)
- [x] Plumb `retrieved_chunk_ids` (pre-rerank) and `reranked_chunk_ids` (post-rerank, post-threshold) through `GraphState` — verified live through multiple graph iterations including a correction-loop firing
- [x] Set chunk-level `doc_id` metadata; kept `source_file` for this release
- [x] Verify stateless headless graph invocation — confirmed no `st.session_state`/mutable-global coupling in `agent.py`
- [x] Add a CPU-viable reranker option (`bge-reranker-v2-m3`, now the default); CPU fallback loud/logged via shared `build_reranker()`, never silent — downloaded and verified live on this CPU machine
- [x] Add `ruff`, `pytest`, `pre-commit`, `pyproject.toml`
- [x] Pin `requirements.txt` exactly; generate `requirements.lock`; Python 3.13.2 recorded
- [x] Add vLLM as a config-only `LLM_PROVIDERS` entry — `base_url` is a runtime override (`vllm_base_url`), not fixed, since it's an ephemeral tunnel URL
- [x] Create `src/api.py` (FastAPI): `/query` (full trace: route, both chunk ID lists, correction signal, per-stage latency), `/health`, `/metrics`, `/trace/{id}` (404 until Phase 5) — verified live via `uvicorn`, real query against the live collection
- [x] Check per-paper licenses (arXiv HTML `license` field, read directly per paper); wrote `artifacts/SOURCES.md`; decided uniformly not-committed rather than split by license

**Acceptance criteria** (`PROJECT_SPEC.md` §7, Phase 0 section) — all verified for real:
- [x] `pytest` passes (32 tests when corpus present; 18 pass + 13 skip cleanly on a fresh clone), including a headless end-to-end smoke query (`tests/test_smoke.py`, real LLM call)
- [x] `ruff check .` clean
- [x] Metadata validator clean, full bijection, mtime-shuffle invariance confirmed by direct simulation
- [x] Double-build chunk-ID determinism test passes (`tests/test_chunk_id_determinism.py`)
- [x] Config-only runtime construction (`config_rag(overrides=...)` → `get_runtime(config)`)
- [x] Fresh-clone reproducibility from `requirements.lock` — actually done: cloned locally into a scratch dir, fresh venv, `pip install -r requirements.lock`, `ruff check .` clean, 18/31 tests pass and 13 skip cleanly (no corpus/credentials shipped, by design)

## Target directory structure

See `PROJECT_SPEC.md` §6 for the complete target tree. Achieved as of Phase 0:
`src/` (`runtime.py`, `api.py`, `agent.py`, `hybrid_database.py`, `parse.py`,
`configuration.py`, `helper.py`, `logging_utils.py`), `deploy/build_metadata.py`,
`artifacts/` (`corpus_seed.csv`, `metadata.jsonl`, `SOURCES.md`, `parsed_md/` —
git-ignored, local-only), `tests/`, `pyproject.toml`, `.pre-commit-config.yaml`,
`requirements.lock`. Not yet built (later phases): `eval/`, `configs/`,
`observability/`, `docs/` (besides this file), `.github/workflows/`,
`deploy/build_ingest_artifacts.py`, `deploy/fetch_corpus.py`, `deploy/record_demo_traces.py`,
`openapi.json`.

## Current architecture (post-Phase-0)

Condensed from `PROJECT_SPEC.md` §3 — read that section for the full node inventory and
stack table. This section describes what's real as of Phase 0's close; update it again
at the end of each subsequent phase rather than letting it drift.

- **Modules, all under `src/`:** `runtime.py` (`Runtime` dataclass + `get_runtime()` —
  the single construction path for database/embedding_model/rerank_model/llm),
  `api.py` (FastAPI: `/query`, `/health`, `/metrics`, `/trace/{id}`), `agent.py`
  (`GraphState` + graph nodes + `build_agent_graph()` + `run_query_with_state()`),
  `hybrid_database.py` (Milvus schema/search/rebuild, `build_chunks()` for pure
  chunk-ID-assignment), `parse.py` (LlamaParse + Qwen2.5-VL captioning, ingest-only),
  `configuration.py` (`Config` singleton, `LLM_PROVIDERS`, `build_llm_client()`,
  `build_reranker()`), `helper.py`, `logging_utils.py`. `app.py` (Streamlit) and
  `pages/1_Setup.py` (BYOK dashboard) stay at repo root. `config.py` shim deleted.
- **Graph:** `query_router` → `chitchat_node` / `web_search` / `retrieve_and_rerank`;
  no-docs path → `rewrite_query` (capped at `max_rewrites`, now config-driven) →
  `web_search`; `generate` → hallucination/relevance grading → `END` / regenerate /
  `rewrite_query`. Upload-driven active-document scoping (`active_document`/
  `scope_to_active_document` in `GraphState`) still exists in the graph but nothing
  sets it anymore (see Uploads below) — vestigial, not removed, since it's general
  architecture rather than upload-specific code.
- **Structural gap closed:** `get_models()`/`create_agent()` (duplicated model
  construction) deleted outright. `build_reranker()` and `get_runtime()` are the only
  paths that construct the reranker/database/embedding-model/LLM client; `app.py` and
  `src/api.py` both call `get_runtime()` in-process (invariant 14).
- **Uploads:** rejected. Corpus is frozen for this upgrade (invariant 2); `app.py`'s
  upload UI (file uploader, `ingest_uploaded_pdf()`, VLM preload) was removed rather
  than given a `doc_id`-assignment scheme. `hybrid_database.py`'s
  `append_parsed_file_to_database()` and `parse.py`'s `parse_single_file()` are
  unreachable now but left in place (legitimate library functions, not broken code).
- **Corpus:** 15 papers, frozen for the duration of the upgrade (invariant 2). Not
  committed to git (see Corpus text is not committed, above) — 2,694 chunks total when
  built, matching `PROJECT_SPEC.md`'s own audited count exactly.

## Commands

```bash
# setup
pip install -r requirements.lock      # exact reproducibility (or requirements.txt for direct deps only)

# dev
pytest                                # 32 tests when corpus present; skips corpus-dependent ones otherwise
ruff check . && ruff format .

# corpus / metadata (requires data/raw_pdfs/*.pdf + artifacts/parsed_md/*.md locally)
python deploy/build_metadata.py --seed artifacts/corpus_seed.csv --out artifacts/metadata.jsonl
python -m src.hybrid_database          # (re)build ./milvus.db -- DESTRUCTIVE, drops + recreates arag_project_v2

# serve
uvicorn src.api:app --reload --port 8000
streamlit run app.py
```

Not yet available (later phases — see `PROJECT_SPEC.md` §9 for the full target list):
`python -m eval.harness`, `python -m eval.noise_floor`, `python -m eval.ablations`,
`python -m eval.failure_taxonomy`, `python -m eval.resolve_passages`,
`python -m eval.validate_golden`, `python -m eval.tavily_cache`,
`python deploy/fetch_corpus.py`, `python deploy/build_ingest_artifacts.py`,
`python deploy/record_demo_traces.py`.

## Working agreements for this upgrade

- **Every phase ends with a commit and a `CLAUDE.md` update** (spec §7) — update the
  Phase state section above and check off completed checklist items as part of that
  phase's final commit, not as an afterthought later.
- **Config-only ablations, no exceptions** (invariant 10) — if implementing an ablation
  row requires editing code rather than a config value, the config promotion work
  (Phase 0 task) is incomplete; fix that instead of hand-editing for one run.
- **Say when something is unverified, and verify for real when you can.** Phase 0 set
  the pattern: every milestone that could be run locally (model downloads, live LLM
  calls, a real `uvicorn`/`streamlit run` launch, a fresh-clone reproducibility check)
  was actually run, not just reasoned through — state plainly which is which per the
  spec's "provably works" goal.
- Never commit `api_keys.json`, `milvus.db`, `logs/`, `fail_logs.txt`, corpus text
  (`data/`, `artifacts/parsed_md/`), or any real key/secret.
- Prefer `git add <path>` over `git add -A`; check `git status` before staging.
- Message format: short imperative subject (`feat:`/`fix:`/`refactor:`/`docs:`/`chore:`), body explaining why when not obvious from the diff.
- Pause for user confirmation between phase milestones before proceeding to the next one — established during Phase 0, keep doing it.
