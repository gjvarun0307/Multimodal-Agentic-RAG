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

**Phase 0 (Unblock) and Phase 1 (golden evaluation set) are complete. Phase 2
(metrics harness) is next.** Full detail in `PROJECT_SPEC.md` §7 Phase 2; this
section is the working checklist, not a restatement.

Full phase list: 0 Unblock (done) → 1 Golden set (done) → **2 Metrics harness (next)** →
3 Noise floor → 4 Eval-as-CI → 5 Deploy & observability → 6 Ablations & failure taxonomy →
7 README rewrite.

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
  `agent.py`'s active-document-scoping fields/logic still exist in the graph but
  nothing sets them anymore.

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
```

Not yet available (later phases — see `PROJECT_SPEC.md` §9 for the full target list):
`eval/run_eval.py` and `eval/metrics/` (Phase 2), `deploy/fetch_corpus.py`,
`deploy/build_ingest_artifacts.py`, `deploy/record_demo_traces.py`.

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
