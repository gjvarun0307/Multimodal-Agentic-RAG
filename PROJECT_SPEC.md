# PROJECT SPEC — Multimodal Agentic RAG: Evaluation & Production Hardening

> **How to use this file.** This is the authoritative blueprint for this project.
> Read it fully, then generate `CLAUDE.md` in the project root by distilling it
> into working instructions (conventions, commands, invariants, phase state).
> `CLAUDE.md` is a *living* document updated as phases complete; this file stays
> frozen as the reference.
>
> **When this file and the existing code disagree, this file wins.**
>
> This spec covers **one project only**. It contains no context from any other
> repository or workstream.

---

## 1. What this project is

An existing, working LangGraph-driven agentic RAG system over a fixed corpus of
15 AI research papers. It performs adaptive routing (vectorstore / web search /
chitchat), Self-RAG-style self-correction (hallucination + relevance grading,
query rewriting, regeneration), hybrid sparse+dense retrieval over Milvus Lite
with BGE-M3 embeddings, BGE reranking, and multimodal PDF ingestion via
LlamaParse + Qwen2.5-VL figure captioning. LLM inference is BYOK to hosted
providers.

**~1,761 lines across 5 core files. It runs. It has never been measured.**

## 2. What this upgrade is

Convert a system that *probably* works into one that **provably** works, with
published numbers, regression-gated CI, a public deployment, and an honest
failure analysis.

### 2.1 Non-goals — read this twice

This upgrade adds **zero user-facing features**. Explicitly **out of scope**:

- New retrieval strategies (HyDE, RAPTOR, GraphRAG, ColBERT)
- Multi-document scoping / multi-tenancy
- MCP integration
- Streaming token output
- Conversation persistence
- Docling migration
- Image-input multimodal queries
- UI redesign
- Any additional agent hop or tool

If a change does not serve *measurement, reliability, deployment, or
reproducibility*, it does not belong here. Park the idea in `docs/BACKLOG.md`.

**Three exceptions, in scope:**
- Defects in Section 5 — they corrupt measurement or block deployment.
- The selectable vLLM backend (Section 4A) — a *measurement instrument*, not a
  feature. It produces the structured-output reliability comparison.
- CPU-viable reranking — the deployment target has no GPU.

---

## 3. Current architecture (as-is)

### 3.1 Module map

| File | Lines | Role |
|---|---|---|
| `app.py` | 242 | Streamlit UI, `load_runtime()` (model loading), ingestion glue |
| `agent.py` | 705 | `GraphState`, all graph nodes, `build_agent_graph()` |
| `hybrid_database.py` | 291 | Milvus schema, splitting, hybrid search, insert/rebuild |
| `parse.py` | 218 | LlamaParse + Qwen2.5-VL captioning pipeline |
| `configuration.py` | 305 | Config singleton, `LLM_PROVIDERS`, `build_llm_client()` |
| `config.py` | — | Back-compat shim re-exporting `configuration.py` |
| `helper.py` | — | JSON parsing, JSONL loading |
| `logging_utils.py` | — | Logger setup → `logs/app.log` |
| `pages/1_Setup.py` | — | BYOK dashboard for provider + API keys |

### 3.2 Graph shape

```
query_router ──┬──► chitchat_node ─────────────────────────────► END
               ├──► web_search ──────────────────► generate
               └──► retrieve_and_rerank ──┬── docs ──► generate
                                          └── no docs ──► rewrite_query
                                                            │ (capped at 3)
                                                            └──► web_search

generate ──► hallucinations_and_relevence_router ──┬── PASS ──────► END
                                                   ├── regenerate ─► generate
                                                   └── rewrite ────► rewrite_query
```

### 3.3 Node inventory

| Node | Function |
|---|---|
| `query_router` | Routes to `vectorstore` / `websearch` / `chitchat` |
| `retrieve_and_rerank` | Milvus hybrid search → BGE reranker, score `> 0.5`, top 5 |
| `generate` | Answers from retrieved context |
| `rewrite_query` | Rewrites failing queries for retrieval |
| `web_search` | Tavily fallback |
| `chitchat_node` | Greetings / small talk |
| `hallucinations_and_relevence_router` | Grounding + relevance validation |

### 3.4 Stack

- **Orchestration:** LangGraph (manual `StateGraph`), LangChain
- **LLM:** BYOK via `build_llm_client()` — Anthropic, OpenAI, Groq, OpenRouter,
  NVIDIA NIM. Structured outputs via `with_structured_output` (tool-calling)
- **Embeddings:** BGE-M3 (local, dense + sparse)
- **Vector store:** Milvus Lite (file-based, `./milvus.db`)
- **Reranker:** `BAAI/bge-reranker-v2-gemma` via `FlagLLMReranker` — **CUDA-only**
- **Ingestion:** LlamaParse → Markdown; Qwen2.5-VL for figure captions
  (custom `######` header marks captions for the Markdown splitter)
- **Web search:** Tavily
- **UI:** Streamlit
- **Chunking:** size 1024, overlap 128

### 3.5 Corpus (frozen)

15 research papers, source PDFs retained and high-quality:
- **LLM architecture & serving** — vLLM, PagedAttention, Transformers
- **RAG methodology** — CRAG, Self-RAG, Adaptive-RAG, vector databases
- **Model building** — ConvNeXt

The corpus is **frozen** for the duration of this upgrade. Changing corpus
content invalidates every gold label. Any change is an eval-set version bump
plus a full re-verification pass.

---

## 4. Deployment target: Hugging Face Spaces

**BYOK solves LLM inference — it does not solve hosting.** The system needs an
always-on host for: the Streamlit app, the headless query API, BGE-M3 query
embedding (local model, per query), the reranker (local model, per query), and
Milvus Lite.

**Decision: Hugging Face Spaces, free CPU tier.** Rationale: 2 vCPU / 16 GB,
Streamlit is first-class, public HTTPS is built in (no tunnel setup), instant
provisioning, and OTLP metrics push to Grafana Cloud's free tier removes the
need to self-host an observability stack.

**Accepted tradeoffs, all of which must be engineered around:**

| Constraint | Consequence |
|---|---|
| Sleeps on inactivity | Cold start on wake — see Section 4B |
| Ephemeral filesystem | Runtime writes are lost on restart; artifacts must be baked in or fetched at boot |
| No root | Cannot self-host Prometheus/Grafana — push OTLP to Grafana Cloud instead |
| CPU only | Reranker and embeddings must be CPU-viable |
| Public Space | API key exposure risk — see Section 4C |

### 4.1 Ingest / serve split

| Environment | Runs | Hardware |
|---|---|---|
| **Ingest (offline, one-shot)** | LlamaParse, Qwen2.5-VL captioning, BGE-M3 embedding, Milvus build | Colab Pro / Kaggle T4 |
| **Serve (always-on)** | Query embedding, retrieval, reranking, graph execution | HF Spaces, CPU |

Ingest artifacts are built once on GPU and shipped to the Space. Milvus Lite is
file-based, so this is an artifact transfer, not a rebuild.

Artifacts must be versioned and content-hashed. Store them either as Git LFS
objects in the Space repo or as a separate HF Dataset fetched at boot. **Prefer
Git LFS in the Space repo** — it removes a network dependency from the cold
start path, which is the dominant latency risk.

### 4.2 Required serving-side changes

1. **Reranker must become CPU-viable.** `bge-reranker-v2-gemma` via
   `FlagLLMReranker` is a multi-billion-parameter LLM reranker, CUDA-only, and
   currently *hard-skipped* on CPU. Replace the default with
   `BAAI/bge-reranker-v2-m3` (568M, CrossEncoder-style) or `bge-reranker-base`
   (278M). Keep `v2-gemma` config-selectable so it can still be benchmarked on
   GPU — it becomes an ablation row.
2. **Reranker must never silently skip.** The CPU hard-skip becomes an explicit,
   loud, logged fallback surfaced in the results JSON. A silently disabled
   reranker invalidates every measurement taken while it was off.
2b. **Expect reranking to dominate latency.** Scoring ~50 candidates on 2 vCPU
   means ~50 cross-encoder forward passes — plausibly 1–3 seconds, likely your
   largest single stage. This is exactly what the ablation quantifies. You may
   end up running `search_limit=50` in eval for measurement headroom but lower
   in production; record both values in config and report them separately.
3. **BGE-M3 query embedding on 2 vCPU** is a real latency component. Measure it.
   If p95 is unacceptable, or if cold-start model load is too slow, export to
   ONNX int8 and re-measure. Report both numbers. ONNX also cuts resident memory
   substantially, which helps the 16 GB budget and the load time.
4. **Memory budget.** Document steady-state RSS. Rough fp32 CPU estimates:
   BGE-M3 ~2.5 GB, `v2-m3` reranker ~2.3 GB, `base` reranker ~1.1 GB, Milvus Lite
   + index modest, Streamlit + Python ~1 GB. Comfortable in 16 GB, but measure
   rather than assume — and note that resident memory drives cold-start load time.
5. **LLM stays BYOK to hosted APIs.** No change needed.
6. **Qwen2.5-VL and LlamaParse are ingest-only.** They never run at query time,
   so their CUDA requirement is irrelevant to serving.

---

## 4A. The vLLM backend — a measurement instrument

**Background.** An earlier version of this system served LLM inference locally
via vLLM with prefix caching and **xgrammar guided decoding** against an
OpenAI-compatible API. It was replaced with BYOK because local serving required
Colab and too many manual steps. BYOK is the correct default and stays the
deployed path.

**However:** vLLM returns as a *selectable, benchmarkable backend*, because it
enables the most interesting measurement available in this project.

### 4A.1 The structured-output reliability experiment

Four nodes depend on structured output: `query_router`, `rewrite_query`, and
both graders inside `hallucinations_and_relevence_router`. Two mechanisms can
produce it:

| Mechanism | Guarantee |
|---|---|
| Hosted tool-calling (`with_structured_output`) | **Probabilistic** — the model is trained to comply; nothing enforces it |
| xgrammar guided decoding (vLLM) | **Mathematical** — invalid tokens masked at sampling time; schema violation impossible |

**The measurement:** run the full golden set through both backends and record
schema-violation rate per grading node.

- `structured_output_validity_rate` — % of grader calls returning schema-valid output
- `structured_output_retry_rate` — % requiring retry or repair
- `silent_coercion_rate` — % where a parser "fixed" malformed output, potentially
  changing routing behavior
- `malformed_to_misroute_rate` — malformed router output that caused a wrong route

This converts an unverifiable claim ("mathematically enforcing zero-error
Pydantic JSON outputs") into a published number with an experiment behind it.

**Be prepared for a boring result.** Modern hosted tool-calling may hit 99.9%+
validity, making the guarantee academically real but practically irrelevant.
**Publish that too.** "The theoretical guarantee bought 0.1% in practice, so I
kept the simpler hosted path" is a mature conclusion and reads better than an
unexamined assertion.

### 4A.2 How to run it without permanent GPU

vLLM is **not** always-on. It is exercised only during ablation runs:

1. Launch vLLM on Colab Pro / Kaggle with the target model, prefix caching on,
   xgrammar guided decoding configured for the four Pydantic grader schemas
2. Expose the OpenAI-compatible endpoint via a tunnel (ngrok / equivalent)
3. Point the eval harness at it as another BYOK-style base URL —
   `build_llm_client()` already abstracts OpenAI-compatible endpoints, so this
   is a **config entry, not a refactor**
4. Run the full golden set, record results, tear down

Also record from these runs: prefix-cache hit rate, tokens/sec, and
time-to-first-token vs. hosted providers — completing the comparison across
correctness, latency, and cost.

---

## 4B. Cold starts — first-class concern

HF Spaces free tier sleeps on inactivity. Wake requires container start,
artifact availability, and model load into memory. **This must be engineered
and measured, not ignored.**

### 4B.1 Mitigation

1. **Bake artifacts into the Space repo (Git LFS).** No network fetch on the
   cold path. This is the single highest-leverage mitigation.
2. **ONNX int8 for BGE-M3.** Smaller file, faster load, lower RSS. Measure load
   time before and after; the delta is worth publishing.
3. **Eager model load at boot behind a readiness gate.** Load everything during
   container start, expose `/health` returning not-ready until warm. Do not
   lazy-load on first query — that transfers cold-start cost onto a real user
   and pollutes your latency distribution.
4. **Warm-up query at boot.** After models load, run one canned query through
   the full graph (retrieval only, no LLM call) to force lazy initialisation
   paths. Discard the result.
5. **Explicit warming UI state.** The Streamlit app must show a clear "warming
   up, ~Ns" state rather than appearing broken or hanging. A portfolio visitor
   who sees a hang closes the tab.
6. **Optional keep-warm ping.** A scheduled GitHub Actions workflow hitting
   `/health` at a low frequency (e.g. every 6 hours) prevents the inactivity
   sleep. Keep it genuinely low-frequency and non-abusive — this is a courtesy
   boundary on free infrastructure, not a rate-limit game. Document that the
   demo is kept warm and disclose the mechanism in the README.

### 4B.2 Measurement — mandatory

**Warm and cold latency are separate numbers and must never be blended.**
Blending them produces a p95 that describes nothing real.

Record in `docs/DEPLOYMENT.md`:
- `cold_start_seconds` — p50 / p95 over ≥ 5 forced cold starts
- Stage breakdown: container boot → artifact availability → model load →
  warm-up query → ready
- `warm_p50_ms` / `warm_p95_ms` / `warm_p99_ms` — steady-state query latency
- Time-to-ready after wake, as experienced by a first visitor

### 4B.3 The eval harness must exclude warm-up — **critical**

The harness runs a warm-up phase before timed measurement, and **warm-up queries
are excluded from all latency statistics**. Otherwise the first query's model
load contaminates p50 and every published latency number is wrong.

- Warm-up: ≥ 3 queries, discarded
- Explicit `warmup_excluded: true` field in the results JSON
- CI runs the pipeline **in-process inside the Actions runner**, not against the
  deployed Space. CI must be fast, deterministic, and independent of Space
  availability. The Space is for the demo and for real-world latency figures —
  never for gating merges.

---

## 4C. Public demo — three tiers

The Space is public (it is a portfolio artifact). The system is BYOK. These two
facts conflict and the conflict must be resolved deliberately.

**The risk:** if your provider key sits in HF Secrets and serves all visitors,
any stranger can burn your Groq/NIM quota, and a scripted visitor can exhaust it
in minutes.

**Resolution: three tiers, all present simultaneously.** A visitor lands on
Tier 1 and can escalate.

### Tier 1 — Precached traces (default, keyless, zero cost)

5–8 curated questions with their **full recorded pipeline output**, replayed
from disk. No API calls, no local inference, instant response even on a cold
container.

Each cached trace stores the complete execution record:

```json
{
  "trace_id": "demo_01",
  "question": "How does PagedAttention reduce KV-cache fragmentation?",
  "recorded_at": "2026-08-11T09:14:22Z",
  "git_sha": "a3f9c21",
  "config_hash": "…",
  "route_decision": "vectorstore",
  "retrieved_chunk_ids": ["…"],
  "reranked_chunk_ids": ["…"],
  "reranker_scores": [0.91, 0.78, 0.61],
  "correction_fired": false,
  "stage_latencies_ms": { "embed": 210, "retrieve": 45, "rerank": 1840, "generate": 1120 },
  "final_answer": "…"
}
```

**This is a better portfolio artifact than a live query.** A live query shows an
answer. An annotated trace shows Adaptive-RAG routing and Self-RAG correction
*actually working* — the interesting part, and the part invisible from a README.
Render the trace as a step-by-step walkthrough, not a chat bubble.

Include at least one trace where the correction loop **fires**, and one
`unanswerable_refuse` question where the system correctly declines. The refusal
demo is the most persuasive thing on the page.

**Two hard rules:**
- Precached responses are **clearly labeled as precached**. Never presented as live.
- Traces are **recorded from real pipeline runs**, never hand-written. Regenerate
  them whenever the pipeline changes materially; stale traces are a lie.

### Tier 2 — Live retrieval, no generation (keyless, zero cost)

Embedding, hybrid search, and reranking are all local models — they cost nothing
per query. Let visitors type any question and see **real** retrieval: routing
decision, retrieved chunks, reranker scores, threshold filtering.

This is unlimited and safe. Only generation costs money.

### Tier 3 — Full pipeline (visitor-supplied key)

`pages/1_Setup.py` already supports BYOK. Visitor pastes their own key and gets
unlimited real queries end to end.

**Optional demo-key fallback:** if you want generation available without a
visitor key, add your key in HF Secrets behind hard caps — per-session query
limit, global daily budget counter that disables generation when exceeded, and a
cheap model pinned for demo traffic. When the budget is exhausted, fall back to
Tier 2 rather than erroring. Only add this if you're willing to monitor it.

**In all cases:** never commit `api_keys.json`; use HF Secrets; verify
`.gitignore` carries over to the new repo before the first push.

---

## 4D. Service architecture — FastAPI + Streamlit

**Both, not either.** FastAPI is required regardless of preference: the eval
harness needs a headless query path, cold-start handling needs `/health` as a
readiness gate, and the keep-warm ping needs an endpoint. That is an API whether
or not it's called one.

**Replacing Streamlit is out of scope.** Building a frontend is a UI project, not
an evaluation project — exactly the scope creep Section 2.1 exists to prevent.
Streamlit stays because it makes the Space demo work for free.

### 4D.1 Shape

```
src/runtime.py  →  get_runtime()  (single source of truth)
      │
      ├── src/api.py     FastAPI  — /query, /health, /metrics, /trace
      └── app.py         Streamlit — demo surface (Tiers 1–3)
```

**Both adapters call `get_runtime()` directly, in-process.** Streamlit must not
call the FastAPI service over HTTP on the same container — that is a pointless
network hop that inflates latency and corrupts measurement.

### 4D.2 Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /query` | Headless query. Returns answer + full trace (route, both chunk ID lists, reranker scores, correction events, per-stage latency). Used by nothing in production — it exists for eval and for anyone reading the repo. |
| `GET /health` | Readiness gate. Returns not-ready until models are loaded and the warm-up query has run (Section 4B). Target of the keep-warm ping. |
| `GET /metrics` | Prometheus-format scrape endpoint (complements OTLP push). |
| `GET /trace/{trace_id}` | Serves a Tier 1 precached trace. |

`/query` returning the **full trace, not just the answer**, is what makes the API
worth reading. It's also what the harness consumes.

### 4D.3 Why this matters beyond function

A documented FastAPI service with a generated OpenAPI schema is a materially
different signal from a Streamlit script. Commit the generated `openapi.json`
and link it from the README. Low cost, and it addresses the weakest part of the
existing stack story.

**CI calls neither adapter.** It runs the pipeline in-process in the Actions
runner (Section 4B.3).

---

## 5. Known defects — Phase 0 blockers

From the project's own audit. Items 5.1–5.4 **must** be fixed before any
measurement begins, because they corrupt results or block debugging.

### 5.1 `metadata.jsonl` must be regenerated from scratch — **CRITICAL**

**Current state, confirmed by audit.** `metadata.jsonl` contains no stable
per-document key of any kind. Each line holds only
`{title|paper_name, authors, year, topic}` — free text generated by the VLM from
the paper's page-1 image (`parse.py:152`), or a placeholder dict when no VLM is
available (`parse.py:116-121`). Nothing in the record references the source file.

`prepare_input_data()` (`hybrid_database.py:29-37`) **mtime-sorts** the `.md`
files and zips them against `metadata.jsonl` line order by index. This is worse
than filename sorting: mtime changes on file copy, git checkout, rsync, and
re-parse. Any skipped, reordered, or re-parsed file shifts every subsequent
paper's metadata by one, silently.

A stable identifier does exist but is bolted on separately at insert time:
`doc.metadata["source_file"] = input_path.name` (`hybrid_database.py:81`, and
`app.py:116` for uploads). The active-document filter scopes on this. But it is
derived from the `.md` filename on disk — which is the **LlamaCloud job name,
not the original PDF filename** — and it lives nowhere in `metadata.jsonl`, so
there is nothing to join against.

**Decision: regenerate `metadata.jsonl` from scratch.** With 15 papers this is
an afternoon, and every downstream label depends on it.

#### 5.1.1 Target schema

One JSON object per line:

```json
{
  "schema_version": 1,
  "doc_id": "vllm_paged_attention",
  "source_pdf": "2309.06180v1.pdf",
  "source_md": "job_a3f9c21b_result.md",
  "content_sha256": "e3b0c44298fc1c149afbf4c8996fb924...",
  "n_chunks": 187,
  "title": "Efficient Memory Management for Large Language Model Serving with PagedAttention",
  "authors": ["Woosuk Kwon", "Zhuohan Li", "..."],
  "year": 2023,
  "topic": "llm_serving",
  "arxiv_id": "2309.06180",
  "parsed_at": "2026-08-11T09:14:22Z",
  "parser": "llamaparse",
  "vlm_captioner": "Qwen2.5-VL-7B",
  "verified_by": "human"
}
```

#### 5.1.2 Rules for each field

**`doc_id` — the primary key. Get this right and everything else follows.**

- **Hand-assign all 15.** Do not derive it from a filename, a hash, or the
  LlamaCloud job name. Fifteen readable slugs take ten minutes and pay for
  themselves every time you debug.
- Format: `^[a-z0-9_]+$`. No hyphens, no dots, no spaces — it becomes a chunk ID
  prefix and must survive string composition and filesystem paths.
- Make it human-readable: `vllm_paged_attention`, `self_rag`, `adaptive_rag`,
  `crag`, `convnext`. When a failure log says
  `RET_MISS on self_rag::0042`, you know instantly what broke. With a hash you
  would be looking things up.
- **Immutable forever.** Changing a `doc_id` invalidates every gold label
  referencing that paper. If one must change, it is an eval-set version bump.

**`source_pdf`** — the original PDF filename, canonical provenance. **`source_md`**
— the LlamaCloud job filename actually on disk. Record **both**; they differ, and
conflating them is the root of the current mess.

**`content_sha256`** — SHA-256 of the `.md` file bytes. This is your drift
detector: if a paper is silently re-parsed and the markdown changes, the hash
changes and the ingest fails loudly instead of quietly invalidating gold labels.

**`n_chunks`** — written back after ingest as a cross-check. If it doesn't match
on rebuild, chunking parameters or content changed.

**`title` / `authors` / `year` / `topic`** — keep them, but treat them as
**descriptive metadata only. They must never be used for joins.** They are
VLM-generated from a page-1 image, which means they will be wrong sometimes —
author lists get truncated, years get read off the wrong line, titles pick up
subtitle fragments. Hand-correct all 15 against the actual PDFs. Thirty minutes,
and it prevents a demo where the system confidently cites the wrong authors.

**`arxiv_id`** — optional but worth adding; it's the real-world canonical ID and
useful in citations.

#### 5.1.3 Procedure

**Hand-write the identity fields; compute the rest.** Do not hand-write
`content_sha256` (you can't) or `n_chunks` (unknown until after ingest). Instead:

- Hand-author `artifacts/corpus_seed.csv` — one row per paper, columns
  `doc_id, source_pdf, source_md, title, authors, year, topic, arxiv_id, license`
- `deploy/build_metadata.py` reads the seed, hashes each `.md`, and emits
  `metadata.jsonl`

This makes regeneration repeatable. If a paper is re-parsed, re-run the script
rather than hand-editing a hash and getting it wrong.

1. List the 15 `.md` files and the 15 source PDFs side by side. Manually
   establish the true PDF ↔ MD correspondence **once**, by opening them. Do not
   trust any existing ordering — the current mapping may already be wrong, so
   verify rather than migrate.
2. Assign a `doc_id` slug to each pair.
3. Compute `content_sha256` per `.md` file.
4. Extract or hand-write `title`/`authors`/`year`/`arxiv_id` from the PDFs
   directly, not from the VLM output. Use the existing VLM values only as a
   starting draft.
5. Write `metadata.jsonl`, one line per document.
6. Run the validator (5.1.4). Fix until clean.
7. Commit it. This file is now ground truth.

#### 5.1.4 Validator — required, runs in CI

`tests/test_metadata_alignment.py` must assert:

- [ ] **Bijection:** every `.md` on disk has exactly one metadata row and vice
      versa. Any orphan on either side is a hard failure, never a warning.
- [ ] `doc_id` unique across all rows
- [ ] `doc_id` matches `^[a-z0-9_]+$`
- [ ] `source_md` resolves to an existing file
- [ ] `content_sha256` matches the file's actual hash
- [ ] `schema_version` present and recognised
- [ ] Shuffling file mtimes or directory listing order produces identical
      doc→chunk mapping

#### 5.1.5 Code changes

- `parse.py:154-156` — write `doc_id`, `source_pdf`, `source_md`,
  `content_sha256` at metadata-write time. VLM output populates only the
  descriptive fields.
- `hybrid_database.py:29-37` — **delete the mtime sort and the index-zip
  entirely.** Replace with a dict keyed on `doc_id`, built by joining on
  `source_md`. Raise on any unmatched entry; never fall back to positional.
- `hybrid_database.py:81` — set `doc.metadata["doc_id"]` as the canonical
  chunk-level identifier. Keep `source_file` populated for one release so the
  active-document filter keeps working, then **migrate that filter to `doc_id`
  and remove `source_file`.** Two parallel identifier systems is how this class
  of bug returns.
- `app.py:116` — the upload path must assign a `doc_id` too, or reject uploads
  for the duration of this upgrade (the corpus is frozen anyway — rejecting is
  simpler and safer).

### 5.2 Chunk IDs must be deterministic — **CRITICAL**

**Current state:** the Milvus schema uses `auto_id=True`, so the database assigns
int64 primary keys on insert. Rebuild the collection and every ID changes.

**Requirement:** `auto_id=False`, with a VARCHAR primary key holding a
deterministic chunk ID:

```
{doc_id}::{chunk_index:04d}::{content_sha8}
e.g. vllm_paged_attention::0042::a3f9c21b
```

- `doc_id` — readable, from `metadata.jsonl` (5.1)
- `chunk_index` — zero-padded ordinal within the document, so IDs sort naturally
- `content_sha8` — first 8 hex chars of the chunk text's SHA-256. This is the
  drift tripwire: if chunking parameters or source content change, the ID
  changes, and stale gold labels **fail to resolve loudly** instead of silently
  pointing at different text.

**This is a collection recreate, not an in-place alter** — Milvus cannot change
a primary key type on an existing collection. Fine, since the index is being
rebuilt anyway. Bump the collection name (`arag_project_v2`) so the old and new
indexes can coexist while you verify.

**Test** (`tests/test_chunk_id_determinism.py`, runs in CI): build the index
twice from identical sources, assert byte-identical ID sets. This single test
protects every gold label in the project.

### 5.2.1 The chunk-size ablation conflict — read before Phase 1

Putting a content hash in the chunk ID creates a problem the ablation matrix
must solve: **the chunk 512 / 1024 / 2048 ablation rows produce entirely
different chunk IDs**, so `gold_chunk_ids` recorded against the 1024 config
won't resolve under any other config. Naively, chunk-size ablation becomes
impossible.

**Resolution: gold labels reference passages, not chunk IDs.**

Ground truth is `gold_passages` — verbatim supporting text plus `doc_id`. Chunk
IDs are *derived* at eval time by resolving passages against whatever chunking
configuration is active.

```
gold_passages (ground truth, chunking-independent)
        │
        ▼  resolver: locate passage in source .md → char span
   char spans (doc_id, start, end)
        │
        ▼  overlap test against active chunking config
   gold_chunk_ids (derived, per-config)
```

A retrieved chunk counts as relevant if it fully contains the gold span, or
overlaps ≥ 50% of the span's characters. Record the criterion in
`eval/golden/SCHEMA.md`; do not change it mid-project.

This is also **better annotation ergonomics.** While writing a question you are
already reading the `.md` — copying the supporting sentence is easier than
looking up a chunk ID, and it stays valid across every reconfiguration.

Cache resolved chunk IDs per config in `eval/golden/resolved/<config_hash>.json`
so resolution runs once, not every eval. A passage that fails to resolve
(text drift, bad copy-paste) is a **hard error**, never a skip.

### 5.3 `logger.debug` is a no-op — **CRITICAL**

A level mismatch in `logging_utils.py` silences every `logger.debug` call. The
graph's routing trace — the primary debugging aid — is invisible.

**Fix first, before anything else.** You cannot debug retrieval failures blind,
and Phase 1 will generate many "why did this fail" questions. Add `LOG_LEVEL`
env var, default `INFO`, `DEBUG` during eval runs.

### 5.4 Duplicated model construction

`agent.py:get_models()` and `app.py:load_runtime()` independently build
`database` / `embedding_model` / `rerank_model`. Only the LLM client goes through
shared `build_llm_client()`. `agent.py:create_agent()` is dead — **confirmed at
`agent.py:694`, the name appears nowhere else in the repo.**

This already caused a real bug: a CPU-reranker fix landed in one path and
silently missed the other.

**Fix:** single `src/runtime.py` exposing `get_runtime()` returning a `Runtime`
dataclass (`llm`, `embedding_model`, `rerank_model`, `database`, `config`). Both
entry points call it. Delete `create_agent()` outright — do not deprecate.

**Prerequisite for the eval harness**, which adds a third (headless) entry point
and must not become a third copy.

### 5.5 Config is decorative

`config_rag()` returns 7 keys; 11+ others (`search_limit`, `reranker_top_k`,
`max_gen_retries`, `sparse_weight`, `dense_weight`, score threshold, chunk size,
overlap) are hardcoded at call sites and ignore env vars.

**Why this blocks ablations:** you cannot ablate parameters that are hardcoded —
every ablation row would be a code edit.

**Fix:** every tunable becomes a config field, precedence
`defaults → api_keys.json → env → explicit override arg`. The harness must
construct a run from a config dict alone.

### 5.6 Graph must be invocable statelessly

Eval requires each query to be independent. Confirm `GraphState` carries no
cross-query residue and that `build_agent_graph()` can be invoked headlessly
without Streamlit's `st.session_state`. If conversation history leaks between
invocations, eval results are contaminated.

### 5.7 `retrieve_and_rerank` must surface chunk IDs

The harness needs retrieved chunk IDs, pre- and post-rerank, to compute recall
and to distinguish `RET_MISS` from `RET_DEMOTED`. If the node currently returns
only concatenated text, plumb IDs through `GraphState`.

**Record both lists:** `retrieved_chunk_ids` (post-hybrid-search, pre-rerank) and
`reranked_chunk_ids` (post-rerank, post-threshold). The difference between them
is what makes reranker ablation and failure classification possible.

### 5.8 No tests, no CI, no lint

Every change is verified by reading, not running. Add `pytest` + `ruff` +
`pre-commit`. Phase 0 target: a smoke test that builds the runtime and answers
one query end-to-end headlessly.

### 5.9 `requirements.txt` pins almost nothing

Reproducibility risk. Pin every direct dependency exactly. Generate
`requirements.lock` via `pip freeze` from a clean install. Record the Python
version. Non-negotiable before publishing any number.

### 5.10 Single active-document scalar

No true multi-document scoping. **Out of scope** — note in README limitations.

---

## 6. Target directory structure

```
project-root/
├── CLAUDE.md                     # generated from this spec; working doc
├── PROJECT_SPEC.md               # this file; frozen reference
├── README.md                     # rewritten in Phase 7
├── pyproject.toml                # ruff + pytest config
├── requirements.txt              # pinned
├── requirements.lock
├── .pre-commit-config.yaml
├── .gitignore                    # verify api_keys.json is covered
├── .github/
│   └── workflows/
│       ├── fast-eval.yml         # every push: retrieval-only subset
│       ├── full-eval.yml         # nightly + on-label
│       ├── keep-warm.yml         # low-frequency /health ping
│       └── lint.yml
├── src/
│   ├── runtime.py                # NEW — unified runtime construction
│   ├── api.py                    # NEW — FastAPI: /query /health /metrics /trace
│   ├── agent.py                  # graph nodes + build_agent_graph()
│   ├── hybrid_database.py
│   ├── parse.py
│   ├── configuration.py
│   ├── helper.py
│   └── logging_utils.py
├── configs/
│   ├── default.yaml
│   └── ablations.yaml
├── eval/
│   ├── golden/
│   │   ├── golden_set.jsonl      # THE eval set — versioned, frozen
│   │   ├── dev_split.jsonl       # tuning split, kept separate
│   │   ├── resolved/             # derived chunk IDs, cached per config hash
│   │   └── SCHEMA.md
│   ├── resolve_passages.py       # gold_passages -> char spans -> chunk IDs
│   ├── harness.py                # run_eval(config) -> results dict
│   ├── metrics/
│   │   ├── retrieval.py          # recall@k, MRR, nDCG — no LLM
│   │   ├── generation.py         # faithfulness, correctness, refusal
│   │   ├── structured.py         # schema validity (Section 4A)
│   │   ├── router.py             # router accuracy + misroute cost
│   │   └── system.py             # latency, tokens, cost
│   ├── judge.py                  # LLM-as-judge, versioned prompts
│   ├── tavily_cache.py           # frozen web-search fixtures
│   ├── noise_floor.py            # Phase 3
│   ├── ablations.py              # Phase 6 matrix runner
│   ├── failure_taxonomy.py       # Phase 6 classifier
│   ├── baselines/
│   │   └── main.json             # committed CI baseline
│   └── results/                  # gitignored run outputs
├── observability/
│   └── otel_setup.py             # OTLP export → Grafana Cloud
├── deploy/
│   ├── build_ingest_artifacts.py # GPU-side; produces milvus.db bundle
│   ├── build_metadata.py         # corpus_seed.csv -> metadata.jsonl
│   ├── fetch_corpus.py           # re-download PDFs from arXiv, re-parse
│   ├── record_demo_traces.py     # produces Tier 1 precached traces
│   └── SPACE_README.md           # HF Space card (title, sdk, secrets)
├── artifacts/
│   ├── corpus_seed.csv           # hand-authored identity fields
│   ├── SOURCES.md                # per-paper attribution + license
│   ├── parsed_md/                # plain git, ~1.5 MB
│   ├── metadata.jsonl
│   ├── milvus.db                 # Git LFS
│   └── demo_traces/              # Tier 1 recorded traces
├── openapi.json                  # generated; linked from README
├── tests/
│   ├── test_metadata_alignment.py
│   ├── test_chunk_id_determinism.py
│   ├── test_config_precedence.py
│   ├── test_runtime.py
│   └── test_smoke.py
├── docs/
│   ├── ABLATIONS.md
│   ├── FAILURE_ANALYSIS.md
│   ├── NOISE_FLOOR.md
│   ├── DEPLOYMENT.md             # incl. cold-start measurements
│   └── BACKLOG.md
└── app.py                        # Streamlit entry point
```

---

## 7. Phase plan

Phases are dependency-ordered. Do not start a phase until its predecessor meets
acceptance criteria. Each phase ends with a commit and a `CLAUDE.md` update.

---

### PHASE 0 — Unblock (est. 15–18h)

**Goal:** make the system measurable, debuggable, and reproducible.

**Tasks**
1. **Fresh repo hygiene.** Seed from the existing repo. Audit every import
   against `requirements.txt`; strip unused dependencies. Delete dead modules.
   Remove the `config.py` shim if nothing imports it. Verify `.gitignore` covers
   `api_keys.json`. A clean first commit beats preserved history.
2. Fix `logging_utils.py` level mismatch. Add `LOG_LEVEL`. Verify the routing
   trace is visible at DEBUG.
3. **Regenerate `metadata.jsonl` from scratch** per 5.1: hand-assign 15
   `doc_id` slugs, verify PDF↔MD correspondence by opening files, hand-correct
   VLM-generated titles/authors/years, add hashes. Write the validator.
4. Delete the mtime sort in `hybrid_database.py:29-37`; replace with a
   `doc_id`-keyed dict join. Raise on unmatched entries.
5. Migrate Milvus schema to `auto_id=False` with VARCHAR deterministic chunk
   IDs (5.2). New collection name `arag_project_v2`. Add the double-build test.
6. Create `src/runtime.py`; migrate `app.py` and `agent.py`; delete
   `create_agent()` at `agent.py:694`.
7. Promote all hardcoded tunables to config with documented precedence.
8. Plumb `retrieved_chunk_ids` and `reranked_chunk_ids` through `GraphState`.
9. Set chunk-level `doc_id` metadata; plan `source_file` retirement.
10. Verify stateless headless graph invocation (5.6).
11. Add CPU-capable reranker option; make CPU fallback loud and logged.
12. Add `ruff`, `pytest`, `pre-commit`, `pyproject.toml`.
13. Pin `requirements.txt`; generate `requirements.lock`.
14. Add vLLM as a provider entry in `LLM_PROVIDERS` (config only — Section 4A).
15. Create `src/api.py` (FastAPI) per Section 4D. `/query` returns the full
    trace, not just the answer. Both adapters call `get_runtime()` in-process.
16. Check licenses for all 15 papers; write `artifacts/SOURCES.md`; decide
    per paper whether the `.md` is committed or fetched (Section 4.1.1).

**Acceptance criteria**
- [ ] `pytest` passes; smoke test answers one query end-to-end headlessly
- [ ] `ruff check .` clean
- [ ] `metadata.jsonl` regenerated, validator clean, bijection with `.md` files
- [ ] All 15 titles/authors/years hand-verified against source PDFs
- [ ] Shuffling file mtimes yields identical doc→chunk mapping
- [ ] Two index builds from identical sources yield identical chunk IDs
- [ ] Milvus schema on `auto_id=False` with VARCHAR chunk IDs
- [ ] A config dict alone constructs a full runtime — no code edits
- [ ] Routing trace visible at `LOG_LEVEL=DEBUG`
- [ ] Reranker runs on CPU, or fails loudly with a logged reason
- [ ] Both pre- and post-rerank chunk ID lists retrievable from a headless run
- [ ] `POST /query` returns a full trace headlessly; `GET /health` gates on readiness
- [ ] `artifacts/SOURCES.md` complete; licence decision recorded per paper
- [ ] Fresh clone + `pip install -r requirements.lock` reproduces a working system

---

### PHASE 1 — Golden evaluation set (est. 10–14h)

**Goal:** 150–300 hand-verified questions over the frozen 15-paper corpus.

**Composition targets** (n ≈ 180)

| Category | Share | n | `expected_route` | Purpose |
|---|---|---|---|---|
| `single_hop` | 30% | 54 | vectorstore | Baseline retrieval |
| `multi_hop` | 19% | 34 | vectorstore | Does routing/rewriting earn its complexity |
| `table_figure` | 15% | 27 | vectorstore | Exercises the Qwen2.5-VL caption path |
| `unanswerable_refuse` | 17% | 31 | refuse | **Hallucination rate** |
| `ambiguous` | 10% | 18 | vectorstore | Clarification behavior |
| `adversarial` | 5% | 9 | vectorstore | Conflicting/misleading context robustness |
| `unanswerable_websearch` | 2% | 4 | websearch | Tavily fallback fires correctly |
| `chitchat` | 2% | 4 | chitchat | Third route represented |

**Deliberately capped at 8 non-vectorstore items.** The web-search path is
simple and does not warrant deep evaluation; 4 items confirm the route fires and
keeps the Tavily fixture trivial to freeze. The 4 `chitchat` items exist purely
so `router_accuracy` has all three classes represented — without them the
confusion matrix has an empty row.

**The `unanswerable_refuse` bucket is the most important and the most commonly
skipped.** Correct behavior is refusal or explicit "not in the provided
documents." Refusal accuracy here is your defensible hallucination metric, and at
31 items it has enough mass to move a CI gate.

**Index scale context: 2,694 chunks across 15 papers (~180 chunks/paper).** Two
consequences for labelling:

- Chunk-level recall is the real signal. **Document-level recall is
  near-ceiling and uninformative** — with only 15 documents, hybrid search will
  almost always surface *something* from the right paper. Report `gold_doc_ids`
  recall for diagnostics only; never gate CI on it.
- At 2,694 chunks, `recall@1` and `MRR` will discriminate between ablations far
  more sharply than `recall@10`. Weight interpretation accordingly.

**Item schema** (`eval/golden/golden_set.jsonl`, one object per line):

```json
{
  "id": "gs_0001",
  "question": "What memory fragmentation problem does PagedAttention solve?",
  "gold_answer": "...",
  "gold_passages": [
    {
      "doc_id": "vllm_paged_attention",
      "passage_text": "verbatim supporting text copied from the .md, long enough to be unique"
    }
  ],
  "gold_doc_ids": ["vllm_paged_attention"],
  "category": "single_hop",
  "expected_route": "vectorstore",
  "difficulty": "easy",
  "requires_multimodal": false,
  "notes": "answer spans a chunk boundary at chunk_size=1024",
  "verified_by": "human",
  "version": 1
}
```

Field rules:
- `gold_passages` — **the ground truth, and mandatory** for every item with
  `expected_route: vectorstore`. Chunk IDs are derived from these at eval time
  (5.2.1), never stored as ground truth. Without them you cannot separate
  retrieval failure from generation failure, and you cannot ablate chunk size.
- `passage_text` — verbatim copy from the source `.md`, long enough to be
  unique within the document. If it appears more than once, the resolver raises;
  extend the passage until unique.
- `expected_route` — `vectorstore` / `websearch` / `chitchat` / `refuse`.
- `gold_answer` — empty string for `unanswerable_refuse` items.
- `gold_doc_ids` — diagnostics only, never a gating metric (near-ceiling at 15 docs).

**Construction procedure**
1. Sample chunks stratified across all 15 papers; ensure every paper is covered.
2. LLM-generate candidate Q/A pairs from sampled chunks.
3. **Hand-verify and correct every item.** LLM-generated *and* LLM-judged with
   no human in the loop is circular and worthless.
4. Hand-author the `unanswerable`, `ambiguous`, and `adversarial` buckets — these
   do not generate well.
5. Hold out ~30 items as `dev_split.jsonl`. **Never tune against the main set.**
6. Freeze. Commit. Version bump on any change.

**Acceptance criteria**
- [ ] 150+ items, buckets within ±3% of target share
- [ ] Every paper represented
- [ ] 100% hand-verified
- [ ] Every `gold_passages` entry resolves to exactly one span in its source `.md`
- [ ] Resolver produces chunk IDs for chunk_size 512 / 1024 / 2048 without error
- [ ] Dev split disjoint from main set
- [ ] `eval/golden/SCHEMA.md` documents every field

---

### PHASE 2 — Metrics harness (est. 10–12h)

**Goal:** `run_eval(config) -> results` — headless, deterministic, config-driven.

**Critical design decision: retrieval and generation metrics are measured and
reported separately.** When end-to-end accuracy drops you must know immediately
which half broke. End-to-end-only reporting makes every regression a mystery.

**Retrieval** (`eval/metrics/retrieval.py`) — no LLM, fast, free.

**Measure both pipeline stages separately.** The reranker keeps only top 5 above
threshold, so a single `recall@10` is ill-defined — post-rerank it cannot exceed
5. Report:

| Stage | Metrics | Bounded by |
|---|---|---|
| **Stage 1** — hybrid search, pre-rerank | `recall@10`, `recall@20`, `recall@50`, `MRR`, `nDCG@10` | `search_limit` |
| **Stage 2** — post-rerank, post-threshold | `recall@1`, `recall@3`, `recall@5`, `MRR` | `reranker_top_k` (5) |

- `rerank_lift` = stage-2 `recall@5` minus stage-1 `recall@5` (i.e. the reranker's
  top 5 vs. hybrid search's raw top 5). **This is the number that justifies the
  reranker's existence** — it can legitimately be negative, which is exactly
  what you want to find out.
- `threshold_loss` = gold chunks retrieved and correctly ranked by the reranker
  but dropped by the `> 0.5` score filter. A high value means the threshold is
  miscalibrated and is a cheap win.
- Set `search_limit` ≥ 50 during eval regardless of production value, so stage-1
  recall has headroom to measure. Record both values in the config.

Both stages are computable because Phase 0 plumbed `retrieved_chunk_ids` and
`reranked_chunk_ids` separately (5.7). At 2,694 chunks, `recall@1` and `MRR`
discriminate more sharply than `recall@10`.

**Generation** (`eval/metrics/generation.py`) — LLM judge:
- `faithfulness` — every claim supported by retrieved context
- `answer_correctness` — vs `gold_answer`
- `refusal_accuracy` — on `unanswerable_refuse` (**hallucination guard**)
- `citation_precision`

**Structured output** (`eval/metrics/structured.py`) — pure schema validation,
no judge. Instrument all four grading nodes:
- `structured_output_validity_rate` (per node and aggregate)
- `structured_output_retry_rate`
- `silent_coercion_rate`
- `malformed_to_misroute_rate`

Collect these on **every** run, not just the vLLM ablation — the BYOK baseline
is half of the Section 4A comparison.

**Router** (`eval/metrics/router.py`):
- `router_accuracy` — predicted vs `expected_route`
- `misroute_cost` — mean added latency and tokens when wrong
- Confusion matrix across the three routes

**Self-RAG correction** — likely the most interesting finding:
- `correction_fire_rate`, `correction_improve_rate`, `correction_degrade_rate`
- `mean_retries` and distribution

> If correction sometimes degrades answers, **publish it**. "Fires on 18%,
> improves 71%, degrades 9%" is more interesting and more senior than a clean
> number. Do not hide it.

**System** (`eval/metrics/system.py`):
- `warm_p50` / `warm_p95` / `warm_p99` (warm-up excluded — Section 4B.3)
- Per-stage latency: embed → retrieve → rerank → route → generate → correct
- Tokens per query (prompt/completion split)
- `cost_per_query`, `cost_per_1k_queries`
- Error rate by stage

**Tavily determinism** (`eval/tavily_cache.py`) — **required**. Live web search
is non-deterministic and drifts over time, which breaks reproducibility and
makes noise-floor measurement meaningless for any `websearch`-routed item.

- Record Tavily responses once into a frozen fixture keyed by query hash
- Eval runs replay from cache by default (`TAVILY_MODE=replay`)
- `TAVILY_MODE=live` available for occasional freshness checks, never for CI or
  baselines
- Fixture version recorded in results JSON
- Cache misses fail loudly rather than falling through to live

**Judge design** (`eval/judge.py`)
- Versioned prompts; a judge prompt change invalidates baseline comparability
- `judge_version` in every results file
- Temperature 0
- **The judge must be a different model family than the generation model.** LLM
  judges show measurable self-preference bias. **Pin the judge to Groq and keep
  it fixed across all runs**, including the vLLM ablation, or the backend
  comparison is uninterpretable. Groq's free tier also suits sustained runs
  better than NIM's credit allowance — a full run makes hundreds of judge calls.
- Calibrate: hand-label 40 items, report judge-human agreement (Cohen's κ) in
  the README. A judge with unknown agreement is decoration.

**Results schema** (`eval/results/<timestamp>_<git_sha>.json`):

```json
{
  "run_id": "2026-08-11T14:22:03Z_a3f9c21",
  "git_sha": "a3f9c21",
  "config": { "...full resolved config..." },
  "golden_set_version": 1,
  "judge_version": "v2",
  "judge_model": "groq/<pinned>",
  "tavily_fixture_version": "v1",
  "backend": "byok-hosted",
  "warmup_excluded": true,
  "n_items": 187,
  "metrics": {
    "retrieval": { "recall@5": 0.812 },
    "generation": { "faithfulness": 0.91, "refusal_accuracy": 0.87 },
    "structured": { "validity_rate": 0.979 },
    "router": { "accuracy": 0.94 },
    "correction": { "fire_rate": 0.18, "improve_rate": 0.71, "degrade_rate": 0.09 },
    "system": { "warm_p95_ms": 3410, "cost_per_query_usd": 0.0021 }
  },
  "per_item": [ { "id": "gs_0001", "retrieved_chunk_ids": [], "reranked_chunk_ids": [] } ]
}
```

`per_item` is mandatory — Phase 6's failure taxonomy consumes it.

**Acceptance criteria**
- [ ] `python -m eval.harness --config configs/default.yaml` runs headless
- [ ] Retrieval-only mode makes **zero LLM calls**
- [ ] Warm-up phase runs and is excluded from latency stats
- [ ] Tavily replays from frozen cache; misses fail loudly
- [ ] Results JSON matches schema; `per_item` populated with both ID lists
- [ ] Judge-human agreement (κ) computed and recorded
- [ ] Full run completes within free-tier rate limits (backoff implemented)

---

### PHASE 3 — Noise floor (est. 2–3h)

**Goal:** know your measurement precision before gating on it.

Almost everyone skips this, and it is why most portfolio CI suites are flaky and
then ignored.

**Procedure**
1. Temperature 0. Fix every available seed. Tavily in replay mode.
2. Run the identical commit against the full eval set **5 times**.
3. Compute mean, σ, and min–max range for every headline metric.
4. Record in `docs/NOISE_FLOOR.md`.

Variance persists regardless — reranker score ties, non-greedy paths, hosted-API
nondeterminism.

**Every Phase 4 threshold must exceed the observed noise floor.** If recall@10
varies ±1.5 points across identical runs, a 1-point threshold fails randomly and
the suite gets ignored within two weeks — wasting Phases 2–4.

**Acceptance criteria**
- [ ] 5 identical runs recorded with full results JSON each
- [ ] Per-metric σ and range documented
- [ ] Proposed CI thresholds all ≥ 2σ, justified in writing
- [ ] README line drafted, e.g. *"Run-to-run variance ±1.2 pts recall@10 over 5
      identical runs; regression threshold set at 2.5 pts."*

---

### PHASE 4 — Eval-as-CI (est. 6–8h)

**Goal:** the headline artifact. Regressions block merges automatically.

**CI runs the pipeline in-process in the Actions runner, never against the
deployed Space.** CI must be fast, deterministic, and independent of Space
availability or cold starts.

**Two tiers**

| Tier | Trigger | Scope | Metrics | Budget |
|---|---|---|---|---|
| **Fast** | Every push / PR | ~40-item stratified subset | Retrieval + structured only, no judge | < 3 min, $0 |
| **Full** | Nightly + `run-full-eval` label | Complete set | All metrics | < 25 min |

Fast tier catches most regressions because most breakage is retrieval-side, and
costs nothing because it makes no LLM calls.

**Gate logic** — compare against `eval/baselines/main.json`:

| Metric | Threshold | Action |
|---|---|---|
| `recall@10` | drop > noise-floor-derived value | **block** |
| `faithfulness` | drop > threshold | **block** |
| `refusal_accuracy` | **any drop** | **block** (strict — hallucination guard) |
| `chunk_id_determinism` | any failure | **block** |
| `router_accuracy` | drop > threshold | warn |
| `structured_validity` | drop > threshold | warn |
| `warm_p95` | increase > 20% | warn |
| `cost_per_query` | increase > 20% | warn |

**PR comment** — post a diff table:

```
| Metric            | Baseline | This PR | Δ      | Status |
|-------------------|----------|---------|--------|--------|
| recall@10         | 0.847    | 0.812   | -0.035 | ❌ FAIL |
| faithfulness      | 0.910    | 0.913   | +0.003 | ✅      |
| refusal_accuracy  | 0.870    | 0.870   |  0.000 | ✅      |
```

**Screenshot a real blocked merge for the README.** A CI run failing because
retrieval recall dropped is worth more than any architecture diagram.

**Baseline update:** only via explicit PR editing `main.json` with written
justification. Never automatic.

**Cost control:** GitHub Actions is unlimited on public repos — keep the repo
public. API keys as repo secrets. Rate-limit backoff, and rate-limit errors must
be reported **distinctly from regressions** so a quota blip never looks like a
quality drop.

**Acceptance criteria**
- [ ] Fast eval on every push, < 3 min, zero LLM calls
- [ ] Full eval nightly and on label
- [ ] A deliberately broken retrieval config **blocks** a test PR
- [ ] PR comment renders the diff table
- [ ] Rate-limit errors distinguishable from regressions
- [ ] Screenshot captured for README

---

### PHASE 5 — Deploy & observability (est. 8–10h)

**Goal:** public demo URL, live dashboard, real per-stage latency, measured
cold starts.

**Ingest** — `deploy/build_ingest_artifacts.py` runs on Colab/Kaggle GPU:
LlamaParse → Qwen2.5-VL captioning → BGE-M3 embedding → Milvus build. Emits a
bundle (`milvus.db` + parsed markdown + `metadata.jsonl` + manifest with content
hashes, model versions, chunk parameters). The Space never runs LlamaParse or
the VLM.

**Space setup**
- `deploy/SPACE_README.md` with correct Space card (sdk, python version, secrets)
- Artifacts in Git LFS (Section 4.1)
- CPU reranker; ONNX int8 BGE-M3 if load time or p95 demands it
- Eager model load at boot behind `/health` readiness gate
- Warm-up query at boot
- Explicit "warming up" UI state
- Three-tier demo implemented per Section 4C; Tier 1 traces recorded from real
  runs via `deploy/record_demo_traces.py` and clearly labeled as precached
- Tier 1 includes at least one correction-loop-fires trace and one refusal trace
- `openapi.json` generated and linked from README
- `keep-warm.yml` low-frequency `/health` ping

**Observability** — OTLP export to Grafana Cloud free tier
(`observability/otel_setup.py`). One span per stage:
`embed → retrieve → rerank → route → generate → self_correct`

The per-stage breakdown **will probably surprise you.** Most people discover the
reranker or an unnecessary LLM hop eats 60% of p95. That discovery is the
interview story — capture it in `docs/FAILURE_ANALYSIS.md`.

**Dashboard panels:** QPS, warm latency percentiles, per-stage breakdown, route
distribution, correction fire rate, structured-output validity, cost per query,
error rate by stage, reranker-enabled status, cold-start events.

**Acceptance criteria**
- [ ] Public Space URL live and documented in README
- [ ] Ingest bundle reproducible from manifest; hashes verified on load
- [ ] All stages emit spans; breakdown visible in Grafana Cloud
- [ ] Dashboard screenshot captured
- [ ] Cold start measured over ≥ 5 forced cold starts, stage-broken-down
- [ ] Warm and cold latency reported as separate numbers
- [ ] Steady-state RSS documented
- [ ] All three demo tiers work; Tier 1 loads instantly on a cold container
- [ ] Precached traces visibly labeled; regenerated against current git SHA
- [ ] Key-exposure mitigation implemented and documented
- [ ] `docs/DEPLOYMENT.md` written with tradeoffs

---

### PHASE 6 — Ablations & failure taxonomy (est. 8–10h)

**Goal:** convert every architectural claim into a measured number.

**Ablation matrix** (`eval/ablations.py` → `docs/ABLATIONS.md`). Every row is a
config-only change — no code edits. This is why Phase 0 item 6 was mandatory.

| Configuration | recall@10 | Faithfulness | warm p95 | $/1k | Δ vs full |
|---|---|---|---|---|---|
| Full pipeline (baseline) | | | | | — |
| − BGE reranker | | | | | |
| Reranker: `v2-gemma` (GPU) | | | | | |
| Reranker: `v2-m3` (CPU) | | | | | |
| Reranker: `base` (CPU) | | | | | |
| − hybrid (dense only) | | | | | |
| − hybrid (sparse only) | | | | | |
| sparse/dense weight sweep | | | | | |
| − Self-RAG correction | | | | | |
| − adaptive routing (always vectorstore) | | | | | |
| − query rewriting | | | | | |
| Chunk 512 / 1024 / 2048 | | | | | |
| Score threshold 0.3 / 0.5 / 0.7 | | | | | |
| top_k 3 / 5 / 10 | | | | | |
| ONNX int8 embeddings vs fp32 | | | | | |
| **Backend: BYOK hosted (tool-calling)** | | | | | |
| **Backend: vLLM + xgrammar guided decoding** | | | | | |
| **Backend: vLLM, prefix cache off** | | | | | |

The three backend rows carry two extra columns — `structured_output_validity_rate`
and `time_to_first_token` — and belong in their own sub-table in
`docs/ABLATIONS.md` with written interpretation (Section 4A).

**Be genuinely prepared for a component to not earn its place.** If Self-RAG
costs 900ms and buys 1.2 points, say so publicly and make it config-optional.
That finding makes you look *better* — it shows you measure rather than assume.
Same if the expensive `v2-gemma` reranker barely beats the 278M CPU model.

**Failure taxonomy** (`eval/failure_taxonomy.py` → `docs/FAILURE_ANALYSIS.md`).
Classify every failed item from a full run:

| Code | Root cause |
|---|---|
| `RET_MISS` | Gold chunk absent from hybrid search results entirely |
| `RET_DEMOTED` | Retrieved, but reranker pushed it below threshold |
| `CHUNK_SPLIT` | Answer spans a chunk boundary; no single chunk suffices |
| `GEN_IGNORED_CTX` | Correct context retrieved, generation ignored it |
| `GEN_HALLUCINATED` | Unsupported claim despite adequate context |
| `ROUTE_WRONG` | Router chose the wrong path |
| `SCHEMA_INVALID` | Malformed grader output changed control flow |
| `REWRITE_DRIFT` | Query rewriting moved away from the answer |
| `CORRECTION_DEGRADED` | Self-RAG loop made a passing answer worse |
| `PARSE_ERROR` | Upstream LlamaParse/VLM ingestion defect |
| `JUDGE_DISAGREE` | Manual review says the judge was wrong |

`RET_MISS` vs `RET_DEMOTED` is only separable because Phase 0 plumbed both chunk
ID lists — this distinction is the single most actionable output of the taxonomy.

Publish the distribution. Hand-review a sample to validate the classifier — an
auto-classified taxonomy nobody checked is just more unverified output.

**This is the rarest artifact in a junior portfolio** and converts directly into
interview answers. "What would you improve?" gets a ranked list backed by counts.

**Acceptance criteria**
- [ ] Every ablation row filled with real numbers, config-only
- [ ] `docs/ABLATIONS.md` published with written interpretation per row
- [ ] Backend sub-table complete (Section 4A experiment)
- [ ] Every failure classified; distribution charted
- [ ] ≥ 30 classifications hand-validated; classifier accuracy reported
- [ ] Top 3 failure modes have a written proposed fix

---

### PHASE 7 — README rewrite (est. 3–4h)

The README is what actually gets read. Twenty seconds, then they open the code
or close the tab.

**Required order:**
1. One-line description
2. **Headline numbers table** — recall@10, faithfulness, refusal accuracy, warm
   p95, cold start, cost/query, judge κ, noise floor
3. Live Space link (note it may take ~Ns to wake) + Grafana dashboard link
4. Architecture diagram
5. **Reproduce in 3 commands** — must work from a clean clone
6. Eval methodology: golden set composition, judge calibration, noise floor,
   Tavily fixture approach
7. CI screenshot showing a blocked merge
8. Ablation table (link to full doc), including the backend comparison
9. Failure analysis summary (link to full doc)
10. **Honest limitations** — single-doc scoping, 15-paper corpus, judge caveats,
    CPU reranker quality tradeoff, cold-start behavior, cached web search

The limitations section signals more maturity than any other paragraph.

**Acceptance criteria**
- [ ] Numbers table above the fold
- [ ] Clean-clone reproduce path verified in a fresh container
- [ ] Every claim traces to a committed results file
- [ ] Warm and cold latency listed separately
- [ ] Limitations section present and specific

---

## 8. Global invariants

1. **No new features.** Section 2.1 governs. Park ideas in `docs/BACKLOG.md`.
2. **Corpus is frozen.** Changes require an eval-set version bump plus
   re-verification.
3. **Chunk IDs are deterministic.** Rebuilds from identical sources produce
   identical IDs. Enforced in CI.
4. **`gold_passages` is ground truth; chunk IDs are derived.** Never hand-edit a
   resolved chunk ID. A passage that fails to resolve is a hard error, not a skip.
5. **`doc_id` is immutable.** Changing one is an eval-set version bump.
6. **Never tune against the main golden set.** Dev split only.
7. **Every published number traces to a committed results JSON** with git SHA,
   config, golden-set version, judge version, Tavily fixture version, and backend.
8. **Retrieval and generation metrics are always reported separately.**
9. **Warm and cold latency are always reported separately.** Never blended.
10. **Config-only ablations.** If an ablation needs a code edit, config is incomplete.
11. **Judge is pinned and independent** of the generation model. Prompt changes
   invalidate baseline comparability — version and re-baseline.
12. **Tavily replays from frozen fixtures** in every eval and CI run.
13. **Precached demo output is always labeled as precached** and always recorded
    from real runs, never hand-written.
14. **Streamlit and FastAPI both call `get_runtime()` in-process.** No self-HTTP.
15. **Silent fallbacks are forbidden.** A disabled reranker, a skipped model, a
    rate-limited call, a cache miss — all loud, logged, and surfaced in results.
16. **Negative results get published.** A component that doesn't earn its cost is
    a finding, not an embarrassment.

---

## 9. Commands

```bash
# dev
pytest
ruff check . && ruff format .

# eval
python -m eval.harness --config configs/default.yaml --split full
python -m eval.harness --config configs/default.yaml --split fast --retrieval-only
python -m eval.noise_floor --runs 5
python -m eval.ablations --matrix configs/ablations.yaml
python -m eval.failure_taxonomy --results eval/results/<file>.json

# golden set
python -m eval.resolve_passages --config configs/default.yaml   # -> resolved/<hash>.json
python -m eval.validate_golden                                   # schema + uniqueness

# tavily fixtures
python -m eval.tavily_cache --record   # one-time, live
TAVILY_MODE=replay python -m eval.harness ...   # default

# corpus / metadata
python deploy/build_metadata.py --seed artifacts/corpus_seed.csv --out artifacts/metadata.jsonl
python deploy/fetch_corpus.py            # re-download + re-parse from arXiv

# ingest (GPU host)
python deploy/build_ingest_artifacts.py --pdfs data/test_pdfs --out artifacts/

# demo traces
python deploy/record_demo_traces.py --questions artifacts/demo_questions.txt

# local serve
uvicorn src.api:app --reload --port 8000    # FastAPI
streamlit run app.py                         # Streamlit demo
```

---

## 10. Effort summary

| Phase | Est. hours | Blocks |
|---|---|---|
| 0 — Unblock | 15–18 | everything |
| 1 — Golden set | 10–14 | 2, 3, 4, 6 |
| 2 — Metrics harness | 10–12 | 3, 4, 6 |
| 3 — Noise floor | 2–3 | 4 |
| 4 — CI | 6–8 | — |
| 5 — Deploy & observability | 8–10 | 6 (latency data) |
| 6 — Ablations & taxonomy | 8–10 | 7 |
| 7 — README | 3–4 | — |
| **Total** | **62–79** | |

Phases 1 and 5 can overlap — deployment work slots into the gaps while
hand-verifying eval items. Phase 1 is the largest single block; start it
immediately after Phase 0.

---

## 11. What this unlocks

Before: *"Architected an enterprise-grade document Q&A system using a
deterministic LangGraph state machine, implementing Adaptive-RAG routing and
Self-RAG correction..."* — a description of tools used.

After, each claim backed by a committed results file:
- Built a 187-question hand-verified evaluation set with a 20% unanswerable
  bucket, measuring hallucination rate directly via refusal accuracy
- Wired retrieval regression testing into CI, blocking merges on recall
  degradation beyond a measured ±1.2-point noise floor
- Quantified every architectural component via config-only ablation: reranker
  contributed +X recall@5 at +Yms; Self-RAG correction fired on Z% of queries
- Measured guided decoding vs. hosted tool-calling for structured output:
  A% vs. B% schema validity across four grading nodes
- Deployed to free CPU infrastructure at $0/month, warm p95 Xms, $Y per 1k
  queries, with per-stage OpenTelemetry tracing

The difference is the difference between a tool list and engineering judgment.
