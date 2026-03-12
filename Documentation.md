# Multimodal Agentic RAG

A production-grade **Agentic Retrieval-Augmented Generation (RAG)** pipeline that combines hybrid vector search, multi-stage self-critique, adaptive routing, and web search fallback — all orchestrated with **LangGraph**.

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [1. Document Parsing & Multimodal Enrichment](#1-document-parsing--multimodal-enrichment)
  - [2. Database Construction (Hybrid Search)](#2-database-construction-hybrid-search)
  - [3. Agentic RAG Graph](#3-agentic-rag-graph)
- [Graph Nodes & Edges](#graph-nodes--edges)
  - [Nodes](#nodes)
  - [Conditional Edges (Routing Logic)](#conditional-edges-routing-logic)
- [Component Details](#component-details)
  - [Embedding & Retrieval](#embedding--retrieval)
  - [Reranking](#reranking)
  - [LLM (vLLM Serving)](#llm-vllm-serving)
  - [Structured Outputs (Pydantic)](#structured-outputs-pydantic)
- [Setup & Installation](#setup--installation)
  - [Prerequisites](#prerequisites)
  - [Install Dependencies](#install-dependencies)
  - [API Keys](#api-keys)
  - [Start the vLLM Server](#start-the-vllm-server)
  - [Build the Database](#build-the-database)
  - [Run the RAG Pipeline](#run-the-rag-pipeline)
- [Configuration Reference](#configuration-reference)
- [Knowledge Base](#knowledge-base)
- [Future Work](#future-work)

---

## Architecture Overview

```
                          ┌──────────────────────────────────────┐
                          │            User Question              │
                          └──────────────┬───────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    Query Router       │  ← LLM decides route
                              └──┬──────────┬────────┘
                      vectorstore│          │websearch    chitchat│
              ┌─────────────────▼─┐  ┌─────▼──────┐  ┌─────────▼──────┐
              │  Retrieve & Rerank│  │ Web Search  │  │    Chitchat     │
              │  (BGE-M3 Hybrid + │  │  (Tavily)   │  │  (Direct LLM)  │
              │  BGE Reranker)    │  └──────┬──────┘  └────────────────┘
              └──────────┬────────┘         │
              docs found?│                  │
         ┌───────────────┴──────────────────┘
         │                    │ no docs
    ┌────▼──────┐       ┌─────▼──────┐
    │  Generate │       │ Rewrite    │◄──── loop_count > 3 → web search
    │   (LLM)   │       │  Query     │
    └────┬──────┘       └────────────┘
         │
   ┌─────▼──────────────────────────┐
   │   Hallucination Check (LLM)    │
   │   + Relevance Check (LLM)      │
   └──────┬──────────────┬──────────┘
     pass │         fail │ (gen_retries < 2 → regenerate, else rewrite)
          │              │
        END          Generate / Rewrite Query
```

---

## Project Structure

```
Multimodal Agentic RAG/
│
├── graph_rag.ipynb          # Main agentic RAG pipeline (LangGraph)
├── hybrid_database.py       # Database build, hybrid search, and embedding logic
├── config.py                # Centralized configuration for parsing and RAG
├── helper.py                # Utility functions (JSON cleaning, JSONL loading)
├── vllm_serve_start.sh      # Shell script to launch the vLLM server
├── requirements.txt         # Python dependencies
├── api_keys.json            # API keys (git-ignored)
│
└── data/                    # Git-ignored data folder
    ├── raw_pdfs_parsed/     # Parsed markdown files + metadata.jsonl
    └── test_pdfs/           # Raw PDFs for parsing
```

---

## Pipeline Walkthrough

### 1. Document Parsing & Multimodal Enrichment

PDFs are uploaded to **LlamaParse** (via the `llama_cloud` API) which converts them to Markdown. During this step:

- **Images** are extracted from PDFs and passed through **Qwen 2.5 VL** (a multimodal LLM) to generate structured captions.
- Each caption is a JSON object with fields: `title`, `type` (`chart`/`diagram`/`irrelevant`), `features`, and `description`.
- Captions are appended to the Markdown under the special heading `###### Image captions`, allowing the Markdown splitter to correctly tag these chunks as image-derived content.
- A **metadata generation** step uses the VLM on the first page of each paper to extract `title`, `authors`, `year`, and `topic`, which is saved to `metadata.jsonl`.

> Prompts for both image captioning and metadata extraction are defined in `config.py` under `config_parse()`.

---

### 2. Database Construction (Hybrid Search)

Handled by `hybrid_database.py`.

**Splitting Strategy:**
1. **Markdown Header Splitter** — first splits by `#`, `##`, `###`, and `######` (image captions) to preserve document structure.
2. **Recursive Character Splitter** — further splits large chunks to `chunk_size=1024` tokens with `overlap=128`.

**Embedding:**
- Model: `BGE-M3` (`BAAI/bge-m3`) via `pymilvus.model.hybrid.BGEM3EmbeddingFunction`
- Produces both **dense** (COSINE) and **sparse** (SPARSE_INVERTED_INDEX, Inner Product) vectors simultaneously.

**Database:**
- **Milvus Lite** stored locally at `./milvus.db`
- Collection: `arag_project`
- Schema fields: `id`, `text`, `metadata` (JSON), `sparse_vector`, `dense_vector`
- Batch insert size: 50 documents

**Hybrid Search:**
- Combines sparse + dense ANN search via `WeightedRanker`
- Default weights: `sparse=0.7`, `dense=1.0`, retrieves top 20 candidates

---

### 3. Agentic RAG Graph

The graph is defined and compiled in `graph_rag.ipynb` using **LangGraph**'s `StateGraph`.

**State (`GraphState`):**

| Field | Type | Description |
|---|---|---|
| `question` | `str` | Current (possibly rewritten) user question |
| `generation` | `str` | LLM's latest generated answer |
| `documents` | `list` | Retrieved/web documents |
| `missing_concept_query` | `str` | Reserved for concept gap queries |
| `loop_count` | `int` | Tracks rewrite iterations (max 3 before web fallback) |
| `gen_retries` | `int` | Tracks generation retries (max 2 before rewrite) |

---

## Graph Nodes & Edges

### Nodes

| Node | Function | Description |
|---|---|---|
| `retrieve_and_rerank` | `retrieve_and_rerank()` | Hybrid search (BGE-M3) → BGE Reranker v2 Gemma filter (score > 0.5, top 5) |
| `generate` | `generate()` | Generates an answer from documents using the LLM |
| `rewrite_query` | `rewrite_query()` | Rewrites the query for better semantic search; increments `loop_count` |
| `web_search` | `web_search()` | Fetches web results via Tavily API |
| `chitchat` | `chitchat_node()` | Handles greetings/off-topic messages directly with the LLM |

### Conditional Edges (Routing Logic)

| Edge | Function | Logic |
|---|---|---|
| `START → *` | `query_router()` | LLM classifies query → `vectorstore`, `websearch`, or `chitchat` |
| `retrieve_and_rerank → *` | `post_retrieve_router()` | If 0 docs → `rewrite_query`; else → `generate` |
| `rewrite_query → *` | `rewrite_router()` | If `loop_count > 3` → `web_search`; else → `retrieve_and_rerank` |
| `generate → *` | `hallucinations_and_relevence_router()` | Hallucination check → Relevance check → `all_pass` (END) / `generate` / `rewrite_query` |

---

## Component Details

### Embedding & Retrieval

```python
# hybrid_database.py
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cuda")
# Returns both dense and sparse vectors in a single pass
```

Hybrid search merges sparse BM25-style retrieval with dense semantic retrieval, providing robustness for both keyword and semantic queries.

### Reranking

```python
# graph_rag.ipynb
rerank_model = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=False, devices=config["device"])
# Scores (question, document) pairs, filters score > 0.5, keeps top 5
```

Cross-encoder reranking significantly improves precision after the initial bi-encoder retrieval.

### LLM (vLLM Serving)

The LLM is served locally via **vLLM**:

- **Model:** `Qwen/Qwen3-8B-AWQ` (quantized)
- **Reasoning parser:** `deepseek_r1`
- **Endpoint:** `http://localhost:8000/v1` (OpenAI-compatible)
- **Structured output backend:** `xgrammar` for guaranteed JSON schema compliance

```python
llm_model = ChatOpenAI(
    model_name="Qwen/Qwen3-8B-AWQ",
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
    model_kwargs={"extra_body": {"guided_decoding_backend": "xgrammar"}}
)
```

### Structured Outputs (Pydantic)

All LLM decision nodes use `with_structured_output(..., method="json_schema", strict=True)` for reliable, typed outputs:

| Model | Fields | Used In |
|---|---|---|
| `RouteDecision` | `reasoning`, `is_in_domain`, `route` | Query Router |
| `RewrittenQuery` | `reasoning`, `query` | Query Rewriter |
| `HallucinationScore` | `reasoning`, `is_grounded` | Hallucination Checker |
| `RelevanceScore` | `reasoning`, `is_relevant` | Relevance Checker |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 24GB+ VRAM for Qwen3-8B-AWQ)
- `vllm` installed in the environment

### Install Dependencies

```bash
pip install -r requirements.txt
```

### API Keys

Create `api_keys.json` in the project root (this file is git-ignored):

```json
{
    "llama_parse": "YOUR_LLAMA_CLOUD_API_KEY",
    "tavilly_web": "YOUR_TAVILY_API_KEY"
}
```

### Start the vLLM Server

```bash
bash vllm_serve_start.sh
```

This will:
1. Launch `Qwen/Qwen3-8B-AWQ` on port `8000`
2. Enable prefix caching, `deepseek_r1` reasoning parser
3. Poll `vllm_server.log` and print a success message when ready

Configuration variables at the top of the script:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-8B-AWQ` | HuggingFace model ID |
| `PORT` | `8000` | Serving port |
| `MAX_MODEL_LEN` | `8192` | Max context length |
| `GPU_MEM_UTIL` | `0.85` | GPU memory utilization |

### Build the Database

Run `hybrid_database.py` directly to parse data and build the Milvus collection:

```bash
python hybrid_database.py
```

This will:
1. Load all `.md` files from `data/raw_pdfs_parsed/`
2. Split, embed (BGE-M3), and insert into `./milvus.db`

Alternatively, call `data_preprocessing(config)` from within the notebook.

### Run the RAG Pipeline

Open and run `graph_rag.ipynb` cell by cell:

1. **Cell 1–3:** Imports, model loading, config initialization
2. **Cell 4:** Tavily web search client test
3. **Cells 7–11:** Individual LLM chain definitions (router, rewriter, generator, checkers)
4. **Cell 12:** Graph state definition
5. **Cell 13:** Node function definitions
6. **Cell 14:** Edge/routing function definitions
7. **Cell 15:** Graph compilation
8. **Cell 16:** Run the graph with a sample query

---

## Configuration Reference

### `config_rag()` — RAG Pipeline Config

| Key | Default | Description |
|---|---|---|
| `device` | `"cuda"` | Device for embedding and reranker models |
| `tavilly_api_key` | from `api_keys.json` | Tavily web search API key |
| `database_path` | `"./milvus.db"` | Path to Milvus Lite database file |
| `input_folder_path` | `"data/raw_pdfs_parsed"` | Folder with parsed markdown files |
| `chunk_size` | `1024` | Max characters per text chunk |
| `overlap_size` | `128` | Overlap between consecutive chunks |

### `config_parse()` — Parsing Pipeline Config

| Key | Description |
|---|---|
| `api_key` | LlamaParse API key |
| `input_folder` | Folder of raw PDFs |
| `output_folder` | Output folder for parsed markdown |
| `device` | Device for VLM inference |
| `prompt_imgcap` | System + user prompts for image captioning |
| `prompt_metagen` | System + user prompts for metadata extraction |

---

## Knowledge Base

The vectorstore contains **15 research papers** covering:

| Topic Area | Examples |
|---|---|
| LLM Architecture & Serving | vLLM, PagedAttention, Transformers |
| Advanced RAG Methodologies | CRAG, Self-RAG, Adaptive-RAG, Vector Databases |
| Model Building from Scratch | ConvNeXt |

Queries about topics outside this scope are automatically routed to **Tavily web search**.

---

## Future Work

- [ ] Replace LlamaParse with **Docling** for a fully local, offline parsing pipeline
- [ ] Add a PDF upload API endpoint for dynamic database updates
- [ ] Expand the knowledge base with more papers
- [ ] Add conversation memory / multi-turn chat support
- [ ] Implement a Gradio or Streamlit UI
- [ ] Containerize the full stack with Docker
