# 🤖 Multimodal Agentic RAG

An advanced **Retrieval-Augmented Generation (RAG)** system built with **LangGraph**, combining hybrid vector search, multimodal document parsing, and an agentic workflow with self-correction capabilities.

---

## 📌 Overview

This project implements a production-ready, agentic RAG pipeline that:

- Parses academic PDFs (including figures/images) using **LlamaParse** and a **Qwen 2.5 VLM**
- Stores parsed content in a **Milvus** hybrid vector database (sparse + dense search via BGE-M3)
- Serves a **LangGraph**-based agent with adaptive routing, query rewriting, hallucination checking, and web search fallback
- Serves LLM inference locally via **vLLM** (Qwen3-8B-AWQ)

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│   Query Router  │  ──► chitchat ──► LLM Response
│  (Adaptive RAG) │  ──► websearch ──► Tavily ──► Generate
└─────────────────┘  ──► vectorstore
         │
         ▼
┌──────────────────────┐
│  Retrieve & Rerank   │  (Milvus Hybrid Search + BGE Reranker)
└──────────────────────┘
         │
    ┌────┴─────┐
    │          │
  docs      no docs
    │          │
    ▼          ▼
 Generate   Rewrite Query ──► loop > 3 ──► Web Search
    │
    ▼
┌──────────────────────────┐
│ Hallucination + Relevance│
│       Checker            │
└──────────────────────────┘
    │           │           │
  PASS       regenerate  rewrite_query
    │
    ▼
  END ✅
```

### Agent Nodes

| Node | Description |
|---|---|
| `query_router` | Routes to `vectorstore`, `websearch`, or `chitchat` |
| `retrieve_and_rerank` | Hybrid search (sparse + dense) then BGE reranker with score filtering |
| `generate` | Answers the question using retrieved context |
| `rewrite_query` | Rewrites failing queries for better semantic retrieval |
| `web_search` | Falls back to Tavily web search |
| `chitchat_node` | Handles simple greetings/small talk |
| `hallucinations_and_relevence_router` | Validates the generation for grounding and relevance |

---

## 📂 Project Structure

```
Multimodal Agentic RAG/
├── graph_rag.ipynb          # Main LangGraph agentic RAG notebook
├── hybrid_database.py       # Milvus hybrid database construction & search
├── config.py                # Configuration for parsing and RAG pipeline
├── helper.py                # Utility functions (JSON parsing, JSONL loading)
├── vllm_serve_start.sh      # Shell script to start vLLM server
├── requirements.txt         # Python dependencies
├── .gitignore
└── data/                    # (gitignored) PDFs, parsed markdowns, metadata
    ├── raw_pdfs_parsed/     # Parsed markdown files + metadata.jsonl
    └── test_pdfs/           # Raw PDF inputs
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: 24GB+ VRAM)
- [vLLM](https://docs.vllm.ai/) installed

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create an `api_keys.json` file in the project root:

```json
{
    "llama_parse": "YOUR_LLAMA_CLOUD_API_KEY",
    "tavilly_web": "YOUR_TAVILY_API_KEY"
}
```

> ⚠️ `api_keys.json` is gitignored. Never commit it.

### 4. Start the vLLM Server

```bash
bash vllm_serve_start.sh
```

This starts **Qwen3-8B-AWQ** on `localhost:8000` with:
- Prefix caching enabled
- `deepseek_r1` reasoning parser
- FP16 precision

### 5. Build the Vector Database

Place your parsed markdown files and `metadata.jsonl` into `data/raw_pdfs_parsed/`, then run:

```bash
python hybrid_database.py
```

This will embed all documents using **BGE-M3** and store them in a local Milvus Lite database (`milvus.db`).

---

## 🚀 Usage

Open and run `graph_rag.ipynb`:

```python
inputs = {"question": "What is the AlphaCodium paper about?"}

for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")

pprint(value["generation"])
```

The agent will automatically:
1. Route the query
2. Retrieve and rerank relevant documents
3. Generate a grounded answer
4. Self-correct if hallucinations are detected

---

## 🧠 Knowledge Base

The vectorstore contains **15 research papers** covering:

- **LLM Architecture & Serving** — vLLM, PagedAttention, Transformers
- **Advanced RAG Methodologies** — CRAG, Self-RAG, Adaptive-RAG, Vector Databases
- **Model Building from Scratch** — ConvNeXt

Queries outside these topics are automatically routed to **Tavily web search**.

---

## 🔧 Key Components

### Hybrid Search (`hybrid_database.py`)
- Uses **BGE-M3** for both sparse (BM25-like) and dense embeddings
- Stores data in **Milvus Lite** (local file-based)
- Hybrid retrieval with configurable `sparse_weight` and `dense_weight`

### Reranker
- Uses **`BAAI/bge-reranker-v2-gemma`** (`FlagLLMReranker`)
- Filters documents with score `> 0.5`, keeps top 5

### LLM
- **Qwen3-8B-AWQ** served via vLLM
- Structured outputs via `json_schema` for routing, rewriting, and grading nodes

### Document Parsing
- PDFs parsed with **LlamaParse** → Markdown
- Figures/images captioned with **Qwen 2.5 VLM**
- Custom `######` header used for image captions in Markdown splitter

---

## 🛠️ Configuration (`config.py`)

| Key | Description |
|---|---|
| `device` | `"cuda"` for GPU inference |
| `database_path` | Path to Milvus Lite DB (`./milvus.db`) |
| `input_folder_path` | Path to parsed markdown folder |
| `chunk_size` | Token chunk size for splitting (`1024`) |
| `overlap_size` | Chunk overlap (`128`) |
| `tavilly_api_key` | Tavily search API key |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langgraph`, `langchain` | Agentic graph framework |
| `langchain_openai` | LLM client (vLLM-compatible) |
| `pymilvus` | Vector database |
| `FlagEmbedding` | BGE-M3 embeddings & reranker |
| `langchain_tavily` | Web search tool |
| `llama_cloud` | PDF parsing |
| `qwen-vl-utils`, `torchvision` | Multimodal VLM utilities |
| `vllm` | Local LLM serving |

---

## 🔮 Future Work

- Replace LlamaParse with **Docling** for a fully local, offline pipeline
- Add a PDF upload pipeline for dynamic knowledge base expansion
- Support multimodal queries (image inputs to the RAG agent)
- Persist conversation history in graph state

---

## 📄 License

This project is for research and educational purposes.
