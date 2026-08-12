"""End-to-end smoke test: builds the real runtime and answers one query
headlessly (no Streamlit, no FastAPI) -- the "does the whole system
actually work" check PROJECT_SPEC.md's Phase 0 acceptance criteria calls
for.

This is a real integration test: it loads the actual embedding model and
reranker, connects to the real Milvus collection, and makes one live LLM
call. Skipped when the corpus or a configured LLM provider isn't
available locally (e.g. a fresh clone before the ingest step has run) --
not something CI should attempt without those prerequisites.
"""

from pathlib import Path

import pytest

from src.agent import build_agent_graph, run_query_with_state
from src.configuration import config_rag
from src.runtime import get_runtime

REPO_ROOT = Path(__file__).resolve().parent.parent
PARSED_MD_DIR = REPO_ROOT / "artifacts" / "parsed_md"

ERROR_FALLBACK_PREFIX = "I encountered an error processing your request"


def _corpus_present() -> bool:
    return PARSED_MD_DIR.exists() and any(PARSED_MD_DIR.glob("*.md"))


def _llm_configured() -> bool:
    config = config_rag()
    return bool(config.get("llm_provider")) and bool(config.get("llm_api_key"))


pytestmark = pytest.mark.skipif(
    not _corpus_present() or not _llm_configured(),
    reason=(
        "requires the local corpus (artifacts/parsed_md/) and a configured LLM "
        "provider (api_keys.json) -- not available on a fresh clone before ingest"
    ),
)


def test_end_to_end_headless_query():
    runtime = get_runtime()
    graph = build_agent_graph(runtime.database, runtime.embedding_model, runtime.rerank_model, runtime.llm, runtime.config)

    answer, final_state, trace_info = run_query_with_state(graph, "What is LoRA?", [])

    assert answer
    assert not answer.startswith(ERROR_FALLBACK_PREFIX), f"got fallback error answer: {answer}"
    assert trace_info["node_sequence"], "graph should have executed at least one node"
    assert "retrieved_chunk_ids" in final_state
    assert "reranked_chunk_ids" in final_state
