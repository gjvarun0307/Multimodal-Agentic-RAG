"""Unit tests for src.agent.retrieve_and_rerank_core -- the retrieval logic
pulled out of build_agent_graph()'s retrieve_and_rerank node closure so
eval.harness's --retrieval-only mode can call the exact same code instead
of a separately maintained reimplementation. hybrid_search is
monkeypatched (fake Milvus results); no real database/embedding model.
"""

import src.agent as agent_module
from src.agent import retrieve_and_rerank_core


class _FakeRerankModel:
    def __init__(self, scores):
        self._scores = scores

    def compute_score(self, pairs, **kwargs):
        return self._scores


class _FailingRerankModel:
    def compute_score(self, pairs, **kwargs):
        raise RuntimeError("simulated rerank failure")


def _fake_docs(n):
    return [{"id": f"doc::{i:04d}", "text": f"text {i}", "metadata": {}} for i in range(n)]


def test_empty_hybrid_search_returns_all_empty(monkeypatch):
    monkeypatch.setattr(agent_module, "hybrid_search", lambda *a, **k: [])
    result = retrieve_and_rerank_core("q", database=None, embedding_model=None, rerank_model=None, config={})
    assert result == {
        "documents": [],
        "retrieved_chunk_ids": [],
        "retrieved_chunk_scores": [],
        "reranked_chunk_ids": [],
    }


def test_no_reranker_takes_top_k_raw(monkeypatch):
    docs = _fake_docs(5)
    monkeypatch.setattr(agent_module, "hybrid_search", lambda *a, **k: docs)
    result = retrieve_and_rerank_core(
        "q", database=None, embedding_model=None, rerank_model=None, config={"reranker_top_k": 2}
    )
    assert result["retrieved_chunk_ids"] == [d["id"] for d in docs]
    assert result["retrieved_chunk_scores"] == []
    assert result["reranked_chunk_ids"] == [docs[0]["id"], docs[1]["id"]]
    assert "fallback_events" not in result


def test_reranker_filters_by_threshold_and_top_k(monkeypatch):
    docs = _fake_docs(4)
    monkeypatch.setattr(agent_module, "hybrid_search", lambda *a, **k: docs)
    # doc 0 below threshold, docs 1-3 above, keep top_k=2 by score
    rerank_model = _FakeRerankModel([0.1, 0.9, 0.6, 0.8])
    result = retrieve_and_rerank_core(
        "q",
        database=None,
        embedding_model=None,
        rerank_model=rerank_model,
        config={"reranker_score_threshold": 0.5, "reranker_top_k": 2},
    )
    assert result["retrieved_chunk_scores"] == [0.1, 0.9, 0.6, 0.8]
    assert result["reranked_chunk_ids"] == ["doc::0001", "doc::0003"]  # scores 0.9, 0.8
    assert "fallback_events" not in result


def test_reranker_failure_falls_back_loudly(monkeypatch):
    docs = _fake_docs(3)
    monkeypatch.setattr(agent_module, "hybrid_search", lambda *a, **k: docs)
    result = retrieve_and_rerank_core(
        "q", database=None, embedding_model=None, rerank_model=_FailingRerankModel(), config={"reranker_top_k": 2}
    )
    assert result["fallback_events"] == ["rerank_fallback"]
    assert result["retrieved_chunk_scores"] == []
    assert result["reranked_chunk_ids"] == [docs[0]["id"], docs[1]["id"]]
