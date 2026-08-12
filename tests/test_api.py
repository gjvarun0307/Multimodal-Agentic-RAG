"""Tests for src/api.py. Monkeypatches runtime construction and query
execution so these run fast without loading real models or making live
LLM calls -- CI must stay fast and deterministic (PROJECT_SPEC.md §4B.3).
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api as api_module


@pytest.fixture
def client(monkeypatch):
    fake_runtime = SimpleNamespace(database=None, embedding_model=None, rerank_model=None, llm=None, config={})
    monkeypatch.setattr(api_module, "get_runtime", lambda: fake_runtime)
    monkeypatch.setattr(api_module, "build_agent_graph", lambda *a, **k: "fake_graph")
    with TestClient(api_module.app) as c:
        yield c


def test_health_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_metrics_reports_ready(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "arag_ready 1" in response.text


def test_query_returns_full_trace(client, monkeypatch):
    fake_final_state = {
        "retrieved_chunk_ids": ["adaptive_rag::0000::deadbeef"],
        "reranked_chunk_ids": ["adaptive_rag::0000::deadbeef"],
    }
    fake_trace_info = {
        "node_sequence": ["retrieve_and_rerank", "generate"],
        "stage_latencies_ms": {"retrieve_and_rerank": 100.0, "generate": 200.0},
        "fallback_events": ["rerank_fallback"],
    }
    monkeypatch.setattr(
        api_module, "run_query_with_state", lambda *a, **k: ("the answer", fake_final_state, fake_trace_info)
    )

    response = client.post("/query", json={"question": "What is LoRA?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "the answer"
    assert body["route_decision"] == "retrieve_and_rerank"
    assert body["node_sequence"] == ["retrieve_and_rerank", "generate"]
    assert body["retrieved_chunk_ids"] == ["adaptive_rag::0000::deadbeef"]
    assert body["reranked_chunk_ids"] == ["adaptive_rag::0000::deadbeef"]
    assert body["correction_fired"] is False
    assert body["fallback_events"] == ["rerank_fallback"]
    assert body["stage_latencies_ms"] == {"retrieve_and_rerank": 100.0, "generate": 200.0}
    assert body["total_latency_ms"] >= 0
    assert "git_sha" in body


def test_query_detects_correction_fired(client, monkeypatch):
    fake_final_state = {"retrieved_chunk_ids": [], "reranked_chunk_ids": []}
    fake_trace_info = {
        "node_sequence": ["retrieve_and_rerank", "generate", "rewrite_query", "retrieve_and_rerank", "generate"],
        "stage_latencies_ms": {},
        "fallback_events": [],
    }
    monkeypatch.setattr(
        api_module, "run_query_with_state", lambda *a, **k: ("answer", fake_final_state, fake_trace_info)
    )

    response = client.post("/query", json={"question": "..."})
    body = response.json()
    assert body["correction_fired"] is True
    assert body["fallback_events"] == []


def test_trace_endpoint_404_until_phase_5(client):
    response = client.get("/trace/demo_01")
    assert response.status_code == 404


def test_not_ready_returns_503(monkeypatch):
    def failing_get_runtime():
        raise RuntimeError("boom")

    monkeypatch.setattr(api_module, "get_runtime", failing_get_runtime)
    with TestClient(api_module.app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/query", json={"question": "hi"}).status_code == 503
