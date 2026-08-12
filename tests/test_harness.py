"""Tests for eval/harness.py. Pure helpers (stratified sampling, resolved-
cache lookup, document formatting, config redaction) are tested directly.
_run_full/_run_retrieval_only/run_eval's LLM/graph/database dependencies
are monkeypatched -- same convention as tests/test_api.py -- so these run
fast without loading real models or making live calls. See
tests/test_harness_live.py for a real end-to-end retrieval-only run
against the actual corpus (no LLM key needed).
"""

from types import SimpleNamespace

import pytest

import eval.harness as harness_module
from eval.harness import (
    DEFAULT_RESOLVED_DIR,
    _format_documents,
    _redact_config,
    find_resolved_cache,
    gold_chunk_ids_by_item,
    stratified_fast_subset,
)


def _mk_items(counts: dict[str, int]) -> list[dict]:
    items = []
    i = 0
    for category, n in counts.items():
        for _ in range(n):
            items.append({"id": f"gs_{i:04d}", "category": category})
            i += 1
    return items


# --------------------------------------------------------------------------
# stratified_fast_subset
# --------------------------------------------------------------------------


def test_stratified_fast_subset_proportional_and_deterministic():
    items = _mk_items({"single_hop": 60, "multi_hop": 40})
    subset1 = stratified_fast_subset(items, target_n=20)
    subset2 = stratified_fast_subset(items, target_n=20)
    assert [it["id"] for it in subset1] == [it["id"] for it in subset2]

    n_single = sum(1 for it in subset1 if it["category"] == "single_hop")
    n_multi = sum(1 for it in subset1 if it["category"] == "multi_hop")
    assert n_single > n_multi  # 60/40 split should stay roughly proportional
    assert n_single + n_multi == len(subset1)


def test_stratified_fast_subset_never_drops_a_category():
    items = _mk_items({"chitchat": 2, "single_hop": 100})
    subset = stratified_fast_subset(items, target_n=10)
    assert any(it["category"] == "chitchat" for it in subset)


def test_stratified_fast_subset_caps_at_category_size():
    items = _mk_items({"tiny": 1, "big": 100})
    subset = stratified_fast_subset(items, target_n=50)
    n_tiny = sum(1 for it in subset if it["category"] == "tiny")
    assert n_tiny == 1  # can't sample more than exists


# --------------------------------------------------------------------------
# find_resolved_cache / gold_chunk_ids_by_item (against the real, committed cache)
# --------------------------------------------------------------------------


def test_find_resolved_cache_matches_default_config():
    doc = find_resolved_cache(DEFAULT_RESOLVED_DIR, chunk_size=1024, overlap_size=128)
    assert doc["chunk_size"] == 1024
    assert doc["overlap_size"] == 128
    assert "items" in doc


def test_find_resolved_cache_raises_for_unknown_config():
    with pytest.raises(FileNotFoundError):
        find_resolved_cache(DEFAULT_RESOLVED_DIR, chunk_size=99999, overlap_size=1)


def test_gold_chunk_ids_by_item_has_entries_for_known_item():
    doc = find_resolved_cache(DEFAULT_RESOLVED_DIR, chunk_size=1024, overlap_size=128)
    mapping = gold_chunk_ids_by_item(doc)
    assert "gs_0001" in mapping
    assert isinstance(mapping["gs_0001"], list)


# --------------------------------------------------------------------------
# _format_documents / _redact_config
# --------------------------------------------------------------------------


def test_format_documents_handles_dict_and_string_lists():
    assert _format_documents([]) == "No context available."
    assert _format_documents([{"text": "a"}, {"text": "b"}]) == "a\n\nb"
    assert _format_documents(["x", "y"]) == "x\n\ny"


def test_redact_config_hides_secrets_not_other_fields():
    config = {
        "llm_api_key": "sk-real-secret",
        "judge_api_key": "gsk-real",
        "tavilly_api_key": "tvly-real",
        "chunk_size": 1024,
    }
    redacted = _redact_config(config)
    assert redacted["llm_api_key"] == "***redacted***"
    assert redacted["judge_api_key"] == "***redacted***"
    assert redacted["tavilly_api_key"] == "***redacted***"
    assert redacted["chunk_size"] == 1024
    assert "real-secret" not in str(redacted)


def test_redact_config_leaves_empty_keys_empty():
    config = {"llm_api_key": "", "chunk_size": 1024}
    redacted = _redact_config(config)
    assert redacted["llm_api_key"] == ""


# --------------------------------------------------------------------------
# run_eval retrieval-only, fully monkeypatched (no models, no real Milvus)
# --------------------------------------------------------------------------


def test_run_eval_retrieval_only_writes_results_and_metrics(monkeypatch, tmp_path):
    fake_items = [
        {
            "id": "gs_0001",
            "question": "q1",
            "expected_route": "vectorstore",
            "gold_answer": "a1",
            "category": "single_hop",
        },
        {"id": "gs_9999", "category": "chitchat", "expected_route": "chitchat", "question": "hi", "gold_answer": ""},
    ]
    monkeypatch.setattr(harness_module, "_load_items", lambda split: fake_items)
    monkeypatch.setattr(
        harness_module,
        "find_resolved_cache",
        lambda resolved_dir, **kw: {"items": {"gs_0001": {"gold_chunk_ids_union": ["doc::0001::aaaa"]}}},
    )
    monkeypatch.setattr(harness_module, "load_or_create_database", lambda config: (None, None))
    monkeypatch.setattr(harness_module, "build_reranker", lambda config: None)
    monkeypatch.setattr(
        harness_module,
        "retrieve_and_rerank_core",
        lambda question, **kw: {
            "documents": [],
            "retrieved_chunk_ids": ["doc::0001::aaaa"],
            "retrieved_chunk_scores": [],
            "reranked_chunk_ids": ["doc::0001::aaaa"],
        },
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text("golden_set_version: 1\noverrides:\n  chunk_size: 1024\n  overlap_size: 128\n")
    results_dir = tmp_path / "results"

    results, out_path = harness_module.run_eval(
        config_path=config_path, split="full", retrieval_only=True, results_dir=results_dir
    )

    assert out_path.exists()
    assert results["backend"] == "retrieval-only"
    assert results["warmup_excluded"] is True
    assert results["n_items"] == 2
    assert results["metrics"]["retrieval"]["n_items_scored"] == 1  # only gs_0001 has gold chunk ids
    assert results["metrics"]["retrieval"]["stage1"]["recall@10"] == 1.0
    assert "generation" not in results["metrics"]  # zero LLM calls in retrieval-only mode
    assert results["judge_version"] is None


def test_run_eval_full_mode_wires_router_and_structured(monkeypatch, tmp_path):
    from eval.metrics.structured import StructuredCallEvent

    fake_items = [
        {
            "id": "gs_0001",
            "question": "q1",
            "expected_route": "vectorstore",
            "gold_answer": "a1",
            "category": "single_hop",
        }
    ]

    monkeypatch.setattr(harness_module, "_load_items", lambda split: fake_items)
    monkeypatch.setattr(
        harness_module,
        "find_resolved_cache",
        lambda resolved_dir, **kw: {"items": {"gs_0001": {"gold_chunk_ids_union": ["doc::0001::aaaa"]}}},
    )
    fake_runtime = SimpleNamespace(database=None, embedding_model=None, rerank_model=None, llm=None, config={})
    monkeypatch.setattr(harness_module, "get_runtime", lambda config: fake_runtime)
    monkeypatch.setattr(harness_module, "build_agent_graph", lambda *a, **k: "fake_graph")
    monkeypatch.setattr(harness_module, "build_tavily_tool", lambda *a, **k: "fake_tool")

    def fake_build_judge_llm(config):
        raise harness_module.JudgeConfigError("no key in test")

    monkeypatch.setattr(harness_module, "build_judge_llm", fake_build_judge_llm)

    def fake_run_query_with_state(graph, question, chat_history):
        final_state = {
            "retrieved_chunk_ids": ["doc::0001::aaaa"],
            "reranked_chunk_ids": ["doc::0001::aaaa"],
            "retrieved_chunk_scores": [0.9],
            "documents": [{"text": "ctx"}],
        }
        trace_info = {
            "node_sequence": ["retrieve_and_rerank", "generate"],
            "stage_latencies_ms": {"retrieve_and_rerank": 100.0, "generate": 200.0},
            "fallback_events": [],
        }
        return "an answer", final_state, trace_info

    monkeypatch.setattr(harness_module, "run_query_with_state", fake_run_query_with_state)

    # Patch InstrumentingLLM so query_router registers as "valid" without a
    # real LLM -- verifies misroute_context wiring end to end.
    class _FakeInstrumentingLLM:
        def __init__(self, real_llm, recorder):
            recorder.record(
                StructuredCallEvent(node="query_router", valid=True, retried=False, coerced=None, error=None)
            )

    monkeypatch.setattr(harness_module, "InstrumentingLLM", _FakeInstrumentingLLM)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "golden_set_version: 1\nbackend: byok-hosted\noverrides:\n  chunk_size: 1024\n  overlap_size: 128\n"
    )
    results_dir = tmp_path / "results"

    results, out_path = harness_module.run_eval(
        config_path=config_path, split="full", retrieval_only=False, results_dir=results_dir
    )

    assert out_path.exists()
    assert results["backend"] == "byok-hosted"
    assert results["metrics"]["router"]["accuracy"] == 1.0
    assert results["metrics"]["structured"]["aggregate"]["n_calls"] == 1
    assert results["metrics"]["generation"]["faithfulness"]["rate"] is None  # judge unavailable
    assert results["judge_version"] is None  # judge never actually used
    assert results["per_item"][0]["total_latency_ms"] > 0
