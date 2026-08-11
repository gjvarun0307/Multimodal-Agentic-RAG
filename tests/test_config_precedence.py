import pytest

from src.configuration import config_rag, reload_config

PROMOTED_TUNABLES = {
    "search_limit",
    "reranker_top_k",
    "sparse_weight",
    "dense_weight",
    "reranker_score_threshold",
    "reranker_model",
    "use_fp16",
    "max_gen_retries",
    "max_rewrites",
    "max_chat_turns",
    "insert_batch_size",
    "web_search_max_results",
    "web_search_topic",
}


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset the Config singleton before and after each test so env var
    mutations in one test can't leak into another."""
    reload_config()
    yield
    reload_config()


def test_config_rag_exposes_promoted_tunables():
    """These used to be hardcoded literals scattered across agent.py and
    hybrid_database.py; config_rag() must expose all of them so a config
    dict alone can drive a run or an ablation, per invariant 10."""
    resolved = config_rag()
    assert PROMOTED_TUNABLES.issubset(resolved.keys())


def test_defaults_used_when_no_env_or_override():
    resolved = config_rag()
    assert resolved["search_limit"] == 20
    assert resolved["reranker_top_k"] == 5
    assert resolved["sparse_weight"] == 0.7
    assert resolved["dense_weight"] == 1.0
    assert resolved["reranker_score_threshold"] == 0.5
    assert resolved["max_gen_retries"] == 2
    assert resolved["max_rewrites"] == 3


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("SEARCH_LIMIT", "50")
    monkeypatch.setenv("RERANKER_TOP_K", "10")
    monkeypatch.setenv("MAX_REWRITES", "7")
    reload_config()

    resolved = config_rag()
    assert resolved["search_limit"] == 50
    assert resolved["reranker_top_k"] == 10
    assert resolved["max_rewrites"] == 7


def test_explicit_override_wins_over_env(monkeypatch):
    monkeypatch.setenv("SEARCH_LIMIT", "50")
    reload_config()

    resolved = config_rag(overrides={"search_limit": 999})
    assert resolved["search_limit"] == 999


def test_max_rewrites_key_name_matches_what_agent_reads():
    """Regression test: agent.py's rewrite_router() used to read
    config_rag().get("max_rewrite_attempts", 3) -- a key config_rag()
    never populated -- so MAX_REWRITES/config always silently fell back
    to the literal default. Assert the exposed key matches the field
    name agent.py actually reads (`max_rewrites`)."""
    resolved = config_rag()
    assert "max_rewrites" in resolved
    assert "max_rewrite_attempts" not in resolved


def test_config_dict_alone_constructs_a_full_run_config():
    """Acceptance criterion (PROJECT_SPEC.md Phase 0): a config dict alone
    must be sufficient to construct a full run config, without touching
    env vars or api_keys.json -- this is what the eval harness needs for
    config-only ablations."""
    overrides = {
        "search_limit": 50,
        "reranker_top_k": 3,
        "sparse_weight": 0.5,
        "dense_weight": 0.8,
        "reranker_score_threshold": 0.3,
        "max_gen_retries": 1,
        "max_rewrites": 2,
    }
    resolved = config_rag(overrides=overrides)
    for key, value in overrides.items():
        assert resolved[key] == value
