"""Tests for eval/tavily_cache.py's replay/record fixture (PROJECT_SPEC.md
invariant 12). No network access -- RecordingTavilyTool's underlying
TavilySearch/TavilySearchAPIWrapper is monkeypatched, never invoked for
real, matching this module's own no-live-call-in-tests intent.
"""

import json
from types import SimpleNamespace

import pytest

import eval.tavily_cache as tavily_cache_module
from eval.tavily_cache import (
    DEFAULT_FIXTURE_PATH,
    FIXTURE_VERSION,
    ReplayTavilyTool,
    TavilyCacheMissError,
    _query_key,
    build_tavily_tool,
    record_from_harness_run,
)


def _write_fixture(path, entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fixture_version": FIXTURE_VERSION, "entries": entries}), encoding="utf-8")


def test_replay_hits_cached_entry(tmp_path):
    fixture_path = tmp_path / "tavily_v1.json"
    key = _query_key("what is vLLM?", max_results=5, topic="general")
    _write_fixture(fixture_path, {key: {"query": "what is vLLM?", "response": {"results": [{"content": "vLLM is..."}]}}})

    tool = ReplayTavilyTool(max_results=5, topic="general", fixture_path=fixture_path)
    result = tool.invoke({"query": "what is vLLM?"})
    assert result == {"results": [{"content": "vLLM is..."}]}


def test_replay_miss_raises_loudly_not_silently(tmp_path):
    fixture_path = tmp_path / "tavily_v1.json"
    _write_fixture(fixture_path, {})

    tool = ReplayTavilyTool(max_results=5, topic="general", fixture_path=fixture_path)
    with pytest.raises(TavilyCacheMissError):
        tool.invoke({"query": "never recorded query"})


def test_replay_key_includes_max_results_and_topic():
    key_a = _query_key("q", max_results=5, topic="general")
    key_b = _query_key("q", max_results=10, topic="general")
    key_c = _query_key("q", max_results=5, topic="news")
    assert len({key_a, key_b, key_c}) == 3


def test_build_tavily_tool_defaults_to_replay(monkeypatch, tmp_path):
    monkeypatch.delenv("TAVILY_MODE", raising=False)
    fixture_path = tmp_path / "tavily_v1.json"
    _write_fixture(fixture_path, {})
    tool = build_tavily_tool({"tavilly_api_key": "unused"}, fixture_path=fixture_path)
    assert isinstance(tool, ReplayTavilyTool)


def test_build_tavily_tool_rejects_unknown_mode(tmp_path):
    fixture_path = tmp_path / "tavily_v1.json"
    with pytest.raises(ValueError):
        build_tavily_tool({"tavilly_api_key": "unused"}, mode="bogus", fixture_path=fixture_path)


def test_recording_tool_writes_fixture_replay_can_then_read(monkeypatch, tmp_path):
    """RecordingTavilyTool.invoke() should write an entry that
    ReplayTavilyTool can immediately look up by the same key -- the whole
    point of keying both by the same (query, max_results, topic) hash."""
    fixture_path = tmp_path / "tavily_v1.json"

    class _FakeLiveTool:
        def invoke(self, inputs):
            return {"results": [{"content": f"live answer for {inputs['query']}"}]}

    from eval import tavily_cache

    monkeypatch.setattr(tavily_cache, "RecordingTavilyTool", tavily_cache.RecordingTavilyTool)
    recorder = tavily_cache.RecordingTavilyTool.__new__(tavily_cache.RecordingTavilyTool)
    recorder.max_results = 5
    recorder.topic = "general"
    recorder.fixture_path = fixture_path
    recorder._fixture = {"fixture_version": FIXTURE_VERSION, "entries": {}}
    recorder._live_tool = _FakeLiveTool()

    response = recorder.invoke({"query": "latest vLLM release"})
    assert response == {"results": [{"content": "live answer for latest vLLM release"}]}
    assert fixture_path.exists()

    replay_tool = ReplayTavilyTool(max_results=5, topic="general", fixture_path=fixture_path)
    assert replay_tool.invoke({"query": "latest vLLM release"}) == response


def test_default_fixture_path_matches_fixture_version():
    assert DEFAULT_FIXTURE_PATH.name == f"tavily_{FIXTURE_VERSION}.json"


def test_record_from_harness_run_exercises_every_item_no_judge(monkeypatch, tmp_path):
    """No judge involved -- discovering web_search queries only needs the
    graph to run, not grading. Verifies the wiring (config -> items ->
    graph -> run_query_with_state per item) without any live calls."""
    import eval.harness as harness_module
    import src.agent as agent_module
    import src.runtime as runtime_module

    fake_items = [{"id": "gs_0001", "question": "q1"}, {"id": "gs_0002", "question": "q2"}]
    monkeypatch.setattr(harness_module, "load_harness_config", lambda path: ({}, {"chunk_size": 1024}))
    monkeypatch.setattr(harness_module, "_load_items", lambda split: fake_items)

    fake_runtime = SimpleNamespace(database=None, embedding_model=None, rerank_model=None, llm=None)
    monkeypatch.setattr(runtime_module, "get_runtime", lambda config: fake_runtime)
    monkeypatch.setattr(tavily_cache_module, "build_tavily_tool", lambda *a, **k: "fake_tavily_tool")
    monkeypatch.setattr(agent_module, "build_agent_graph", lambda *a, **k: "fake_graph")

    calls = []
    monkeypatch.setattr(
        agent_module, "run_query_with_state", lambda graph, question, history: calls.append(question) or ("ans", {}, {})
    )

    n = record_from_harness_run(config_path=tmp_path / "config.yaml", split="fast")

    assert n == 2
    assert calls == ["q1", "q2"]
