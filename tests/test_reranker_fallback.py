"""Regression tests for build_reranker()'s loud-fallback behavior
(PROJECT_SPEC.md invariant 15: silent fallbacks are forbidden -- a
disabled reranker must be loud, logged, and never silent).

Uses lightweight fake reranker classes instead of downloading real
models, so these run fast and without network access.
"""

import logging

from src.configuration import CPU_VIABLE_RERANKER_DEFAULT, LLM_STYLE_RERANKERS, build_reranker


class _FakeReranker:
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs


class _FakeLLMReranker(_FakeReranker):
    pass


class _FailingReranker:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("simulated load failure")


def test_llm_style_reranker_on_cpu_falls_back_loudly(monkeypatch, caplog):
    monkeypatch.setattr("FlagEmbedding.FlagReranker", _FakeReranker)
    monkeypatch.setattr("FlagEmbedding.FlagLLMReranker", _FakeLLMReranker)

    gemma_model = next(iter(LLM_STYLE_RERANKERS))
    config = {"reranker_model": gemma_model, "device": "cpu", "use_fp16": False}

    with caplog.at_level(logging.WARNING):
        result = build_reranker(config)

    assert isinstance(result, _FakeReranker)
    assert not isinstance(result, _FakeLLMReranker)
    assert result.model_name_or_path == CPU_VIABLE_RERANKER_DEFAULT
    assert any("Falling back to CPU-viable default" in msg for msg in caplog.messages)


def test_cpu_viable_reranker_used_directly_no_fallback(monkeypatch, caplog):
    monkeypatch.setattr("FlagEmbedding.FlagReranker", _FakeReranker)
    monkeypatch.setattr("FlagEmbedding.FlagLLMReranker", _FakeLLMReranker)

    config = {"reranker_model": CPU_VIABLE_RERANKER_DEFAULT, "device": "cpu", "use_fp16": False}

    with caplog.at_level(logging.WARNING):
        result = build_reranker(config)

    assert isinstance(result, _FakeReranker)
    assert result.model_name_or_path == CPU_VIABLE_RERANKER_DEFAULT
    assert not any("Falling back" in msg for msg in caplog.messages)


def test_llm_style_reranker_on_cuda_uses_llm_reranker_directly(monkeypatch):
    monkeypatch.setattr("FlagEmbedding.FlagReranker", _FakeReranker)
    monkeypatch.setattr("FlagEmbedding.FlagLLMReranker", _FakeLLMReranker)

    gemma_model = next(iter(LLM_STYLE_RERANKERS))
    config = {"reranker_model": gemma_model, "device": "cuda", "use_fp16": False}

    result = build_reranker(config)

    assert isinstance(result, _FakeLLMReranker)
    assert result.model_name_or_path == gemma_model


def test_reranker_load_failure_returns_none_and_logs_loudly(monkeypatch, caplog):
    monkeypatch.setattr("FlagEmbedding.FlagReranker", _FailingReranker)
    monkeypatch.setattr("FlagEmbedding.FlagLLMReranker", _FailingReranker)

    config = {"reranker_model": CPU_VIABLE_RERANKER_DEFAULT, "device": "cpu", "use_fp16": False}

    with caplog.at_level(logging.ERROR):
        result = build_reranker(config)

    assert result is None
    assert any("Failed to load reranker" in msg for msg in caplog.messages)
