"""Tests for eval/judge.py. No real LLM calls -- build_llm_client is
monkeypatched, and grading functions are exercised against a fake judge_llm
whose .with_structured_output(Schema) returns a canned-response runnable,
matching this repo's existing reranker/config test conventions (fast,
network-free).
"""

import httpx
import openai
import pytest

from eval.judge import (
    CitationJudgment,
    JudgeConfigError,
    JudgeGradingError,
    JudgeRateLimitError,
    build_judge_llm,
    citation_precision_score,
    grade_citation_precision,
    grade_correctness,
    grade_faithfulness,
    grade_refusal,
)


class _FakeStructuredChain:
    """Needs __call__ (not just .invoke()) so LangChain's `prompt | chain`
    composition can coerce it into a RunnableLambda -- see
    eval/metrics/structured.py's InstrumentedStructuredRunnable for the
    same pattern, verified against real ChatPromptTemplate composition."""

    def __init__(self, response):
        self._response = response
        self.invoked_with = None

    def __call__(self, inputs):
        return self.invoke(inputs)

    def invoke(self, inputs):
        self.invoked_with = inputs
        return self._response


class _FakeJudgeLLM:
    def __init__(self, response):
        self._response = response
        self.last_schema = None
        self.last_method = None

    def with_structured_output(self, schema, method=None):
        self.last_schema = schema
        self.last_method = method
        return _FakeStructuredChain(self._response)


class _RaisingStructuredChain:
    """Simulates the real failure hit live during judge calibration: the
    provider raises mid-invoke (Groq's output_parse_failed 400) instead of
    returning a parsed object."""

    def __call__(self, inputs):
        return self.invoke(inputs)

    def invoke(self, inputs):
        raise RuntimeError("Parsing failed. The model generated output that could not be parsed.")


class _RaisingJudgeLLM:
    def with_structured_output(self, schema, method=None):
        return _RaisingStructuredChain()


class _RateLimitedStructuredChain:
    """Simulates a real Groq 429 -- openai.RateLimitError is what every
    provider call raises in this project (all routed through
    langchain_openai.ChatOpenAI, see src.helper.is_rate_limit_error)."""

    def __call__(self, inputs):
        return self.invoke(inputs)

    def invoke(self, inputs):
        resp = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/x"))
        raise openai.RateLimitError("rate limit exceeded", response=resp, body=None)


class _RateLimitedJudgeLLM:
    def with_structured_output(self, schema, method=None):
        return _RateLimitedStructuredChain()


def test_build_judge_llm_raises_without_api_key():
    with pytest.raises(JudgeConfigError):
        build_judge_llm({"judge_api_key": ""})


def test_grade_faithfulness_wraps_provider_failure_in_judge_grading_error():
    with pytest.raises(JudgeGradingError):
        grade_faithfulness(_RaisingJudgeLLM(), context="ctx", generation="gen")


def test_grade_faithfulness_raises_judge_rate_limit_error_on_429():
    # JudgeRateLimitError is a JudgeGradingError subclass, so callers that
    # only catch JudgeGradingError keep working -- eval.harness catches it
    # specifically to count it separately (CLAUDE.md Phase 4 checklist:
    # rate-limit errors distinguishable from regressions).
    with pytest.raises(JudgeRateLimitError):
        grade_faithfulness(_RateLimitedJudgeLLM(), context="ctx", generation="gen")


def test_judge_rate_limit_error_is_a_judge_grading_error():
    assert issubclass(JudgeRateLimitError, JudgeGradingError)


def test_build_judge_llm_never_falls_back_to_generation_key(monkeypatch):
    captured = {}

    def fake_build_llm_client(cfg):
        captured.update(cfg)

        class _Client:
            def bind(self, **kwargs):
                captured["bind_kwargs"] = kwargs
                return self

        return _Client()

    monkeypatch.setattr("eval.judge.build_llm_client", fake_build_llm_client)

    build_judge_llm({"judge_api_key": "judge-key", "llm_api_key": "generation-key", "llm_provider": "anthropic"})

    assert captured["llm_api_key"] == "judge-key"
    assert captured["llm_provider"] == "groq"
    assert captured["bind_kwargs"] == {"temperature": 0}


def test_build_judge_llm_uses_configured_judge_model(monkeypatch):
    captured = {}

    def fake_build_llm_client(cfg):
        captured.update(cfg)

        class _Client:
            def bind(self, **kwargs):
                return self

        return _Client()

    monkeypatch.setattr("eval.judge.build_llm_client", fake_build_llm_client)
    build_judge_llm({"judge_api_key": "k", "judge_model": "custom/model"})
    assert captured["llm_model"] == "custom/model"


def test_grade_faithfulness_wires_context_and_generation():
    from eval.judge import FaithfulnessJudgment

    expected = FaithfulnessJudgment(reasoning="ok", unsupported_claims=[], is_faithful=True)
    fake_llm = _FakeJudgeLLM(expected)
    result = grade_faithfulness(fake_llm, context="the sky is blue", generation="the sky is blue")
    assert result is expected


def test_grade_faithfulness_uses_function_calling_method():
    # Regression: Groq's default with_structured_output method (json_schema)
    # 400s outright on every model except the gpt-oss family (live-verified
    # 2026-08-13) -- every grade_* call must request method="function_calling"
    # explicitly, never rely on the provider default.
    from eval.judge import FaithfulnessJudgment

    expected = FaithfulnessJudgment(reasoning="ok", unsupported_claims=[], is_faithful=True)
    fake_llm = _FakeJudgeLLM(expected)
    grade_faithfulness(fake_llm, context="ctx", generation="gen")
    assert fake_llm.last_method == "function_calling"


def test_grade_correctness_wires_gold_answer():
    from eval.judge import CorrectnessJudgment

    expected = CorrectnessJudgment(reasoning="matches", is_correct=True)
    fake_llm = _FakeJudgeLLM(expected)
    result = grade_correctness(fake_llm, question="q", gold_answer="42", generation="the answer is 42")
    assert result is expected


def test_grade_refusal_wires_question_and_generation():
    from eval.judge import RefusalJudgment

    expected = RefusalJudgment(reasoning="declines", is_refusal=True)
    fake_llm = _FakeJudgeLLM(expected)
    result = grade_refusal(fake_llm, question="q", generation="I cannot answer this based on the provided context.")
    assert result is expected


def test_grade_citation_precision_returns_judgment():
    expected = CitationJudgment(reasoning="two cites, one bad", cited_claims_total=2, cited_claims_supported=1)
    fake_llm = _FakeJudgeLLM(expected)
    result = grade_citation_precision(fake_llm, context="ctx", generation="gen")
    assert result is expected


def test_citation_precision_score_undefined_when_no_citations():
    judgment = CitationJudgment(reasoning="none", cited_claims_total=0, cited_claims_supported=0)
    assert citation_precision_score(judgment) is None


def test_citation_precision_score_computes_ratio():
    judgment = CitationJudgment(reasoning="x", cited_claims_total=4, cited_claims_supported=3)
    assert citation_precision_score(judgment) == 0.75
