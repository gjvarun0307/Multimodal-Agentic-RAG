"""LLM-as-judge for generation-quality grading (PROJECT_SPEC.md §7 Phase 2,
invariant 11: "Judge is pinned (Groq) and independent of the generation
model; prompt changes force a re-baseline.").

The judge is a SEPARATE LLM client from whatever the run under test uses
for generation -- even when the generation backend also happens to be
Groq, the judge's API key is resolved independently (`judge_api_key`, never
`llm_api_key`) so a judge-vs-generation self-preference bias (documented in
the LLM-judge literature) can never creep in through a shared client or
shared credentials. Temperature is pinned to 0 for every judge call so
grading itself isn't a source of run-to-run noise ahead of Phase 3's
noise-floor measurement.

JUDGE_VERSION is embedded in every prompt's system message and recorded in
every results JSON (invariant 7). Any prompt wording change invalidates
comparability with already-committed results -- bump this constant
whenever a prompt changes, in the same commit, never separately.

This module only grades ONE item at a time and returns typed Pydantic
judgments -- eval/metrics/generation.py aggregates those into rates
(faithfulness, answer_correctness, refusal_accuracy, citation_precision)
across a run; judge.py has no concept of a "run", only of "grade this one
(question, context, generation[, gold_answer])".
"""

from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.configuration import build_llm_client
from src.helper import is_rate_limit_error

JUDGE_VERSION = "v2"

# Kept in sync with configs/default.yaml's judge_model -- update both
# together (module docstring: a prompt or model change forces a
# re-baseline either way, so there's no cost to keeping them the same edit).
#
# v2 (2026-08-19): llama-3.3-70b-versatile (v1's judge) was deprecated by
# Groq for free/developer-tier usage and now 404s (model_not_found) on
# every call -- confirmed live, mid-run, in a real full-eval.yml attempt.
# Replaced with openai/gpt-oss-120b, live-calibrated against all four
# grading functions (faithfulness/correctness/refusal/citation_precision,
# method="function_calling") on real golden-set content the same day: 6/6
# calls returned clean structured output with substantively correct
# judgments (including correctly flagging a deliberately-flawed answer as
# incorrect) -- no output_parse_failed, no chain-of-thought leakage. This
# is NOT the same failure the smaller openai/gpt-oss-20b hit during the
# original 2026-08-13 calibration (a hard 400 output_parse_failed from
# Groq, chain-of-thought monologue instead of a clean tool call) -- 120b
# was verified separately, not assumed to inherit the fix by family
# resemblance. Separately (also confirmed 2026-08-13): Groq's `json_schema`
# structured-output response format is ONLY supported by the gpt-oss
# family -- every other model 400s outright if asked for it. That's why
# every with_structured_output() call in this module passes
# method="function_calling" explicitly instead of relying on the
# provider's default method -- tool/function calling is supported by
# effectively every model Groq hosts (confirmed via Groq's own tool-use
# docs), unlike the narrower json_schema mode.
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_PROVIDER = "groq"  # fixed by invariant 11, never config-selectable


class JudgeConfigError(Exception):
    """Raised when the judge can't be constructed -- e.g. no judge_api_key
    resolved. Never falls back to the generation model's key (that would
    silently violate invariant 11's independence requirement)."""


class JudgeGradingError(Exception):
    """Raised when a single grading call fails -- wraps whatever the
    underlying provider raised (e.g. openai.BadRequestError's
    output_parse_failed, seen live from Groq when a reasoning-style model
    emitted chain-of-thought text instead of a clean tool call) into one
    exception type so callers grading many items don't need to know every
    possible provider-specific exception class to catch and skip a single
    failed row loudly (invariant 15) rather than letting it crash an
    entire run."""


class JudgeRateLimitError(JudgeGradingError):
    """A JudgeGradingError whose cause is specifically a Groq/provider
    quota 429 (see src.helper.is_rate_limit_error), not a real grading
    failure -- callers (eval.harness) count these separately so
    eval.gate can report a quota blip distinctly from a real regression
    (CLAUDE.md Phase 4 checklist) instead of both looking like the same
    generic skip."""


def _invoke_judge_chain(chain, inputs: dict, *, dimension: str):
    try:
        return chain.invoke(inputs)
    except Exception as e:
        error_cls = JudgeRateLimitError if is_rate_limit_error(e) else JudgeGradingError
        raise error_cls(f"Judge grading failed for dimension={dimension!r}: {e}") from e


class FaithfulnessJudgment(BaseModel):
    reasoning: str = Field(..., description="Compare each concrete claim in the answer against the provided context.")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Concrete claims in the answer not supported by the context; empty if none."
    )
    is_faithful: bool = Field(
        ..., description="True only if every concrete claim in the answer is supported by the context."
    )


class CorrectnessJudgment(BaseModel):
    reasoning: str = Field(..., description="Compare the answer's substantive content against the gold answer.")
    is_correct: bool = Field(
        ...,
        description="True if the answer conveys the same substantive information as the gold answer, even if phrased differently.",
    )


class RefusalJudgment(BaseModel):
    reasoning: str = Field(
        ..., description="Determine whether the answer declines to answer or attempts a substantive answer."
    )
    is_refusal: bool = Field(
        ...,
        description="True if the answer declines to answer or states the information isn't in the provided context, in any phrasing. False if it attempts a substantive answer.",
    )


class CitationJudgment(BaseModel):
    reasoning: str = Field(
        ..., description="Identify every source/citation cue in the answer and check it against the context."
    )
    cited_claims_total: int = Field(
        ..., ge=0, description="Number of distinct source citations/attributions made in the answer."
    )
    cited_claims_supported: int = Field(
        ..., ge=0, description="Of those, the number actually backed by the provided context."
    )


def build_judge_llm(config: dict, *, judge_model: Optional[str] = None):
    """Constructs the judge's own Groq client, independent of whatever
    llm_provider/llm_model/llm_api_key the generation model under test is
    configured with. `config` should come from src.configuration.config_rag()
    -- reads config["judge_api_key"] only, never config["llm_api_key"]."""
    judge_api_key = config.get("judge_api_key")
    if not judge_api_key:
        raise JudgeConfigError(
            "No judge_api_key resolved (set it in api_keys.json or the JUDGE_API_KEY env var). "
            "The judge must use its own Groq key, independent of llm_api_key (invariant 11) -- "
            "there is no fallback to the generation model's key."
        )
    judge_config = {
        "llm_provider": JUDGE_PROVIDER,
        "llm_model": judge_model or config.get("judge_model") or DEFAULT_JUDGE_MODEL,
        "llm_api_key": judge_api_key,
    }
    return build_llm_client(judge_config).bind(temperature=0)


def grade_faithfulness(judge_llm, *, context: str, generation: str) -> FaithfulnessJudgment:
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                f"[judge_version={JUDGE_VERSION}] You are a strict factual-grounding judge for a RAG "
                "system evaluation. Mark is_faithful=True only if every concrete claim in the answer "
                "is supported by the supplied context. Any unsupported claim (a fact, number, or "
                "citation not present in the context) makes is_faithful=False, even if the claim "
                "happens to be true in the real world.",
            ),
            ("human", "Context:\n{context}\n\nAnswer to evaluate:\n{generation}"),
        ],
        input_variables=["context", "generation"],
    )
    chain = prompt | judge_llm.with_structured_output(FaithfulnessJudgment, method="function_calling")
    return _invoke_judge_chain(chain, {"context": context, "generation": generation}, dimension="faithfulness")


def grade_correctness(judge_llm, *, question: str, gold_answer: str, generation: str) -> CorrectnessJudgment:
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                f"[judge_version={JUDGE_VERSION}] You are a strict answer-correctness judge for a RAG "
                "system evaluation. Compare the candidate answer against the gold answer for the same "
                "question. Mark is_correct=True if the candidate conveys the same substantive "
                "information as the gold answer, even if phrased very differently, more concisely, or "
                "with extra (non-contradictory) detail. Mark is_correct=False if it omits the core "
                "answer, contradicts the gold answer, or answers a different question.",
            ),
            ("human", "Question:\n{question}\n\nGold answer:\n{gold_answer}\n\nCandidate answer:\n{generation}"),
        ],
        input_variables=["question", "gold_answer", "generation"],
    )
    chain = prompt | judge_llm.with_structured_output(CorrectnessJudgment, method="function_calling")
    return _invoke_judge_chain(
        chain, {"question": question, "gold_answer": gold_answer, "generation": generation}, dimension="correctness"
    )


def grade_refusal(judge_llm, *, question: str, generation: str) -> RefusalJudgment:
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                f"[judge_version={JUDGE_VERSION}] You are grading whether a RAG system correctly "
                "declined to answer a question it shouldn't be able to answer from its provided "
                "context. Mark is_refusal=True for any phrasing that declines to answer or states the "
                "information isn't available in the provided documents/context -- it does not need to "
                "match any exact wording. Mark is_refusal=False if the answer attempts a substantive "
                "response to the question, even a partial or hedged one.",
            ),
            ("human", "Question:\n{question}\n\nAnswer:\n{generation}"),
        ],
        input_variables=["question", "generation"],
    )
    chain = prompt | judge_llm.with_structured_output(RefusalJudgment, method="function_calling")
    return _invoke_judge_chain(chain, {"question": question, "generation": generation}, dimension="refusal")


def grade_citation_precision(judge_llm, *, context: str, generation: str) -> CitationJudgment:
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                f"[judge_version={JUDGE_VERSION}] You are grading citation precision for a RAG system "
                "answer. Identify every place the answer attributes a fact to a source (a paper name, "
                "author, or explicit source cue) and check whether that attribution is actually "
                "supported by the provided context. cited_claims_total is the number of such "
                "attributions found; cited_claims_supported is how many are correctly backed by the "
                "context. If the answer makes no citations at all, return 0 for both.",
            ),
            ("human", "Context:\n{context}\n\nAnswer:\n{generation}"),
        ],
        input_variables=["context", "generation"],
    )
    chain = prompt | judge_llm.with_structured_output(CitationJudgment, method="function_calling")
    return _invoke_judge_chain(chain, {"context": context, "generation": generation}, dimension="citation_precision")


def citation_precision_score(judgment: CitationJudgment) -> Optional[float]:
    """None (not 0.0 or 1.0) when the answer made no citations at all --
    undefined, not a failure, and must not be averaged in as either."""
    if judgment.cited_claims_total == 0:
        return None
    return judgment.cited_claims_supported / judgment.cited_claims_total
