"""Generation-quality metrics: faithfulness, answer_correctness,
refusal_accuracy, citation_precision (PROJECT_SPEC.md §7 Phase 2). Pure
aggregation over already-graded per-item judge outputs -- eval/judge.py
makes the actual LLM calls (one per item per judgment type); this module
never calls the judge itself, so it stays fast/free/testable without
network access, the same separation eval/metrics/router.py and
eval/metrics/structured.py already use for their own upstream signal.

Item scoping (see eval/golden/SCHEMA.md):
  - faithfulness / citation_precision: only meaningful for items that
    actually retrieved context and generated a substantive answer --
    typically vectorstore-routed items with a non-empty `documents` list.
    The caller filters before calling compute_generation_metrics(); this
    module never guesses eligibility from a category string.
  - answer_correctness: only meaningful for items with a non-empty
    gold_answer (SCHEMA.md: gold_answer == "" iff
    category == unanswerable_refuse).
  - refusal_accuracy: only meaningful for unanswerable_refuse items
    (SCHEMA.md: expected_route == "refuse").

None entries anywhere below mean "not graded for this item" -- excluded
from the corresponding rate/mean, never counted as a failure (0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class GenerationItem:
    id: str
    is_faithful: Optional[bool] = None
    is_correct: Optional[bool] = None
    is_refusal: Optional[bool] = None
    expected_refusal: Optional[bool] = None  # True for unanswerable_refuse items
    citation_precision: Optional[float] = None  # from judge.citation_precision_score(); None if no citations made


def _rate(values: list[bool]) -> Optional[float]:
    return (sum(1 for v in values if v) / len(values)) if values else None


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def compute_generation_metrics(items: Sequence[GenerationItem]) -> dict:
    faithfulness_values = [it.is_faithful for it in items if it.is_faithful is not None]
    correctness_values = [it.is_correct for it in items if it.is_correct is not None]
    citation_values = [it.citation_precision for it in items if it.citation_precision is not None]

    # refusal_accuracy is scoped to items expected to refuse -- an item
    # that refused but wasn't supposed to (a false refusal on an answerable
    # question) is a different failure mode this metric doesn't cover.
    refusal_correctness = [it.is_refusal for it in items if it.expected_refusal is True and it.is_refusal is not None]

    return {
        "n_items": len(items),
        "faithfulness": {"n_scored": len(faithfulness_values), "rate": _rate(faithfulness_values)},
        "answer_correctness": {"n_scored": len(correctness_values), "rate": _rate(correctness_values)},
        "refusal_accuracy": {"n_scored": len(refusal_correctness), "rate": _rate(refusal_correctness)},
        "citation_precision": {"n_scored": len(citation_values), "mean": _mean(citation_values)},
    }


def correctness_delta(pre_correct: Optional[bool], post_correct: Optional[bool]) -> Optional[float]:
    """+1.0 if correction turned an incorrect answer correct, -1.0 if it
    turned a correct answer incorrect, 0.0 if unchanged, None if either
    side wasn't graded. The sole consumer is
    eval.metrics.system.compute_correction_metrics's correctness_deltas
    parameter -- not otherwise part of compute_generation_metrics.
    """
    if pre_correct is None or post_correct is None:
        return None
    if pre_correct == post_correct:
        return 0.0
    return 1.0 if post_correct else -1.0
