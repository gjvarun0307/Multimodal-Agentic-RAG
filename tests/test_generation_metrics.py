"""Unit tests for eval/metrics/generation.py's judge-output aggregation.
Pure functions over already-graded GenerationItem records -- no judge
calls, no LLM.
"""

from eval.metrics.generation import GenerationItem, compute_generation_metrics, correctness_delta


def test_faithfulness_rate_excludes_ungraded_items():
    items = [
        GenerationItem(id="1", is_faithful=True),
        GenerationItem(id="2", is_faithful=False),
        GenerationItem(id="3", is_faithful=None),  # not scored, e.g. no documents retrieved
    ]
    result = compute_generation_metrics(items)
    assert result["faithfulness"]["n_scored"] == 2
    assert result["faithfulness"]["rate"] == 0.5


def test_answer_correctness_rate():
    items = [
        GenerationItem(id="1", is_correct=True),
        GenerationItem(id="2", is_correct=True),
        GenerationItem(id="3", is_correct=False),
    ]
    result = compute_generation_metrics(items)
    assert result["answer_correctness"]["n_scored"] == 3
    assert abs(result["answer_correctness"]["rate"] - 2 / 3) < 1e-9


def test_refusal_accuracy_scoped_to_expected_refusal_items():
    items = [
        GenerationItem(id="1", expected_refusal=True, is_refusal=True),  # correct refusal
        GenerationItem(id="2", expected_refusal=True, is_refusal=False),  # should have refused, didn't
        GenerationItem(id="3", expected_refusal=False, is_refusal=True),  # false refusal on answerable item -- excluded
        GenerationItem(id="4", expected_refusal=None, is_refusal=None),  # unrelated item -- excluded
    ]
    result = compute_generation_metrics(items)
    assert result["refusal_accuracy"]["n_scored"] == 2
    assert result["refusal_accuracy"]["rate"] == 0.5


def test_citation_precision_excludes_no_citation_items():
    items = [
        GenerationItem(id="1", citation_precision=1.0),
        GenerationItem(id="2", citation_precision=0.5),
        GenerationItem(id="3", citation_precision=None),  # answer made no citations
    ]
    result = compute_generation_metrics(items)
    assert result["citation_precision"]["n_scored"] == 2
    assert result["citation_precision"]["mean"] == 0.75


def test_compute_generation_metrics_all_none_when_nothing_scored():
    items = [GenerationItem(id="1")]
    result = compute_generation_metrics(items)
    assert result["faithfulness"]["rate"] is None
    assert result["answer_correctness"]["rate"] is None
    assert result["refusal_accuracy"]["rate"] is None
    assert result["citation_precision"]["mean"] is None


def test_correctness_delta_improve_degrade_unchanged():
    assert correctness_delta(False, True) == 1.0
    assert correctness_delta(True, False) == -1.0
    assert correctness_delta(True, True) == 0.0
    assert correctness_delta(False, False) == 0.0


def test_correctness_delta_none_when_ungraded():
    assert correctness_delta(None, True) is None
    assert correctness_delta(True, None) is None
