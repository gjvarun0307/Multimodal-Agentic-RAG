"""Tests for eval/judge_calibration.py's pure logic -- sampling, CSV row
shaping, and Cohen's kappa scoring. No LLM calls; run_sample (which does
make real calls) is exercised manually via the CLI, not here.
"""

import csv

from eval.judge_calibration import (
    CSV_FIELDS,
    _parse_bool,
    _row,
    _truncate,
    cohens_kappa,
    sample_calibration_items,
    score_calibration_csv,
)


def _mk_items(counts: dict[str, int]) -> list[dict]:
    items = []
    i = 0
    for category, n in counts.items():
        for _ in range(n):
            items.append({"id": f"gs_{i:04d}", "category": category, "question": "q", "expected_route": "vectorstore"})
            i += 1
    return items


def test_sample_calibration_items_excludes_chitchat():
    items = _mk_items({"single_hop": 30, "chitchat": 4})
    sampled = sample_calibration_items(items, target_n=10)
    assert all(it["category"] != "chitchat" for it in sampled)


def test_sample_calibration_items_deterministic():
    items = _mk_items({"single_hop": 30, "multi_hop": 20})
    s1 = sample_calibration_items(items, target_n=10)
    s2 = sample_calibration_items(items, target_n=10)
    assert [it["id"] for it in s1] == [it["id"] for it in s2]


def test_truncate_short_text_unchanged():
    assert _truncate("short", limit=100) == "short"


def test_truncate_long_text_marks_truncation():
    text = "x" * 200
    result = _truncate(text, limit=50)
    assert result.startswith("x" * 50)
    assert "truncated" in result


def test_row_shape_matches_csv_fields():
    item = {"id": "gs_0001", "category": "single_hop", "question": "q?", "gold_answer": "a"}
    row = _row(item, "faithfulness", "ctx", "gen", True, "reasoning text")
    assert set(row.keys()) == set(CSV_FIELDS)
    assert row["judge_verdict"] == "TRUE"
    assert row["human_verdict"] == ""


def test_parse_bool_variants():
    assert _parse_bool("TRUE") is True
    assert _parse_bool("true") is True
    assert _parse_bool("Y") is True
    assert _parse_bool("FALSE") is False
    assert _parse_bool("n") is False
    assert _parse_bool("") is None
    assert _parse_bool("maybe") is None


def test_cohens_kappa_perfect_agreement():
    judge = [True, False, True, False]
    human = [True, False, True, False]
    assert cohens_kappa(judge, human) == 1.0


def test_cohens_kappa_systematic_disagreement_is_negative():
    # Balanced marginals (50/50 True on both sides) but every pair
    # disagrees -- worse than chance, kappa must go negative.
    judge = [True, True, False, False]
    human = [False, False, True, True]
    kappa = cohens_kappa(judge, human)
    assert kappa == -1.0


def test_cohens_kappa_empty_is_none():
    assert cohens_kappa([], []) is None


def test_cohens_kappa_all_same_label_both_sides_is_one():
    # Degenerate case: both judge and human say True every time -- pe=1.0,
    # would divide by zero without the guard.
    judge = [True, True, True]
    human = [True, True, True]
    assert cohens_kappa(judge, human) == 1.0


def test_score_calibration_csv_computes_per_dimension(tmp_path):
    csv_path = tmp_path / "calibration.csv"
    rows = [
        {
            "id": "gs_0001",
            "dimension": "faithfulness",
            "category": "single_hop",
            "question": "q",
            "gold_answer": "",
            "context": "ctx",
            "generation": "gen",
            "judge_verdict": "TRUE",
            "judge_reasoning": "r",
            "human_verdict": "TRUE",
            "human_notes": "",
        },
        {
            "id": "gs_0002",
            "dimension": "faithfulness",
            "category": "single_hop",
            "question": "q",
            "gold_answer": "",
            "context": "ctx",
            "generation": "gen",
            "judge_verdict": "TRUE",
            "judge_reasoning": "r",
            "human_verdict": "FALSE",
            "human_notes": "",
        },
        {
            "id": "gs_0003",
            "dimension": "refusal",
            "category": "unanswerable_refuse",
            "question": "q",
            "gold_answer": "",
            "context": "",
            "generation": "gen",
            "judge_verdict": "TRUE",
            "judge_reasoning": "r",
            "human_verdict": "",  # not yet labeled
            "human_notes": "",
        },
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = score_calibration_csv(csv_path)
    assert summary["n_rows"] == 3
    assert summary["n_unlabeled"] == 1
    assert summary["dimensions"]["faithfulness"]["n"] == 2
    assert summary["dimensions"]["faithfulness"]["raw_agreement"] == 0.5
    assert "refusal" not in summary["dimensions"]  # only unlabeled row for this dimension
