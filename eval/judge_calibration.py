"""Judge calibration (PROJECT_SPEC.md §7 Phase 2: "Calibrate: hand-label 40
items, report judge-human agreement (Cohen's kappa) in the README. A judge
with unknown agreement is decoration.").

Two steps, split into two subcommands because step 1 makes real LLM +
judge calls (cost, non-deterministic) and step 2 is pure arithmetic over
whatever a human filled in -- re-scoring after fixing a typo in the CSV
must never re-spend API budget.

Step 1 -- python -m eval.judge_calibration sample [--config configs/default.yaml] [--n 40] [--out eval/judge_calibration/calibration_v1.csv]
    Stratified-samples N items from golden_set.jsonl (chitchat excluded --
    it has no judged dimension; never dev_split.jsonl, same as every other
    golden-set consumer), runs each through the REAL graph to get a
    genuine generation, grades every applicable judge dimension
    (faithfulness / correctness / refusal -- NOT citation_precision, a
    continuous ratio that Cohen's kappa isn't suited to; spot-check that
    by eye instead), and writes one CSV row per (item, dimension) with an
    empty `human_verdict` column.

    This is a real, costed, multi-minute run: one full graph invocation
    per item (LLM generation + router + graders) plus one judge call per
    applicable dimension.

Step 2 -- python -m eval.judge_calibration score --csv <path>
    Reads the completed CSV, computes Cohen's kappa + raw agreement per
    dimension from judge_verdict vs. human_verdict, and prints a summary.
    Rows with a blank human_verdict are skipped with a warning, not
    silently treated as agreement or disagreement.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

from eval.harness import DEFAULT_CONFIG, _format_documents, load_harness_config, stratified_fast_subset
from eval.judge import (
    JUDGE_VERSION,
    JudgeGradingError,
    build_judge_llm,
    grade_correctness,
    grade_faithfulness,
    grade_refusal,
)
from src.agent import build_agent_graph, run_query_with_state
from src.helper import open_jsonl
from src.runtime import get_runtime

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = REPO_ROOT / "eval" / "golden" / "golden_set.jsonl"
DEFAULT_CALIBRATION_DIR = REPO_ROOT / "eval" / "judge_calibration"
DEFAULT_CALIBRATION_CSV = DEFAULT_CALIBRATION_DIR / f"calibration_{JUDGE_VERSION}.csv"

CALIBRATION_TARGET_N = 40
# Distinct from eval.harness.FAST_SPLIT_SEED so the calibration sample
# isn't identical to the fast-tier CI sample -- calibration should probe a
# separately drawn slice of the golden set, not double as the CI subset.
CALIBRATION_SEED = 20260813

CONTEXT_TRUNCATE_CHARS = 3000

CSV_FIELDS = [
    "id",
    "dimension",
    "category",
    "question",
    "gold_answer",
    "context",
    "generation",
    "judge_verdict",
    "judge_reasoning",
    "human_verdict",
    "human_notes",
]


def sample_calibration_items(
    golden_items: list[dict], *, target_n: int = CALIBRATION_TARGET_N, seed: int = CALIBRATION_SEED
) -> list[dict]:
    """chitchat items are excluded from the pool before sampling -- they
    have no judged dimension (no context to check faithfulness against, no
    gold_answer, not an unanswerable_refuse item), so including them would
    waste calibration slots on items with nothing to grade."""
    eligible = [it for it in golden_items if it.get("category") != "chitchat"]
    return stratified_fast_subset(eligible, target_n=target_n, seed=seed)


def _truncate(text: str, limit: int = CONTEXT_TRUNCATE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated, {len(text) - limit} more chars]"


def _row(item: dict, dimension: str, context: str, generation: str, judge_verdict: bool, judge_reasoning: str) -> dict:
    return {
        "id": item["id"],
        "dimension": dimension,
        "category": item.get("category", ""),
        "question": item["question"],
        "gold_answer": item.get("gold_answer", ""),
        "context": _truncate(context),
        "generation": generation,
        "judge_verdict": str(bool(judge_verdict)).upper(),
        "judge_reasoning": judge_reasoning,
        "human_verdict": "",
        "human_notes": "",
    }


def build_calibration_rows(item: dict, answer: str, final_state: dict, judge_llm) -> list[dict]:
    """A single dimension failing to grade (e.g. the judge model emitting
    unparseable output -- a real failure mode hit live during calibration,
    see eval/judge.py's JudgeGradingError) must not lose the whole item's
    other dimensions or crash the run for every remaining item. Failures
    are logged loudly to stderr and simply produce one fewer row, never
    silently skipped without a trace (invariant 15)."""
    rows: list[dict] = []
    documents = final_state.get("documents", [])
    context_text = _format_documents(documents) if documents else ""

    if documents:
        try:
            faithfulness = grade_faithfulness(judge_llm, context=context_text, generation=answer)
            rows.append(
                _row(item, "faithfulness", context_text, answer, faithfulness.is_faithful, faithfulness.reasoning)
            )
        except JudgeGradingError as e:
            print(f"  SKIPPED {item['id']}/faithfulness: {e}", file=sys.stderr)

    if item.get("gold_answer"):
        try:
            correctness = grade_correctness(
                judge_llm, question=item["question"], gold_answer=item["gold_answer"], generation=answer
            )
            rows.append(_row(item, "correctness", context_text, answer, correctness.is_correct, correctness.reasoning))
        except JudgeGradingError as e:
            print(f"  SKIPPED {item['id']}/correctness: {e}", file=sys.stderr)

    if item.get("expected_route") == "refuse":
        try:
            refusal = grade_refusal(judge_llm, question=item["question"], generation=answer)
            rows.append(_row(item, "refusal", context_text, answer, refusal.is_refusal, refusal.reasoning))
        except JudgeGradingError as e:
            print(f"  SKIPPED {item['id']}/refusal: {e}", file=sys.stderr)

    return rows


def run_sample(*, config_path: Path, n: int, out_path: Path) -> Path:
    _doc, resolved_config = load_harness_config(config_path)
    golden_items = open_jsonl(DEFAULT_GOLDEN)
    sampled = sample_calibration_items(golden_items, target_n=n)

    runtime = get_runtime(resolved_config)
    graph = build_agent_graph(
        runtime.database, runtime.embedding_model, runtime.rerank_model, runtime.llm, resolved_config
    )
    judge_llm = build_judge_llm(resolved_config)

    rows: list[dict] = []
    for i, item in enumerate(sampled, start=1):
        print(f"[{i}/{len(sampled)}] {item['id']}: {item['question'][:80]!r}", file=sys.stderr)
        answer, final_state, _trace_info = run_query_with_state(graph, item["question"], [])
        rows.extend(build_calibration_rows(item, answer, final_state, judge_llm))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} calibration rows ({len(sampled)} items) to {out_path}", file=sys.stderr)
    return out_path


def _parse_bool(value: str) -> Optional[bool]:
    v = value.strip().upper()
    if v in ("TRUE", "T", "YES", "Y", "1"):
        return True
    if v in ("FALSE", "F", "NO", "N", "0"):
        return False
    return None


def cohens_kappa(judge_labels: list[bool], human_labels: list[bool]) -> Optional[float]:
    """None if there's nothing to score. When observed agreement is
    perfect and expected-by-chance agreement is also 1.0 (every label
    identical on both sides), kappa's denominator is 0 -- treated as 1.0
    (perfect agreement), not raised as a division error."""
    n = len(judge_labels)
    if n == 0:
        return None
    po = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n
    p_judge_true = sum(judge_labels) / n
    p_human_true = sum(human_labels) / n
    pe = p_judge_true * p_human_true + (1 - p_judge_true) * (1 - p_human_true)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def score_calibration_csv(csv_path: Path) -> dict:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_dimension: dict[str, list[dict]] = {}
    n_unlabeled = 0
    for row in rows:
        human = _parse_bool(row.get("human_verdict", ""))
        if human is None:
            n_unlabeled += 1
            continue
        by_dimension.setdefault(row["dimension"], []).append(
            {"judge": _parse_bool(row["judge_verdict"]), "human": human}
        )

    summary: dict = {"n_rows": len(rows), "n_unlabeled": n_unlabeled, "dimensions": {}}
    for dimension, pairs in sorted(by_dimension.items()):
        judge_labels = [p["judge"] for p in pairs]
        human_labels = [p["human"] for p in pairs]
        n = len(pairs)
        raw_agreement = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n if n else None
        summary["dimensions"][dimension] = {
            "n": n,
            "raw_agreement": raw_agreement,
            "cohens_kappa": cohens_kappa(judge_labels, human_labels),
        }
    return summary


def run_score(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"{csv_path} does not exist -- run `python -m eval.judge_calibration sample` first.", file=sys.stderr)
        sys.exit(1)
    summary = score_calibration_csv(csv_path)
    if summary["n_unlabeled"]:
        print(
            f"WARNING: {summary['n_unlabeled']} row(s) have no human_verdict filled in -- excluded from scoring.",
            file=sys.stderr,
        )
    print(f"judge_version: {JUDGE_VERSION}")
    print(f"n_rows: {summary['n_rows']}")
    for dimension, stats in summary["dimensions"].items():
        kappa = stats["cohens_kappa"]
        agreement = stats["raw_agreement"]
        # kappa/agreement are only None when n==0, which can't happen here --
        # a dimension only appears in summary["dimensions"] when it has at
        # least one labeled row (see score_calibration_csv).
        print(f"  {dimension}: n={stats['n']}, raw_agreement={agreement:.3f}, cohens_kappa={kappa:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="Generate the calibration CSV (real LLM + judge calls).")
    sample_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sample_parser.add_argument("--n", type=int, default=CALIBRATION_TARGET_N)
    sample_parser.add_argument("--out", type=Path, default=DEFAULT_CALIBRATION_CSV)

    score_parser = subparsers.add_parser("score", help="Compute Cohen's kappa from a completed calibration CSV.")
    score_parser.add_argument("--csv", type=Path, default=DEFAULT_CALIBRATION_CSV)

    args = parser.parse_args()
    if args.command == "sample":
        run_sample(config_path=args.config, n=args.n, out_path=args.out)
    else:
        run_score(args.csv)


if __name__ == "__main__":
    main()
