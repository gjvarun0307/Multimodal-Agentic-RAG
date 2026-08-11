"""Validate eval/golden/golden_set.jsonl (+ dev_split.jsonl) against the schema
and acceptance criteria in eval/golden/SCHEMA.md / PROJECT_SPEC.md SS7 Phase 1.

Composition/coverage targets apply to golden_set.jsonl UNION dev_split.jsonl
(dev items are carved FROM the ~180-item target, not additional to it). Paper
coverage is additionally checked on golden_set.jsonl alone as a warning, since
that's the set later phases actually gate CI on.

Usage:
    python -m eval.validate_golden
    python -m eval.validate_golden --skip-resolution        # fast path while drafting
    python -m eval.validate_golden --require-verified        # freeze gate (Checkpoint 4 only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from eval.resolve_passages import PassageResolutionError, resolve_golden_set
from src.helper import open_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = REPO_ROOT / "eval" / "golden" / "golden_set.jsonl"
DEFAULT_DEV = REPO_ROOT / "eval" / "golden" / "dev_split.jsonl"
DEFAULT_METADATA = REPO_ROOT / "artifacts" / "metadata.jsonl"
DEFAULT_PARSED_MD = REPO_ROOT / "artifacts" / "parsed_md"

CATEGORIES = (
    "single_hop",
    "multi_hop",
    "table_figure",
    "unanswerable_refuse",
    "ambiguous",
    "adversarial",
    "unanswerable_websearch",
    "chitchat",
)
ROUTES = ("vectorstore", "websearch", "chitchat", "refuse")
CATEGORY_TO_ROUTE = {
    "single_hop": "vectorstore",
    "multi_hop": "vectorstore",
    "table_figure": "vectorstore",
    "ambiguous": "vectorstore",
    "adversarial": "vectorstore",
    "unanswerable_refuse": "refuse",
    "unanswerable_websearch": "websearch",
    "chitchat": "chitchat",
}
# PROJECT_SPEC.md SS7 composition target shares (n ~= 180)
TARGET_SHARE_PCT = {
    "single_hop": 30.0,
    "multi_hop": 19.0,
    "table_figure": 15.0,
    "unanswerable_refuse": 17.0,
    "ambiguous": 10.0,
    "adversarial": 5.0,
    "unanswerable_websearch": 2.0,
    "chitchat": 2.0,
}
COMPOSITION_TOLERANCE_PP = 3.0
REQUIRED_FIELDS = (
    "id",
    "question",
    "gold_answer",
    "gold_passages",
    "gold_doc_ids",
    "category",
    "expected_route",
    "difficulty",
    "requires_multimodal",
    "verified_by",
    "version",
)
DIFFICULTIES = ("easy", "medium", "hard")
VERIFIED_BY_VALUES = ("draft", "human")
CHUNK_SIZES = (512, 1024, 2048)


class Check:
    def __init__(self, name: str):
        self.name = name
        self.status = "PASS"  # PASS | FAIL | WARN | SKIP
        self.detail = ""
        self.errors: list[str] = []

    def fail(self, detail: str, errors: Optional[list[str]] = None):
        self.status = "FAIL"
        self.detail = detail
        self.errors = errors or []

    def warn(self, detail: str, errors: Optional[list[str]] = None):
        if self.status != "FAIL":
            self.status = "WARN"
        self.detail = detail
        self.errors = errors or []

    def skip(self, detail: str):
        self.status = "SKIP"
        self.detail = detail

    def ok(self, detail: str):
        self.detail = detail


def _load(path: Path) -> list[dict]:
    return open_jsonl(path) if path.exists() else []


def check_schema(items: list[dict]) -> Check:
    c = Check("schema")
    bad = []
    for item in items:
        missing = [f for f in REQUIRED_FIELDS if f not in item]
        if missing:
            bad.append(f"{item.get('id', '<no id>')}: missing fields {missing}")
            continue
        if not isinstance(item["question"], str) or not item["question"].strip():
            bad.append(f"{item['id']}: question must be non-empty str")
        if not isinstance(item["gold_answer"], str):
            bad.append(f"{item['id']}: gold_answer must be str")
        if not isinstance(item["gold_passages"], list):
            bad.append(f"{item['id']}: gold_passages must be list")
        elif any(
            not isinstance(p, dict) or "doc_id" not in p or "passage_text" not in p for p in item["gold_passages"]
        ):
            bad.append(f"{item['id']}: gold_passages entries must be {{doc_id, passage_text}}")
        if not isinstance(item["gold_doc_ids"], list):
            bad.append(f"{item['id']}: gold_doc_ids must be list")
        if item["category"] not in CATEGORIES:
            bad.append(f"{item['id']}: category {item['category']!r} not in {CATEGORIES}")
        if item["expected_route"] not in ROUTES:
            bad.append(f"{item['id']}: expected_route {item['expected_route']!r} not in {ROUTES}")
        if item["difficulty"] not in DIFFICULTIES:
            bad.append(f"{item['id']}: difficulty {item['difficulty']!r} not in {DIFFICULTIES}")
        if not isinstance(item["requires_multimodal"], bool):
            bad.append(f"{item['id']}: requires_multimodal must be bool")
        if item["verified_by"] not in VERIFIED_BY_VALUES:
            bad.append(f"{item['id']}: verified_by {item['verified_by']!r} not in {VERIFIED_BY_VALUES}")
        if not isinstance(item["version"], int):
            bad.append(f"{item['id']}: version must be int")
    if bad:
        c.fail(f"{len(bad)}/{len(items)} items failed schema checks", bad)
    else:
        c.ok(f"{len(items)}/{len(items)} well-formed")
    return c


def check_id_format_and_uniqueness(golden: list[dict], dev: list[dict]) -> Check:
    c = Check("id format + uniqueness")
    import re

    pattern = re.compile(r"^gs_\d{4}$")
    all_items = golden + dev
    bad_format = [item["id"] for item in all_items if "id" in item and not pattern.match(item["id"])]
    ids = [item["id"] for item in all_items if "id" in item]
    seen = set()
    dupes = set()
    for i in ids:
        (dupes if i in seen else seen).add(i)
    errors = []
    if bad_format:
        errors.append(f"bad format: {bad_format}")
    if dupes:
        errors.append(f"duplicates: {sorted(dupes)}")
    if errors:
        c.fail(f"{len(bad_format)} bad format, {len(dupes)} duplicates", errors)
    else:
        c.ok(f"{len(ids)} unique, 0 collisions")
    return c


def check_dev_disjoint(golden: list[dict], dev: list[dict]) -> Check:
    c = Check("dev/main disjointness")
    golden_ids = {item["id"] for item in golden if "id" in item}
    dev_ids = {item["id"] for item in dev if "id" in item}
    overlap = golden_ids & dev_ids
    if overlap:
        c.fail(f"{len(overlap)} overlapping ids", sorted(overlap))
    else:
        c.ok(f"0 overlapping ids ({len(golden_ids)} main, {len(dev_ids)} dev)")
    return c


def check_doc_id_validity(items: list[dict], known_doc_ids: set[str]) -> Check:
    c = Check("doc_id validity")
    bad = []
    for item in items:
        for p in item.get("gold_passages", []):
            if p.get("doc_id") not in known_doc_ids:
                bad.append(f"{item['id']}: unknown doc_id {p.get('doc_id')!r} in gold_passages")
        for doc_id in item.get("gold_doc_ids", []):
            if doc_id not in known_doc_ids:
                bad.append(f"{item['id']}: unknown doc_id {doc_id!r} in gold_doc_ids")
    if bad:
        c.fail(f"{len(bad)} unknown doc_id references", bad)
    else:
        used = {p["doc_id"] for item in items for p in item.get("gold_passages", [])}
        c.ok(f"all references resolve ({len(used)}/{len(known_doc_ids)} known doc_ids used)")
    return c


def check_gold_doc_ids_consistency(items: list[dict]) -> Check:
    c = Check("gold_doc_ids consistency")
    bad = []
    for item in items:
        expected = sorted({p["doc_id"] for p in item.get("gold_passages", [])})
        actual = sorted(item.get("gold_doc_ids", []))
        if expected != actual:
            bad.append(f"{item['id']}: gold_doc_ids={actual} != derived {expected}")
    if bad:
        c.fail(f"{len(bad)}/{len(items)} mismatched", bad)
    else:
        c.ok(f"{len(items)}/{len(items)} match derived set")
    return c


def check_category_route_consistency(items: list[dict]) -> Check:
    c = Check("category/route consistency")
    bad = [
        f"{item['id']}: category={item['category']!r} expects route "
        f"{CATEGORY_TO_ROUTE.get(item['category'])!r}, got {item['expected_route']!r}"
        for item in items
        if CATEGORY_TO_ROUTE.get(item.get("category")) != item.get("expected_route")
    ]
    if bad:
        c.fail(f"{len(bad)}/{len(items)} mismatched", bad)
    else:
        c.ok(f"{len(items)}/{len(items)} match")
    return c


def check_gold_answer_emptiness(items: list[dict]) -> Check:
    c = Check("gold_answer emptiness rule")
    bad = []
    for item in items:
        is_refuse = item.get("category") == "unanswerable_refuse"
        is_empty = item.get("gold_answer", None) == ""
        if is_refuse != is_empty:
            bad.append(f"{item['id']}: category={item['category']!r} but gold_answer empty={is_empty}")
    if bad:
        c.fail(f"{len(bad)}/{len(items)} incorrect", bad)
    else:
        c.ok(f"{len(items)}/{len(items)} correct")
    return c


def check_gold_passages_presence(items: list[dict]) -> Check:
    c = Check("gold_passages presence rule")
    bad = []
    for item in items:
        n_passages = len(item.get("gold_passages", []))
        if item.get("expected_route") == "vectorstore" and n_passages == 0:
            bad.append(f"{item['id']}: expected_route=vectorstore but gold_passages is empty")
        if item.get("expected_route") == "refuse" and n_passages != 0:
            bad.append(f"{item['id']}: expected_route=refuse but gold_passages is non-empty")
    if bad:
        c.fail(f"{len(bad)}/{len(items)} incorrect", bad)
    else:
        c.ok(f"{len(items)}/{len(items)} correct")
    return c


def check_bucket_composition(items: list[dict]) -> Check:
    c = Check(f"bucket composition (n={len(items)}, target +/-{COMPOSITION_TOLERANCE_PP}pp)")
    if not items:
        c.skip("no items")
        return c
    total = len(items)
    counts = {cat: 0 for cat in CATEGORIES}
    for item in items:
        if item.get("category") in counts:
            counts[item["category"]] += 1
    rows = []
    out_of_range = []
    for cat in CATEGORIES:
        actual_pct = 100 * counts[cat] / total
        target_pct = TARGET_SHARE_PCT[cat]
        delta = actual_pct - target_pct
        status = "OK" if abs(delta) <= COMPOSITION_TOLERANCE_PP else f"OUT OF RANGE ({delta:+.1f}pp)"
        if status != "OK":
            out_of_range.append(cat)
        rows.append(f"  {cat:24s} target={target_pct:5.1f}%  actual={actual_pct:5.1f}%  n={counts[cat]:3d}  {status}")
    detail = "\n".join(rows)
    if out_of_range:
        c.fail(detail, [])
    else:
        c.ok(detail)
    return c


def check_paper_coverage(golden: list[dict], dev: list[dict], known_doc_ids: set[str]) -> Check:
    c = Check("paper coverage")
    union_doc_ids = {p["doc_id"] for item in (golden + dev) for p in item.get("gold_passages", [])}
    missing_union = known_doc_ids - union_doc_ids
    if missing_union:
        c.fail(f"{len(missing_union)}/{len(known_doc_ids)} papers missing from golden+dev union", sorted(missing_union))
        return c
    golden_doc_ids = {p["doc_id"] for item in golden for p in item.get("gold_passages", [])}
    missing_golden = known_doc_ids - golden_doc_ids
    if missing_golden:
        c.warn(
            f"{len(known_doc_ids)}/{len(known_doc_ids)} papers in union, but "
            f"{len(missing_golden)} missing from golden_set.jsonl alone (the CI-gated set)",
            sorted(missing_golden),
        )
    else:
        c.ok(f"{len(known_doc_ids)}/{len(known_doc_ids)} papers represented (golden_set.jsonl alone)")
    return c


def check_passage_resolution(
    golden: list[dict], dev: list[dict], parsed_md_dir: Path, metadata_path: Path
) -> list[Check]:
    checks = []
    for chunk_size in CHUNK_SIZES:
        c = Check(f"passage resolution @{chunk_size}")
        try:
            result = resolve_golden_set(
                golden + dev,
                chunk_size=chunk_size,
                overlap_size=128,
                parsed_md_dir=parsed_md_dir,
                metadata_path=metadata_path,
            )
        except PassageResolutionError as exc:
            c.fail("resolution error", [str(exc)])
            checks.append(c)
            continue
        unresolved_pct = 100 * result.n_chunks_unresolved / result.n_chunks_total if result.n_chunks_total else 0.0
        canary_bad = unresolved_pct > 3.0
        detail = (
            f"{result.n_gold_passages}/{result.n_gold_passages} gold passages resolved, "
            f"{result.n_chunks_unresolved}/{result.n_chunks_total} chunks unresolved ({unresolved_pct:.1f}%)"
        )
        if canary_bad:
            c.fail(f"{detail} -- exceeds 3% unresolved-chunk canary threshold")
        else:
            c.ok(detail)
        checks.append(c)
    return checks


def check_freeze_gate(golden: list[dict], dev: list[dict]) -> Check:
    c = Check("freeze gate (verified_by == human)")
    not_human = [item["id"] for item in (golden + dev) if item.get("verified_by") != "human"]
    if not_human:
        c.fail(f"{len(not_human)}/{len(golden) + len(dev)} items not yet verified_by=human", not_human)
    else:
        c.ok(f"{len(golden) + len(dev)}/{len(golden) + len(dev)} verified_by=human")
    return c


def print_report(checks: list[Check], golden_path: Path, dev_path: Path, golden: list[dict], dev: list[dict]) -> int:
    print("=== eval/validate_golden.py report ===")
    print(f"golden_set: {golden_path} ({len(golden)} items)   dev_split: {dev_path} ({len(dev)} items)")
    print()
    n_fail = n_warn = n_pass = n_skip = 0
    for c in checks:
        label = f"[{c.status}]"
        print(f"{label:<6} {c.name:<38} {c.detail.splitlines()[0] if c.detail else ''}")
        for extra_line in c.detail.splitlines()[1:]:
            print(extra_line)
        for err in c.errors[:10]:
            print(f"         - {err}")
        if len(c.errors) > 10:
            print(f"         ... and {len(c.errors) - 10} more")
        if c.status == "FAIL":
            n_fail += 1
        elif c.status == "WARN":
            n_warn += 1
        elif c.status == "SKIP":
            n_skip += 1
        else:
            n_pass += 1
    print()
    exit_code = 1 if n_fail else 0
    print(f"SUMMARY: {n_fail} FAIL, {n_warn} WARN, {n_skip} SKIP, {n_pass} PASS -- exit {exit_code}")
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--parsed-md", type=Path, default=DEFAULT_PARSED_MD)
    parser.add_argument("--skip-resolution", action="store_true")
    parser.add_argument("--require-verified", action="store_true")
    args = parser.parse_args()

    golden = _load(args.golden)
    dev = _load(args.dev)
    all_items = golden + dev

    known_doc_ids = {r["doc_id"] for r in open_jsonl(args.metadata)} if args.metadata.exists() else set()

    checks = [
        check_schema(all_items),
        check_id_format_and_uniqueness(golden, dev),
        check_dev_disjoint(golden, dev),
        check_doc_id_validity(all_items, known_doc_ids),
        check_gold_doc_ids_consistency(all_items),
        check_category_route_consistency(all_items),
        check_gold_answer_emptiness(all_items),
        check_gold_passages_presence(all_items),
        check_bucket_composition(all_items),
        check_paper_coverage(golden, dev, known_doc_ids),
    ]

    if args.skip_resolution:
        skip = Check("passage resolution")
        skip.skip("--skip-resolution set")
        checks.append(skip)
    else:
        checks.extend(check_passage_resolution(golden, dev, args.parsed_md, args.metadata))

    if args.require_verified:
        checks.append(check_freeze_gate(golden, dev))
    else:
        skip = Check("freeze gate (verified_by == human)")
        skip.skip("--require-verified not set")
        checks.append(skip)

    exit_code = print_report(checks, args.golden, args.dev, golden, dev)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
