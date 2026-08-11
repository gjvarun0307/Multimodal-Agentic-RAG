"""Regenerate artifacts/metadata.jsonl from artifacts/corpus_seed.csv.

Hand-authored identity fields (doc_id, source_pdf, source_md, title,
authors, year, topic, arxiv_id, license) live in the seed CSV -- this
script computes everything that can't be hand-written (content_sha256,
parsed_at) and emits one JSON object per line. If a paper is re-parsed,
re-run this rather than hand-editing metadata.jsonl: the hash will
change and downstream consumers see it immediately instead of silently
serving stale text against a fresh chunk index.

n_chunks is written as null here -- it's unknown until the Milvus index
is (re)built, which writes it back (see hybrid_database.py).

Usage:
    python deploy/build_metadata.py --seed artifacts/corpus_seed.csv --out artifacts/metadata.jsonl
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
PARSER = "llamaparse"
VLM_CAPTIONER = "Qwen2.5-VL-7B-Instruct"
DOC_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")

REQUIRED_COLUMNS = {
    "doc_id",
    "source_pdf",
    "source_md",
    "title",
    "authors",
    "year",
    "topic",
    "arxiv_id",
}


def build_metadata(seed_path: Path, parsed_md_dir: Path) -> list[dict]:
    """Read the seed CSV, validate it against the .md files on disk, and
    return the list of metadata records (sorted by doc_id, so output is
    independent of seed row order and filesystem listing order alike).
    Raises loudly on any mismatch -- never falls back to a partial join.
    """
    with open(seed_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Seed file {seed_path} is missing columns: {sorted(missing_columns)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in seed file {seed_path}")

    seen_doc_ids = set()
    records = []
    for row in rows:
        doc_id = row["doc_id"].strip()
        if not DOC_ID_PATTERN.match(doc_id):
            raise ValueError(f"doc_id {doc_id!r} does not match ^[a-z0-9_]+$")
        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate doc_id in seed file: {doc_id}")
        seen_doc_ids.add(doc_id)

        source_md = row["source_md"].strip()
        md_path = parsed_md_dir / source_md
        if not md_path.exists():
            raise FileNotFoundError(f"doc_id {doc_id!r}: source_md {source_md!r} not found under {parsed_md_dir}")

        content = md_path.read_bytes()
        content_sha256 = hashlib.sha256(content).hexdigest()
        parsed_at = datetime.fromtimestamp(md_path.stat().st_mtime, tz=timezone.utc).isoformat()

        authors = [a.strip() for a in row["authors"].split(";") if a.strip()]
        year_raw = row["year"].strip()
        if not year_raw.isdigit():
            raise ValueError(f"doc_id {doc_id!r}: year {year_raw!r} is not a 4-digit integer")

        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "doc_id": doc_id,
                "source_pdf": row["source_pdf"].strip(),
                "source_md": source_md,
                "content_sha256": content_sha256,
                "n_chunks": None,
                "title": row["title"].strip(),
                "authors": authors,
                "year": int(year_raw),
                "topic": row["topic"].strip(),
                "arxiv_id": row["arxiv_id"].strip() or None,
                "parsed_at": parsed_at,
                "parser": PARSER,
                "vlm_captioner": VLM_CAPTIONER,
                "verified_by": "human",
            }
        )

    md_files_on_disk = {p.name for p in parsed_md_dir.glob("*.md")}
    claimed_md_files = {r["source_md"] for r in records}
    orphan_md = md_files_on_disk - claimed_md_files
    orphan_seed = claimed_md_files - md_files_on_disk
    if orphan_md:
        raise ValueError(f".md files under {parsed_md_dir} with no seed row: {sorted(orphan_md)}")
    if orphan_seed:
        raise ValueError(f"Seed rows referencing .md files missing from {parsed_md_dir}: {sorted(orphan_seed)}")

    records.sort(key=lambda r: r["doc_id"])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=Path, default=Path("artifacts/corpus_seed.csv"))
    parser.add_argument("--parsed-md-dir", type=Path, default=Path("artifacts/parsed_md"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/metadata.jsonl"))
    args = parser.parse_args()

    records = build_metadata(args.seed, args.parsed_md_dir)

    with open(args.out, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
