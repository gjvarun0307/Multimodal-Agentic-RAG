"""Validates artifacts/metadata.jsonl per PROJECT_SPEC.md §5.1.4.

This is the regression guard for the exact bug that made regeneration
necessary: metadata used to be joined against .md files by mtime-sort +
positional zip, with no shared key on either side, so any re-parse,
re-copy, or filesystem mtime change silently mispaired a paper's metadata
with a different paper's text.
"""

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import pytest

from deploy.build_metadata import build_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "artifacts" / "metadata.jsonl"
PARSED_MD_DIR = REPO_ROOT / "artifacts" / "parsed_md"
SEED_PATH = REPO_ROOT / "artifacts" / "corpus_seed.csv"
DOC_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")

# Corpus text isn't committed (license terms vary per paper -- see
# artifacts/SOURCES.md); it's regenerated locally from data/raw_pdfs/ via
# the parse pipeline. A fresh clone or CI without that ingest step won't
# have it -- skip rather than fail confusingly.
pytestmark = pytest.mark.skipif(
    not PARSED_MD_DIR.exists() or not any(PARSED_MD_DIR.glob("*.md")),
    reason=f"{PARSED_MD_DIR} not populated -- run the ingest pipeline first (see artifacts/SOURCES.md)",
)


def _load_records():
    with open(METADATA_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_metadata_file_exists():
    assert METADATA_PATH.exists()


def test_bijection_between_metadata_and_md_files():
    """Every .md on disk has exactly one metadata row and vice versa. An
    orphan on either side is a hard failure, never a warning."""
    records = _load_records()
    metadata_files = {r["source_md"] for r in records}
    disk_files = {p.name for p in PARSED_MD_DIR.glob("*.md")}
    orphan_metadata = metadata_files - disk_files
    orphan_disk = disk_files - metadata_files
    assert not orphan_metadata, f"metadata rows with no .md on disk: {orphan_metadata}"
    assert not orphan_disk, f".md files with no metadata row: {orphan_disk}"


def test_doc_id_unique():
    records = _load_records()
    doc_ids = [r["doc_id"] for r in records]
    assert len(doc_ids) == len(set(doc_ids)), "duplicate doc_id in metadata.jsonl"


def test_doc_id_format():
    records = _load_records()
    for r in records:
        assert DOC_ID_PATTERN.match(r["doc_id"]), f"bad doc_id: {r['doc_id']}"


def test_source_md_resolves_to_existing_file():
    records = _load_records()
    for r in records:
        assert (PARSED_MD_DIR / r["source_md"]).exists(), r["source_md"]


def test_content_sha256_matches_actual_file():
    records = _load_records()
    for r in records:
        content = (PARSED_MD_DIR / r["source_md"]).read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        assert actual == r["content_sha256"], f"hash drift for {r['doc_id']}"


def test_schema_version_present_and_recognised():
    records = _load_records()
    for r in records:
        assert r.get("schema_version") == 1


def test_fifteen_papers_present():
    """Regression guard for the frozen corpus size (invariant 2)."""
    records = _load_records()
    assert len(records) == 15


def test_rebuild_is_independent_of_filesystem_listing_order(tmp_path):
    """Shuffling every file's mtime (and therefore directory listing order)
    must not change the doc_id -> content mapping. Regression test for the
    mtime-sort/positional-zip bug this regeneration replaces. Operates on a
    scratch copy so it never touches real file mtimes / provenance."""
    tmp_md_dir = tmp_path / "parsed_md"
    shutil.copytree(PARSED_MD_DIR, tmp_md_dir)
    tmp_seed = tmp_path / "corpus_seed.csv"
    shutil.copy(SEED_PATH, tmp_seed)

    first = build_metadata(tmp_seed, tmp_md_dir)

    md_files = sorted(tmp_md_dir.glob("*.md"), reverse=True)
    base = time.time()
    for i, p in enumerate(md_files):
        os.utime(p, (base + i, base + i))

    second = build_metadata(tmp_seed, tmp_md_dir)

    first_map = {r["doc_id"]: r["content_sha256"] for r in first}
    second_map = {r["doc_id"]: r["content_sha256"] for r in second}
    assert first_map == second_map
    assert len(first_map) == 15
