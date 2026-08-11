"""Regression test for PROJECT_SPEC.md §5.2: chunk IDs must be
deterministic -- rebuilding from identical sources produces identical
IDs. This is what lets gold labels resolve to stable chunk_ids across
reruns instead of silently shifting whenever the index is rebuilt.

Uses build_chunks() (splitting + ID assignment only, no embeddings or
Milvus) so the double-build comparison is fast and has no side effects.
"""

import re

from src.configuration import config_rag
from src.hybrid_database import build_chunks

CHUNK_ID_PATTERN = re.compile(r"^[a-z0-9_]+::\d{4}::[0-9a-f]{8}$")


def test_double_build_produces_identical_chunk_ids():
    config = config_rag()
    first = build_chunks(config)
    second = build_chunks(config)

    first_ids = [doc.metadata["chunk_id"] for doc in first]
    second_ids = [doc.metadata["chunk_id"] for doc in second]

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids)), "chunk_ids must be unique"


def test_chunk_ids_match_expected_format():
    config = config_rag()
    chunks = build_chunks(config)
    assert chunks, "expected at least one chunk from the corpus"
    for doc in chunks:
        assert CHUNK_ID_PATTERN.match(doc.metadata["chunk_id"]), doc.metadata["chunk_id"]


def test_chunk_ids_are_prefixed_by_their_doc_id():
    config = config_rag()
    chunks = build_chunks(config)
    for doc in chunks:
        assert doc.metadata["chunk_id"].startswith(f"{doc.metadata['doc_id']}::")


def test_every_corpus_document_produces_at_least_one_chunk():
    config = config_rag()
    chunks = build_chunks(config)
    doc_ids_seen = {doc.metadata["doc_id"] for doc in chunks}
    assert len(doc_ids_seen) == 15
