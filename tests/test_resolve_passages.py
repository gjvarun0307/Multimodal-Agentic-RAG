"""Tests for eval/resolve_passages.py.

Two tiers, following the corpus-skip convention established in
test_chunk_id_determinism.py:

  - pure-function unit tests (no corpus needed): the whitespace-tolerant
    aligner, the header-stripping raw-body reconstruction, the fuzzy-anchor
    fallback, the overlap/containment test, and gold-passage resolution.
  - a corpus-gated integration test that runs the full resolver against a
    real passage from the real corpus at all three ablation chunk sizes.
"""

from pathlib import Path

import pytest

from eval.resolve_passages import (
    PassageResolutionError,
    build_raw_body,
    overlap_chunk_is_gold,
    resolve_chunk_span,
    resolve_gold_passage,
    resolve_golden_set,
    strict_align,
)
from src.hybrid_database import MARKDOWN_HEADERS_TO_SPLIT_ON

_PARSED_MD_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "parsed_md"
_METADATA_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "metadata.jsonl"

pytestmark_corpus = pytest.mark.skipif(
    not _PARSED_MD_DIR.exists() or not any(_PARSED_MD_DIR.glob("*.md")),
    reason=f"{_PARSED_MD_DIR} not populated -- run the ingest pipeline first (see artifacts/SOURCES.md)",
)


# --------------------------------------------------------------------------
# overlap_chunk_is_gold
# --------------------------------------------------------------------------


def test_overlap_full_containment_is_gold():
    assert overlap_chunk_is_gold(chunk_span=(0, 100), passage_span=(10, 20)) is True


def test_overlap_exactly_half_is_gold():
    # overlap=10, passage_len=20 -> ratio exactly 0.5, boundary is inclusive
    assert overlap_chunk_is_gold(chunk_span=(0, 10), passage_span=(0, 20)) is True


def test_overlap_below_half_is_not_gold():
    # overlap=5, passage_len=20 -> ratio 0.25
    assert overlap_chunk_is_gold(chunk_span=(0, 10), passage_span=(5, 25)) is False


def test_no_overlap_is_not_gold():
    assert overlap_chunk_is_gold(chunk_span=(0, 5), passage_span=(10, 20)) is False


def test_overlap_only_considers_same_span_math_not_doc_id():
    # doc_id filtering happens one layer up in resolve_golden_set; this
    # function is pure span math.
    assert overlap_chunk_is_gold(chunk_span=(100, 200), passage_span=(100, 200)) is True


# --------------------------------------------------------------------------
# build_raw_body -- header stripping + offset map correctness
# --------------------------------------------------------------------------


def test_build_raw_body_strips_header_lines():
    raw_text = "# Title Line\nFirst body sentence.\n\nSecond body sentence.\n"
    raw_body, offsets = build_raw_body(raw_text, MARKDOWN_HEADERS_TO_SPLIT_ON)
    assert "Title Line" not in raw_body
    assert "First body sentence." in raw_body
    assert "Second body sentence." in raw_body


def test_build_raw_body_offsets_are_faithful():
    raw_text = "## Section\nAlpha beta gamma.\n\nDelta epsilon.\n"
    raw_body, offsets = build_raw_body(raw_text, MARKDOWN_HEADERS_TO_SPLIT_ON)
    assert len(raw_body) == len(offsets)
    for i, off in enumerate(offsets):
        if off is not None:
            assert raw_text[off] == raw_body[i]


def test_build_raw_body_does_not_strip_hash_inside_code_fence():
    raw_text = "```python\n# not a header, it's a comment\n```\nBody text.\n"
    raw_body, _ = build_raw_body(raw_text, MARKDOWN_HEADERS_TO_SPLIT_ON)
    assert "not a header" in raw_body
    assert "Body text." in raw_body


# --------------------------------------------------------------------------
# strict_align
# --------------------------------------------------------------------------


def test_strict_align_exact_match():
    raw_body = "hello world"
    offsets = list(range(len(raw_body)))
    result = strict_align(raw_body, offsets, "hello", 0)
    assert result == (0, 5, 5)


def test_strict_align_tolerates_whitespace_run_mismatch():
    raw_body = "a   b"  # three raw spaces
    offsets = list(range(len(raw_body)))
    result = strict_align(raw_body, offsets, "a b", 0)  # single space in page_content
    assert result is not None
    start, end, _cursor = result
    assert raw_body[start:end] == "a   b"


def test_strict_align_fails_on_content_mismatch():
    raw_body = "hello"
    offsets = list(range(len(raw_body)))
    assert strict_align(raw_body, offsets, "world", 0) is None


def test_strict_align_fails_when_start_past_needed_content():
    raw_body = "hello world"
    offsets = list(range(len(raw_body)))
    assert strict_align(raw_body, offsets, "hello", 6) is None


# --------------------------------------------------------------------------
# resolve_chunk_span -- fuzzy-anchor fallback for a bad hint
# --------------------------------------------------------------------------


def test_resolve_chunk_span_recovers_via_full_document_fallback():
    target = "UNIQUE TARGET PHRASE FOR TEST"
    raw_body = ("A" * 5000) + target + ("B" * 5000)
    offsets = list(range(len(raw_body)))
    # hint of 0 puts the +/-3000 window entirely before the true position (5000)
    result = resolve_chunk_span(raw_body, offsets, target, approx_hint=0)
    assert result is not None
    start, end, _cursor = result
    assert raw_body[start:end] == target


def test_resolve_chunk_span_returns_none_for_absent_content():
    raw_body = "A" * 200
    offsets = list(range(len(raw_body)))
    assert resolve_chunk_span(raw_body, offsets, "not present anywhere", approx_hint=0) is None


def test_resolve_chunk_span_handles_caption_style_whitespace_collapse():
    # simulates a VLM-caption block: raw has hard-break "  \n" between short
    # fields, page_content (post-transform) has a single "\n".
    raw_text = "######  Caption\n\ntitle: Foo  \ntype: bar  \ndescription: baz  \n"
    raw_body, offsets = build_raw_body(raw_text, MARKDOWN_HEADERS_TO_SPLIT_ON)
    page_content = "title: Foo\ntype: bar\ndescription: baz"
    result = resolve_chunk_span(raw_body, offsets, page_content, approx_hint=0)
    assert result is not None


# --------------------------------------------------------------------------
# resolve_gold_passage -- hard-error semantics
# --------------------------------------------------------------------------


def test_resolve_gold_passage_unique_match():
    raw_text = "Some intro. The unique supporting sentence. Some outro."
    span = resolve_gold_passage(raw_text, "The unique supporting sentence.", item_id="gs_test", doc_id="test_doc")
    assert raw_text[span[0] : span[1]] == "The unique supporting sentence."


def test_resolve_gold_passage_zero_matches_raises():
    with pytest.raises(PassageResolutionError):
        resolve_gold_passage("Some text here.", "not present at all", item_id="gs_test", doc_id="test_doc")


def test_resolve_gold_passage_multiple_matches_raises():
    with pytest.raises(PassageResolutionError):
        resolve_gold_passage("dup dup dup", "dup", item_id="gs_test", doc_id="test_doc")


# --------------------------------------------------------------------------
# Corpus-gated integration test
# --------------------------------------------------------------------------


@pytestmark_corpus
@pytest.mark.parametrize("chunk_size", [512, 1024, 2048])
def test_resolve_golden_set_against_real_corpus(chunk_size):
    items = [
        {
            "id": "gs_test_0001",
            "gold_passages": [
                {
                    "doc_id": "vllm_paged_attention",
                    "passage_text": (
                        "PagedAttention divides the request’s KV cache into blocks, "
                        "each of which can contain the attention keys and values of a "
                        "fixed number of tokens."
                    ),
                }
            ],
        }
    ]
    result = resolve_golden_set(
        items,
        chunk_size=chunk_size,
        overlap_size=128,
        parsed_md_dir=_PARSED_MD_DIR,
        metadata_path=_METADATA_PATH,
    )
    assert result.n_gold_passages == 1
    unresolved_pct = 100 * result.n_chunks_unresolved / result.n_chunks_total
    assert unresolved_pct < 3.0, f"unresolved-chunk canary tripped: {unresolved_pct:.2f}%"
    gold_chunk_ids = result.per_item["gs_test_0001"]["gold_chunk_ids_union"]
    assert gold_chunk_ids, "expected at least one gold chunk id for a real, resolvable passage"
    assert all(cid.startswith("vllm_paged_attention::") for cid in gold_chunk_ids)


@pytestmark_corpus
def test_resolve_golden_set_is_deterministic():
    items = [
        {
            "id": "gs_test_0001",
            "gold_passages": [
                {
                    "doc_id": "vllm_paged_attention",
                    "passage_text": (
                        "PagedAttention divides the request’s KV cache into blocks, "
                        "each of which can contain the attention keys and values of a "
                        "fixed number of tokens."
                    ),
                }
            ],
        }
    ]
    kwargs = dict(chunk_size=1024, overlap_size=128, parsed_md_dir=_PARSED_MD_DIR, metadata_path=_METADATA_PATH)
    first = resolve_golden_set(items, **kwargs)
    second = resolve_golden_set(items, **kwargs)
    assert first.per_item == second.per_item
    assert first.config_hash == second.config_hash
