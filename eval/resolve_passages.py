"""Resolve golden-set gold_passages to chunk IDs under an active chunking config.

Ground truth in eval/golden/golden_set.jsonl is verbatim `gold_passages` text,
never chunk IDs (PROJECT_SPEC.md SS5.2.1) -- chunk IDs depend on chunk_size/
overlap_size, and the chunk-size ablation (512/1024/2048, Phase 6) would break
any gold label stored as a fixed chunk ID. This script derives gold_chunk_ids
at resolve time, for any chunking config, and caches the result.

Two independent resolution problems:

  1. Gold passage -> char span in the source .md. Trivial literal substring
     search: the annotator copied `passage_text` straight out of the raw .md,
     so it must appear exactly once. Zero or >1 matches is a hard error
     (invariant 4) -- never a skip.

  2. Chunk -> char span in the source .md, needed only to test overlap against
     a gold passage span. Non-trivial: MarkdownHeaderTextSplitter(strip_headers
     =True) (see src.hybrid_database.split_data) removes header lines and
     strips each kept line, so a chunk's page_content is NOT a literal
     substring of the raw file. Solved by reconstructing the same transform
     with an offset map back to raw .md positions, then aligning page_content
     against it (strict, whitespace-tolerant; falls back to a fuzzy regex
     anchor when the overlap window makes the naive cursor estimate land in
     the wrong place). This resolution is best-effort and logged, not a hard
     error: on this corpus ~99% of chunks resolve cleanly, and the residual is
     concentrated in table-divider rows and bare LaTeX blocks that no real
     prose gold passage will ever land in. See resolve_golden_set()'s returned
     n_chunks_unresolved for the per-run count.

Usage:
    python -m eval.resolve_passages
    python -m eval.resolve_passages --chunk-sizes 512,1024,2048 --overlap-size 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.configuration import config_rag
from src.helper import open_jsonl
from src.hybrid_database import MARKDOWN_HEADERS_TO_SPLIT_ON, build_chunks, prepare_input_data

RESOLVER_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = REPO_ROOT / "eval" / "golden" / "golden_set.jsonl"
DEFAULT_DEV = REPO_ROOT / "eval" / "golden" / "dev_split.jsonl"
DEFAULT_METADATA = REPO_ROOT / "artifacts" / "metadata.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "eval" / "golden" / "resolved"


class PassageResolutionError(Exception):
    """Raised when a gold_passages entry does not resolve to exactly one span."""


# --------------------------------------------------------------------------
# Problem 1: gold passage -> char span (trivial, exact substring match)
# --------------------------------------------------------------------------


def resolve_gold_passage(raw_text: str, passage_text: str, *, item_id: str, doc_id: str) -> tuple[int, int]:
    n_matches = raw_text.count(passage_text)
    if n_matches == 0:
        raise PassageResolutionError(
            f"{item_id}: passage_text not found in {doc_id}'s source .md (0 matches) -- "
            f"check for copy/paste drift, a typo, or a rendered-view whitespace mismatch "
            f"(see eval/golden/SCHEMA.md 'Authoring pitfalls'). Passage: {passage_text[:80]!r}..."
        )
    if n_matches > 1:
        raise PassageResolutionError(
            f"{item_id}: passage_text matches {n_matches} spans in {doc_id}'s source .md -- "
            f"not unique, extend the passage. Passage: {passage_text[:80]!r}..."
        )
    start = raw_text.find(passage_text)
    return start, start + len(passage_text)


# --------------------------------------------------------------------------
# Problem 2: chunk -> char span (best-effort, whitespace-tolerant alignment)
# --------------------------------------------------------------------------


def _headers_sorted_by_length(headers_to_split_on: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted(headers_to_split_on, key=lambda h: len(h[0]), reverse=True)


def _is_header_line(stripped: str, headers_sorted: list[tuple[str, str]]) -> bool:
    for sep, _name in headers_sorted:
        if stripped.startswith(sep) and (len(stripped) == len(sep) or stripped[len(sep)] == " "):
            return True
    return False


def build_raw_body(raw_text: str, headers_to_split_on: list[tuple[str, str]]) -> tuple[str, list[Optional[int]]]:
    """Reproduce MarkdownHeaderTextSplitter(strip_headers=True)'s whitespace
    transform: header lines removed, every kept line .strip()'d, kept lines
    joined by a single '\\n'. Returns (raw_body, offsets) where offsets[i] is
    the raw_text char index raw_body[i] came from, or None for a synthetic
    joiner newline we inserted between kept lines (there is no single raw_text
    position that represents it -- it may span a blank line, stripped
    indentation, or an entire removed header line).
    """
    headers_sorted = _headers_sorted_by_length(headers_to_split_on)

    body_chars: list[str] = []
    offsets: list[Optional[int]] = []
    pos, n = 0, len(raw_text)
    in_code_block, fence = False, ""
    first_kept_line = True

    while pos <= n:
        nl = raw_text.find("\n", pos)
        if nl == -1:
            line, line_start = raw_text[pos:], pos
            pos = n + 1
        else:
            line, line_start = raw_text[pos:nl], pos
            pos = nl + 1
        stripped = line.strip()
        lstrip_count = len(line) - len(line.lstrip())

        if not in_code_block:
            if stripped.startswith("```"):
                in_code_block, fence = True, "```"
            elif stripped.startswith("~~~"):
                in_code_block, fence = True, "~~~"
        elif stripped.startswith(fence):
            in_code_block, fence = False, ""

        excluded = (not in_code_block) and _is_header_line(stripped, headers_sorted)
        if not excluded:
            if not first_kept_line:
                body_chars.append("\n")
                offsets.append(None)
            first_kept_line = False
            for i, ch in enumerate(stripped):
                body_chars.append(ch)
                offsets.append(line_start + lstrip_count + i)

        if nl == -1:
            break

    return "".join(body_chars), offsets


def strict_align(
    raw_body: str, offsets: list[Optional[int]], page_content: str, start: int
) -> Optional[tuple[int, int, int]]:
    """Whitespace-tolerant alignment of page_content against raw_body starting
    at `start`. Non-whitespace characters must match exactly -- the transform
    in build_raw_body never touches non-whitespace content, only whitespace
    runs. Returns (raw_text_span_start, raw_text_span_end, next_raw_body_cursor)
    or None if alignment fails before page_content is fully consumed.
    """
    r, p = start, 0
    body_len, pc_len = len(raw_body), len(page_content)
    first_r: Optional[int] = None
    last_r: Optional[int] = None

    while p < pc_len:
        if r >= body_len:
            return None
        cr, cp = raw_body[r], page_content[p]
        if cr == cp:
            if offsets[r] is not None:
                if first_r is None:
                    first_r = offsets[r]
                last_r = offsets[r]
            r += 1
            p += 1
        elif cr.isspace() and cp.isspace():
            r += 1
            p += 1
        elif cr.isspace():
            r += 1
        elif cp.isspace():
            p += 1
        else:
            return None

    if first_r is None:
        return None
    return first_r, last_r + 1, r


def _fuzzy_anchor_pattern(s: str) -> str:
    """Turn a snippet into a regex that matches it with any single run of
    whitespace standing in for any other -- needed because a literal anchor
    fails whenever it straddles a whitespace-transform boundary (e.g. VLM
    caption blocks: short fields joined by '  \\n' in raw text collapse to a
    single '\\n' in raw_body)."""
    parts = re.split(r"(\s+)", s)
    return "".join(r"\s+" if part[:1].isspace() else re.escape(part) for part in parts if part)


def resolve_chunk_span(
    raw_body: str, offsets: list[Optional[int]], page_content: str, approx_hint: int
) -> Optional[tuple[int, int, int]]:
    result = strict_align(raw_body, offsets, page_content, approx_hint)
    if result is not None:
        return result

    anchor_len = min(80, len(page_content))
    if anchor_len == 0:
        return None
    pattern = _fuzzy_anchor_pattern(page_content[:anchor_len])
    for lo, hi in (
        (max(0, approx_hint - 3000), min(len(raw_body), approx_hint + 3000)),
        (0, len(raw_body)),
    ):
        match = re.search(pattern, raw_body[lo:hi])
        if match:
            retry = strict_align(raw_body, offsets, page_content, lo + match.start())
            if retry is not None:
                return retry
    return None


def _whitespace_normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def build_chunk_spans_for_doc(
    raw_text: str, doc_chunks: list, headers_to_split_on: list[tuple[str, str]]
) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """doc_chunks must be in document order (build_chunks' own iteration
    order). Returns (chunk_id -> (start, end)) for resolved chunks, plus the
    list of chunk_ids that failed to resolve (logged, not raised -- see
    module docstring)."""
    raw_body, offsets = build_raw_body(raw_text, headers_to_split_on)

    spans: dict[str, tuple[int, int]] = {}
    unresolved: list[str] = []
    cursor = 0

    for doc in doc_chunks:
        chunk_id = doc.metadata["chunk_id"]
        page_content = doc.page_content
        hint = max(0, cursor - 200)  # allow backing up into the previous chunk's overlap window
        result = resolve_chunk_span(raw_body, offsets, page_content, hint)
        if result is None:
            unresolved.append(chunk_id)
            continue
        start, end, next_cursor = result
        candidate = raw_text[start:end]
        if _whitespace_normalize(candidate) != _whitespace_normalize(page_content):
            unresolved.append(chunk_id)
            continue
        spans[chunk_id] = (start, end)
        cursor = next_cursor

    return spans, unresolved


def overlap_chunk_is_gold(chunk_span: tuple[int, int], passage_span: tuple[int, int]) -> bool:
    c0, c1 = chunk_span
    p0, p1 = passage_span
    overlap = max(0, min(c1, p1) - max(c0, p0))
    fully_contains = c0 <= p0 and c1 >= p1
    return fully_contains or (overlap / (p1 - p0)) >= 0.5


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class ResolutionResult:
    chunk_size: int
    overlap_size: int
    config_hash: str
    n_items: int
    n_gold_passages: int
    n_chunks_total: int
    n_chunks_unresolved: int
    unresolved_chunk_ids: list[str]
    per_item: dict[str, dict]  # item_id -> {"gold_passages": [...], "gold_chunk_ids_union": [...]}


def _load_metadata_by_doc_id(metadata_path: Path) -> dict[str, dict]:
    records = open_jsonl(metadata_path)
    return {r["doc_id"]: r for r in records}


def compute_config_hash(chunk_size: int, overlap_size: int, corpus_hashes: dict[str, str]) -> str:
    payload = {
        "chunk_size": chunk_size,
        "overlap_size": overlap_size,
        "resolver_version": RESOLVER_VERSION,
        "headers": MARKDOWN_HEADERS_TO_SPLIT_ON,
        "corpus": dict(sorted(corpus_hashes.items())),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def resolve_golden_set(
    items: list[dict],
    *,
    chunk_size: int,
    overlap_size: int,
    parsed_md_dir: Path = REPO_ROOT / "artifacts" / "parsed_md",
    metadata_path: Path = DEFAULT_METADATA,
) -> ResolutionResult:
    """Resolve every gold_passages entry across `items` to gold_chunk_ids under
    the given chunking config. Raises PassageResolutionError on any gold
    passage that doesn't resolve to exactly one span (hard error). Chunk-span
    resolution failures are logged into the returned result, not raised.
    """
    config = config_rag(
        overrides={
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "input_folder_path": str(parsed_md_dir),
            "metadata_path": str(metadata_path),
        }
    )
    metadata_by_doc_id = _load_metadata_by_doc_id(metadata_path)
    input_data = prepare_input_data(str(parsed_md_dir), str(metadata_path))
    all_chunks = build_chunks(config)

    chunks_by_doc: dict[str, list] = {}
    for doc in all_chunks:
        chunks_by_doc.setdefault(doc.metadata["doc_id"], []).append(doc)

    # Resolve chunk spans for every document in the corpus up front, not just
    # ones referenced by `items` -- otherwise n_chunks_unresolved (numerator,
    # scoped to whatever's touched) and n_chunks_total (denominator, always
    # corpus-wide from build_chunks) drift out of sync and the unresolved-
    # chunk canary silently under-reports whenever the golden set doesn't yet
    # cover every paper (e.g. mid-drafting, or a future single-doc ablation).
    raw_text_by_doc: dict[str, str] = {}
    chunk_spans_by_doc: dict[str, dict[str, tuple[int, int]]] = {}
    unresolved_chunk_ids: list[str] = []

    for doc_id, entry in input_data.items():
        raw_text_by_doc[doc_id] = entry["path"].read_text(encoding="utf-8")
        spans, unresolved = build_chunk_spans_for_doc(
            raw_text_by_doc[doc_id], chunks_by_doc.get(doc_id, []), MARKDOWN_HEADERS_TO_SPLIT_ON
        )
        chunk_spans_by_doc[doc_id] = spans
        unresolved_chunk_ids.extend(unresolved)

    per_item: dict[str, dict] = {}
    n_gold_passages = 0

    for item in items:
        item_id = item["id"]
        resolved_passages = []
        gold_chunk_ids_union: set[str] = set()

        for passage in item.get("gold_passages", []):
            doc_id = passage["doc_id"]
            if doc_id not in input_data:
                raise PassageResolutionError(
                    f"{item_id}: doc_id {doc_id!r} not found in metadata.jsonl / {parsed_md_dir}"
                )
            n_gold_passages += 1
            raw_text, chunk_spans = raw_text_by_doc[doc_id], chunk_spans_by_doc[doc_id]
            passage_span = resolve_gold_passage(raw_text, passage["passage_text"], item_id=item_id, doc_id=doc_id)

            gold_chunk_ids = sorted(
                chunk_id
                for chunk_id, chunk_span in chunk_spans.items()
                if chunk_id.startswith(f"{doc_id}::") and overlap_chunk_is_gold(chunk_span, passage_span)
            )
            gold_chunk_ids_union.update(gold_chunk_ids)
            resolved_passages.append(
                {
                    "doc_id": doc_id,
                    "passage_char_span": list(passage_span),
                    "gold_chunk_ids": gold_chunk_ids,
                }
            )

        per_item[item_id] = {
            "gold_passages": resolved_passages,
            "gold_chunk_ids_union": sorted(gold_chunk_ids_union),
        }

    corpus_hashes = {doc_id: metadata_by_doc_id[doc_id]["content_sha256"] for doc_id in input_data}
    n_chunks_total = len(all_chunks)

    return ResolutionResult(
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        config_hash=compute_config_hash(chunk_size, overlap_size, corpus_hashes),
        n_items=len(items),
        n_gold_passages=n_gold_passages,
        n_chunks_total=n_chunks_total,
        n_chunks_unresolved=len(unresolved_chunk_ids),
        unresolved_chunk_ids=sorted(unresolved_chunk_ids),
        per_item=per_item,
    )


def write_resolution_cache(result: ResolutionResult, out_dir: Path, corpus_hashes: dict[str, str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.config_hash}.json"
    payload = {
        "config_hash": result.config_hash,
        "chunk_size": result.chunk_size,
        "overlap_size": result.overlap_size,
        "resolver_version": RESOLVER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_content_hashes": dict(sorted(corpus_hashes.items())),
        "n_items": result.n_items,
        "n_gold_passages": result.n_gold_passages,
        "n_chunks_total": result.n_chunks_total,
        "n_chunks_unresolved": result.n_chunks_unresolved,
        "unresolved_chunk_ids": result.unresolved_chunk_ids,
        "items": result.per_item,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
        f.write("\n")
    return out_path


def _load_items(golden_path: Path, dev_path: Optional[Path]) -> list[dict]:
    items = open_jsonl(golden_path) if golden_path.exists() else []
    if dev_path is not None and dev_path.exists():
        items = items + open_jsonl(dev_path)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--parsed-md", type=Path, default=REPO_ROOT / "artifacts" / "parsed_md")
    parser.add_argument("--chunk-sizes", type=str, default="512,1024,2048")
    parser.add_argument("--overlap-size", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    items = _load_items(args.golden, args.dev)
    if not items:
        print(f"No items found at {args.golden} (or {args.dev}) -- nothing to resolve.", file=sys.stderr)
        sys.exit(1)

    chunk_sizes = [int(s) for s in args.chunk_sizes.split(",") if s.strip()]
    metadata_by_doc_id = _load_metadata_by_doc_id(args.metadata)
    corpus_hashes = {doc_id: rec["content_sha256"] for doc_id, rec in metadata_by_doc_id.items()}

    exit_code = 0
    for chunk_size in chunk_sizes:
        try:
            result = resolve_golden_set(
                items,
                chunk_size=chunk_size,
                overlap_size=args.overlap_size,
                parsed_md_dir=args.parsed_md,
                metadata_path=args.metadata,
            )
        except PassageResolutionError as exc:
            print(f"chunk_size={chunk_size}: FAILED -- {exc}", file=sys.stderr)
            exit_code = 1
            continue

        out_path = write_resolution_cache(result, args.out_dir, corpus_hashes)
        unresolved_pct = 100 * result.n_chunks_unresolved / result.n_chunks_total if result.n_chunks_total else 0.0
        print(
            f"chunk_size={chunk_size}: {result.n_items} items, "
            f"{result.n_gold_passages}/{result.n_gold_passages} gold passages resolved, "
            f"{result.n_chunks_unresolved}/{result.n_chunks_total} chunks unresolved "
            f"({unresolved_pct:.1f}%) -> {out_path}",
            file=sys.stderr,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
