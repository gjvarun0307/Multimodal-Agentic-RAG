# Golden set schema

Defines every field in `eval/golden/golden_set.jsonl` and `eval/golden/dev_split.jsonl`
(same schema, disjoint by `id` — see `PROJECT_SPEC.md` §7 Phase 1). One JSON object
per line. This document is authoritative for field meaning; `eval/validate_golden.py`
enforces it mechanically and must be kept in sync if a field is added.

## Fields

| field | type | required | notes |
|---|---|---|---|
| `id` | str | always | pattern `^gs_\d{4}$`, unique across `golden_set.jsonl` ∪ `dev_split.jsonl` |
| `question` | str | always | non-empty |
| `gold_answer` | str | always | `""` **iff** `category == "unanswerable_refuse"`; non-empty otherwise |
| `gold_passages` | list of `{doc_id, passage_text}` | required when `expected_route == "vectorstore"` | `[]` for `refuse`/`websearch`/`chitchat` items; see "Ground truth and resolution" below |
| `gold_doc_ids` | list[str] | always | must equal `sorted(set(p["doc_id"] for p in gold_passages))` when `gold_passages` is non-empty, else `[]`. **Diagnostics only — never a CI-gating metric** (near-ceiling at 15 docs, per spec §7) |
| `category` | str enum | always | one of: `single_hop`, `multi_hop`, `table_figure`, `unanswerable_refuse`, `ambiguous`, `adversarial`, `unanswerable_websearch`, `chitchat` |
| `expected_route` | str enum | always | `vectorstore` \| `websearch` \| `chitchat` \| `refuse`; must match the category→route mapping below |
| `difficulty` | str enum | always | `easy` \| `medium` \| `hard` — informational only, not gated by any Phase 1 acceptance criterion |
| `requires_multimodal` | bool | always | true when the answer depends on a VLM-captioned table/figure block. Correlated with, but not mechanically forced to equal, `category == "table_figure"` |
| `notes` | str or null | optional | freeform, e.g. chunk-boundary caveats |
| `verified_by` | str enum | always | `"draft"` \| `"human"` — see "verified_by lifecycle" below |
| `version` | int | always | `1` at initial freeze; a project-wide bump on any post-freeze edit (invariant 2/5) |

### category → expected_route mapping

| category | expected_route |
|---|---|
| `single_hop` | `vectorstore` |
| `multi_hop` | `vectorstore` |
| `table_figure` | `vectorstore` |
| `ambiguous` | `vectorstore` |
| `adversarial` | `vectorstore` |
| `unanswerable_refuse` | `refuse` |
| `unanswerable_websearch` | `websearch` |
| `chitchat` | `chitchat` |

## Ground truth and resolution (PROJECT_SPEC.md §5.2.1)

`gold_passages` is the ground truth. **Chunk IDs are never stored in this file** —
they are derived at eval time by `eval/resolve_passages.py`, because storing them
directly would break under the chunk-size ablation (512/1024/2048): different chunk
sizes produce entirely different chunk IDs for the same source text.

A chunk counts as a gold chunk for a passage if:

```
gold_chunk = chunk fully contains passage_span
             OR overlap_chars(chunk_span, passage_span) / len(passage_span) >= 0.5
```

This criterion is **frozen for the project's lifetime** (spec's own instruction — do
not change it mid-project). Any future change is a version bump of every
already-resolved cache file under `eval/golden/resolved/`, not a silent edit.

A `passage_text` that fails to resolve to exactly one span in its source `.md` is a
**hard error**, never a skip (invariant 4) — either it's not found (typo/drift) or
it's found more than once (extend the passage until it's unique within the document).

## `verified_by` lifecycle

- `"draft"` — Claude-authored, not yet reviewed by a human.
- `"human"` — the user has read this exact item (question, gold_answer,
  gold_passages, category/route/difficulty labels) and approved it, or edited it
  themselves. Substantial rewrites still land as `"human"` once the user is
  satisfied — the diff is visible in git history, no separate `"human_edited"`
  value is tracked.

**This is item-level**, distinct from `artifacts/metadata.jsonl`'s `verified_by`
field, which is **document-level** (whether title/authors/year were hand-checked
against the source PDF, per spec §5.1). Don't conflate the two files' semantics.

`eval/validate_golden.py --require-verified` hard-fails if any item is not
`"human"` — this is the Phase 1 freeze gate implementing the acceptance criterion
"100% hand-verified." Off by default so validation can run throughout drafting.

## Authoring pitfalls

- **Copy `passage_text` from the raw `.md` source file directly**, never from a
  rendered markdown viewer or by retyping. A rendered view can silently reflow
  whitespace/newlines in a way that breaks the literal substring match the
  resolver performs. Open the `.md` file itself and copy a contiguous span.
- Extend a passage until it is unique in its document — the resolver hard-fails on
  ambiguous matches rather than guessing.
- `gold_doc_ids` is derived from `gold_passages`, not independently chosen — don't
  hand-pick it separately or it can drift out of sync (validator checks this).

## Worked example

```json
{
  "id": "gs_0001",
  "question": "What memory fragmentation problem does PagedAttention solve?",
  "gold_answer": "PagedAttention addresses internal and external fragmentation of the KV cache by managing memory in fixed-size blocks analogous to OS virtual memory paging, rather than requiring contiguous memory for each request's KV cache.",
  "gold_passages": [
    {
      "doc_id": "vllm_paged_attention",
      "passage_text": "verbatim supporting text copied from the .md, long enough to be unique"
    }
  ],
  "gold_doc_ids": ["vllm_paged_attention"],
  "category": "single_hop",
  "expected_route": "vectorstore",
  "difficulty": "easy",
  "requires_multimodal": false,
  "notes": "answer spans a chunk boundary at chunk_size=1024",
  "verified_by": "human",
  "version": 1
}
```
