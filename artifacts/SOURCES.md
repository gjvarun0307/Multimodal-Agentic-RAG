# Corpus sources and licenses

15 papers, frozen for the duration of this upgrade (`PROJECT_SPEC.md` invariant 2).
Identity fields (`doc_id`, title, authors, year, license) are hand-authored in
`artifacts/corpus_seed.csv`; `deploy/build_metadata.py` derives `artifacts/metadata.jsonl`
from it plus each `.md` file's content hash.

## Not committed: source PDFs or parsed full text

Neither `data/raw_pdfs/*.pdf` nor `artifacts/parsed_md/*.md` is committed to this
repository (`.gitignore`: `data`, `artifacts/parsed_md/`). Licenses vary per paper (see
table below) — several only carry arXiv's default non-exclusive-to-arXiv license, which
doesn't itself authorize third-party redistribution of the full text, and one (Milvus)
carries an ACM notice that explicitly requires permission for server-posting beyond
personal/classroom use. Rather than sorting papers into "committed" vs. "fetched" by
license, the corpus is uniformly not committed — simpler than a per-paper split, and it
means the repo's licensing posture doesn't depend on getting that classification right
for all 15.

`artifacts/metadata.jsonl` and `artifacts/corpus_seed.csv` stay committed: bibliographic
metadata (title, authors, year, topic, arXiv ID) and content hashes, not the papers'
substantial expression.

**Reproducing the corpus locally:** the PDFs and parsed markdown must exist on disk
(`data/raw_pdfs/*.pdf`, `artifacts/parsed_md/*.md`) for `pytest`, `hybrid_database.py`,
and the app to work. `tests/test_metadata_alignment.py` and
`tests/test_chunk_id_determinism.py` skip (not fail) when `artifacts/parsed_md/` is
empty or absent — a fresh clone or CI run without the corpus present will see 13 skips,
not failures. Today that means re-running the existing `parse.py` pipeline
(LlamaParse + Qwen2.5-VL captioning) against locally-held PDFs; a `deploy/fetch_corpus.py`
that re-downloads the PDFs fresh from arXiv is listed in the target directory structure
(`PROJECT_SPEC.md` §6) but not yet built — Phase 5 (ingest/deploy) territory, not Phase 0.

## Per-paper licenses

License strings below were read directly from each paper's arXiv abstract page HTML
(the `license` field, not paraphrased) on 2026-08-11, except Milvus (no arXiv posting —
SIGMOD '21 / ACM DOI). "Full-text redistribution clearly permitted?" reflects the arXiv
license alone; it does not resolve PDF-embedded publisher notices layered on top (noted
separately below where they exist).

| doc_id | Title | License | Full-text redistribution clearly permitted? |
|---|---|---|---|
| adaptive_rag | Adaptive-RAG | CC0 1.0 (public domain dedication) | Yes |
| attention_is_all_you_need | Attention Is All You Need | arXiv non-exclusive (no CC grant) | No |
| chain_of_thought_prompting | Chain-of-Thought Prompting Elicits Reasoning in LLMs | CC BY 4.0 | Yes (attribution required) |
| colbertv2 | ColBERTv2 | CC BY 4.0 | Yes (attribution required) |
| crag | Corrective Retrieval Augmented Generation | arXiv non-exclusive (no CC grant) | No |
| flashattention | FlashAttention | arXiv non-exclusive (no CC grant) | No |
| graphrag | From Local to Global: A GraphRAG Approach | CC BY 4.0 | Yes (attribution required) |
| llama3 | The Llama 3 Herd of Models | arXiv non-exclusive (no CC grant) | No |
| lora | LoRA | arXiv non-exclusive (no CC grant) | No |
| milvus | Milvus: A Purpose-Built Vector Data Management System | ACM SIGMOD '21 permission notice | No (server-posting needs permission) |
| hyde | Precise Zero-Shot Dense Retrieval without Relevance Labels | arXiv non-exclusive (no CC grant) | No |
| rag_knowledge_intensive | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | arXiv non-exclusive (no CC grant) | No |
| self_rag | Self-RAG | CC BY 4.0 | Yes (attribution required) |
| llava | Visual Instruction Tuning | CC BY 4.0 | Yes (attribution required) |
| vllm_paged_attention | Efficient Memory Management for LLM Serving with PagedAttention | CC BY 4.0 | Yes (attribution required) |

## Notices beyond the arXiv license field

- **attention_is_all_you_need** — the PDF's first page carries a red banner: *"Provided
  proper attribution is provided, Google hereby grants permission to reproduce the
  tables and figures in this paper solely for use in journalistic or scholarly works."*
  Narrower than full-text redistribution on both axes (figures/tables only; journalistic
  or scholarly use only) — doesn't change the "No" above.
- **milvus** — ACM notice on the PDF's first page: *"Permission to make digital or hard
  copies of all or part of this work for personal or classroom use is granted without
  fee... To copy otherwise, or republish, to post on servers or to redistribute to
  lists, requires prior specific permission and/or a fee."* Explicit permission
  requirement for server-posting.
- **vllm_paged_attention** — the PDF's first page (SOSP '23) carries a similar ACM-style
  notice (*"Permission to make digital or hard copies of part or all of this work for
  personal or classroom use is granted without fee... For all other uses, contact the
  owner/author(s)"*), but the paper's arXiv posting is explicitly licensed CC BY 4.0 —
  the author's own selection, and the stronger signal of their licensing intent for that
  specific posting. Treated as CC BY 4.0 above; noted for transparency rather than
  treated as a conflict.

## Attribution

For the 7 CC BY 4.0 papers, attribution is: title, authors (from `corpus_seed.csv`),
and a link to the arXiv abstract page (`https://arxiv.org/abs/<arxiv_id>`) wherever the
paper is cited in generated output, the README, or any published eval artifact.
