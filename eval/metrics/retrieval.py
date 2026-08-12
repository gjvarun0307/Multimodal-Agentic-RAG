"""No-LLM retrieval metrics: Stage 1 (hybrid search, pre-rerank) and Stage 2
(post-rerank, post-threshold), reported separately per PROJECT_SPEC.md
invariant 8. Pure functions over chunk-ID lists -- no Milvus, no LLM calls,
fast and free, so this module is exercised by eval/harness.py's
--retrieval-only mode with zero API cost (PROJECT_SPEC.md §7 Phase 2
acceptance criterion: "Retrieval-only mode makes zero LLM calls").

Input contract (one RetrievalItem per golden-set item that has
gold_chunk_ids):
    id: str
    gold_chunk_ids: list[str]           -- union across gold_passages, from
        eval/resolve_passages.py's resolved/<config_hash>.json cache
        ("gold_chunk_ids_union"), never derived independently here.
    retrieved_chunk_ids: list[str]      -- Stage 1: hybrid search, pre-rerank,
        GraphState["retrieved_chunk_ids"] (already truncated to search_limit).
    retrieved_chunk_scores: list[float] -- reranker score aligned to
        retrieved_chunk_ids, pre-threshold; [] if no reranker ran for this item.
    reranked_chunk_ids: list[str]       -- Stage 2: post-rerank, post-threshold,
        GraphState["reranked_chunk_ids"] (bounded by reranker_top_k).

Items with expected_route != "vectorstore", or with no gold_passages (refuse /
websearch / chitchat items -- see eval/golden/SCHEMA.md), carry no
gold_chunk_ids and must be excluded by the caller before these functions see
them; there is no retrieval ground truth to score them against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class RetrievalItem:
    id: str
    gold_chunk_ids: Sequence[str]
    retrieved_chunk_ids: Sequence[str]
    reranked_chunk_ids: Sequence[str]
    retrieved_chunk_scores: Sequence[float] = field(default_factory=list)


# --------------------------------------------------------------------------
# Per-item primitives
# --------------------------------------------------------------------------


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> Optional[float]:
    """Fraction of gold_ids present in the top k of retrieved_ids.
    None (not 0.0) when gold_ids is empty -- an item with no gold chunks
    is not a zero-recall failure, it's undefined and must be excluded
    from the mean, never silently counted as a miss."""
    if not gold_ids:
        return None
    gold_set = set(gold_ids)
    top_k = set(retrieved_ids[:k])
    return len(gold_set & top_k) / len(gold_set)


def reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> Optional[float]:
    """1 / (rank of first gold hit, 1-indexed); 0.0 if no gold id appears
    anywhere in retrieved_ids; None if gold_ids is empty (undefined)."""
    if not gold_ids:
        return None
    gold_set = set(gold_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> Optional[float]:
    """Binary-relevance nDCG@k (relevance=1 for any gold chunk, 0 otherwise).
    None if gold_ids is empty."""
    if not gold_ids:
        return None
    gold_set = set(gold_ids)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, cid in enumerate(retrieved_ids[:k], start=1) if cid in gold_set)
    n_relevant_possible = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_relevant_possible + 1))
    return dcg / idcg if idcg > 0 else 0.0


def threshold_loss_chunks(item: RetrievalItem, *, score_threshold: float, reranker_top_k: int) -> list[str]:
    """Gold chunks the reranker scored high enough to belong in the top
    reranker_top_k by score, but the `> score_threshold` filter dropped
    before generate() ever saw them (PROJECT_SPEC.md §7 Phase 2:
    a high threshold_loss means "the threshold is miscalibrated and is a
    cheap win"). Requires retrieved_chunk_scores -- returns [] (not None)
    when scores are unavailable (no reranker ran for this item), since "no
    measurable loss" and "no reranker" are different states the caller must
    track via threshold_loss's items_scoreable, not conflate here.
    """
    if not item.retrieved_chunk_scores or len(item.retrieved_chunk_scores) != len(item.retrieved_chunk_ids):
        return []
    gold_set = set(item.gold_chunk_ids)
    if not gold_set:
        return []

    scored = list(zip(item.retrieved_chunk_ids, item.retrieved_chunk_scores))
    rank_by_score = sorted(scored, key=lambda pair: pair[1], reverse=True)
    would_rank_in_top_k = {cid for cid, _ in rank_by_score[:reranker_top_k]}
    reranked_set = set(item.reranked_chunk_ids)

    lost = []
    for cid, score in scored:
        if cid in gold_set and cid in would_rank_in_top_k and score <= score_threshold and cid not in reranked_set:
            lost.append(cid)
    return lost


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def compute_retrieval_metrics(
    items: Sequence[RetrievalItem],
    *,
    stage1_ks: Sequence[int] = (10, 20, 50),
    stage2_ks: Sequence[int] = (1, 3, 5),
    ndcg_k: int = 10,
    score_threshold: float = 0.5,
    reranker_top_k: int = 5,
) -> dict:
    """Compute Stage 1 (hybrid search, pre-rerank) and Stage 2 (post-rerank,
    post-threshold) retrieval metrics, reported separately (invariant 8) --
    a single recall@10 is meaningless once the reranker keeps only
    reranker_top_k documents, so Stage 2 recall is only ever measured up to
    that bound (PROJECT_SPEC.md §7 Phase 2). `items` should already be
    filtered to those with non-empty gold_chunk_ids (module docstring); any
    that slip through with empty gold_chunk_ids are excluded here too, not
    zero-scored, so n_items_scored may be lower than len(items).
    """
    scoreable = [it for it in items if it.gold_chunk_ids]
    n_scoreable = len(scoreable)

    stage1: dict = {}
    for k in stage1_ks:
        values = [
            v for v in (recall_at_k(it.retrieved_chunk_ids, it.gold_chunk_ids, k) for it in scoreable) if v is not None
        ]
        stage1[f"recall@{k}"] = _mean(values)
    stage1["mrr"] = _mean(
        [v for v in (reciprocal_rank(it.retrieved_chunk_ids, it.gold_chunk_ids) for it in scoreable) if v is not None]
    )
    stage1[f"ndcg@{ndcg_k}"] = _mean(
        [v for v in (ndcg_at_k(it.retrieved_chunk_ids, it.gold_chunk_ids, ndcg_k) for it in scoreable) if v is not None]
    )
    # Stage 1's top-5, for rerank_lift below -- computed against the exact
    # hybrid-search ranking, not resorted, so the comparison is apples-to-apples.
    stage1_recall_at_5 = _mean(
        [v for v in (recall_at_k(it.retrieved_chunk_ids, it.gold_chunk_ids, 5) for it in scoreable) if v is not None]
    )

    stage2: dict = {}
    for k in stage2_ks:
        values = [
            v for v in (recall_at_k(it.reranked_chunk_ids, it.gold_chunk_ids, k) for it in scoreable) if v is not None
        ]
        stage2[f"recall@{k}"] = _mean(values)
    stage2["mrr"] = _mean(
        [v for v in (reciprocal_rank(it.reranked_chunk_ids, it.gold_chunk_ids) for it in scoreable) if v is not None]
    )

    stage2_recall_at_5 = stage2.get("recall@5")
    rerank_lift = (
        stage2_recall_at_5 - stage1_recall_at_5
        if stage2_recall_at_5 is not None and stage1_recall_at_5 is not None
        else None
    )

    scored_for_threshold = [it for it in scoreable if it.retrieved_chunk_scores]
    n_scored_for_threshold = len(scored_for_threshold)
    lost_counts = [
        len(threshold_loss_chunks(it, score_threshold=score_threshold, reranker_top_k=reranker_top_k))
        for it in scored_for_threshold
    ]
    items_with_loss = sum(1 for n in lost_counts if n > 0)
    threshold_loss = {
        "items_affected": items_with_loss,
        "items_scoreable": n_scored_for_threshold,
        "rate": (items_with_loss / n_scored_for_threshold) if n_scored_for_threshold else None,
        "total_chunks_lost": sum(lost_counts),
    }

    return {
        "n_items_scored": n_scoreable,
        "stage1": stage1,
        "stage2": stage2,
        "rerank_lift": rerank_lift,
        "threshold_loss": threshold_loss,
    }
