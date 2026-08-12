"""Unit tests for eval/metrics/retrieval.py's pure chunk-ID-list metrics.
No Milvus, no LLM -- synthetic chunk IDs only, matching this module's own
"no-LLM, fast, free" framing (PROJECT_SPEC.md §7 Phase 2).
"""

import math

from eval.metrics.retrieval import (
    RetrievalItem,
    compute_retrieval_metrics,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    threshold_loss_chunks,
)


def test_recall_at_k_basic():
    assert recall_at_k(["a", "b", "c"], ["b"], k=2) == 1.0
    assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
    assert recall_at_k(["a", "b"], ["b", "d"], k=2) == 0.5


def test_recall_at_k_empty_gold_is_none_not_zero():
    assert recall_at_k(["a", "b"], [], k=5) is None


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == 1.0 / 3
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0
    assert reciprocal_rank(["a"], []) is None


def test_ndcg_at_k_perfect_and_zero():
    assert ndcg_at_k(["a", "b"], ["a"], k=2) == 1.0
    assert ndcg_at_k(["x", "y"], ["a"], k=2) == 0.0
    # gold hit at rank 2 discounts vs. rank 1
    ndcg_rank2 = ndcg_at_k(["x", "a"], ["a"], k=2)
    assert math.isclose(ndcg_rank2, 1.0 / math.log2(3))


def test_threshold_loss_chunks_flags_dropped_gold_chunk():
    item = RetrievalItem(
        id="gs_0001",
        gold_chunk_ids=["doc::0001::aaaa"],
        retrieved_chunk_ids=["doc::0001::aaaa", "doc::0002::bbbb", "doc::0003::cccc"],
        retrieved_chunk_scores=[0.4, 0.9, 0.8],  # gold chunk scores below threshold
        reranked_chunk_ids=["doc::0002::bbbb", "doc::0003::cccc"],
    )
    lost = threshold_loss_chunks(item, score_threshold=0.5, reranker_top_k=5)
    assert lost == ["doc::0001::aaaa"]


def test_threshold_loss_chunks_no_scores_returns_empty_not_none():
    item = RetrievalItem(
        id="gs_0002",
        gold_chunk_ids=["doc::0001::aaaa"],
        retrieved_chunk_ids=["doc::0001::aaaa"],
        retrieved_chunk_scores=[],
        reranked_chunk_ids=["doc::0001::aaaa"],
    )
    assert threshold_loss_chunks(item, score_threshold=0.5, reranker_top_k=5) == []


def test_threshold_loss_chunks_gold_chunk_outside_rerank_topk_not_counted():
    # gold chunk scores above threshold but ranks 6th -- a RET_DEMOTED case,
    # not a threshold-calibration issue, so it must not be counted here.
    item = RetrievalItem(
        id="gs_0003",
        gold_chunk_ids=["doc::gold::0000"],
        retrieved_chunk_ids=["doc::gold::0000"] + [f"doc::filler::{i:04d}" for i in range(6)],
        retrieved_chunk_scores=[0.6] + [0.95] * 6,
        reranked_chunk_ids=[f"doc::filler::{i:04d}" for i in range(5)],
    )
    lost = threshold_loss_chunks(item, score_threshold=0.5, reranker_top_k=5)
    assert lost == []


def test_compute_retrieval_metrics_excludes_items_without_gold():
    items = [
        RetrievalItem(id="gs_0001", gold_chunk_ids=["a"], retrieved_chunk_ids=["a"], reranked_chunk_ids=["a"]),
        RetrievalItem(id="gs_0002", gold_chunk_ids=[], retrieved_chunk_ids=["z"], reranked_chunk_ids=["z"]),
    ]
    result = compute_retrieval_metrics(items)
    assert result["n_items_scored"] == 1
    assert result["stage1"]["recall@10"] == 1.0
    assert result["stage2"]["recall@5"] == 1.0


def test_compute_retrieval_metrics_rerank_lift_can_be_negative():
    # Stage 1 already has the gold chunk in the raw top 5; the reranker
    # demotes it out of its own top 5 entirely -- rerank_lift must go negative,
    # not clamp at zero, per PROJECT_SPEC.md ("it can legitimately be negative,
    # which is exactly what you want to find out").
    items = [
        RetrievalItem(
            id="gs_0001",
            gold_chunk_ids=["gold"],
            retrieved_chunk_ids=["gold", "b", "c", "d", "e"],
            reranked_chunk_ids=["b", "c", "d", "e", "f"],
        )
    ]
    result = compute_retrieval_metrics(items)
    assert result["rerank_lift"] == -1.0


def test_compute_retrieval_metrics_threshold_loss_aggregation():
    items = [
        RetrievalItem(
            id="gs_0001",
            gold_chunk_ids=["gold"],
            retrieved_chunk_ids=["gold", "b"],
            retrieved_chunk_scores=[0.3, 0.9],
            reranked_chunk_ids=["b"],
        ),
        RetrievalItem(
            id="gs_0002",
            gold_chunk_ids=["gold2"],
            retrieved_chunk_ids=["gold2"],
            retrieved_chunk_scores=[0.9],
            reranked_chunk_ids=["gold2"],
        ),
    ]
    result = compute_retrieval_metrics(items, score_threshold=0.5, reranker_top_k=5)
    tl = result["threshold_loss"]
    assert tl["items_scoreable"] == 2
    assert tl["items_affected"] == 1
    assert tl["total_chunks_lost"] == 1
    assert tl["rate"] == 0.5
