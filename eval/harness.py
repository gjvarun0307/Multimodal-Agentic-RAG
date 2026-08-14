"""python -m eval.harness --config configs/default.yaml [--split full|fast|dev] [--retrieval-only]

Headless, config-driven eval run -> a results JSON matching the schema in
PROJECT_SPEC.md §7 Phase 2. See that section for the full field list and
the acceptance criteria this module targets.

Retrieval-only mode (--retrieval-only) makes zero LLM calls: no LLM client
is constructed at all -- only database/embedding_model/reranker via
src.hybrid_database.load_or_create_database + src.configuration.
build_reranker, calling src.agent.retrieve_and_rerank_core directly, the
exact retrieval code path the real graph uses (see that function's
docstring for why it isn't reimplemented here). Only vectorstore-routed
items with resolved gold_chunk_ids are scored; router/generation/
structured/system metrics are all absent, never fabricated, since they
need either the LLM-driven route decision or a real generation.

Full mode builds the whole graph via get_runtime() + build_agent_graph(),
wraps the LLM with InstrumentingLLM (structured-output instrumentation,
eval/metrics/structured.py) and the web-search tool with
eval.tavily_cache.build_tavily_tool() (frozen fixture replay by default --
TAVILY_MODE=live only for the explicit recording pass), and grades
eligible items with the pinned Groq judge (eval/judge.py). If no
judge_api_key is configured, generation metrics are skipped for the run
with a loud warning -- never silently faked.

Warm-up: the first WARMUP_N items of whichever split is running execute
once, through a freshly built graph/recorder, and are discarded; the SAME
items are then re-run for real as part of the normal loop.
warmup_excluded=true is recorded in the results JSON either way
(PROJECT_SPEC.md §4B.3).

Known gap: correction_improve_rate/degrade_rate (PROJECT_SPEC.md's
"Self-RAG correction" metrics) are NOT computed by this harness yet --
they need the pre-correction generation text, which GraphState currently
overwrites rather than preserves across a rewrite_query loop iteration.
eval.metrics.system.compute_correction_metrics still reports
correction_fire_rate and mean_retries (trace-derivable alone); improve/
degrade stay None, not a fabricated number. Preserving generation history
across loop iterations is a separate, not-yet-made GraphState change.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from eval.judge import (
    JUDGE_VERSION,
    JudgeConfigError,
    JudgeGradingError,
    build_judge_llm,
    citation_precision_score,
    grade_citation_precision,
    grade_correctness,
    grade_faithfulness,
    grade_refusal,
)
from eval.metrics.generation import GenerationItem, compute_generation_metrics
from eval.metrics.retrieval import RetrievalItem, compute_retrieval_metrics
from eval.metrics.router import RouterItem, compute_router_metrics, expected_route_label, predicted_route_label
from eval.metrics.structured import InstrumentingLLM, StructuredOutputRecorder, compute_structured_metrics
from eval.metrics.system import (
    SystemItem,
    compute_correction_metrics,
    compute_error_rate_by_stage,
    compute_latency_metrics,
    compute_token_and_cost_metrics,
)
from eval.tavily_cache import build_tavily_tool
from src.agent import build_agent_graph, retrieve_and_rerank_core, run_query_with_state
from src.configuration import build_reranker, config_rag
from src.helper import open_jsonl
from src.hybrid_database import load_or_create_database
from src.logging_utils import get_logger
from src.runtime import get_runtime

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = REPO_ROOT / "eval" / "golden" / "golden_set.jsonl"
DEFAULT_DEV = REPO_ROOT / "eval" / "golden" / "dev_split.jsonl"
DEFAULT_RESOLVED_DIR = REPO_ROOT / "eval" / "golden" / "resolved"
DEFAULT_RESULTS_DIR = REPO_ROOT / "eval" / "results"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

FAST_SPLIT_TARGET_N = 40
FAST_SPLIT_SEED = 20260812
WARMUP_N = 3

_REDACTED_CONFIG_KEYS = ("llm_api_key", "judge_api_key", "tavilly_api_key")


def _git_sha() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2)
        return result.stdout.strip() or None
    except Exception:
        return None


def _redact_config(config: dict) -> dict:
    """Never persist raw secrets into a results JSON, even one that's
    gitignored -- results get pasted, shared, and uploaded in ways a repo
    file doesn't."""
    redacted = dict(config)
    for key in _REDACTED_CONFIG_KEYS:
        if redacted.get(key):
            redacted[key] = "***redacted***"
    return redacted


def load_harness_config(path: Path) -> tuple[dict, dict]:
    """Loads a configs/*.yaml file. Returns (yaml_doc, resolved_config) --
    yaml_doc's top-level fields (golden_set_version, judge_model, backend,
    ...) are eval-run provenance, read directly by run_eval(); resolved_config
    is yaml_doc["overrides"] applied on top of config_rag()'s normal
    defaults -> api_keys.json -> env resolution (CLAUDE.md "Config-only
    overrides" -- the mechanism the harness and ablation runner both use).
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    resolved = config_rag(overrides=doc.get("overrides", {}))
    return doc, resolved


def _load_items(split: str, *, golden_path: Path = DEFAULT_GOLDEN, dev_path: Path = DEFAULT_DEV) -> list[dict]:
    if split == "dev":
        return open_jsonl(dev_path)
    golden_items = open_jsonl(golden_path)
    if split == "full":
        return golden_items
    if split == "fast":
        return stratified_fast_subset(golden_items)
    raise ValueError(f"Unknown split {split!r}; expected 'full', 'fast', or 'dev'.")


def stratified_fast_subset(
    items: list[dict], *, target_n: int = FAST_SPLIT_TARGET_N, seed: int = FAST_SPLIT_SEED
) -> list[dict]:
    """Deterministic, proportional-per-category sample of golden_set.jsonl
    for CI's fast tier (PROJECT_SPEC.md §7 Phase 4: "~40-item stratified
    subset"). Never samples dev_split.jsonl -- invariant 6 reserves the dev
    split for tuning only, and "fast" numbers ARE reported (gating every
    push), just computed over fewer items than the nightly full run."""
    by_category: dict[str, list[dict]] = {}
    for item in items:
        by_category.setdefault(item["category"], []).append(item)

    rng = random.Random(seed)
    total = len(items)
    subset: list[dict] = []
    for category in sorted(by_category):
        cat_items = by_category[category]
        share = len(cat_items) / total
        n_take = min(len(cat_items), max(1, round(share * target_n)))
        sampled = sorted(cat_items, key=lambda it: it["id"])
        rng.shuffle(sampled)
        subset.extend(sampled[:n_take])
    subset.sort(key=lambda it: it["id"])
    return subset


def find_resolved_cache(resolved_dir: Path, *, chunk_size: int, overlap_size: int) -> dict:
    """The harness never resolves gold passages itself -- it only reads
    the cache eval/resolve_passages.py already wrote (PROJECT_SPEC.md
    §5.2.1: "Cache resolved chunk IDs per config ... so resolution runs
    once, not every eval"). Matches on the cache files' own recorded
    chunk_size/overlap_size rather than recomputing the config_hash
    independently, so there's only one place that hash is computed."""
    candidates = []
    if resolved_dir.exists():
        for path in sorted(resolved_dir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            if doc.get("chunk_size") == chunk_size and doc.get("overlap_size") == overlap_size:
                candidates.append(doc)
    if not candidates:
        raise FileNotFoundError(
            f"No resolved gold-chunk-id cache for chunk_size={chunk_size}, overlap_size={overlap_size} in "
            f"{resolved_dir}. Run `python -m eval.resolve_passages` first -- the harness never resolves "
            "passages itself, only reads the cache (PROJECT_SPEC.md §5.2.1)."
        )
    # Corpus is supposed to be frozen, so >1 match should only happen if
    # resolve_passages ran twice at the same chunk_size/overlap_size --
    # defensive tie-break on the most recent, not an expected case.
    candidates.sort(key=lambda doc: doc.get("generated_at", ""))
    return candidates[-1]


def gold_chunk_ids_by_item(resolved_cache: dict) -> dict[str, list[str]]:
    return {
        item_id: entry.get("gold_chunk_ids_union", []) for item_id, entry in resolved_cache.get("items", {}).items()
    }


def _format_documents(documents: list) -> str:
    """Mirrors build_agent_graph()'s private _format_documents closure.
    Duplicated (not extracted like retrieve_and_rerank_core) because it's
    pure text formatting with no decision logic to drift -- only used to
    build judge context here, not to reproduce a measured code path."""
    if not documents:
        return "No context available."
    if isinstance(documents[0], dict):
        return "\n\n".join(str(doc.get("text", "")) for doc in documents if doc.get("text"))
    return "\n\n".join(str(doc) for doc in documents if str(doc).strip())


def _run_retrieval_only(
    items: list[dict], *, database, embedding_model, rerank_model, config: dict, gold_by_item: dict[str, list[str]]
) -> list[RetrievalItem]:
    retrieval_items: list[RetrievalItem] = []
    for item in items:
        gold_ids = gold_by_item.get(item["id"], [])
        if item.get("expected_route") != "vectorstore" or not gold_ids:
            continue
        core_result = retrieve_and_rerank_core(
            item["question"],
            database=database,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
            config=config,
        )
        retrieval_items.append(
            RetrievalItem(
                id=item["id"],
                gold_chunk_ids=gold_ids,
                retrieved_chunk_ids=core_result["retrieved_chunk_ids"],
                retrieved_chunk_scores=core_result["retrieved_chunk_scores"],
                reranked_chunk_ids=core_result["reranked_chunk_ids"],
            )
        )
    return retrieval_items


def _run_full(
    items: list[dict], *, runtime, resolved_config: dict, gold_by_item: dict[str, list[str]], tavily_mode: Optional[str]
) -> dict:
    """Builds one instrumented graph, runs every item through it, grades
    eligible items with the judge, and returns per-item records ready for
    metric aggregation. A fresh graph + StructuredOutputRecorder per call
    is what keeps warm-up items' structured-output events out of the real
    run's stats -- run_eval() calls this once for warm-up (discarded) and
    once for real."""
    recorder = StructuredOutputRecorder()
    instrumented_llm = InstrumentingLLM(runtime.llm, recorder)
    tavily_tool = build_tavily_tool(resolved_config, mode=tavily_mode)
    graph = build_agent_graph(
        runtime.database,
        runtime.embedding_model,
        runtime.rerank_model,
        instrumented_llm,
        resolved_config,
        web_search_tool=tavily_tool,
    )

    try:
        judge_llm = build_judge_llm(resolved_config)
    except JudgeConfigError as e:
        judge_llm = None
        logger.warning(f"Judge unavailable, generation metrics will be skipped for this run: {e}")

    retrieval_items: list[RetrievalItem] = []
    router_items: list[RouterItem] = []
    system_items: list[SystemItem] = []
    generation_items: list[GenerationItem] = []
    misroute_context: list[dict] = []
    per_item_records: list[dict] = []

    for item in items:
        events_start = len(recorder.events)
        t0 = time.time()
        answer, final_state, trace_info = run_query_with_state(graph, item["question"], [])
        total_latency_ms = (time.time() - t0) * 1000
        item_events = recorder.events[events_start:]

        node_sequence = trace_info.get("node_sequence", [])
        first_node = node_sequence[0] if node_sequence else None

        router_items.append(
            RouterItem(
                id=item["id"], expected_route=item["expected_route"], first_node=first_node, latency_ms=total_latency_ms
            )
        )

        router_call_events = [e for e in item_events if e.node == "query_router"]
        if router_call_events:
            misroute_context.append(
                {
                    "router_valid": router_call_events[0].valid,
                    "predicted_route": predicted_route_label(first_node),
                    "expected_route": expected_route_label(item["expected_route"]),
                }
            )

        system_items.append(
            SystemItem(
                id=item["id"],
                total_latency_ms=total_latency_ms,
                stage_latencies_ms=trace_info.get("stage_latencies_ms", {}),
                node_sequence=node_sequence,
                fallback_events=trace_info.get("fallback_events", []),
            )
        )

        gold_ids = gold_by_item.get(item["id"], [])
        if item.get("expected_route") == "vectorstore" and gold_ids:
            retrieval_items.append(
                RetrievalItem(
                    id=item["id"],
                    gold_chunk_ids=gold_ids,
                    retrieved_chunk_ids=final_state.get("retrieved_chunk_ids", []),
                    retrieved_chunk_scores=final_state.get("retrieved_chunk_scores", []),
                    reranked_chunk_ids=final_state.get("reranked_chunk_ids", []),
                )
            )

        gen_item = GenerationItem(id=item["id"])
        documents = final_state.get("documents", [])
        if judge_llm is not None:
            # A single dimension's grading failure (e.g. the judge model
            # emitting unparseable output -- a real failure mode hit live
            # during judge calibration, see eval/judge.py's
            # JudgeGradingError) must not lose an item's other dimensions
            # or crash the whole run. Logged loudly, never silent
            # (invariant 15) -- the field just stays None for this item.
            if documents:
                context_text = _format_documents(documents)
                try:
                    faithfulness = grade_faithfulness(judge_llm, context=context_text, generation=answer)
                    gen_item.is_faithful = faithfulness.is_faithful
                except JudgeGradingError as e:
                    logger.warning(f"{item['id']}: faithfulness grading failed, skipping: {e}")
                try:
                    citation = grade_citation_precision(judge_llm, context=context_text, generation=answer)
                    gen_item.citation_precision = citation_precision_score(citation)
                except JudgeGradingError as e:
                    logger.warning(f"{item['id']}: citation_precision grading failed, skipping: {e}")
            if item.get("gold_answer"):
                try:
                    correctness = grade_correctness(
                        judge_llm, question=item["question"], gold_answer=item["gold_answer"], generation=answer
                    )
                    gen_item.is_correct = correctness.is_correct
                except JudgeGradingError as e:
                    logger.warning(f"{item['id']}: correctness grading failed, skipping: {e}")
            if item.get("expected_route") == "refuse":
                try:
                    refusal = grade_refusal(judge_llm, question=item["question"], generation=answer)
                    gen_item.is_refusal = refusal.is_refusal
                    gen_item.expected_refusal = True
                except JudgeGradingError as e:
                    logger.warning(f"{item['id']}: refusal grading failed, skipping: {e}")
        generation_items.append(gen_item)

        per_item_records.append(
            {
                "id": item["id"],
                "category": item.get("category"),
                "expected_route": item.get("expected_route"),
                "route_decision": first_node,
                "node_sequence": node_sequence,
                "retrieved_chunk_ids": final_state.get("retrieved_chunk_ids", []),
                "reranked_chunk_ids": final_state.get("reranked_chunk_ids", []),
                "fallback_events": trace_info.get("fallback_events", []),
                "total_latency_ms": total_latency_ms,
                "answer": answer,
            }
        )

    return {
        "retrieval_items": retrieval_items,
        "router_items": router_items,
        "system_items": system_items,
        "generation_items": generation_items,
        "structured_events": list(recorder.events),
        "misroute_context": misroute_context,
        "per_item_records": per_item_records,
        "judge_used": judge_llm is not None,
    }


def run_eval(
    *,
    config_path: Path = DEFAULT_CONFIG,
    split: str = "full",
    retrieval_only: bool = False,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    resolved_dir: Path = DEFAULT_RESOLVED_DIR,
    tavily_mode: Optional[str] = None,
) -> tuple[dict, Path]:
    doc, resolved_config = load_harness_config(config_path)
    items = _load_items(split)
    if not items:
        raise ValueError(f"No items for split={split!r}")

    resolved_cache = find_resolved_cache(
        resolved_dir, chunk_size=resolved_config["chunk_size"], overlap_size=resolved_config["overlap_size"]
    )
    gold_by_item = gold_chunk_ids_by_item(resolved_cache)

    n_warmup = min(WARMUP_N, len(items))
    warmup_items = items[:n_warmup]

    if retrieval_only:
        database, embedding_model = load_or_create_database(resolved_config)
        rerank_model = build_reranker(resolved_config)

        _run_retrieval_only(
            warmup_items,
            database=database,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
            config=resolved_config,
            gold_by_item=gold_by_item,
        )
        retrieval_items = _run_retrieval_only(
            items,
            database=database,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
            config=resolved_config,
            gold_by_item=gold_by_item,
        )

        metrics: dict[str, Any] = {
            "retrieval": compute_retrieval_metrics(
                retrieval_items,
                score_threshold=resolved_config.get("reranker_score_threshold", 0.5),
                reranker_top_k=resolved_config.get("reranker_top_k", 5),
            )
        }
        per_item = [
            {
                "id": it.id,
                "gold_chunk_ids": list(it.gold_chunk_ids),
                "retrieved_chunk_ids": list(it.retrieved_chunk_ids),
                "reranked_chunk_ids": list(it.reranked_chunk_ids),
            }
            for it in retrieval_items
        ]
        backend = "retrieval-only"
        judge_used = False
    else:
        runtime = get_runtime(resolved_config)
        _run_full(
            warmup_items,
            runtime=runtime,
            resolved_config=resolved_config,
            gold_by_item=gold_by_item,
            tavily_mode=tavily_mode,
        )
        run_result = _run_full(
            items, runtime=runtime, resolved_config=resolved_config, gold_by_item=gold_by_item, tavily_mode=tavily_mode
        )

        metrics = {
            "retrieval": compute_retrieval_metrics(
                run_result["retrieval_items"],
                score_threshold=resolved_config.get("reranker_score_threshold", 0.5),
                reranker_top_k=resolved_config.get("reranker_top_k", 5),
            ),
            "router": compute_router_metrics(run_result["router_items"]),
            "structured": compute_structured_metrics(
                run_result["structured_events"], misroute_context=run_result["misroute_context"]
            ),
            "generation": compute_generation_metrics(run_result["generation_items"]),
            "system": {
                **compute_latency_metrics(run_result["system_items"]),
                **compute_token_and_cost_metrics(run_result["system_items"]),
                "error_rate_by_stage": compute_error_rate_by_stage(run_result["system_items"]),
            },
            # improve_rate/degrade_rate always None here -- see module docstring's "Known gap".
            "correction": compute_correction_metrics(run_result["system_items"]),
        }
        per_item = run_result["per_item_records"]
        backend = doc.get("backend", "byok-hosted")
        judge_used = run_result["judge_used"]

    results = {
        "run_id": f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{_git_sha() or 'nogit'}",
        "git_sha": _git_sha(),
        "config_path": str(config_path),
        "config": _redact_config(resolved_config),
        "split": split,
        "golden_set_version": doc.get("golden_set_version"),
        "judge_version": JUDGE_VERSION if judge_used else None,
        "judge_model": doc.get("judge_model") if judge_used else None,
        "tavily_fixture_version": doc.get("tavily_fixture_version"),
        "backend": backend,
        "warmup_excluded": True,
        "n_warmup": n_warmup,
        "n_items": len(items),
        "metrics": metrics,
        "per_item": per_item,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{results['run_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, sort_keys=True)
        f.write("\n")

    return results, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=["full", "fast", "dev"], default="full")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--tavily-mode", choices=["replay", "live"], default=None, help="Overrides TAVILY_MODE for this run."
    )
    args = parser.parse_args()

    results, out_path = run_eval(
        config_path=args.config,
        split=args.split,
        retrieval_only=args.retrieval_only,
        results_dir=args.out_dir,
        tavily_mode=args.tavily_mode,
    )
    print(f"Wrote results to {out_path}", file=sys.stderr)
    print(json.dumps(results["metrics"], indent=2, default=str))


if __name__ == "__main__":
    main()
