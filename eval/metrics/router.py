"""Router accuracy, confusion matrix, and misroute cost (PROJECT_SPEC.md §7
Phase 2). Compares the graph's actual first-node choice --
trace_info["node_sequence"][0] from agent.run_query_with_state(), per
CLAUDE.md "Implementation notes: query_router is a conditional-edge
selector, its choice isn't itself written into GraphState" -- against each
golden-set item's `expected_route` (eval/golden/SCHEMA.md).

query_router is a conditional-edge selector with exactly three literal
choices (RouteDecision.route: "vectorstore" | "websearch" | "chitchat",
src/agent.py) -- there is no fourth "refuse" edge. eval/golden/SCHEMA.md's
`expected_route: "refuse"` (unanswerable_refuse items) describes the
*generation*-level expected outcome, not a router edge --
eval/metrics/generation.py's refusal_accuracy measures that. For
router_accuracy purposes only, a "refuse" item's correct router target is
"vectorstore": these questions read as in-domain (the router should still
send them to retrieval), and it's retrieval + hallucination/relevance
grading that must then produce a refusal, not the router. This mapping
(EXPECTED_ROUTE_FOR_ROUTER) is the single place that decision lives --
don't duplicate it elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

NODE_TO_ROUTE_LABEL = {
    "retrieve_and_rerank": "vectorstore",
    "web_search": "websearch",
    "chitchat": "chitchat",
}

# See module docstring: "refuse" is a generation-level expectation, not a
# router edge -- its router target is vectorstore.
EXPECTED_ROUTE_FOR_ROUTER = {
    "refuse": "vectorstore",
}

ROUTE_LABELS = ("vectorstore", "websearch", "chitchat")
UNKNOWN_LABEL = "unknown"


@dataclass
class RouterItem:
    id: str
    expected_route: str  # raw eval/golden/SCHEMA.md value, including "refuse"
    first_node: Optional[str]  # trace_info["node_sequence"][0]; None if the run produced no nodes
    latency_ms: Optional[float] = None  # total latency for this item, if the caller tracked it
    tokens: Optional[float] = None  # total tokens for this item, if the caller tracked it


def predicted_route_label(first_node: Optional[str]) -> Optional[str]:
    """None only when first_node itself is None (no nodes ran). An
    unrecognized (but non-None) node name maps to UNKNOWN_LABEL rather than
    raising -- a genuinely new node name showing up here is a signal this
    module's mapping is stale, and it should surface as a visible
    "unknown" bucket in the confusion matrix, not crash the whole run."""
    if first_node is None:
        return None
    return NODE_TO_ROUTE_LABEL.get(first_node, UNKNOWN_LABEL)


def expected_route_label(expected_route: str) -> str:
    return EXPECTED_ROUTE_FOR_ROUTER.get(expected_route, expected_route)


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _empty_confusion_matrix() -> dict[str, dict[str, int]]:
    return {actual: {predicted: 0 for predicted in ROUTE_LABELS + (UNKNOWN_LABEL,)} for actual in ROUTE_LABELS}


def compute_router_metrics(items: Sequence[RouterItem]) -> dict:
    n = len(items)
    confusion = _empty_confusion_matrix()

    correct_latencies: list[float] = []
    misroute_latencies: list[float] = []
    correct_tokens: list[float] = []
    misroute_tokens: list[float] = []
    n_correct = 0

    for item in items:
        expected = expected_route_label(item.expected_route)
        predicted = predicted_route_label(item.first_node) or UNKNOWN_LABEL

        row = confusion.setdefault(expected, {p: 0 for p in ROUTE_LABELS + (UNKNOWN_LABEL,)})
        row[predicted] = row.get(predicted, 0) + 1

        is_correct = predicted == expected
        if is_correct:
            n_correct += 1
        if item.latency_ms is not None:
            (correct_latencies if is_correct else misroute_latencies).append(item.latency_ms)
        if item.tokens is not None:
            (correct_tokens if is_correct else misroute_tokens).append(item.tokens)

    mean_correct_latency = _mean(correct_latencies)
    mean_misroute_latency = _mean(misroute_latencies)
    mean_correct_tokens = _mean(correct_tokens)
    mean_misroute_tokens = _mean(misroute_tokens)

    return {
        "n_items": n,
        "n_misrouted": n - n_correct,
        "accuracy": (n_correct / n) if n else None,
        "confusion_matrix": confusion,
        "misroute_cost": {
            "latency_ms": (
                mean_misroute_latency - mean_correct_latency
                if mean_misroute_latency is not None and mean_correct_latency is not None
                else None
            ),
            "tokens": (
                mean_misroute_tokens - mean_correct_tokens
                if mean_misroute_tokens is not None and mean_correct_tokens is not None
                else None
            ),
        },
    }
