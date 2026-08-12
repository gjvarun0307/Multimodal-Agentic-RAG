"""Unit tests for eval/metrics/router.py's accuracy/confusion-matrix/
misroute-cost computation. Pure functions over (expected_route, first_node)
pairs -- no graph execution, no LLM.
"""

from eval.metrics.router import (
    RouterItem,
    compute_router_metrics,
    expected_route_label,
    predicted_route_label,
)


def test_predicted_route_label_maps_known_nodes():
    assert predicted_route_label("retrieve_and_rerank") == "vectorstore"
    assert predicted_route_label("web_search") == "websearch"
    assert predicted_route_label("chitchat") == "chitchat"


def test_predicted_route_label_none_for_no_nodes():
    assert predicted_route_label(None) is None


def test_predicted_route_label_unknown_for_unrecognized_node():
    assert predicted_route_label("some_new_node") == "unknown"


def test_expected_route_label_maps_refuse_to_vectorstore():
    assert expected_route_label("refuse") == "vectorstore"
    assert expected_route_label("vectorstore") == "vectorstore"
    assert expected_route_label("websearch") == "websearch"
    assert expected_route_label("chitchat") == "chitchat"


def test_compute_router_metrics_perfect_accuracy():
    items = [
        RouterItem(id="gs_0001", expected_route="vectorstore", first_node="retrieve_and_rerank"),
        RouterItem(id="gs_0002", expected_route="websearch", first_node="web_search"),
        RouterItem(id="gs_0003", expected_route="chitchat", first_node="chitchat"),
        RouterItem(id="gs_0004", expected_route="refuse", first_node="retrieve_and_rerank"),
    ]
    result = compute_router_metrics(items)
    assert result["n_items"] == 4
    assert result["accuracy"] == 1.0
    assert result["n_misrouted"] == 0
    assert result["confusion_matrix"]["vectorstore"]["vectorstore"] == 2  # incl. the refuse item
    assert result["confusion_matrix"]["websearch"]["websearch"] == 1
    assert result["confusion_matrix"]["chitchat"]["chitchat"] == 1


def test_compute_router_metrics_misroute_recorded_in_confusion_matrix():
    items = [
        RouterItem(id="gs_0001", expected_route="vectorstore", first_node="web_search"),  # misrouted
        RouterItem(id="gs_0002", expected_route="chitchat", first_node="chitchat"),
    ]
    result = compute_router_metrics(items)
    assert result["accuracy"] == 0.5
    assert result["n_misrouted"] == 1
    assert result["confusion_matrix"]["vectorstore"]["websearch"] == 1


def test_compute_router_metrics_misroute_cost_latency_and_tokens():
    items = [
        RouterItem(id="gs_0001", expected_route="vectorstore", first_node="retrieve_and_rerank", latency_ms=1000, tokens=200),
        RouterItem(id="gs_0002", expected_route="vectorstore", first_node="web_search", latency_ms=3000, tokens=800),
    ]
    result = compute_router_metrics(items)
    cost = result["misroute_cost"]
    assert cost["latency_ms"] == 2000.0
    assert cost["tokens"] == 600.0


def test_compute_router_metrics_misroute_cost_none_without_data():
    items = [RouterItem(id="gs_0001", expected_route="vectorstore", first_node="retrieve_and_rerank")]
    result = compute_router_metrics(items)
    assert result["misroute_cost"]["latency_ms"] is None
    assert result["misroute_cost"]["tokens"] is None


def test_compute_router_metrics_empty_items():
    result = compute_router_metrics([])
    assert result["n_items"] == 0
    assert result["accuracy"] is None
