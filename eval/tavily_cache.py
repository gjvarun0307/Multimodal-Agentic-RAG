"""Frozen Tavily web-search fixtures for deterministic eval (PROJECT_SPEC.md
§7 Phase 2, invariant 12: "Tavily replays from frozen fixtures in every
eval/CI run -- never live."). Live web search is non-deterministic and
drifts over time, which would break reproducibility and make noise-floor
measurement (Phase 3) meaningless for any websearch-routed item.

Two modes, selected by the TAVILY_MODE env var:
  - "replay" (default) -- look up a cached response by query hash; a miss
    is a hard, loud failure (TavilyCacheMissError), never a silent
    fall-through to a live call (invariant 15).
  - "live" -- call the real Tavily API via TavilySearchAPIWrapper and
    record the response into the fixture, keyed by the same hash replay
    will use. Never used in CI or for baselines -- only for the
    one-time/occasional freshness-refresh pass run via:
        python -m eval.tavily_cache --record

build_tavily_tool() returns a drop-in replacement for
langchain_tavily.TavilySearch -- anything with a `.invoke({"query": ...})
-> dict` matching TavilySearch's own response shape -- so
src.agent.build_agent_graph()'s `web_search_tool=` override (added
alongside this module) can take either the real tool or this one with zero
changes to the web_search node itself.

The fixture is deliberately trivial to freeze (CLAUDE.md Phase 1 outcome:
only 4 `unanswerable_websearch` golden items) -- one JSON file, keyed by a
hash of (query, max_results, topic), record()'d straight from those items'
literal `question` field, since query_router routes them to web_search
directly without a query rewrite in the common case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.helper import open_jsonl

FIXTURE_VERSION = "v1"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE_PATH = REPO_ROOT / "eval" / "fixtures" / f"tavily_{FIXTURE_VERSION}.json"
DEFAULT_GOLDEN = REPO_ROOT / "eval" / "golden" / "golden_set.jsonl"
DEFAULT_DEV = REPO_ROOT / "eval" / "golden" / "dev_split.jsonl"


class TavilyCacheMissError(Exception):
    """Raised in replay mode when a query has no frozen fixture entry --
    invariant 15 (silent fallbacks forbidden): a cache miss must fail
    loudly, never fall through to a live call."""


def _query_key(query: str, *, max_results: int, topic: str) -> str:
    payload = json.dumps({"query": query, "max_results": max_results, "topic": topic}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_fixture(path: Path) -> dict:
    if not path.exists():
        return {"fixture_version": FIXTURE_VERSION, "entries": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_fixture(path: Path, fixture: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")


class ReplayTavilyTool:
    """Drop-in TavilySearch replacement that only ever reads the frozen
    fixture -- never makes a network call."""

    def __init__(self, *, max_results: int, topic: str, fixture_path: Path = DEFAULT_FIXTURE_PATH):
        self.max_results = max_results
        self.topic = topic
        self.fixture_path = fixture_path
        self._fixture = _load_fixture(fixture_path)

    def invoke(self, inputs: dict) -> dict:
        query = inputs["query"]
        key = _query_key(query, max_results=self.max_results, topic=self.topic)
        entry = self._fixture.get("entries", {}).get(key)
        if entry is None:
            raise TavilyCacheMissError(
                f"No frozen Tavily fixture entry for query {query!r} "
                f"(max_results={self.max_results}, topic={self.topic!r}, key={key}) in "
                f"{self.fixture_path}. Record it first: python -m eval.tavily_cache --record "
                "-- replay mode never falls through to a live call (invariant 15)."
            )
        return entry["response"]


class RecordingTavilyTool:
    """Live wrapper: calls the real Tavily API and writes the response into
    the fixture, keyed by the same hash ReplayTavilyTool looks up. Only
    ever constructed by build_tavily_tool() with TAVILY_MODE=live, or by
    this module's --record CLI -- never the default, never used in CI."""

    def __init__(self, *, tavily_api_key: str, max_results: int, topic: str, fixture_path: Path = DEFAULT_FIXTURE_PATH):
        from langchain_tavily import TavilySearch
        from langchain_tavily.tavily_search import TavilySearchAPIWrapper

        self.max_results = max_results
        self.topic = topic
        self.fixture_path = fixture_path
        self._fixture = _load_fixture(fixture_path)
        self._live_tool = TavilySearch(
            api_wrapper=TavilySearchAPIWrapper(tavily_api_key=tavily_api_key),
            max_results=max_results,
            topic=topic,
            include_images=False,
        )

    def invoke(self, inputs: dict) -> dict:
        query = inputs["query"]
        response = self._live_tool.invoke(inputs)
        key = _query_key(query, max_results=self.max_results, topic=self.topic)
        self._fixture.setdefault("entries", {})[key] = {
            "query": query,
            "max_results": self.max_results,
            "topic": self.topic,
            "response": response,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_fixture(self.fixture_path, self._fixture)
        return response


def build_tavily_tool(config: dict, *, mode: Optional[str] = None, fixture_path: Path = DEFAULT_FIXTURE_PATH):
    """Factory used by src.agent.build_agent_graph(web_search_tool=...) and
    eval.harness -- selects replay vs. live from TAVILY_MODE (mode=None) or
    an explicit override. Defaults to "replay" so omitting TAVILY_MODE can
    never silently reach the live API (PROJECT_SPEC.md §7 Phase 2: "Eval
    runs replay from cache by default")."""
    resolved_mode = mode or os.environ.get("TAVILY_MODE", "replay")
    max_results = config.get("web_search_max_results", 5)
    topic = config.get("web_search_topic", "general")
    if resolved_mode == "live":
        return RecordingTavilyTool(
            tavily_api_key=config["tavilly_api_key"], max_results=max_results, topic=topic, fixture_path=fixture_path
        )
    if resolved_mode != "replay":
        raise ValueError(f"Unknown TAVILY_MODE {resolved_mode!r}; expected 'replay' or 'live'.")
    return ReplayTavilyTool(max_results=max_results, topic=topic, fixture_path=fixture_path)


def _load_websearch_questions(golden_path: Path, dev_path: Path) -> list[str]:
    items: list[dict] = []
    if golden_path.exists():
        items += open_jsonl(golden_path)
    if dev_path.exists():
        items += open_jsonl(dev_path)
    return [item["question"] for item in items if item.get("expected_route") == "websearch"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--record", action="store_true", required=True, help="Make live Tavily calls and freeze responses -- costs API quota."
    )
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument(
        "--questions", type=str, default=None, help="Comma-separated extra queries to record beyond the golden set's websearch items."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    args = parser.parse_args()

    from src.configuration import config_rag

    cfg = config_rag()
    if not cfg.get("tavilly_api_key"):
        print(
            "No Tavily API key resolved (tavilly_web in api_keys.json / TAVILY_API_KEY env) -- cannot record.",
            file=sys.stderr,
        )
        sys.exit(1)

    questions = _load_websearch_questions(args.golden, args.dev)
    if args.questions:
        questions += [q.strip() for q in args.questions.split(",") if q.strip()]
    if not questions:
        print("No websearch-routed golden items and no --questions given -- nothing to record.", file=sys.stderr)
        sys.exit(1)

    tool = build_tavily_tool(cfg, mode="live", fixture_path=args.fixture)
    for question in questions:
        print(f"Recording: {question[:80]!r}", file=sys.stderr)
        tool.invoke({"query": question})

    print(f"Recorded {len(questions)} queries to {args.fixture}", file=sys.stderr)


if __name__ == "__main__":
    main()
