"""FastAPI service -- headless query path, readiness gate, and (from
Phase 5 onward) precached demo traces. Both this and app.py (Streamlit)
call get_runtime() in-process; neither calls the other over HTTP
(invariant 14, PROJECT_SPEC.md §8).

CI runs the pipeline in-process in the Actions runner, never against a
deployed instance of this service (spec §4B.3) -- this module is for the
`/query` headless path, the live demo, and the keep-warm ping, not for
gating merges.
"""

import subprocess
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .agent import build_agent_graph, run_query_with_state
from .logging_utils import get_logger
from .runtime import get_runtime

logger = get_logger(__name__)


def _git_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2
        )
        return result.stdout.strip() or None
    except Exception:
        return None


_GIT_SHA = _git_sha()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eager load at startup -- readiness should reflect real model load,
    not lazily transfer that cost onto the first real request (spec §4B.1
    item 3). A real warm-up query and explicit "warming" state land in
    Phase 5; for now readiness means get_runtime() + graph compile
    succeeded."""
    logger.info("API starting up -- constructing runtime")
    try:
        runtime = get_runtime()
        graph = build_agent_graph(runtime.database, runtime.embedding_model, runtime.rerank_model, runtime.llm)
        app.state.runtime = runtime
        app.state.graph = graph
        app.state.ready = True
        logger.info(f"API ready (reranker_enabled={runtime.rerank_model is not None})")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        app.state.runtime = None
        app.state.graph = None
        app.state.ready = False
    yield


app = FastAPI(title="Multimodal Agentic RAG API", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    chat_history: list[str] = []


class QueryResponse(BaseModel):
    question: str
    answer: str
    route_decision: Optional[str] = None
    node_sequence: list[str] = []
    retrieved_chunk_ids: list[str] = []
    reranked_chunk_ids: list[str] = []
    correction_fired: bool = False
    stage_latencies_ms: dict[str, float] = {}
    total_latency_ms: float
    git_sha: Optional[str] = None


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Headless query with a full trace: route decision, both chunk ID
    lists (pre/post-rerank), correction signal, per-stage latency. Used
    by nothing in production -- exists for eval (Phase 2) and for anyone
    reading the repo (spec §4D.1)."""
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="Runtime not ready")

    t0 = time.time()
    answer, final_state, trace_info = run_query_with_state(app.state.graph, request.question, request.chat_history)
    total_latency_ms = (time.time() - t0) * 1000

    node_sequence = trace_info["node_sequence"]
    correction_fired = "rewrite_query" in node_sequence or node_sequence.count("generate") > 1

    return QueryResponse(
        question=request.question,
        answer=answer,
        route_decision=node_sequence[0] if node_sequence else None,
        node_sequence=node_sequence,
        retrieved_chunk_ids=final_state.get("retrieved_chunk_ids", []),
        reranked_chunk_ids=final_state.get("reranked_chunk_ids", []),
        correction_fired=correction_fired,
        stage_latencies_ms=trace_info["stage_latencies_ms"],
        total_latency_ms=total_latency_ms,
        git_sha=_GIT_SHA,
    )


@app.get("/health")
def health():
    """Readiness gate -- target of the keep-warm ping (Phase 5). Real
    warm-up-query semantics land in Phase 5; for now this reflects
    whether get_runtime() + graph compile succeeded at startup."""
    if app.state.ready:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="not ready")


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus-format scrape endpoint. Minimal for Phase 0 -- real
    per-stage span/counter wiring is Phase 5 (observability/otel_setup.py)."""
    ready_value = 1 if app.state.ready else 0
    return (
        "# HELP arag_ready Whether the runtime finished loading at startup\n"
        "# TYPE arag_ready gauge\n"
        f"arag_ready {ready_value}\n"
    )


@app.get("/trace/{trace_id}")
def trace(trace_id: str):
    """Serves a Tier 1 precached demo trace (spec §4C). None recorded yet
    -- deploy/record_demo_traces.py (Phase 5) populates
    artifacts/demo_traces/; this endpoint is structured now so that phase
    only adds data, not code."""
    raise HTTPException(status_code=404, detail=f"No precached trace {trace_id!r} -- Tier 1 traces land in Phase 5")
