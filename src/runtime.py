"""Single source of truth for constructing the query-serving runtime --
database, embedding model, reranker, and LLM client. Both app.py
(Streamlit) and src/api.py (FastAPI) call get_runtime() in-process;
neither builds its own copy of these objects.

This closes the gap PROJECT_SPEC.md's audit flagged (§5.4): agent.py's
get_models() and app.py's load_runtime() used to each independently
construct database/embedding_model/rerank_model, which already caused
one real bug (a CPU-reranker fix landed in one path and silently missed
the other).

Ingest-only concerns (LlamaParse, the Qwen2.5-VL figure-captioning model)
are deliberately not part of this runtime -- they never run at query
time (spec §4.2 item 6), so they stay local to whatever ingestion
surface needs them.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .configuration import build_llm_client, build_reranker, config_rag
from .hybrid_database import load_database_and_embedding
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Runtime:
    config: Dict[str, Any]
    database: Any
    embedding_model: Any
    rerank_model: Optional[Any]
    llm: Any


def get_runtime(config: Optional[Dict[str, Any]] = None) -> Runtime:
    """Construct a full Runtime.

    :param config: explicit config dict. Defaults to config_rag() (env /
        api_keys.json / defaults) when not given. Callers needing a
        fully config-driven run (the eval harness, an ablation runner)
        pass one explicitly instead of relying on ambient env state.
    """
    if config is None:
        config = config_rag()

    logger.info("Constructing runtime")

    database, embedding_model = load_database_and_embedding(config["database_path"], config["device"])
    logger.info(f"Loaded database with {database.num_entities} entities")

    rerank_model = build_reranker(config)

    llm = build_llm_client(config)
    logger.info(f"Loaded LLM client: {config.get('llm_provider')}/{config.get('llm_model')}")

    logger.info(f"Runtime constructed (reranker_enabled={rerank_model is not None})")
    return Runtime(
        config=config,
        database=database,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        llm=llm,
    )
