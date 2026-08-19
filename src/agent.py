"""
LangGraph Agent Module for Multimodal Agentic RAG

This module contains the core LangGraph agent implementation extracted from the Jupyter notebook.
It provides a modular, reusable agent implementation for the RAG system.
"""

import heapq
import operator
import time
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langchain_tavily.tavily_search import TavilySearchAPIWrapper
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .configuration import config_rag
from .helper import is_rate_limit_error
from .hybrid_database import hybrid_search
from .logging_utils import get_logger

# Set up logger
logger = get_logger(__name__)


class RouteDecision(BaseModel):
    """Routes the user query to the appropriate data source."""
    reasoning: str = Field(
        ...,
        description="Briefly explain what the user is asking and check if it matches the explicit database topics."
    )
    is_in_domain: bool = Field(
        ...,
        description="True ONLY if the query is strictly about the specific topics in our database (e.g., vLLM, CRAG, Self-RAG). False for general ML topics like Federated Learning, Vision, etc."
    )
    route: Literal["vectorstore", "websearch", "chitchat"] = Field(
        ...,
        description="Choose 'vectorstore' if is_in_domain is True. Choose 'websearch' if is_in_domain is False or for current events. Choose 'chitchat' for greetings."
    )


class RewrittenQuery(BaseModel):
    """The optimized search query for the vector database."""
    reasoning: str = Field(..., description="Briefly explain what keywords were extracted or expanded.")
    query: str = Field(..., description="The highly optimized, keyword-dense search query.")


class HallucinationScore(BaseModel):
    """Checks if the generated answer contains hallucinations."""
    reasoning: str = Field(..., description="Briefly compare the generated facts against the source documents.")
    is_grounded: bool = Field(..., description="True if all facts in the generation are present in the documents. False if any outside info was added.")


class RelevanceScore(BaseModel):
    """Checks if the generated answer actually addresses the user's question."""
    reasoning: str = Field(..., description="Explain how the generated text does or does not answer the core premise of the user's question.")
    is_relevant: bool = Field(..., description="True if the answer directly addresses the question. False if it is evasive or off-topic.")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        loop_count: number to track loops
        gen_retries: number of generates for hallucinations
        chat_history: list of previous messages
        retrieved_chunk_ids: chunk IDs from hybrid search, pre-rerank
        retrieved_chunk_scores: reranker score per retrieved_chunk_ids entry (same
            order/length), pre-threshold -- lets eval/metrics/retrieval.py compute
            threshold_loss (a gold chunk the reranker scored above other kept
            chunks but the score threshold still dropped) without re-running the
            reranker. Empty list whenever no per-candidate score exists (no
            reranker configured, or the rerank_fallback path fired) -- never
            partially populated.
        reranked_chunk_ids: chunk IDs post-rerank, post-threshold (what generate() actually saw)
        fallback_events: tags appended by node-level except blocks when a silent
            fallback would otherwise occur (e.g. "rerank_fallback") -- invariant 15
            (PROJECT_SPEC.md) requires fallbacks be surfaced in results, not just
            logged. Uses operator.add so nodes only need to return the new tag(s)
            for their own step; LangGraph concatenates across the run.
    """
    question: str
    generation: str
    documents: List
    loop_count: int
    gen_retries: int
    chat_history: List
    retrieved_chunk_ids: List[str]
    retrieved_chunk_scores: List[float]
    reranked_chunk_ids: List[str]
    fallback_events: Annotated[List[str], operator.add]


def retrieve_and_rerank_core(question: str, *, database, embedding_model, rerank_model, config: dict) -> dict:
    """Hybrid search + rerank + threshold filter, with no GraphState/graph
    dependency -- the pure retrieval logic the `retrieve_and_rerank` node
    below wraps. Pulled out to a module-level function so
    eval.harness's --retrieval-only mode (PROJECT_SPEC.md §7 Phase 2:
    "Retrieval-only mode makes zero LLM calls") exercises this *exact* code
    path instead of a separately maintained reimplementation that could
    silently drift from what the real graph does -- duplicating this logic
    in eval/ would risk measuring something other than the real pipeline.

    Returns {"documents", "retrieved_chunk_ids", "retrieved_chunk_scores",
    "reranked_chunk_ids"}, plus "fallback_events": ["rerank_fallback"] only
    when the reranker call itself failed (never present otherwise -- the
    caller merges this dict into GraphState directly).
    """
    raw_docs = hybrid_search(
        database,
        embedding_model,
        question,
        sparse_weight=config.get("sparse_weight", 0.7),
        dense_weight=config.get("dense_weight", 1.0),
        limit=config.get("search_limit", 20),
    )

    retrieved_chunk_ids = [doc.get("id") for doc in raw_docs]

    if not raw_docs:
        logger.warning("No documents found in database.")
        return {
            "documents": [],
            "retrieved_chunk_ids": [],
            "retrieved_chunk_scores": [],
            "reranked_chunk_ids": [],
        }

    fallback_tag = None
    retrieved_chunk_scores: list = []
    if rerank_model is None:
        keep_top_k = config.get("reranker_top_k", 5)
        final_documents = raw_docs[:keep_top_k]
        logger.debug(f"No reranker available; using top {keep_top_k} raw hybrid-search results")
    else:
        logger.debug(f"Reranking {len(raw_docs)} retrieved documents...")
        try:
            question_and_docs = [[question, doc["text"]] for doc in raw_docs]
            scores = rerank_model.compute_score(question_and_docs, normalize=True, batch_size=4, max_length=1024)
            score_threshold = config.get("reranker_score_threshold", 0.5)
            filtered_pairs = [(doc, score) for doc, score in zip(raw_docs, scores) if score > score_threshold]
            keep_top_k = config.get("reranker_top_k", 5)
            reranked_docs = heapq.nlargest(keep_top_k, filtered_pairs, key=lambda x: x[1])

            final_documents = [doc for doc, _ in reranked_docs]
            # Aligned with retrieved_chunk_ids (raw_docs order), pre-threshold --
            # this is what makes threshold_loss computable in eval without
            # re-running the reranker (see GraphState docstring).
            retrieved_chunk_scores = [float(s) for s in scores]

            logger.info(f"Milvus found {len(raw_docs)} docs. Reranker kept {len(final_documents)} docs > {score_threshold} score.")
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback to just taking first k documents
            keep_top_k = config.get("reranker_top_k", 5)
            final_documents = raw_docs[:keep_top_k]
            fallback_tag = "rerank_fallback"
            logger.warning(f"Using fallback: taking first {keep_top_k} documents")

    reranked_chunk_ids = [doc.get("id") for doc in final_documents]

    result = {
        "documents": final_documents,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_chunk_scores": retrieved_chunk_scores,
        "reranked_chunk_ids": reranked_chunk_ids,
    }
    if fallback_tag:
        result["fallback_events"] = [fallback_tag]
    return result


def build_agent_graph(database, embedding_model, rerank_model, llm_model, config: Optional[dict] = None, web_search_tool=None):
    """
    Build and compile the LangGraph agent.

    Args:
        database: Milvus database collection
        embedding_model: BGE-M3 embedding model
        rerank_model: BGE reranker model
        llm_model: chat model client for the configured LLM provider
        web_search_tool: optional override for the web-search tool, anything
            with a `.invoke({"query": ...}) -> dict` matching TavilySearch's
            response shape. When omitted, the real TavilySearch is built as
            before. eval.tavily_cache.build_tavily_tool() supplies a
            replay/record wrapper here so the eval harness never makes a
            live Tavily call outside an explicit recording pass
            (PROJECT_SPEC.md invariant 12) -- this parameter is the only
            injection point the harness needs; web_search() itself is
            unchanged.

    Returns:
        Compiled StateGraph ready for use
    """
    logger.info("Building agent graph")

    cfg = config or config_rag()

    if web_search_tool is not None:
        web_tool = web_search_tool
        logger.info("Using injected web search tool (%s)", type(web_tool).__name__)
    else:
        # Key must go through api_wrapper -- TavilySearch itself has no
        # api_key field (extra="ignore" on its pydantic model silently
        # drops one), so passing api_key= directly is a no-op and it falls
        # back to reading TAVILY_API_KEY from the process env instead of cfg.
        web_tool = TavilySearch(
            api_wrapper=TavilySearchAPIWrapper(tavily_api_key=cfg["tavilly_api_key"]),
            max_results=cfg.get("web_search_max_results", 5),
            topic=cfg.get("web_search_topic", "general"),
            include_images=False
        )
        logger.info("Initialized web search tool")

    # Router node prompt
    domain_topics = cfg.get("domain_topics", [])
    router_node_prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a strict query router for a specialized AI-research RAG system.\n"
                "Decide the best route using only these options: vectorstore, websearch, chitchat.\n\n"
                "Vectorstore scope (in-domain) is LIMITED to these paper themes/topics: "
                + ", ".join(domain_topics)
                + ".\n\n"
                "Routing policy:\n"
                "1) Use vectorstore only when user intent clearly matches the in-domain topics.\n"
                "2) Use websearch for out-of-domain AI/ML topics, live/current information, or when unsure.\n"
                "3) Use chitchat for greetings, thanks, compliments, or casual conversation.\n"
                "4) Be conservative: if uncertain, prefer websearch over vectorstore.\n"
                "5) Use chat history to resolve references like 'that paper' or 'explain more'."
            ),
            ("human", "Chat History:\n{chat_history}\n\nCurrent Query:\n{question}"),
        ],
        input_variables=["question", "chat_history"],
    )

    router_node_llm = router_node_prompt | llm_model.with_structured_output(RouteDecision)

    # Rewrite node prompt
    rewrite_node_prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You optimize failed retrieval queries for dense+sparse semantic search.\n"
                "Rewrite into a compact technical query that maximizes recall and precision.\n"
                "Rules:\n"
                "- Keep original intent unchanged.\n"
                "- Remove filler words and conversational phrasing.\n"
                "- Expand relevant acronyms if helpful (e.g., RAG -> Retrieval-Augmented Generation).\n"
                "- Include core entities: model names, methods, mechanisms, and metrics when present.\n"
                "- Return one best query only."
            ),
            ("human", "Original query:\n{question}"),
        ],
        input_variables=["question"],
    )

    rewrite_node_llm = rewrite_node_prompt | llm_model.with_structured_output(RewrittenQuery)

    # Generate node prompt
    generate_node_prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are an expert AI research assistant. Answer using ONLY the provided context.\n"
                "Quality requirements:\n"
                "- Be correct, concise, and directly responsive to the question.\n"
                "- Prefer bullet points for multi-part answers.\n"
                "- Include short source cues (paper/topic names) when available.\n"
                "- If context is insufficient, reply exactly: I cannot answer this based on the provided context.\n"
                "- Do not invent facts, numbers, or citations."
            ),
            (
                "human",
                "Chat History:\n{chat_history}\n\nContext:\n{documents}\n\n"
                "Question:\n{question}\n\n"
                "Now produce the final answer."
            ),
        ],
        input_variables=["documents", "question", "chat_history"],
    )

    generate_node_llm = generate_node_prompt | llm_model | StrOutputParser()

    # Hallucination checker node
    hallucination_check_node_prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a strict factual-grounding judge.\n"
                "Mark is_grounded=True only if every concrete claim in the answer is supported by the supplied context.\n"
                "If any unsupported claim appears, set is_grounded=False."
            ),
            ("human", "Context:\n{documents}\n\nAnswer to evaluate:\n{generation}"),
        ],
        input_variables=["documents", "generation"],
    )

    hallucination_check_node_llm = hallucination_check_node_prompt | llm_model.with_structured_output(HallucinationScore)

    # Relevance checker node
    relevance_check_node_prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a strict relevance grader.\n"
                "Set is_relevant=True only if the answer directly resolves the user's current question.\n"
                "Fail answers that are generic summaries, evasive, or only partially responsive.\n"
                "Exception: an honest refusal because the provided context doesn't contain the "
                "answer (e.g. 'I cannot answer this based on the provided context') is NOT evasive -- "
                "it is the correct response to an unanswerable question. Grade that as is_relevant=True."
            ),
            ("human", "Question:\n{question}\n\nAnswer:\n{generation}"),
        ],
        input_variables=["question", "generation"],
    )

    relevance_check_node_llm = relevance_check_node_prompt | llm_model.with_structured_output(RelevanceScore)

    # Chitchat prompt
    chitchat_prompt = ChatPromptTemplate(
        [
            ("system", "You are a friendly assistant for a technical RAG app. "
                     "For greetings/small talk, respond briefly and naturally in 1-2 sentences."),
            ("human", "Chat History:\n{chat_history}\n\nUser Message: {question}"),
        ],
        input_variables=["chat_history", "question"],
    )

    chitchat_llm = chitchat_prompt | llm_model | StrOutputParser()

    # Helper functions
    def _safe_join_history(chat_history: List[str]) -> str:
        return "\n".join(chat_history) if chat_history else "None"

    def _format_documents(documents: list) -> str:
        if not documents:
            return "No context available."

        if isinstance(documents[0], dict):
            return "\n\n".join([str(doc.get("text", "")) for doc in documents if doc.get("text")])

        return "\n\n".join([str(doc) for doc in documents if str(doc).strip()])

    def _extract_web_results(docs: dict) -> list:
        results = docs.get("results", []) if isinstance(docs, dict) else []
        extracted = []
        for item in results:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or item.get("snippet") or ""
            if content:
                extracted.append(content)
        return extracted

    # Node implementations
    def retrieve_and_rerank(state):
        """
        Retrieves documents from the index and immediately reranks them,
        applying a score filter and keeping only the top k.

        Args:
            state (dict): Current graph state

        Returns:
            dict: The updated state containing strictly the highly relevant documents.
        """
        logger.debug("---RETRIEVE & RERANK---")

        question = state["question"]
        loop_count = state.get("loop_count", 0)
        chat_history = state.get("chat_history", [])

        logger.debug(f"Retrieving for question: {question[:100]}...")
        core_result = retrieve_and_rerank_core(
            question, database=database, embedding_model=embedding_model, rerank_model=rerank_model, config=cfg
        )

        result = {
            "documents": core_result["documents"],
            "question": question,
            "loop_count": loop_count,
            "chat_history": chat_history,
            "retrieved_chunk_ids": core_result["retrieved_chunk_ids"],
            "retrieved_chunk_scores": core_result["retrieved_chunk_scores"],
            "reranked_chunk_ids": core_result["reranked_chunk_ids"],
        }
        if "fallback_events" in core_result:
            result["fallback_events"] = core_result["fallback_events"]
        return result

    def generate(state):
        """
        Generate answer for query

        Args:
            state (dict): Current graph state

        Returns:
            dict: retrieved documents added to the state
        """
        logger.debug("---GENERATE---")

        question = state["question"]
        documents = state["documents"]
        loop_count = state.get("loop_count", 0)
        gen_retries = state.get("gen_retries", 0)
        chat_history = state.get("chat_history", [])

        # Format documents consistently
        if documents:
            if isinstance(documents[0], dict):
                # From vectorstore
                formatted_docs = "\n\n".join([doc.get("text", str(doc)) for doc in documents])
            else:
                # From web search
                formatted_docs = "\n\n".join([str(doc) for doc in documents])
        else:
            formatted_docs = "No context available."

        fallback_tag = None
        try:
            result = generate_node_llm.invoke({
                "question": question,
                "documents": formatted_docs,
                "chat_history": _safe_join_history(chat_history)
            })
            generation = result
            logger.debug("Generated response successfully")
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            generation = "I encountered an error while generating the response."
            fallback_tag = "generate_error"

        gen_retries += 1

        output = {"documents": documents, "question": question, "generation": generation, "loop_count": loop_count, "gen_retries": gen_retries, "chat_history": chat_history}
        if fallback_tag:
            output["fallback_events"] = [fallback_tag]
        return output

    def rewrite_query(state):
        """
        Rewrite query for better question

        Args:
            state (dict): Current graph state

        Returns:
            dict: retrieved documents added to the state
        """
        logger.debug("---REWRITE_QUERY---")

        question = state["question"]
        documents = state.get("documents", [])
        loop_count = state.get("loop_count", 0)
        loop_count += 1
        chat_history = state.get("chat_history", [])
        gen_retries = 0

        fallback_tag = None
        try:
            result = rewrite_node_llm.invoke({"question": question})
            rewritten_query = result.query
            logger.debug(f"Rewritten query: {rewritten_query[:100]}...")
        except Exception as e:
            logger.error(f"Error during query rewriting: {e}")
            # Return original query on error to avoid breaking the flow
            rewritten_query = question
            fallback_tag = "rewrite_error"

        output = {"documents": documents, "question": rewritten_query, "loop_count": loop_count, "gen_retries": gen_retries, "chat_history": chat_history}
        if fallback_tag:
            output["fallback_events"] = [fallback_tag]
        return output

    def web_search(state):
        """
        Web search for query to find relevant context

        Args:
            state (dict): Current graph state

        Returns:
            dict: retrieved documents added to the state
        """
        logger.debug("---WEB SEARCH---")
        question = state["question"]
        chat_history = state.get("chat_history", [])
        loop_count = state.get("loop_count", 0)
        gen_retries = 0

        logger.info(f"Performing web search for: {question[:100]}...")
        fallback_tag = None
        try:
            docs = web_tool.invoke({"query": f"{question}"})
            web_results = _extract_web_results(docs)
            logger.info(f"Web search returned {len(web_results)} results")
        except Exception as e:
            logger.error(f"Error during web search: {e}")
            web_results = []
            fallback_tag = "web_search_error"

        output = {"documents": web_results, "question": question, "loop_count": loop_count, "gen_retries": gen_retries, "chat_history": chat_history}
        if fallback_tag:
            output["fallback_events"] = [fallback_tag]
        return output

    def chitchat_node(state):
        """Handles simple greetings without searching the database."""
        logger.debug("---CHITCHAT---")
        question = state["question"]
        chat_history = state.get("chat_history", [])
        loop_count = state.get("loop_count", 0)
        gen_retries = state.get("gen_retries", 0)

        fallback_tag = None
        try:
            result = chitchat_llm.invoke({"question": question, "chat_history": _safe_join_history(chat_history)})
            logger.debug("Generated chitchat response")
        except Exception as e:
            logger.error(f"Error during chitchat generation: {e}")
            result = "Hello! How can I help you today?"
            fallback_tag = "chitchat_error"

        output = {"generation": result, "question": question, "chat_history": chat_history, "loop_count": loop_count, "gen_retries": gen_retries}
        if fallback_tag:
            output["fallback_events"] = [fallback_tag]
        return output

    # Router functions
    def query_router(state):
        """
        [Adaptive RAG]: routes the query to vectorbase or simple llm call or more

        Args:
            state (dict): Current graph state

        Returns:
            str: Next node to route to
        """
        question = state["question"]
        chat_history = state.get("chat_history", [])

        try:
            result = router_node_llm.invoke({"question": question, "chat_history": _safe_join_history(chat_history)})
            route = result.route if result.route in {"vectorstore", "websearch", "chitchat"} else "websearch"
            logger.debug(f"Router decided: {route}")
            return route
        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            return "websearch"  # Default to websearch on error

    def rewrite_router(state):
        max_rewrites = cfg.get("max_rewrites", 3)
        loop_count = state.get("loop_count", 0)
        if loop_count <= max_rewrites:
            logger.debug(f"Rewrite loop count: {loop_count}, max: {max_rewrites}, decision: retrieve")
            return "retrieve"
        logger.debug(f"Rewrite loop count: {loop_count}, max: {max_rewrites}, decision: web_search")
        return "web_search"

    def hallucinations_and_relevance_router(state):
        question = state["question"]
        documents = state.get("documents", [])
        generation = state.get("generation", "")
        gen_retries = state.get("gen_retries", 0)
        max_gen_retries = cfg.get("max_gen_retries", 2)

        logger.debug(f"Checking hallucinations and relevance for generation: {generation[:100]}...")

        if documents:
            formatted_docs = _format_documents(documents)

            try:
                grounded = hallucination_check_node_llm.invoke({"documents": formatted_docs, "generation": generation}).is_grounded
                if not grounded:
                    decision = "rewrite_query" if gen_retries >= max_gen_retries else "generate"
                    logger.debug(f"Grounded: {grounded}, retries: {gen_retries}/{max_gen_retries}, decision: {decision}")
                    return decision
            except Exception as e:
                logger.error(f"Error in hallucination check: {e}")
                # On error, continue generation to avoid getting stuck
                return "generate"

        try:
            relevant = relevance_check_node_llm.invoke({"question": question, "generation": generation}).is_relevant
            if relevant:
                logger.debug("Relevance check passed")
                return "all_pass"
            decision = "rewrite_query" if gen_retries >= max_gen_retries else "generate"
            logger.debug(f"Relevant: {relevant}, retries: {gen_retries}/{max_gen_retries}, decision: {decision}")
            return decision
        except Exception as e:
            logger.error(f"Error in relevance check: {e}")
            # On error, assume relevant to avoid infinite loops
            return "all_pass"

    def post_retrieve_router(state):
        doc_count = len(state.get("documents", []))
        decision = "generate" if doc_count > 0 else "rewrite"
        logger.debug(f"Post-retrieve: {doc_count} documents, decision: {decision}")
        return decision

    # Build the graph
    logger.debug("Building StateGraph workflow")
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve_and_rerank", retrieve_and_rerank)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("chitchat", chitchat_node)

    workflow.add_conditional_edges(
        START,
        query_router,
        {
            "vectorstore": "retrieve_and_rerank",
            "websearch": "web_search",
            "chitchat": "chitchat",
        },
    )

    workflow.add_conditional_edges(
        "retrieve_and_rerank",
        post_retrieve_router,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
        },
    )

    workflow.add_conditional_edges(
        "rewrite_query",
        rewrite_router,
        {
            "retrieve": "retrieve_and_rerank",
            "web_search": "web_search",
            "generate": "generate",
        },
    )

    workflow.add_edge("web_search", "generate")

    workflow.add_conditional_edges(
        "generate",
        hallucinations_and_relevance_router,
        {
            "all_pass": END,
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    workflow.add_edge("chitchat", END)

    logger.info("Agent graph compiled successfully")
    return workflow.compile()


def run_query_with_state(app_graph, question: str, chat_history: List[str]):
    """
    Execute a query through the agent graph, returning the full final
    GraphState and a small trace_info dict (node execution sequence +
    per-node-name latency) alongside the answer -- not just the answer
    string. Used by src/api.py to build a full /query trace (route
    decision, both chunk ID lists, correction signal, per-stage timing).

    node_sequence[0] is the route decision: query_router is a conditional-
    edge selector, not a node, so its choice isn't itself written into
    GraphState -- but it's exactly what determines which node runs first.

    Args:
        app_graph: Compiled StateGraph agent
        question: User question string
        chat_history: List of previous chat messages

    Returns:
        tuple: (answer: str, final_state: dict, trace_info: dict)
    """
    logger.info(f"Processing query: {question[:100]}...")
    inputs = {
        "question": question,
        "chat_history": chat_history,
        "loop_count": 0,
        "gen_retries": 0,
        "retrieved_chunk_ids": [],
        "retrieved_chunk_scores": [],
        "reranked_chunk_ids": [],
        "fallback_events": [],
    }

    final_state = dict(inputs)
    node_sequence: list[str] = []
    stage_latencies_ms: dict[str, float] = {}

    try:
        step_start = time.time()
        for step in app_graph.stream(inputs):
            for node_name, node_output in step.items():
                elapsed_ms = (time.time() - step_start) * 1000
                node_sequence.append(node_name)
                stage_latencies_ms[node_name] = stage_latencies_ms.get(node_name, 0.0) + elapsed_ms
                if "fallback_events" in node_output:
                    final_state["fallback_events"] = final_state.get("fallback_events", []) + node_output["fallback_events"]
                    node_output = {k: v for k, v in node_output.items() if k != "fallback_events"}
                final_state.update(node_output)
                step_start = time.time()
    except Exception as e:
        logger.error(f"Error processing query through agent graph: {e}")
        final_state["generation"] = f"I encountered an error processing your request: {e}"
        # Tagged separately from other crashes so eval.gate can tell a
        # provider quota blip apart from a real regression (CLAUDE.md
        # Phase 4 checklist) rather than lumping both under one tag.
        error_tag = "rate_limited" if is_rate_limit_error(e) else "graph_execution_error"
        fallback_events = final_state.get("fallback_events", []) + [error_tag]
        return final_state["generation"], final_state, {
            "node_sequence": node_sequence,
            "stage_latencies_ms": stage_latencies_ms,
            "fallback_events": fallback_events,
        }

    if "generation" in final_state:
        logger.info("Successfully generated response")
    else:
        logger.warning("No generation produced from agent graph")
        final_state["generation"] = "I encountered an error processing your request."

    trace_info = {
        "node_sequence": node_sequence,
        "stage_latencies_ms": stage_latencies_ms,
        "fallback_events": final_state.get("fallback_events", []),
    }
    return final_state["generation"], final_state, trace_info


def run_query(app_graph, question: str, chat_history: List[str]) -> str:
    """
    Execute a query through the agent graph.

    Args:
        app_graph: Compiled StateGraph agent
        question: User question string
        chat_history: List of previous chat messages

    Returns:
        str: Generated response from the agent
    """
    answer, _, _ = run_query_with_state(app_graph, question, chat_history)
    return answer
