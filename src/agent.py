"""
LangGraph Agent Module for Multimodal Agentic RAG

This module contains the core LangGraph agent implementation extracted from the Jupyter notebook.
It provides a modular, reusable agent implementation for the RAG system.
"""

import heapq
import os
from typing import List, Literal, Optional, TypedDict

from FlagEmbedding import FlagLLMReranker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .configuration import build_llm_client, config_rag
from .hybrid_database import hybrid_search, load_or_create_database
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
        active_document: filename of the most recently uploaded document this session, if any
        scope_to_active_document: whether this turn should be retrieved/routed against active_document only
    """
    question: str
    generation: str
    documents: List
    loop_count: int
    gen_retries: int
    chat_history: List
    active_document: Optional[str]
    scope_to_active_document: bool


def get_models(config):
    """
    Initialize and return all required models and databases.

    Args:
        config: Configuration dictionary from config_rag()

    Returns:
        tuple: (database, embedding_model, rerank_model, llm_model)
    """
    logger.info("Initializing models and databases")
    database, embedding_model = load_or_create_database(config)
    logger.info("Loaded database and embedding model")

    if config["device"] == "cpu":
        rerank_model = None
        logger.info("No CUDA device detected -- skipping reranker, using raw hybrid-search ranking")
    else:
        rerank_model = FlagLLMReranker(
            config.get("reranker_model", "BAAI/bge-reranker-v2-gemma"),
            use_fp16=config.get("use_fp16", False),
            devices=config["device"]
        )
        logger.info("Loaded rerank model")

    llm_model = build_llm_client(config)
    logger.info(f"Loaded LLM model: {config.get('llm_provider')}/{config.get('llm_model')}")
    logger.info("All models loaded!!")
    return database, embedding_model, rerank_model, llm_model


def build_agent_graph(database, embedding_model, rerank_model, llm_model):
    """
    Build and compile the LangGraph agent.

    Args:
        database: Milvus database collection
        embedding_model: BGE-M3 embedding model
        rerank_model: BGE reranker model
        llm_model: chat model client for the configured LLM provider

    Returns:
        Compiled StateGraph ready for use
    """
    logger.info("Building agent graph")

    # Initialize web search tool
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = config_rag()["tavilly_api_key"]
        logger.info("Set TAVILY_API_KEY from config")

    web_tool = TavilySearch(
        max_results=config_rag().get("web_search_max_results", 5),
        topic=config_rag().get("web_search_topic", "general"),
        include_images=False
    )
    logger.info("Initialized web search tool")

    # Router node prompt
    domain_topics = config_rag().get("domain_topics", [])
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
                "Fail answers that are generic summaries, evasive, or only partially responsive."
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
        active_document = state.get("active_document")
        scope_to_active_document = state.get("scope_to_active_document", False)

        filter_expr = None
        if scope_to_active_document and active_document:
            escaped = active_document.replace('"', '\\"')
            filter_expr = f'metadata["source_file"] == "{escaped}"'
            logger.debug(f"Scoping retrieval to active document: {active_document}")

        logger.debug(f"Retrieving for question: {question[:100]}...")
        config = config_rag()
        try:
            raw_docs = hybrid_search(
                database,
                embedding_model,
                question,
                sparse_weight=config.get("sparse_weight", 0.7),
                dense_weight=config.get("dense_weight", 1.0),
                limit=config.get("search_limit", 20),
                filter_expr=filter_expr,
            )
        except Exception as e:
            if filter_expr:
                logger.warning(f"Scoped search failed ({e}); retrying unfiltered")
                raw_docs = hybrid_search(
                    database,
                    embedding_model,
                    question,
                    sparse_weight=config.get("sparse_weight", 0.7),
                    dense_weight=config.get("dense_weight", 1.0),
                    limit=config.get("search_limit", 20),
                )
            else:
                raise

        if not raw_docs:
            logger.warning("No documents found in database.")
            return {"documents": [], "question": question, "loop_count": loop_count, "chat_history": chat_history}

        if rerank_model is None:
            keep_top_k = config.get("reranker_top_k", 5)
            final_documents = raw_docs[:keep_top_k]
            logger.debug(f"No reranker available; using top {keep_top_k} raw hybrid-search results")
        else:
            logger.debug(f"Reranking {len(raw_docs)} retrieved documents...")
            question_and_docs = [[question, doc["text"]] for doc in raw_docs]
            try:
                scores = rerank_model.compute_score(question_and_docs, normalize=True, batch_size=4, max_length=1024)
                score_threshold = config.get("reranker_score_threshold", 0.5)
                filtered_pairs = [(doc, score) for doc, score in zip(raw_docs, scores) if score > score_threshold]
                keep_top_k = config.get("reranker_top_k", 5)
                reranked_docs = heapq.nlargest(keep_top_k, filtered_pairs, key=lambda x: x[1])

                final_documents = [doc for doc, _ in reranked_docs]

                logger.info(f"Milvus found {len(raw_docs)} docs. Reranker kept {len(final_documents)} docs > {score_threshold} score.")
            except Exception as e:
                logger.error(f"Error during reranking: {e}")
                # Fallback to just taking first k documents
                keep_top_k = config.get("reranker_top_k", 5)
                final_documents = raw_docs[:keep_top_k]
                logger.warning(f"Using fallback: taking first {keep_top_k} documents")

        return {"documents": final_documents, "question": question, "loop_count": loop_count, "chat_history": chat_history}

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

        gen_retries += 1

        return {"documents": documents, "question": question, "generation": generation, "loop_count": loop_count, "gen_retries": gen_retries, "chat_history": chat_history}

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

        try:
            result = rewrite_node_llm.invoke({"question": question})
            rewritten_query = result.query
            logger.debug(f"Rewritten query: {rewritten_query[:100]}...")
        except Exception as e:
            logger.error(f"Error during query rewriting: {e}")
            # Return original query on error to avoid breaking the flow
            rewritten_query = question

        return {"documents": documents, "question": rewritten_query, "loop_count": loop_count, "gen_retries": gen_retries, "chat_history": chat_history}

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
        try:
            docs = web_tool.invoke({"query": f"{question}"})
            web_results = _extract_web_results(docs)
            logger.info(f"Web search returned {len(web_results)} results")
        except Exception as e:
            logger.error(f"Error during web search: {e}")
            web_results = []

        return {"documents": web_results, "question": question, "loop_count": loop_count, "gen_retries": gen_retries, "chat_history": chat_history}

    def chitchat_node(state):
        """Handles simple greetings without searching the database."""
        logger.debug("---CHITCHAT---")
        question = state["question"]
        chat_history = state.get("chat_history", [])
        loop_count = state.get("loop_count", 0)
        gen_retries = state.get("gen_retries", 0)

        try:
            result = chitchat_llm.invoke({"question": question, "chat_history": _safe_join_history(chat_history)})
            logger.debug("Generated chitchat response")
        except Exception as e:
            logger.error(f"Error during chitchat generation: {e}")
            result = "Hello! How can I help you today?"

        return {"generation": result, "question": question, "chat_history": chat_history, "loop_count": loop_count, "gen_retries": gen_retries}

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

        # Decided once in run_query() from the original question, before any rewrite
        # can strip the marker words that triggered it.
        if state.get("scope_to_active_document") and state.get("active_document"):
            logger.debug("Routing to vectorstore (active document referenced)")
            return "vectorstore"

        try:
            result = router_node_llm.invoke({"question": question, "chat_history": _safe_join_history(chat_history)})
            route = result.route if result.route in {"vectorstore", "websearch", "chitchat"} else "websearch"
            logger.debug(f"Router decided: {route}")
            return route
        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            return "websearch"  # Default to websearch on error

    def rewrite_router(state):
        max_rewrites = config_rag().get("max_rewrite_attempts", 3)
        loop_count = state.get("loop_count", 0)
        if loop_count <= max_rewrites:
            logger.debug(f"Rewrite loop count: {loop_count}, max: {max_rewrites}, decision: retrieve")
            return "retrieve"
        if state.get("scope_to_active_document") and state.get("active_document"):
            logger.debug("Rewrite budget exhausted for a document-scoped query; skipping web search")
            return "generate"
        logger.debug(f"Rewrite loop count: {loop_count}, max: {max_rewrites}, decision: web_search")
        return "web_search"

    def hallucinations_and_relevance_router(state):
        question = state["question"]
        documents = state.get("documents", [])
        generation = state.get("generation", "")
        gen_retries = state.get("gen_retries", 0)

        logger.debug(f"Checking hallucinations and relevance for generation: {generation[:100]}...")

        if documents:
            formatted_docs = _format_documents(documents)

            try:
                grounded = hallucination_check_node_llm.invoke({"documents": formatted_docs, "generation": generation}).is_grounded
                if not grounded:
                    max_gen_retries = config_rag().get("max_gen_retries", 2)
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
            max_gen_retries = config_rag().get("max_gen_retries", 2)
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


_ACTIVE_DOCUMENT_MARKERS = [
    "uploaded pdf",
    "uploaded file",
    "uploaded document",
    "this pdf",
    "this file",
    "this document",
    "this paper",
    "my pdf",
    "my file",
    "my document",
    "attached",
    "the attachment",
    "document i uploaded",
    "paper i uploaded",
    "file i uploaded",
    "newly added",
    "just uploaded",
    "just added",
]


def _references_active_document(question: str, chat_history: List[str]) -> bool:
    """Heuristic: does this turn appear to reference the most recently uploaded document?"""
    combined_text = f"{question}\n{chr(10).join(chat_history or [])}".lower()
    return any(marker in combined_text for marker in _ACTIVE_DOCUMENT_MARKERS)


def run_query(app_graph, question: str, chat_history: List[str], active_document: Optional[str] = None):
    """
    Execute a query through the agent graph.

    Args:
        app_graph: Compiled StateGraph agent
        question: User question string
        chat_history: List of previous chat messages
        active_document: filename of the most recently uploaded document this
            session, if any -- used to decide whether to scope this turn's
            retrieval/routing to that document specifically

    Returns:
        str: Generated response from the agent
    """
    logger.info(f"Processing query: {question[:100]}...")
    scope_to_active_document = bool(active_document) and _references_active_document(question, chat_history)
    inputs = {
        "question": question,
        "chat_history": chat_history,
        "loop_count": 0,
        "gen_retries": 0,
        "active_document": active_document,
        "scope_to_active_document": scope_to_active_document,
    }

    final_output = None
    try:
        for output in app_graph.stream(inputs):
            for _, value in output.items():
                final_output = value
    except Exception as e:
        logger.error(f"Error processing query through agent graph: {e}")
        return f"I encountered an error processing your request: {e}"

    if final_output and "generation" in final_output:
        logger.info("Successfully generated response")
        return final_output["generation"]
    logger.warning("No generation produced from agent graph")
    return "I encountered an error processing your request."


def create_agent():
    """
    Factory function to create and initialize the complete agent.

    Returns:
        tuple: (app_graph, config) where app_graph is the compiled agent and config is the configuration
    """
    logger.info("Creating agent")
    config = config_rag()
    database, embedding_model, rerank_model, llm_model = get_models(config)
    app_graph = build_agent_graph(database, embedding_model, rerank_model, llm_model)
    logger.info("Agent created successfully")
    return app_graph, config
