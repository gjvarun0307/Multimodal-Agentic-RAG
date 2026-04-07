import heapq
import os
from pathlib import Path
from typing import List
from typing_extensions import TypedDict, Literal

import streamlit as st
from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from FlagEmbedding import FlagLLMReranker

from config import config_parse, config_rag
from hybrid_database import append_parsed_file_to_database, data_preprocessing, hybrid_search, load_database_and_embedding
from parse import parse_single_file, load_model


MAX_CHAT_TURNS = 10


DOMAIN_TOPICS = [
    "vLLM",
    "Transformers",
    "PagedAttention",
    "GraphRAG",
    "CRAG",
    "Self-RAG",
    "Adaptive-RAG",
    "Milvus",
    "HyDE",
    "LoRA",
    "LLaVA",
    "FlashAttention",
    "ColBERTv2",
]


class RouteDecision(BaseModel):
    reasoning: str = Field(..., description="Why this route best serves the user.")
    is_in_domain: bool = Field(..., description="True only for topics clearly covered by the local research corpus.")
    route: Literal["vectorstore", "websearch", "chitchat"] = Field(..., description="Routing destination.")


class RewrittenQuery(BaseModel):
    reasoning: str = Field(..., description="How the query was transformed for retrieval.")
    query: str = Field(..., description="Search-optimized rewrite of the question.")


class HallucinationScore(BaseModel):
    reasoning: str = Field(..., description="Grounding analysis against supplied documents.")
    is_grounded: bool = Field(..., description="True when every factual claim is supported by context.")


class RelevanceScore(BaseModel):
    reasoning: str = Field(..., description="How well the answer addresses the exact user intent.")
    is_relevant: bool = Field(..., description="True when answer directly and sufficiently resolves the question.")


class GraphState(TypedDict):
    question: str
    generation: str
    documents: list
    loop_count: int
    gen_retries: int
    chat_history: list


@st.cache_resource(show_spinner=True)
def load_runtime():
    config = config_rag()

    try:
        database, embedding_model = load_database_and_embedding(config["database_path"], config["device"])
    except Exception:
        database, embedding_model = data_preprocessing(config)

    rerank_model = FlagLLMReranker(
        "BAAI/bge-reranker-v2-gemma",
        use_fp16=False,
        devices=config["device"],
    )

    llm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    llm_api_key = os.environ.get("VLLM_API_KEY", "token-abc123")

    llm_model = ChatOpenAI(
        model_name="Qwen/Qwen3-8B-AWQ",
        base_url=llm_base_url,
        api_key=llm_api_key,
        model_kwargs={
            "extra_body": {
                "guided_decoding_backend": "xgrammar"
            }
        },
    )

    tavily_key = config.get("tavilly_api_key")
    if tavily_key and not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = tavily_key

    web_tool = None
    if os.environ.get("TAVILY_API_KEY"):
        try:
            web_tool = TavilySearch(max_results=5, topic="general", include_images=False)
        except Exception:
            web_tool = None

    parse_model = None
    parse_processor = None
    if os.environ.get("PRELOAD_PARSE_VLM", "1") == "1":
        try:
            parse_model, parse_processor = load_model(config.get("device", "cuda"))
        except Exception:
            parse_model, parse_processor = None, None

    return config, database, embedding_model, rerank_model, llm_model, web_tool, parse_model, parse_processor


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


@st.cache_resource(show_spinner=True)
def build_graph():
    _, database, embedding_model, rerank_model, llm_model, web_tool, _, _ = load_runtime()

    router_node_prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a strict query router for a specialized AI-research RAG system.\n"
                "Decide the best route using only these options: vectorstore, websearch, chitchat.\n\n"
                "Vectorstore scope (in-domain) is LIMITED to these paper themes/topics: "
                + ", ".join(DOMAIN_TOPICS)
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
    router_node_llm = router_node_prompt | llm_model.with_structured_output(RouteDecision, method="json_schema", strict=True)

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
    rewrite_node_llm = rewrite_node_prompt | llm_model.with_structured_output(RewrittenQuery, method="json_schema", strict=True)

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
    hallucination_check_node_llm = hallucination_check_node_prompt | llm_model.with_structured_output(HallucinationScore, method="json_schema", strict=True)

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
    relevance_check_node_llm = relevance_check_node_prompt | llm_model.with_structured_output(RelevanceScore, method="json_schema", strict=True)

    chitchat_prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a friendly assistant for a technical RAG app. "
                "For greetings/small talk, respond briefly and naturally in 1-2 sentences."
            ),
            ("human", "Chat History:\n{chat_history}\n\nUser Message:\n{question}"),
        ],
        input_variables=["chat_history", "question"],
    )
    chitchat_llm = chitchat_prompt | llm_model | StrOutputParser()

    def retrieve_and_rerank(state):
        question = state["question"]
        loop_count = state.get("loop_count", 0)
        chat_history = state.get("chat_history", [])

        raw_docs = hybrid_search(database, embedding_model, question, sparse_weight=0.7, dense_weight=1.0, limit=20)
        if not raw_docs:
            return {"documents": [], "question": question, "loop_count": loop_count, "chat_history": chat_history}

        clean_docs = [doc for doc in raw_docs if isinstance(doc, dict) and doc.get("text")]
        if not clean_docs:
            return {"documents": [], "question": question, "loop_count": loop_count, "chat_history": chat_history}

        question_and_docs = [[question, doc["text"]] for doc in clean_docs]
        try:
            scores = rerank_model.compute_score(question_and_docs, normalize=True, batch_size=4, max_length=1024)
            filtered_pairs = [(doc, score) for doc, score in zip(clean_docs, scores) if score > 0.5]
            reranked_docs = heapq.nlargest(5, filtered_pairs, key=lambda x: x[1])
            final_documents = [doc for doc, _ in reranked_docs]
        except Exception:
            final_documents = clean_docs[:5]

        return {"documents": final_documents, "question": question, "loop_count": loop_count, "chat_history": chat_history}

    def generate(state):
        question = state["question"]
        documents = state.get("documents", [])
        loop_count = state.get("loop_count", 0)
        gen_retries = state.get("gen_retries", 0)
        chat_history = state.get("chat_history", [])

        formatted_docs = _format_documents(documents)

        generation = generate_node_llm.invoke(
            {
                "question": question,
                "documents": formatted_docs,
                "chat_history": _safe_join_history(chat_history),
            }
        )

        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "loop_count": loop_count,
            "gen_retries": gen_retries + 1,
            "chat_history": chat_history,
        }

    def rewrite_query(state):
        question = state["question"]
        documents = state.get("documents", [])
        loop_count = state.get("loop_count", 0) + 1
        chat_history = state.get("chat_history", [])

        rewritten = rewrite_node_llm.invoke({"question": question}).query
        return {
            "documents": documents,
            "question": rewritten,
            "loop_count": loop_count,
            "gen_retries": 0,
            "chat_history": chat_history,
        }

    def web_search(state):
        question = state["question"]
        chat_history = state.get("chat_history", [])
        loop_count = state.get("loop_count", 0)

        if web_tool is None:
            web_results = []
        else:
            try:
                docs = web_tool.invoke({"query": question})
                web_results = _extract_web_results(docs)
            except Exception:
                web_results = []

        return {
            "documents": web_results,
            "question": question,
            "loop_count": loop_count,
            "gen_retries": 0,
            "chat_history": chat_history,
        }

    def chitchat_node(state):
        question = state["question"]
        chat_history = state.get("chat_history", [])

        generation = chitchat_llm.invoke(
            {
                "question": question,
                "chat_history": _safe_join_history(chat_history),
            }
        )
        return {"generation": generation, "question": question, "chat_history": chat_history}

    def query_router(state):
        question = state["question"]
        chat_history = state.get("chat_history", [])

        # Uploaded-document intents should stay on vector retrieval so newly ingested PDFs are immediately queryable.
        combined_text = f"{question}\n{_safe_join_history(chat_history)}".lower()
        upload_markers = [
            "uploaded pdf",
            "uploaded file",
            "this pdf",
            "this file",
            "my pdf",
            "my file",
            "document i uploaded",
            "newly added",
        ]
        if any(marker in combined_text for marker in upload_markers):
            return "vectorstore"

        result = router_node_llm.invoke({"question": question, "chat_history": _safe_join_history(chat_history)})
        return result.route if result.route in {"vectorstore", "websearch", "chitchat"} else "websearch"

    def rewrite_router(state):
        return "web_search" if state.get("loop_count", 0) > 3 else "retrieve"

    def hallucinations_and_relevance_router(state):
        question = state["question"]
        documents = state.get("documents", [])
        generation = state.get("generation", "")
        gen_retries = state.get("gen_retries", 0)

        if documents:
            formatted_docs = _format_documents(documents)

            grounded = hallucination_check_node_llm.invoke({"documents": formatted_docs, "generation": generation}).is_grounded
            if not grounded:
                return "rewrite_query" if gen_retries >= 2 else "generate"

        relevant = relevance_check_node_llm.invoke({"question": question, "generation": generation}).is_relevant
        if relevant:
            return "all_pass"
        return "rewrite_query" if gen_retries >= 2 else "generate"

    def post_retrieve_router(state):
        return "generate" if len(state.get("documents", [])) > 0 else "rewrite"

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
    return workflow.compile()


def run_query(app_graph, question: str, chat_history: List[str]):
    inputs = {
        "question": question,
        "chat_history": chat_history,
        "loop_count": 0,
        "gen_retries": 0,
    }

    final_output = None
    try:
        for output in app_graph.stream(inputs):
            for _, value in output.items():
                final_output = value
    except Exception as e:
        return f"I encountered an error processing your request: {e}"

    if final_output and "generation" in final_output:
        return final_output["generation"]
    return "I encountered an error processing your request."


def ingest_uploaded_pdf(uploaded_file, rag_config, database, embedding_model, parse_model=None, parse_processor=None):
    pdf_dir = Path("data/raw_pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    safe_file_name = Path(uploaded_file.name).name
    saved_pdf_path = pdf_dir / safe_file_name
    with open(saved_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    parse_config = config_parse()
    parse_config["input_folder"] = str(pdf_dir)
    parse_config["output_folder"] = rag_config["input_folder_path"]
    parse_config["device"] = rag_config["device"]
    Path(parse_config["output_folder"]).mkdir(parents=True, exist_ok=True)

    artifact = parse_single_file(
        parse_config,
        str(saved_pdf_path),
        model=parse_model,
        processor=parse_processor,
    )
    if artifact is None:
        raise RuntimeError("PDF parsing failed. Please check fail_logs.txt for details.")

    inserted_chunks = append_parsed_file_to_database(
        markdown_path=artifact["markdown_path"],
        metadata=artifact.get("metadata", {}),
        config=rag_config,
        database=database,
        embedding_model=embedding_model,
    )

    return {
        "pdf_path": str(saved_pdf_path),
        "markdown_path": artifact["markdown_path"],
        "inserted_chunks": inserted_chunks,
    }


def main():
    st.set_page_config(page_title="Multimodal Agentic RAG", page_icon="🤖", layout="wide")
    st.title("Multimodal Agentic RAG")
    st.caption("Adaptive routing across vector search, web search, and conversational mode.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = []

    rag_config, database, embedding_model, _, _, _, parse_model, parse_processor = load_runtime()

    with st.sidebar:
        st.subheader("Add PDF to Knowledge Base")
        uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
        if st.button("Parse and Add", use_container_width=True, disabled=uploaded_pdf is None):
            with st.spinner("Parsing PDF and indexing into Milvus..."):
                try:
                    ingest_result = ingest_uploaded_pdf(
                        uploaded_pdf,
                        rag_config,
                        database,
                        embedding_model,
                        parse_model=parse_model,
                        parse_processor=parse_processor,
                    )
                    st.session_state.ingested_files.append(
                        {
                            "pdf": Path(ingest_result["pdf_path"]).name,
                            "chunks": ingest_result["inserted_chunks"],
                        }
                    )
                    # st.session_state.rag_history.append(
                        f"System: New uploaded document indexed: {Path(ingest_result['pdf_path']).name}"
                    )
                    st.success(
                        f"Added {Path(ingest_result['pdf_path']).name} with {ingest_result['inserted_chunks']} chunks to RAG context."
                    )
                except Exception as e:
                    st.error(f"Failed to ingest PDF: {e}")

        if st.session_state.ingested_files:
            st.caption("Recently ingested")
            for item in st.session_state.ingested_files[-5:][::-1]:
                st.write(f"- {item['pdf']} ({item['chunks']} chunks)")

    app_graph = build_graph()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the RAG papers...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = run_query(app_graph, user_input, st.session_state.rag_history.copy())
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.rag_history.append(f"User: {user_input}")
    st.session_state.rag_history.append(f"Assistant: {answer}")

    if len(st.session_state.rag_history) > MAX_CHAT_TURNS * 2:
        st.session_state.rag_history = st.session_state.rag_history[-(MAX_CHAT_TURNS * 2):]


if __name__ == "__main__":
    main()
