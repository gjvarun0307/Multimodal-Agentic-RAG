import importlib
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Protocol

import streamlit as st


class RAGBackend(Protocol):
    def ingest(self, file_paths: list[str]) -> str: ...
    def answer(self, query: str, history: list[dict[str, str]], **kwargs: Any) -> str: ...


class GenericBackendAdapter:
    def __init__(self, target: Any):
        self.target = target

    def _call_first(self, names: Iterable[str], *args: Any, **kwargs: Any) -> Any:
        for name in names:
            fn = getattr(self.target, name, None)
            if callable(fn):
                return fn(*args, **kwargs)
        raise AttributeError(f"No compatible method found in backend for candidates: {list(names)}")

    def ingest(self, file_paths: list[str]) -> str:
        result = self._call_first(
            ["ingest", "index", "index_documents", "add_documents", "upload_files", "load_documents"],
            file_paths,
        )
        return "Ingestion completed." if result is None else str(result)

    def answer(self, query: str, history: list[dict[str, str]], **kwargs: Any) -> str:
        try:
            result = self._call_first(["answer", "query", "ask", "chat", "run"], query, history=history, **kwargs)
        except TypeError:
            result = self._call_first(["answer", "query", "ask", "chat", "run"], query, **kwargs)
        return str(result)


class FallbackBackend:
    def ingest(self, file_paths: list[str]) -> str:
        return (
            "No backend discovered. Set env var RAG_BACKEND_FACTORY='module:function' "
            "or RAG_BACKEND_OBJECT='module:object' to connect your RAG pipeline."
        )

    def answer(self, query: str, history: list[dict[str, str]], **kwargs: Any) -> str:
        return (
            "Backend not connected yet.\n\n"
            "Set one of:\n"
            "1) RAG_BACKEND_FACTORY='your_module:create_backend'\n"
            "2) RAG_BACKEND_OBJECT='your_module:backend_instance'"
        )


def _load_from_reference(ref: str) -> Any:
    module_name, symbol_name = ref.split(":", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, symbol_name)


def discover_backend() -> RAGBackend:
    factory_ref = os.getenv("RAG_BACKEND_FACTORY", "").strip()
    object_ref = os.getenv("RAG_BACKEND_OBJECT", "").strip()

    if factory_ref:
        factory = _load_from_reference(factory_ref)
        return GenericBackendAdapter(factory())

    if object_ref:
        obj = _load_from_reference(object_ref)
        return GenericBackendAdapter(obj)

    module_candidates = [
        "backend",
        "rag",
        "rag_app",
        "pipeline",
        "main",
        "src.backend",
        "src.rag",
        "src.pipeline",
    ]
    object_candidates = ["backend", "rag", "app", "pipeline"]
    factory_candidates = ["get_backend", "create_backend", "build_backend", "make_backend"]
    class_candidates = ["RAG", "RAGApp", "RAGApplication", "MultimodalRAG", "AgenticRAG"]

    for module_name in module_candidates:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue

        for fn_name in factory_candidates:
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return GenericBackendAdapter(fn())

        for cls_name in class_candidates:
            cls = getattr(mod, cls_name, None)
            if callable(cls):
                try:
                    return GenericBackendAdapter(cls())
                except Exception:
                    pass

        for obj_name in object_candidates:
            obj = getattr(mod, obj_name, None)
            if obj is not None:
                return GenericBackendAdapter(obj)

    return FallbackBackend()


@st.cache_resource
def get_backend() -> RAGBackend:
    return discover_backend()


def _save_uploads(files: list[Any]) -> list[str]:
    workspace = Path(st.session_state.setdefault("_upload_dir", tempfile.mkdtemp(prefix="mm-rag-")))
    saved_paths: list[str] = []
    for f in files:
        dst = workspace / f.name
        dst.write_bytes(f.getvalue())
        saved_paths.append(str(dst))
    return saved_paths


def main() -> None:
    st.set_page_config(page_title="Multimodal Agentic RAG", page_icon="🧠", layout="wide")
    st.title("🧠 Multimodal Agentic RAG")
    st.caption("Upload files, ingest, then chat with your RAG system.")

    backend = get_backend()

    with st.sidebar:
        st.subheader("Settings")
        model = st.text_input("Model", value=os.getenv("RAG_MODEL", "default"))
        top_k = st.slider("Top-K", 1, 20, 5)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.05)
        st.divider()
        st.code(
            "RAG_BACKEND_FACTORY=module:function\nRAG_BACKEND_OBJECT=module:object",
            language="bash",
        )

    uploaded = st.file_uploader(
        "Upload documents/images",
        type=["pdf", "txt", "md", "docx", "png", "jpg", "jpeg", "csv", "json"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Ingest uploaded files", use_container_width=True):
            if not uploaded:
                st.warning("Upload at least one file first.")
            else:
                with st.spinner("Ingesting..."):
                    paths = _save_uploads(uploaded)
                    msg = backend.ingest(paths)
                st.success(msg)

    with col2:
        if st.button("Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    messages = st.session_state.setdefault("messages", [])
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask something about your uploaded content..."):
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = backend.answer(
                    prompt,
                    history=messages,
                    model=model,
                    top_k=top_k,
                    temperature=temperature,
                )
            st.markdown(answer)

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
