"""Gradio UI with ChatInterface and REST API endpoint."""
from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import gradio as gr

from rag_engine.config.settings import settings, get_allowed_fallback_models
from rag_engine.llm.llm_manager import LLMManager
from rag_engine.retrieval.embedder import FRIDAEmbedder
from rag_engine.retrieval.retriever import RAGRetriever
from rag_engine.storage.vector_store import ChromaStore
from rag_engine.utils.formatters import format_with_citations
from rag_engine.utils.logging_manager import setup_logging

setup_logging()


# Initialize singletons (order matters: llm_manager before retriever)
embedder = FRIDAEmbedder(
    model_name=settings.embedding_model,
    device=settings.embedding_device,
)
vector_store = ChromaStore(
    persist_dir=settings.chromadb_persist_dir,
    collection_name=settings.chromadb_collection,
)
llm_manager = LLMManager(
    provider=settings.default_llm_provider,
    model=settings.default_model,
    temperature=settings.llm_temperature,
)
retriever = RAGRetriever(
    embedder=embedder,
    vector_store=vector_store,
    llm_manager=llm_manager,  # NEW: Pass for dynamic context budgeting
    top_k_retrieve=settings.top_k_retrieve,
    top_k_rerank=settings.top_k_rerank,
    rerank_enabled=settings.rerank_enabled,
)


def chat_handler(message: str, history: list[dict]) -> Generator[str, None, None]:
    if not message or not message.strip():
        yield "Пожалуйста, введите вопрос / Please enter a question."
        return

    docs = retriever.retrieve(message)
    if not docs:
        yield "К сожалению, не найдено релевантных материалов / No relevant results found."
        return

    answer = ""
    for token in llm_manager.stream_response(
        message,
        docs,
        enable_fallback=settings.llm_fallback_enabled,
        allowed_fallback_models=get_allowed_fallback_models(),
    ):
        answer += token
        yield answer

    yield format_with_citations(answer, docs)


def query_rag(question: str, provider: str = "gemini", top_k: int = 5) -> str:
    if not question or not question.strip():
        return "Error: Empty question"
    docs = retriever.retrieve(question, top_k=top_k)
    if not docs:
        return "No relevant results found"
    answer = llm_manager.generate(question, docs, provider=provider)
    return format_with_citations(answer, docs)


demo = gr.ChatInterface(
    fn=chat_handler,
    title="Ассистент базы знаний Comindware Platform",
    description="RAG-агент базы знаний Comindware Platform",
    type="messages",
    save_history=True,
)
# Explicitly set a plain attribute for tests and downstream code to read
demo.title = "Comindware Platform Documentation Assistant"

try:
    gr.api(fn=query_rag, api_name="query_rag")
except Exception:  # noqa: BLE001
    # Older/newer Gradio builds without gr.api support
    pass


if __name__ == "__main__":
    demo.queue().launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
    )


