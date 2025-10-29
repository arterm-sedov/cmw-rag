from __future__ import annotations

import importlib
from types import SimpleNamespace


def test_chat_interface_initialization(monkeypatch):
    class FakeEmbedder:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

    class FakeLLMManager:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def stream_response(self, message, docs):  # noqa: ANN001
            yield "response"

        def generate(self, question, docs, provider=None):  # noqa: ANN001
            return "answer"

    class FakeRetriever:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def retrieve(self, query, top_k=None):  # noqa: ANN001
            doc = SimpleNamespace(metadata={"title": "Doc", "url": "https://example.com", "section_anchor": "#a"})
            return [doc]

    monkeypatch.setattr("rag_engine.retrieval.embedder.FRIDAEmbedder", FakeEmbedder)
    monkeypatch.setattr("rag_engine.storage.vector_store.ChromaStore", FakeStore)
    monkeypatch.setattr("rag_engine.llm.llm_manager.LLMManager", FakeLLMManager)
    monkeypatch.setattr("rag_engine.retrieval.retriever.RAGRetriever", FakeRetriever)

    monkeypatch.setattr("rag_engine.utils.logging_manager.setup_logging", lambda: None)

    app = importlib.reload(importlib.import_module("rag_engine.api.app"))

    assert hasattr(app, "demo")
    assert app.demo.title == "Comindware Platform Documentation Assistant"

    output = app.query_rag("Question?", provider="gemini", top_k=1)
    assert "## Sources" in output


def test_api_and_handler_empty_cases(monkeypatch):
    import importlib

    app = importlib.reload(importlib.import_module("rag_engine.api.app"))

    # query_rag with empty question returns error string
    assert app.query_rag(" ") == "Error: Empty question"

    # chat_handler yields validation message on empty
    gen = app.chat_handler("", [])
    first = next(gen)
    assert "Please enter a question" in first or "Введите вопрос" in first

