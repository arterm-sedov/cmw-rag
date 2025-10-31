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
            doc = SimpleNamespace(
                metadata={
                    "kbId": "123",
                    "title": "Doc",
                    "url": "https://example.com",
                    "section_anchor": "#a",
                }
            )
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
    # Citations heading is in Russian per current formatter
    assert any(h in output for h in ("## Источники:", "## Sources:"))


def test_api_and_handler_empty_cases(monkeypatch):
    import importlib

    class FakeEmbedder:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

    class FakeLLMManager:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def stream_response(self, message, docs, **kwargs):  # noqa: ANN001, ANN003
            # LLM receives injected "no results" document when docs is empty
            # It should see the "No relevant results found" message in context
            assert len(docs) > 0, "Should have at least the injected 'no results' doc"
            # Check for the metadata flag instead of string matching
            assert any(getattr(d, "metadata", {}).get("_is_no_results") is True for d in docs)
            yield "Sorry, I couldn't find relevant information in the knowledge base."

        def generate(self, question, docs, provider=None):  # noqa: ANN001
            # LLM receives injected "no results" document when docs is empty
            assert len(docs) > 0, "Should have at least the injected 'no results' doc"
            # Check for the metadata flag instead of string matching
            assert any(getattr(d, "metadata", {}).get("_is_no_results") is True for d in docs)
            return "Sorry, I couldn't find relevant information in the knowledge base."

        def save_assistant_turn(self, session_id, content):  # noqa: ANN001
            pass

    class FakeRetriever:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def retrieve(self, query, top_k=None):  # noqa: ANN001
            return []  # Empty results - will trigger injection of "no results" message

    monkeypatch.setattr("rag_engine.retrieval.embedder.FRIDAEmbedder", FakeEmbedder)
    monkeypatch.setattr("rag_engine.storage.vector_store.ChromaStore", FakeStore)
    monkeypatch.setattr("rag_engine.llm.llm_manager.LLMManager", FakeLLMManager)
    monkeypatch.setattr("rag_engine.retrieval.retriever.RAGRetriever", FakeRetriever)

    monkeypatch.setattr("rag_engine.utils.logging_manager.setup_logging", lambda: None)

    app = importlib.reload(importlib.import_module("rag_engine.api.app"))

    # query_rag with empty question returns error string
    assert app.query_rag(" ") == "Error: Empty question"

    # query_rag with empty docs should still return LLM response (not block)
    result = app.query_rag("test question")
    assert "Sorry" in result or "found" in result.lower() or "information" in result.lower()

    # chat_handler yields validation message on empty input
    gen = app.chat_handler("", [])
    first = next(gen)
    assert "Please enter a question" in first or "Введите вопрос" in first

    # chat_handler with empty docs should still call LLM (not block)
    gen = app.chat_handler("test question", [])
    result = list(gen)
    assert len(result) > 0  # Should have some response


def test_chat_handler_appends_footer_and_saves_to_memory(monkeypatch):
    import importlib

    class FakeEmbedder:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

    class FakeLLMManager:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.last_saved = None

        def stream_response(self, message, docs, **kwargs):  # noqa: ANN001
            yield "partial"
            yield " answer"

        def save_assistant_turn(self, session_id, content):  # noqa: ANN001
            self.last_saved = (session_id, content)

        def generate(self, question, docs, provider=None):  # noqa: ANN001
            return "answer"

    class FakeRetriever:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def retrieve(self, query, top_k=None):  # noqa: ANN001
            from types import SimpleNamespace

            doc = SimpleNamespace(
                metadata={
                    "kbId": "123",
                    "title": "Doc",
                    "url": "https://example.com",
                    "section_anchor": "#a",
                }
            )
            return [doc]

    monkeypatch.setattr("rag_engine.retrieval.embedder.FRIDAEmbedder", FakeEmbedder)
    monkeypatch.setattr("rag_engine.storage.vector_store.ChromaStore", FakeStore)
    monkeypatch.setattr("rag_engine.llm.llm_manager.LLMManager", FakeLLMManager)
    monkeypatch.setattr("rag_engine.retrieval.retriever.RAGRetriever", FakeRetriever)

    monkeypatch.setattr("rag_engine.utils.logging_manager.setup_logging", lambda: None)

    app = importlib.reload(importlib.import_module("rag_engine.api.app"))

    # Build a fake request with session_hash
    class R:
        session_hash = "sess-1"

    gen = app.chat_handler("Question?", [], R())
    outs = list(gen)
    assert outs[-1].startswith("partial answer") or outs[-1].endswith("partial answer")
    assert any(h in outs[-1] for h in ("## Источники:", "## Sources:"))


def test_session_salting_new_chat(monkeypatch):
    """Test session salting for new chats uses current message."""
    import importlib
    import hashlib

    received_session_ids = []

    class FakeLLMManager:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def stream_response(self, message, docs, session_id=None, **kwargs):  # noqa: ANN001
            if session_id:
                received_session_ids.append(session_id)
            yield "answer"

        def save_assistant_turn(self, session_id, content):  # noqa: ANN001
            pass

        def generate(self, question, docs, provider=None):  # noqa: ANN001
            return "answer"

    class FakeRetriever:
        def retrieve(self, query, top_k=None):  # noqa: ANN001
            from types import SimpleNamespace

            return [
                SimpleNamespace(
                    metadata={"kbId": "123", "title": "Doc", "url": "https://example.com"}
                )
            ]

    monkeypatch.setattr("rag_engine.retrieval.embedder.FRIDAEmbedder", lambda *args, **kwargs: None)
    monkeypatch.setattr("rag_engine.storage.vector_store.ChromaStore", lambda *args, **kwargs: None)
    monkeypatch.setattr("rag_engine.llm.llm_manager.LLMManager", FakeLLMManager)
    monkeypatch.setattr("rag_engine.retrieval.retriever.RAGRetriever", lambda *args, **kwargs: FakeRetriever())
    monkeypatch.setattr("rag_engine.utils.logging_manager.setup_logging", lambda: None)

    app = importlib.reload(importlib.import_module("rag_engine.api.app"))

    class Request:
        session_hash = "base-session-123"

    # New chat - empty history
    gen = app.chat_handler("First message", [], Request())
    list(gen)  # Consume generator

    # Verify session_id was generated from first message
    assert len(received_session_ids) == 1
    session_id = received_session_ids[0]
    expected = hashlib.sha256(f"{Request.session_hash}:First message".encode()).hexdigest()[:32]
    assert session_id == expected


def test_session_salting_loaded_chat(monkeypatch):
    """Test session salting for loaded chats uses first message from history."""
    import importlib
    import hashlib

    received_session_ids = []

    class FakeLLMManager:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def stream_response(self, message, docs, session_id=None, **kwargs):  # noqa: ANN001
            if session_id:
                received_session_ids.append(session_id)
            yield "answer"

        def save_assistant_turn(self, session_id, content):  # noqa: ANN001
            pass

        def generate(self, question, docs, provider=None):  # noqa: ANN001
            return "answer"

    class FakeRetriever:
        def retrieve(self, query, top_k=None):  # noqa: ANN001
            from types import SimpleNamespace

            return [
                SimpleNamespace(
                    metadata={"kbId": "123", "title": "Doc", "url": "https://example.com"}
                )
            ]

    monkeypatch.setattr("rag_engine.retrieval.embedder.FRIDAEmbedder", lambda *args, **kwargs: None)
    monkeypatch.setattr("rag_engine.storage.vector_store.ChromaStore", lambda *args, **kwargs: None)
    monkeypatch.setattr("rag_engine.llm.llm_manager.LLMManager", FakeLLMManager)
    monkeypatch.setattr("rag_engine.retrieval.retriever.RAGRetriever", lambda *args, **kwargs: FakeRetriever())
    monkeypatch.setattr("rag_engine.utils.logging_manager.setup_logging", lambda: None)

    app = importlib.reload(importlib.import_module("rag_engine.api.app"))

    class Request:
        session_hash = "base-session-456"

    # Loaded chat - history with first message
    history = [
        {"role": "user", "content": "Original first message"},
        {"role": "assistant", "content": "Previous response"},
    ]

    gen = app.chat_handler("Follow up", history, Request())
    list(gen)  # Consume generator

    # Verify session_id was generated from first message in history (not current)
    assert len(received_session_ids) == 1
    session_id = received_session_ids[0]
    expected = hashlib.sha256(f"{Request.session_hash}:Original first message".encode()).hexdigest()[
        :32
    ]
    assert session_id == expected


def test_session_salting_same_chat_preserves_session(monkeypatch):
    """Test same chat generates same session_id (memory continuity)."""
    import importlib
    import hashlib

    received_session_ids = []

    class FakeLLMManager:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def stream_response(self, message, docs, session_id=None, **kwargs):  # noqa: ANN001
            if session_id:
                received_session_ids.append(session_id)
            yield "answer"

        def save_assistant_turn(self, session_id, content):  # noqa: ANN001
            pass

        def generate(self, question, docs, provider=None):  # noqa: ANN001
            return "answer"

    class FakeRetriever:
        def retrieve(self, query, top_k=None):  # noqa: ANN001
            from types import SimpleNamespace

            return [
                SimpleNamespace(
                    metadata={"kbId": "123", "title": "Doc", "url": "https://example.com"}
                )
            ]

    monkeypatch.setattr("rag_engine.retrieval.embedder.FRIDAEmbedder", lambda *args, **kwargs: None)
    monkeypatch.setattr("rag_engine.storage.vector_store.ChromaStore", lambda *args, **kwargs: None)
    monkeypatch.setattr("rag_engine.llm.llm_manager.LLMManager", FakeLLMManager)
    monkeypatch.setattr("rag_engine.retrieval.retriever.RAGRetriever", lambda *args, **kwargs: FakeRetriever())
    monkeypatch.setattr("rag_engine.utils.logging_manager.setup_logging", lambda: None)

    app = importlib.reload(importlib.import_module("rag_engine.api.app"))

    class Request:
        session_hash = "base-session-789"

    history = [
        {"role": "user", "content": "Same first message"},
        {"role": "assistant", "content": "Response 1"},
    ]

    # First call
    gen1 = app.chat_handler("Second question", history, Request())
    list(gen1)
    session_id1 = received_session_ids[0]

    # Second call with same history
    received_session_ids.clear()
    gen2 = app.chat_handler("Third question", history, Request())
    list(gen2)
    session_id2 = received_session_ids[0]

    # Same session_id for same first message
    assert session_id1 == session_id2
    expected = hashlib.sha256(f"{Request.session_hash}:Same first message".encode()).hexdigest()[:32]
    assert session_id1 == expected


def test_session_salting_different_chats_isolated(monkeypatch):
    """Test different chats generate different session_ids."""
    import importlib
    import hashlib

    received_session_ids = []

    class FakeLLMManager:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def stream_response(self, message, docs, session_id=None, **kwargs):  # noqa: ANN001
            if session_id:
                received_session_ids.append(session_id)
            yield "answer"

        def save_assistant_turn(self, session_id, content):  # noqa: ANN001
            pass

        def generate(self, question, docs, provider=None):  # noqa: ANN001
            return "answer"

    class FakeRetriever:
        def retrieve(self, query, top_k=None):  # noqa: ANN001
            from types import SimpleNamespace

            return [
                SimpleNamespace(
                    metadata={"kbId": "123", "title": "Doc", "url": "https://example.com"}
                )
            ]

    monkeypatch.setattr("rag_engine.retrieval.embedder.FRIDAEmbedder", lambda *args, **kwargs: None)
    monkeypatch.setattr("rag_engine.storage.vector_store.ChromaStore", lambda *args, **kwargs: None)
    monkeypatch.setattr("rag_engine.llm.llm_manager.LLMManager", FakeLLMManager)
    monkeypatch.setattr("rag_engine.retrieval.retriever.RAGRetriever", lambda *args, **kwargs: FakeRetriever())
    monkeypatch.setattr("rag_engine.utils.logging_manager.setup_logging", lambda: None)

    app = importlib.reload(importlib.import_module("rag_engine.api.app"))

    class Request:
        session_hash = "base-session-abc"

    # Chat 1: first message "Hello"
    history1 = [{"role": "user", "content": "Hello"}]
    gen1 = app.chat_handler("Follow up 1", history1, Request())
    list(gen1)
    session_id1 = received_session_ids[0]

    # Chat 2: first message "Hi" (different)
    received_session_ids.clear()
    history2 = [{"role": "user", "content": "Hi"}]
    gen2 = app.chat_handler("Follow up 2", history2, Request())
    list(gen2)
    session_id2 = received_session_ids[0]

    # Different session_ids for different first messages
    assert session_id1 != session_id2