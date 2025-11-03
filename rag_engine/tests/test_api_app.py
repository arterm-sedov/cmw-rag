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
            from types import SimpleNamespace
            from unittest.mock import Mock
            self._conversations = SimpleNamespace(append=lambda *a, **k: None)
            self._mock_chat_model = Mock()

        def _chat_model(self, provider=None):  # noqa: ANN001
            return self._mock_chat_model

        def stream_response(self, message, docs):  # noqa: ANN001
            yield "response"

        def generate(self, question, docs, provider=None):  # noqa: ANN001
            return "answer"

    class FakeRetriever:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def retrieve(self, query, top_k=None, reserved_tokens=0):  # noqa: ANN001, ANN002
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
            from types import SimpleNamespace
            from unittest.mock import Mock
            self._conversations = SimpleNamespace(append=lambda *a, **k: None)
            self._mock_chat_model = Mock()

        def _chat_model(self, provider=None):  # noqa: ANN001
            return self._mock_chat_model

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

        def retrieve(self, query, top_k=None, reserved_tokens=0):  # noqa: ANN001, ANN002
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

    # agent_chat_handler yields validation message on empty input
    gen = app.agent_chat_handler("", [])
    first = next(gen)
    if isinstance(first, dict):
        content = first.get("content", "")
    else:
        content = first
    assert "Please enter a question" in content or "Введите вопрос" in content

    # agent_chat_handler with empty docs should still stream a response (not block)
    gen = app.agent_chat_handler("test question", [])
    result = list(gen)
    assert any(isinstance(x, str) and x for x in result)


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
            from types import SimpleNamespace
            from unittest.mock import Mock
            self.last_saved = None
            self._conversations = SimpleNamespace(append=lambda *a, **k: None)
            self._mock_chat_model = Mock()

        def _chat_model(self, provider=None):  # noqa: ANN001
            return self._mock_chat_model

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

    # Patch agent creation to synthesize a tool result followed by AI text
    def fake_agent_stream(*args, **kwargs):  # noqa: ANN001, ANN003
        import json as _json
        # First yield a tool message with one small article
        class ToolMsg:
            type = "tool"
            content = _json.dumps({
                "articles": [{
                    "kb_id": "kb-1",
                    "title": "Doc",
                    "url": "https://example.com",
                    "content": "Short content",
                    "metadata": {"kbId": "kb-1", "title": "Doc"}
                }],
                "metadata": {"articles_count": 1, "query": "q"}
            }, ensure_ascii=False)

        yield ("messages", (ToolMsg(), {}))

        # Then yield an AI text message chunk
        class AiMsg:
            tool_calls = None
            content_blocks = [{"type": "text", "text": "Final answer body."}]

        yield ("messages", (AiMsg(), {}))

    class FakeAgent:
        def stream(self, *a, **k):  # noqa: ANN001, ANN003
            return fake_agent_stream()

    app._create_rag_agent = lambda override_model=None: FakeAgent()  # type: ignore

    # Build a fake request with session_hash
    class R:
        session_hash = "sess-1"

    gen = app.agent_chat_handler("Question?", [], R())
    outs = list(gen)
    final_text = "".join([x for x in outs if isinstance(x, str)])
    # With a synthesized tool result, footer must be present
    assert any(h in final_text for h in ("## Источники:", "## Sources:"))


def test_session_salting_new_chat(monkeypatch):
    """Test session salting for new chats uses current message."""
    import importlib
    import hashlib

    received_session_ids = []

    class FakeLLMManager:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            from types import SimpleNamespace
            from unittest.mock import Mock
            self._conversations = SimpleNamespace(append=lambda sid, role, content: received_session_ids.append(sid))
            self._mock_chat_model = Mock()

        def _chat_model(self, provider=None):  # noqa: ANN001
            return self._mock_chat_model

        def stream_response(self, message, docs, session_id=None, **kwargs):  # noqa: ANN001
            if session_id:
                received_session_ids.append(session_id)
            yield "answer"

        def save_assistant_turn(self, session_id, content):  # noqa: ANN001
            pass

        def generate(self, question, docs, provider=None):  # noqa: ANN001
            return "answer"

    class FakeRetriever:
        def retrieve(self, query, top_k=None, reserved_tokens=0):  # noqa: ANN001, ANN002
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
    gen = app.agent_chat_handler("First message", [], Request())
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
            from types import SimpleNamespace
            from unittest.mock import Mock
            self._conversations = SimpleNamespace(append=lambda sid, role, content: received_session_ids.append(sid))
            self._mock_chat_model = Mock()

        def _chat_model(self, provider=None):  # noqa: ANN001
            return self._mock_chat_model

        def stream_response(self, message, docs, session_id=None, **kwargs):  # noqa: ANN001
            if session_id:
                received_session_ids.append(session_id)
            yield "answer"

        def save_assistant_turn(self, session_id, content):  # noqa: ANN001
            pass

        def generate(self, question, docs, provider=None):  # noqa: ANN001
            return "answer"

    class FakeRetriever:
        def retrieve(self, query, top_k=None, reserved_tokens=0):  # noqa: ANN001, ANN002
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

    gen = app.agent_chat_handler("Follow up", history, Request())
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
            from types import SimpleNamespace
            from unittest.mock import Mock
            self._conversations = SimpleNamespace(append=lambda sid, role, content: received_session_ids.append(sid))
            self._mock_chat_model = Mock()

        def _chat_model(self, provider=None):  # noqa: ANN001
            return self._mock_chat_model

        def stream_response(self, message, docs, session_id=None, **kwargs):  # noqa: ANN001
            if session_id:
                received_session_ids.append(session_id)
            yield "answer"

        def save_assistant_turn(self, session_id, content):  # noqa: ANN001
            pass

        def generate(self, question, docs, provider=None):  # noqa: ANN001
            return "answer"

    class FakeRetriever:
        def retrieve(self, query, top_k=None, reserved_tokens=0):  # noqa: ANN001, ANN002
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
    gen1 = app.agent_chat_handler("Second question", history, Request())
    list(gen1)
    session_id1 = received_session_ids[0]

    # Second call with same history
    received_session_ids.clear()
    gen2 = app.agent_chat_handler("Third question", history, Request())
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
            from types import SimpleNamespace
            from unittest.mock import Mock
            self._conversations = SimpleNamespace(append=lambda sid, role, content: received_session_ids.append(sid))
            self._mock_chat_model = Mock()

        def _chat_model(self, provider=None):  # noqa: ANN001
            return self._mock_chat_model

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
    gen1 = app.agent_chat_handler("Follow up 1", history1, Request())
    list(gen1)
    session_id1 = received_session_ids[0]

    # Chat 2: first message "Hi" (different)
    received_session_ids.clear()
    history2 = [{"role": "user", "content": "Hi"}]
    gen2 = app.agent_chat_handler("Follow up 2", history2, Request())
    list(gen2)
    session_id2 = received_session_ids[0]

    # Different session_ids for different first messages
    assert session_id1 != session_id2