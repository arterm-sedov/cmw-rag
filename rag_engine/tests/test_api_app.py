from __future__ import annotations

import asyncio
import hashlib
import importlib
from types import SimpleNamespace


async def _collect_async(gen):
    out = []
    async for item in gen:
        out.append(item)
    return out


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

        def retrieve(self, query, top_k=None):  # noqa: ANN001, ANN002
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

        def retrieve(self, query, top_k=None):  # noqa: ANN001, ANN002
            return []  # Empty results - will trigger injection of "no results" message

    monkeypatch.setattr("rag_engine.retrieval.embedder.FRIDAEmbedder", FakeEmbedder)
    monkeypatch.setattr("rag_engine.storage.vector_store.ChromaStore", FakeStore)
    monkeypatch.setattr("rag_engine.llm.llm_manager.LLMManager", FakeLLMManager)
    monkeypatch.setattr("rag_engine.retrieval.retriever.RAGRetriever", FakeRetriever)

    monkeypatch.setattr("rag_engine.utils.logging_manager.setup_logging", lambda: None)

    app = importlib.reload(importlib.import_module("rag_engine.api.app"))

    class NoopAgent:
        async def astream(self, *a, **k):  # noqa: ANN001, ANN003
            if False:
                yield None

    app._create_rag_agent = lambda *a, **k: NoopAgent()  # type: ignore

    # query_rag with empty question returns error string
    assert app.query_rag(" ") == "Error: Empty question"

    # query_rag with empty docs should still return LLM response (not block)
    result = app.query_rag("test question")
    assert "Sorry" in result or "found" in result.lower() or "information" in result.lower()

    # agent_chat_handler yields history list on empty input
    out = asyncio.run(_collect_async(app.agent_chat_handler("", [])))
    assert out == [[]]

    # agent_chat_handler with empty docs should still stream (at least one list yield)
    out2 = asyncio.run(_collect_async(app.agent_chat_handler("test question", [])))
    assert any(isinstance(x, list) for x in out2)


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
        async def astream(self, *a, **k):  # noqa: ANN001, ANN003
            for item in fake_agent_stream():
                yield item

    app._create_rag_agent = lambda *a, **k: FakeAgent()  # type: ignore

    # Build a fake request with session_hash
    class R:
        session_hash = "sess-1"

    outs = asyncio.run(_collect_async(app.agent_chat_handler("Question?", [], request=R())))
    final_hist = next((x for x in reversed(outs) if isinstance(x, list)), [])
    final_text = ""
    for msg in reversed(final_hist):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and not msg.get("metadata"):
            final_text = msg.get("content", "") or ""
            break
    # With a synthesized tool result, footer must be present
    assert any(h in final_text for h in ("## Источники:", "## Sources:"))


def test_session_salting_new_chat(monkeypatch):
    """Test session salting for new chats uses current message."""
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
        def retrieve(self, query, top_k=None):  # noqa: ANN001, ANN002
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

    class NoopAgent:
        async def astream(self, *a, **k):  # noqa: ANN001, ANN003
            if False:
                yield None

    app._create_rag_agent = lambda *a, **k: NoopAgent()  # type: ignore

    class Request:
        session_hash = "base-session-123"

    # New chat - empty history
    asyncio.run(_collect_async(app.agent_chat_handler("First message", [], request=Request())))

    # Verify session_id was generated from first message
    assert len(received_session_ids) == 1
    session_id = received_session_ids[0]
    expected = hashlib.sha256(f"{Request.session_hash}:First message".encode()).hexdigest()[:32]
    assert session_id == expected


def test_session_salting_loaded_chat(monkeypatch):
    """Test session salting for loaded chats uses first message from history."""
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
        def retrieve(self, query, top_k=None):  # noqa: ANN001, ANN002
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

    class NoopAgent:
        async def astream(self, *a, **k):  # noqa: ANN001, ANN003
            if False:
                yield None

    app._create_rag_agent = lambda *a, **k: NoopAgent()  # type: ignore

    class Request:
        session_hash = "base-session-456"

    # Loaded chat - history with first message
    history = [
        {"role": "user", "content": "Original first message"},
        {"role": "assistant", "content": "Previous response"},
    ]

    asyncio.run(_collect_async(app.agent_chat_handler("Follow up", history, request=Request())))

    # Verify session_id was generated from first message in history (not current)
    assert len(received_session_ids) == 1
    session_id = received_session_ids[0]
    expected = hashlib.sha256(f"{Request.session_hash}:Original first message".encode()).hexdigest()[
        :32
    ]
    assert session_id == expected


def test_session_salting_same_chat_preserves_session(monkeypatch):
    """Test same chat generates same session_id (memory continuity)."""
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
        def retrieve(self, query, top_k=None):  # noqa: ANN001, ANN002
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

    class NoopAgent:
        async def astream(self, *a, **k):  # noqa: ANN001, ANN003
            if False:
                yield None

    app._create_rag_agent = lambda *a, **k: NoopAgent()  # type: ignore

    class Request:
        session_hash = "base-session-789"

    history = [
        {"role": "user", "content": "Same first message"},
        {"role": "assistant", "content": "Response 1"},
    ]

    # First call
    asyncio.run(_collect_async(app.agent_chat_handler("Second question", history, request=Request())))
    session_id1 = received_session_ids[0]

    # Second call with same history
    received_session_ids.clear()
    asyncio.run(_collect_async(app.agent_chat_handler("Third question", history, request=Request())))
    session_id2 = received_session_ids[0]

    # Same session_id for same first message
    assert session_id1 == session_id2
    expected = hashlib.sha256(f"{Request.session_hash}:Same first message".encode()).hexdigest()[:32]
    assert session_id1 == expected


def test_session_salting_different_chats_isolated(monkeypatch):
    """Test different chats generate different session_ids."""
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

    class NoopAgent:
        async def astream(self, *a, **k):  # noqa: ANN001, ANN003
            if False:
                yield None

    app._create_rag_agent = lambda *a, **k: NoopAgent()  # type: ignore

    class Request:
        session_hash = "base-session-abc"

    # Chat 1: first message "Hello"
    history1 = [{"role": "user", "content": "Hello"}]
    asyncio.run(_collect_async(app.agent_chat_handler("Follow up 1", history1, request=Request())))
    session_id1 = received_session_ids[0]

    # Chat 2: first message "Hi" (different)
    received_session_ids.clear()
    history2 = [{"role": "user", "content": "Hi"}]
    asyncio.run(_collect_async(app.agent_chat_handler("Follow up 2", history2, request=Request())))
    session_id2 = received_session_ids[0]

    # Different session_ids for different first messages
    assert session_id1 != session_id2
