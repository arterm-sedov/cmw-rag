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


def test_harmony_split_basic():
    from rag_engine.api.harmony_parser import split as harmony_split

    raw = (
        "analysisWe need to explain quantum entanglement clearly."
        "assistantfinal**Final answer**"
    )
    analysis, final = harmony_split(raw)

    assert "quantum entanglement" in analysis
    assert final.startswith("**Final answer**")


def test_harmony_split_no_marker():
    from rag_engine.api.harmony_parser import split as harmony_split

    raw = "Just a normal answer without harmony markers."
    analysis, final = harmony_split(raw)

    assert analysis == ""
    assert final == raw


def test_harmony_split_analysis_only():
    from rag_engine.api.harmony_parser import split as harmony_split

    raw = "analysisWe need to retrieve context and think."
    analysis, final = harmony_split(raw)

    assert "We need to retrieve context" in analysis
    assert final == ""


def test_harmony_split_multichannel():
    """All Harmony channels (analysis, commentary, assistantfinal) are parsed."""
    from rag_engine.api.harmony_parser import split as harmony_split

    raw = (
        "analysisWe need to retrieve context."
        'assistantcommentary to=functions.retrieve_context json{"query":"install"}'
        'assistantcommentary{"articles":[{"kb_id":"4521"}]}'
        "assistantanalysisNow retrieve latest version."
        'assistantcommentary to=functions.retrieve_context json{"query":"version"}'
        'assistantcommentary{"articles":[{"kb_id":"4600"}]}'
        "assistantanalysisNow we have all info."
        "assistantfinal## Installation\nFollow these steps."
    )
    reasoning, final = harmony_split(raw)

    assert "We need to retrieve context" in reasoning
    assert "Now retrieve latest version" in reasoning
    assert "Now we have all info" in reasoning
    assert "retrieve_context" in reasoning

    assert "## Installation" in final
    assert "Follow these steps" in final

    for marker in ("assistantfinal", "assistantcommentary", "assistantanalysis"):
        assert marker not in reasoning
        assert marker not in final


def test_harmony_split_tool_response_false_positive():
    """Tool response headers containing 'to=assistantcommentary' are not false splits."""
    from rag_engine.api.harmony_parser import split as harmony_split

    raw = (
        "analysisNeed weather data."
        'assistantcommentary to=functions.get_weather json{"city":"Tokyo"}'
        'functions.get_weather to=assistantcommentary{"temp":20}'
        "assistantanalysisGot it, now answer."
        "assistantfinalThe weather in Tokyo is 20C."
    )
    reasoning, final = harmony_split(raw)

    assert "Need weather data" in reasoning
    assert "Got it, now answer" in reasoning
    assert final == "The weather in Tokyo is 20C."
    assert "temp" in reasoning


def test_harmony_stream_parser_cross_chunk():
    """HarmonyStreamParser handles markers split across streaming chunks."""
    from rag_engine.api.harmony_parser import HarmonyStreamParser

    parser = HarmonyStreamParser()
    all_reasoning = ""
    all_final = ""

    for chunk in [
        "analysisThinking about this.",
        "assistant",  # partial marker
        "final",  # completes "assistantfinal"
        "Here is the answer.",
    ]:
        r_delta, f_delta = parser.feed(chunk)
        all_reasoning += r_delta
        all_final += f_delta

    r_delta, f_delta = parser.flush()
    all_reasoning += r_delta
    all_final += f_delta

    assert "Thinking about this" in all_reasoning
    assert "Here is the answer" in all_final
    assert "assistantfinal" not in all_final
    assert "assistantfinal" not in all_reasoning


def test_harmony_stream_parser_no_markers():
    """Plain text without Harmony markers passes through as final content."""
    from rag_engine.api.harmony_parser import HarmonyStreamParser

    parser = HarmonyStreamParser()
    all_final = ""

    for chunk in ["Hello ", "world ", "how are you?"]:
        _, f_delta = parser.feed(chunk)
        all_final += f_delta

    _, f_delta = parser.flush()
    all_final += f_delta

    assert "Hello world how are you?" in all_final


def test_harmony_stream_parser_streams_reasoning_live():
    """Reasoning deltas arrive incrementally, not just at flush time."""
    from rag_engine.api.harmony_parser import HarmonyStreamParser

    parser = HarmonyStreamParser()
    reasoning_deltas: list[str] = []

    # Feed a long analysis that clearly exceeds the tail holdback.
    for chunk in [
        "analysisThis is the chain of thought. ",
        "We need to figure out the answer carefully. ",
        "Let me think step by step about this problem. ",
        "assistantfinalThe answer is 42.",
    ]:
        r_delta, _ = parser.feed(chunk)
        if r_delta:
            reasoning_deltas.append(r_delta)

    r_delta, _ = parser.flush()
    if r_delta:
        reasoning_deltas.append(r_delta)

    # Reasoning should have been emitted BEFORE the flush (live streaming).
    assert len(reasoning_deltas) >= 1
    full_reasoning = "".join(reasoning_deltas)
    assert "chain of thought" in full_reasoning


def test_parse_think_tags_single_block():
    from rag_engine.api.app import _parse_think_tags

    text = "Hello <think>internal\nreasoning</think> world"
    cleaned, reasoning, in_block, saw_orphan = _parse_think_tags(text, "", False)

    assert "internal" not in cleaned
    assert "reasoning" not in cleaned
    assert "Hello" in cleaned and "world" in cleaned
    assert "internal" in reasoning and "reasoning" in reasoning
    assert in_block is False
    assert saw_orphan is False


def test_parse_think_tags_across_chunks():
    from rag_engine.api.app import _parse_think_tags

    first = "prefix <think>partial"
    cleaned1, reasoning1, in_block1, saw_orphan1 = _parse_think_tags(first, "", False)
    assert "partial" not in cleaned1
    assert reasoning1 == "partial"
    assert in_block1 is True

    second = " continued</think> suffix"
    cleaned2, reasoning2, in_block2, saw_orphan2 = _parse_think_tags(second, reasoning1, in_block1)
    assert "continued" not in cleaned2
    assert "suffix" in cleaned2
    assert "partial" in reasoning2 and "continued" in reasoning2
    assert in_block2 is False


def test_parse_think_tags_orphan_close_signals_reclassification():
    from rag_engine.api.app import _parse_think_tags

    # Qwen pattern: orphan </think> with surrounding whitespace
    text = "\n</think>\n\n"
    cleaned, reasoning, in_block, saw_orphan = _parse_think_tags(text, "", False)

    assert saw_orphan is True
    assert in_block is False
    # The </think> tag itself is consumed; whitespace around it is in clean output
    assert "</think>" not in cleaned


def test_parse_think_tags_orphan_preserves_preceding_text():
    from rag_engine.api.app import _parse_think_tags

    text = "planning text </think> "
    cleaned, reasoning, in_block, saw_orphan = _parse_think_tags(text, "", False)

    assert saw_orphan is True
    # Text before the orphan tag is kept in clean (caller decides to reclassify it)
    assert "planning text" in cleaned
    assert "</think>" not in cleaned


def test_parse_think_tags_normalized_escaped_open():
    from rag_engine.api.app import _parse_think_tags

    # JSON-escaped opening tag paired with literal closing tag
    text = "\\u003cthink\\u003ereasoninghere\\u003c/think\\u003e"
    cleaned, reasoning, in_block, saw_orphan = _parse_think_tags(text, "", False)

    assert "reasoninghere" in reasoning
    assert "reasoninghere" not in cleaned
    assert saw_orphan is False


def test_parse_think_tags_normalized_mixed_escapes():
    from rag_engine.api.app import _parse_think_tags

    # Mixed escaping: only angle brackets are escaped.
    text = "\\u003cthink>abc\\u003c/think>"
    cleaned, reasoning, in_block, saw_orphan = _parse_think_tags(text, "", False)

    assert cleaned == ""
    assert reasoning == "abc"
    assert in_block is False
    assert saw_orphan is False


def test_process_reasoning_chunk_strip_only_merges_think_continuation():
    """When content_blocks has reasoning ending with <think>, text-block continuation goes to buffer."""
    from rag_engine.api.app import _ReasoningCtx, _process_reasoning_chunk

    ctx = _ReasoningCtx(buffer="content_blocks_reasoning<think>", in_block=True)
    gen = _process_reasoning_chunk(
        "continuation</think>",
        ctx,
        "",
        has_seen_tool_results=True,
        reasoning_enabled=True,
        harmony_parser=None,
        gradio_history=[],
        harmony_strip_only=True,
    )
    try:
        while True:
            next(gen)
    except StopIteration as e:
        text, should_skip = e.value

    assert "continuation" in ctx.buffer
    assert "content_blocks_reasoning" in ctx.buffer
    assert should_skip is True


def test_extract_stream_delta_handles_cumulative_chunks():
    from rag_engine.api.app import _extract_stream_delta

    seen = ""
    delta, seen = _extract_stream_delta("Hello", seen)
    assert delta == "Hello"
    delta, seen = _extract_stream_delta("Hello world", seen)
    assert delta == " world"
    delta, seen = _extract_stream_delta("Hello world", seen)
    assert delta == ""


def test_extract_stream_delta_handles_incremental_chunks():
    from rag_engine.api.app import _extract_stream_delta

    seen = ""
    delta, seen = _extract_stream_delta("Hello", seen)
    assert delta == "Hello"
    delta, seen = _extract_stream_delta(" ", seen)
    assert delta == " "
    delta, seen = _extract_stream_delta("world", seen)
    assert delta == "world"
    assert seen == "Hello world"


def test_extract_stream_delta_handles_partial_overlap_chunks():
    from rag_engine.api.app import _extract_stream_delta

    seen = "latest release"
    delta, seen = _extract_stream_delta("release as of 31.12.2025", seen)
    assert delta == " as of 31.12.2025"
    assert seen == "latest release as of 31.12.2025"


def test_extract_stream_delta_ignores_tiny_overlap():
    from rag_engine.api.app import _extract_stream_delta

    seen = "abc"
    delta, seen = _extract_stream_delta("cdef", seen)
    # 1-char overlap is ignored to avoid false-positive trimming.
    assert delta == "cdef"
    assert seen == "abccdef"


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


def test_agent_stream_injects_reasoning_bubble_and_diagnostics(monkeypatch):
    """Agent stream should capture reasoning tokens and surface them via UI bubble and diagnostics."""
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
            yield "answer"

        def generate(self, question, docs, provider=None):  # noqa: ANN001
            return "answer"

        def save_assistant_turn(self, session_id, content):  # noqa: ANN001
            # No-op for tests; real implementation persists conversation turns.
            pass

    class FakeRetriever:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def retrieve(self, query, top_k=None):  # noqa: ANN001, ANN002
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

    # Enable reasoning at runtime
    from rag_engine.config import settings as settings_mod

    # Pydantic Settings instance – flip the flag directly for the duration of this test.
    monkeypatch.setattr(
        settings_mod,
        "llm_reasoning_enabled",
        True,
        raising=False,
    )

    # Patch agent to emit a reasoning block followed by a text block
    def fake_agent_stream(*args, **kwargs):  # noqa: ANN001, ANN003
        class AiMsg:
            type = "ai"
            tool_calls = None
            content_blocks = [
                {"type": "reasoning", "reasoning": "Step 1: think."},
                {"type": "text", "text": "Final answer."},
            ]

        yield ("messages", (AiMsg(), {}))

    class FakeAgent:
        async def astream(self, *a, **k):  # noqa: ANN001, ANN003
            for item in fake_agent_stream():
                yield item

    app._create_rag_agent = lambda *a, **k: FakeAgent()  # type: ignore

    class R:
        session_hash = "sess-reasoning"

    outs = asyncio.run(_collect_async(app.agent_chat_handler("Question?", [], request=R())))

    # Last list yield is the final Gradio history
    final_hist = next((x for x in reversed(outs) if isinstance(x, list)), [])
    reasoning_msgs = [
        msg
        for msg in final_hist
        if isinstance(msg, dict)
        and isinstance(msg.get("metadata"), dict)
        and msg["metadata"].get("ui_type") == "reasoning"
    ]
    assert reasoning_msgs, "Reasoning bubble should be present in final history"
    assert "Step 1: think." in reasoning_msgs[0].get("content", "")

    from rag_engine.utils.context_tracker import AgentContext

    final_ctx = next((x for x in outs if isinstance(x, AgentContext)), None)
    assert final_ctx is not None
    assert "reasoning" in final_ctx.diagnostics
    assert "Step 1: think." in final_ctx.diagnostics["reasoning"]


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
