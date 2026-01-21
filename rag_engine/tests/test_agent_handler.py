"""Tests for streaming handler + structured context output."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock, patch

from rag_engine.api.app import (
    AgentContext,
    _check_context_fallback,
    _create_rag_agent,
    agent_chat_handler,
)


async def _collect_async(gen):
    out = []
    async for item in gen:
        out.append(item)
    return out


def test_check_context_fallback_smoke(monkeypatch):
    monkeypatch.setattr("rag_engine.api.app.settings.llm_fallback_enabled", False, raising=False)
    assert _check_context_fallback([{"role": "user", "content": "hi"}]) is None


def test_create_rag_agent_wrapper_passes_sgr_flags():
    with patch("rag_engine.llm.agent_factory.create_rag_agent") as mock_create:
        mock_create.return_value = object()
        _create_rag_agent(enable_sgr_planning=True, sgr_spam_threshold=0.9)
        assert mock_create.call_args.kwargs["enable_sgr_planning"] is True
        assert mock_create.call_args.kwargs["sgr_spam_threshold"] == 0.9


def test_agent_chat_handler_empty_message_yields_history_list():
    out = asyncio.run(_collect_async(agent_chat_handler(message="", history=[])))
    assert out == [[]]


def test_agent_chat_handler_yields_agent_context_at_end(monkeypatch):
    class FakeAgent:
        async def astream(self, *args, **kwargs):  # noqa: ANN002, ANN003
            # Stream a single text chunk via content_blocks
            token = Mock()
            token.type = "ai"
            token.tool_calls = None
            token.content_blocks = [{"type": "text", "text": "Final answer."}]
            token.response_metadata = {}
            yield ("messages", (token, {"langgraph_node": "model"}))

    monkeypatch.setattr("rag_engine.api.app._create_rag_agent", lambda *a, **k: FakeAgent())

    out = asyncio.run(_collect_async(agent_chat_handler(message="q", history=[])))
    assert any(isinstance(x, list) for x in out)
    assert isinstance(out[-1], AgentContext)
    assert out[-1].final_answer.strip() != ""
