from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from rag_engine.api import app as api_app
from rag_engine.utils.context_tracker import AgentContext


async def _collect_async(gen):
    out = []
    async for item in gen:
        out.append(item)
    return out


def test_ask_comindware_structured_collects_context(monkeypatch):
    async def fake_handler(*args, **kwargs):  # noqa: ANN001, ANN003
        yield []
        ctx = AgentContext()
        ctx.sgr_plan = {
            "spam_score": 0.1,
            "spam_reason": "ok",
            "user_intent": "intent",
            "topic": "test",
            "category": "help",
            "intent_confidence": 0.8,
            "knowledge_base_search_queries": ["q1"],
            "action": "proceed",
        }
        ctx.query_traces = [{"query": "q1", "confidence": {}, "articles": []}]
        ctx.final_answer = "answer"
        ctx.final_articles = []
        ctx.diagnostics = {"x": 1}
        yield ctx

    # Mock settings to avoid requiring .env for tests
    mock_settings = MagicMock()
    mock_settings.srp_enabled = False
    monkeypatch.setattr(api_app, "settings", mock_settings)

    monkeypatch.setattr(api_app, "agent_chat_handler", fake_handler)

    res = asyncio.run(api_app.ask_comindware_structured("hi"))
    assert res.answer_text == "answer"
    assert res.plan.spam_score == 0.1
    assert res.per_query_results


def test_ask_comindware_structured_with_srp_enabled(monkeypatch):
    """Test with SRP enabled to exercise different code path."""

    async def fake_handler(*args, **kwargs):  # noqa: ANN001, ANN003
        yield []
        ctx = AgentContext()
        ctx.sgr_plan = {
            "spam_score": 0.0,
            "spam_reason": "",
            "user_intent": "test",
            "topic": "test",
            "category": "help",
            "intent_confidence": 0.9,
            "knowledge_base_search_queries": [],
            "action": "proceed",
        }
        ctx.query_traces = []
        ctx.final_answer = "Test answer with headers\n\n## Section 1\nContent"
        ctx.final_articles = []
        ctx.diagnostics = {}
        yield ctx

    # Mock settings with SRP enabled
    mock_settings = MagicMock()
    mock_settings.srp_enabled = True
    monkeypatch.setattr(api_app, "settings", mock_settings)

    monkeypatch.setattr(api_app, "agent_chat_handler", fake_handler)

    res = asyncio.run(api_app.ask_comindware_structured("test"))
    assert res.answer_text
