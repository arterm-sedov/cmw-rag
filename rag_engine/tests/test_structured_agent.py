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
        ctx.diagnostics = {"x": 1, "session_id": "sess-1"}
        ctx.usage_turn_summary = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "reasoning_tokens": 5,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "cost": 0.001,
            "upstream_cost": 0.0005,
        }
        ctx.turn_time_ms = 1000.0
        ctx.model_used = "provider: model-a"
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
    # Usage block and diagnostics are populated for Comindware wiring
    assert res.usage is not None
    assert res.usage.turn is not None
    assert res.usage.conversation is not None
    # Diagnostics contain usage_conversation and timing/model fields
    assert "usage_conversation" in res.diagnostics
    usage_conv = res.diagnostics["usage_conversation"]
    assert usage_conv.get("total_tokens") == 30.0
    assert "total_conversation_time_ms" in usage_conv
    assert res.diagnostics.get("last_turn_time_s") > 0
    assert res.diagnostics.get("total_conversation_time_s") > 0
    assert res.diagnostics.get("model_used") == "provider: model-a"


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
