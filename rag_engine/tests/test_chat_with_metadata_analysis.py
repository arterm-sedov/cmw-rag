"""Test that chat_with_metadata wires timing and model into analysis_data (structured flow)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from rag_engine.api import app as api_app
from rag_engine.utils.context_tracker import AgentContext


async def _collect_async(gen):
    out = []
    async for item in gen:
        out.append(item)
    return out


async def _chat_with_metadata_yields_analysis_with_timing_and_model(*args, **kwargs):
    """Fake agent_chat_handler yields history then AgentContext with turn_time_ms and model_used."""
    yield [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    ctx = AgentContext(
        turn_time_ms=150.5,
        model_used="openrouter: deepseek/deepseek-v3.1-terminus",
        diagnostics={"session_id": "test-session-flow"},
        usage_turn_summary={"prompt_tokens": 10, "completion_tokens": 20},
        sgr_plan={
            "user_intent": "greeting",
            "knowledge_base_search_queries": [],
            "action_plan": [],
        },
        query_traces=[],
        executed_queries=[],
        final_answer="hello",
        final_articles=[],
    )
    yield ctx


def test_chat_with_metadata_analysis_data_contains_timing_and_model(monkeypatch):
    """Programmatically verify analysis_data includes last_turn_time_s, total_conversation_time_s, model_used."""
    monkeypatch.setattr(api_app, "agent_chat_handler", _chat_with_metadata_yields_analysis_with_timing_and_model)

    def fake_accumulate(session_id, turn_summary):
        return {
            "prompt_tokens": turn_summary.get("prompt_tokens", 0),
            "completion_tokens": turn_summary.get("completion_tokens", 0),
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "cost": 0.0,
            "upstream_cost": 0.0,
            "total_conversation_time_ms": 250.0,
        }

    with patch.object(api_app, "accumulate_conversation_usage", side_effect=fake_accumulate):
        outputs = asyncio.run(_collect_async(api_app.chat_with_metadata("hi", [])))

    assert len(outputs) >= 1
    last = outputs[-1]
    assert isinstance(last, tuple), "chat_with_metadata yields tuples (chatbot, analysis_update, ...)"
    analysis_update = last[1]
    assert analysis_update is not None
    analysis_data = analysis_update.get("value") if isinstance(analysis_update, dict) else getattr(analysis_update, "value", None)
    assert analysis_data is not None, "analysis_update should carry value=analysis_data"
    assert "last_turn_time_s" in analysis_data
    assert analysis_data["last_turn_time_s"] == 0.1505  # 150.5 ms -> s, 4 decimals
    assert "total_conversation_time_s" in analysis_data
    assert analysis_data["total_conversation_time_s"] == 0.25  # 250 ms -> s
    assert "model_used" in analysis_data
    assert analysis_data["model_used"] == "openrouter: deepseek/deepseek-v3.1-terminus"
