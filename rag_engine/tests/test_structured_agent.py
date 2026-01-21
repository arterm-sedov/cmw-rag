from __future__ import annotations

import asyncio

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
            "subqueries": ["q1"],
        }
        ctx.query_traces = [{"query": "q1", "confidence": {}, "articles": []}]
        ctx.final_answer = "answer"
        ctx.final_articles = []
        ctx.diagnostics = {"x": 1}
        yield ctx

    monkeypatch.setattr(api_app, "agent_chat_handler", fake_handler)

    res = asyncio.run(api_app.ask_comindware_structured("hi"))
    assert res.answer_text == "answer"
    assert res.plan.spam_score == 0.1
    assert res.per_query_results

