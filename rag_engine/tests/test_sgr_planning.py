from __future__ import annotations

from unittest.mock import Mock

from rag_engine.llm.schemas import SGRPlanResult
from rag_engine.llm.sgr_planning import run_sgr_planning


def test_sgr_plan_schema_constraints():
    plan = SGRPlanResult(
        spam_score=0.1,
        spam_reason="Связано с платформой и настройкой.",
        user_intent="Пользователь хочет настроить форму.",
        subqueries=["настройка форм", "поля формы"],
        action_plan=["Открыть конструктор форм", "Настроить поля"],
        ask_for_clarification=False,
        clarification_suggestion=None,
    )
    assert 0.0 <= plan.spam_score <= 1.0
    assert len(plan.subqueries) >= 1


def test_run_sgr_planning_uses_structured_model(monkeypatch):
    fake_model = Mock()
    fake_model.invoke.return_value = SGRPlanResult(
        spam_score=0.2,
        spam_reason="ok",
        user_intent="intent",
        subqueries=["q1"],
    )
    fake_llm = Mock()
    fake_llm._chat_model.return_value = fake_model

    out = run_sgr_planning("req", fake_llm)
    assert isinstance(out, SGRPlanResult)
    fake_llm._chat_model.assert_called_once()
    fake_model.invoke.assert_called_once()

