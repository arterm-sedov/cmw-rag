"""SGR planning tool (schema-guided request analysis).

This is a lightweight LangChain tool whose *arguments* are the plan itself.
We force the model to call this tool first, so the plan is emitted as a tool call
(structured args), and then stored into `AgentContext` for UI/batch use.
"""

from __future__ import annotations

import json
import logging

from langchain.tools import ToolRuntime, tool

from rag_engine.llm.schemas import SGRPlanResult
from rag_engine.utils.context_tracker import AgentContext

logger = logging.getLogger(__name__)


@tool("sgr_plan", args_schema=SGRPlanResult)
async def sgr_plan(
    spam_score: float,
    spam_reason: str,
    user_intent: str,
    subqueries: list[str],
    action_plan: list[str] | None = None,
    ask_for_clarification: bool = False,
    clarification_suggestion: str | None = None,
    runtime: ToolRuntime[AgentContext, None] | None = None,
) -> str:
    """Analyze the user request and produce the user question resolution plan.

    IMPORTANT:
    - Do NOT echo the plan to the user.
    """
    plan = {
        "spam_score": float(spam_score),
        "spam_reason": spam_reason,
        "user_intent": user_intent,
        "subqueries": list(subqueries),
        "action_plan": list(action_plan) if action_plan else [],
        "ask_for_clarification": bool(ask_for_clarification),
        "clarification_suggestion": clarification_suggestion,
    }

    if runtime and hasattr(runtime, "context") and runtime.context is not None:
        try:
            runtime.context.sgr_plan = plan
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to store sgr_plan into AgentContext: %s", exc)

    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))

