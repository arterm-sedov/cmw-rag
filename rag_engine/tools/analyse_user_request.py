"""User request analysis tool (schema-guided request analysis).

This is a lightweight LangChain tool whose *arguments* are the plan itself.
We force the model to call this tool first, so the plan is emitted as a tool call
(structured args), and then stored into `AgentContext` for UI/batch use.

The tool returns a formatted markdown directive for the LLM context, not raw JSON.
"""

from __future__ import annotations

import logging

from langchain.tools import ToolRuntime, tool

from rag_engine.llm.schemas import SGRAction, SGRCategory, SGRPlanResult
from rag_engine.utils.context_tracker import AgentContext

logger = logging.getLogger(__name__)


def _render_proceed_template(plan: dict, guardian_result: dict | None = None) -> str:
    """Render proceed template with optional content advisory."""
    queries = "\n   - ".join(plan.get("knowledge_base_search_queries", []))

    advisory = ""
    if guardian_result and guardian_result.get("safety_level") == "Controversial":
        categories = ", ".join(guardian_result.get("categories", []))
        advisory = f"""
⚠️ **Content Advisory**: This request was flagged as potentially controversial ({categories}).
Respond with factual, neutral tone. Avoid speculation or subjective opinions."""

    return f"""**Request Analysis**
Intent: {plan["user_intent"]}
Category: {plan["category"]}
Confidence: {plan["intent_confidence"] * 100:.0f}%
{advisory}

**Required next steps**
1. Call retrieve_context tool with these queries:
   - {queries}
2. Review results and provide answer based on retrieved documentation"""


def _render_clarify_template(plan: dict) -> str:
    """Render clarification needed template."""
    questions = "\n   - ".join(plan.get("clarification_questions_to_ask", []))
    return f"""**Request Analysis**
Intent: {plan["user_intent"]}
Category: {plan["category"]}
Confidence: {plan["intent_confidence"] * 100:.0f}% (низкая)

**Required next steps**
1. Ask the user these clarification questions:
   - {questions}
   - [+ add any additional questions if needed]
2. Wait for user response before proceeding"""


def _render_decline_template(plan: dict) -> str:
    """Render decline template (spam/off-topic)."""
    refusal = "Извините, я не могу помочь с этим запросом. Он не относится к Comindware Platform."
    return f"""**Request Analysis**
Assessment: Запрос не относится к Comindware Platform
Reason: {plan["spam_reason"]}
Spam Score: {plan["spam_score"]}

**Required next steps**
Do not process this request. Use this refusal message:

"{refusal}"""


def render_sgr_template(
    action: str,
    plan: dict,
    guardian_result: dict | None = None,
) -> str:
    """Render SGR plan as directive tool result.

    Args:
        action: Action from SGR analysis (proceed, ask_clarification, decline)
        plan: SGR analysis plan dict
        guardian_result: Optional guardian classification for advisory note

    Returns:
        Formatted markdown string for LLM context
    """
    if action == SGRAction.PROCEED:
        return _render_proceed_template(plan, guardian_result)
    elif action == SGRAction.ASK_CLARIFICATION:
        return _render_clarify_template(plan)
    elif action == SGRAction.DECLINE:
        return _render_decline_template(plan)
    return ""


@tool("analyse_user_request", args_schema=SGRPlanResult)
async def analyse_user_request(
    user_intent: str,
    topic: str,
    category: SGRCategory,
    intent_confidence: float,
    clarification_questions_to_ask: list[str],
    spam_score: float,
    spam_reason: str,
    knowledge_base_search_queries: list[str],
    action_plan: list[str],
    action: SGRAction,
    runtime: ToolRuntime[AgentContext, None] | None = None,
) -> str:
    """Analyze the user request and produce the question resolution plan.

    Returns guidance for your further steps.

    Edge cases:
    - Simple greetings (привет, спасибо): Set queries=[], action=proceed, spam_score=0
    - Time/date questions (сколько времени?): Set queries=[], action=proceed
    - General knowledge (2+2=?): Set queries=[], action=proceed
    """
    plan = {
        "user_intent": user_intent,
        "topic": topic,
        "category": category,
        "intent_confidence": intent_confidence,
        "clarification_questions_to_ask": clarification_questions_to_ask,
        "spam_score": spam_score,
        "spam_reason": spam_reason,
        "knowledge_base_search_queries": knowledge_base_search_queries,
        "action_plan": action_plan,
        "action": action,
    }

    if runtime and hasattr(runtime, "context") and runtime.context is not None:
        try:
            runtime.context.sgr_plan = plan
        except Exception as exc:
            logger.warning("Failed to store sgr_plan into AgentContext: %s", exc)

    return render_sgr_template(action.value, plan)
