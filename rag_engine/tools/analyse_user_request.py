"""User request analysis tool (schema-guided request analysis).

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


@tool("analyse_user_request", args_schema=SGRPlanResult)
async def analyse_user_request(
    spam_score: float,
    spam_reason: str,
    user_intent: str,
    subqueries: list[str],
    action_plan: list[str] | None = None,
    ask_for_clarification: bool = False,
    clarification_question: str | None = None,
    runtime: ToolRuntime[AgentContext, None] | None = None,
) -> str:
    """Analyze the user request and produce the user question resolution plan.

    Spam Classification Criteria:
    - 0.0-0.2: Clearly related to Comindware Platform support/development. The request is relevant to
      Comindware Platform, KB, processes, apps, configuration, integrations, or platform features.
    - 0.3-0.5: Ambiguous but could be IT/business/support related. Allow and proceed with
      retrieval. May need paraphrasing to Comindware Platform context.
    - 0.6-0.8: Likely irrelevant but has some technical keywords. Try to paraphrase the search queries. Ask the user for clarification.
    - 0.9-1.0: Obviously spam (e.g., marketing, personal, random email thread, other product,
      completely off-topic). Ask for clarification. Do not retrieve articles.
    - Strategy: We prefer to allow ~2% spam requests rather than block a single valid request.

    Important:
    - Fill all the args in Russian
    - Do NOT echo the plan to the user

    Returns:
        JSON dictionary with the action plan:
        {
            "spam_score": float,  # Spam classification per criteria above
            "spam_reason": str,  # Brief explanation of spam classification
            "user_intent": str,  # Brief explanation of the user request intent
            "subqueries": list[str],  # Use these suggested subqueries to gather relevant articles
            "action_plan": list[str] | None,  # Follow these steps sequentially to resolve the user's request
            "ask_for_clarification": bool,  # True if spam_score >= 0.6 OR request is vague/harmful/off-topic
            "clarification_question": str | None,  # Helpful question when ask_for_clarification=True
        }

        Note: This plan is for internal use only. Do NOT quote, restate, or output this plan/JSON in your final answer to the user.

    """
    plan = {
        "spam_score": float(spam_score),
        "spam_reason": spam_reason,
        "user_intent": user_intent,
        "subqueries": list(subqueries),
        "action_plan": list(action_plan) if action_plan else [],
        "ask_for_clarification": bool(ask_for_clarification),
        "clarification_question": clarification_question,
    }

    if runtime and hasattr(runtime, "context") and runtime.context is not None:
        try:
            runtime.context.sgr_plan = plan
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to store sgr_plan into AgentContext: %s", exc)

    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
