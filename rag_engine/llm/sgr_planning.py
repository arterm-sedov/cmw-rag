"""Schema-Guided Reasoning (SGR) planning step.

Single LLM call, structured output:
- spam_score + spam_reason
- user_intent
- suggested subqueries
- optional action_plan and clarification_suggestion
"""

from __future__ import annotations

from rag_engine.llm.llm_manager import LLMManager
from rag_engine.llm.prompts import (
    SGR_PLANNING_CLARIFICATION,
    SGR_PLANNING_USER_TEMPLATE,
    get_system_prompt,
)
from rag_engine.llm.schemas import SGRPlanResult


def run_sgr_planning(
    request: str,
    llm_manager: LLMManager,
    *,
    include_clarification: bool = True,
) -> SGRPlanResult:
    """Run SGR planning (single structured call).

    Args:
        request: User request text.
        llm_manager: LLM manager to construct the chat model.
        include_clarification: If True, appends a short clarification to system prompt.

    Returns:
        Parsed SGRPlanResult.
    """
    system = get_system_prompt()
    if include_clarification:
        system = f"{system}\n\n{SGR_PLANNING_CLARIFICATION}"

    user_msg = SGR_PLANNING_USER_TEMPLATE.format(request=request)
    model = llm_manager._chat_model(structured_output_schema=SGRPlanResult)

    # LangChain accepts (role, content) tuples as messages.
    return model.invoke([("system", system), ("user", user_msg)])

