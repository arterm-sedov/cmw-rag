"""Support Resolution Plan tool (SRP).

Generates a step-by-step resolution plan for human support engineers
based on the conversation context and final answer.
"""

from __future__ import annotations

import logging

from langchain.tools import ToolRuntime, tool

from rag_engine.llm.schemas import ResolutionOutcome, ResolutionPlanResult
from rag_engine.utils.context_tracker import AgentContext

logger = logging.getLogger(__name__)


def _render_plan_markdown(plan: dict) -> str:
    """Render resolution plan as markdown string."""
    from rag_engine.api.i18n import get_text

    outcome = plan.get("outcome")
    if outcome:
        # Handle both enum value ("escalation_required") and full string ("ResolutionOutcome.ESCALATION_REQUIRED")
        outcome_str = str(outcome)
        if "." in outcome_str:
            outcome_str = outcome_str.split(".")[
                -1
            ].lower()  # Extract "ESCALATION_REQUIRED" -> "escalation_required"
        outcome_key = f"srp_outcome_{outcome_str}"
        outcome_text = get_text(outcome_key)
    else:
        outcome_text = get_text("srp_outcome_unknown")

    steps = plan.get("steps_completed", [])
    steps_text = (
        "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))
        if steps
        else get_text("srp_no_steps")
    )

    next_steps = plan.get("next_steps", [])
    next_steps_text = (
        "\n".join(f"{i + 1}. {step}" for i, step in enumerate(next_steps))
        if next_steps
        else get_text("srp_no_next_steps")
    )

    return f"""# {get_text("srp_section_title")}

## {get_text("srp_issue_summary")}
{plan.get("issue_summary", "")}

## {get_text("srp_steps_completed")}
{steps_text}

## {get_text("srp_next_steps")}
{next_steps_text}

## {get_text("srp_result")}
{outcome_text}"""


@tool("generate_resolution_plan", args_schema=ResolutionPlanResult)
async def generate_resolution_plan(
    engineer_intervention_needed: bool,
    issue_summary: str,
    steps_completed: list[str],
    next_steps: list[str],
    outcome: ResolutionOutcome | None,
    runtime: ToolRuntime[AgentContext, None] | None = None,
) -> str:
    """Generate a support engineer resolution plan.

    Analyze the conversation and YOUR final answer.
    Critically evaluate: did you actually solve the user's problem?

    ALWAYS fill these fields (for structured trace):
    - issue_summary: 20-150 words in Russian - describe user's issue
    - steps_completed: 2-5 items in Russian - what you did
    - next_steps: 1-3 items in Russian - what engineer should do
    - outcome: resolved / partially_resolved / escalation_required / user_followup_needed / not_applicable

    Set engineer_intervention_needed=TRUE if human help needed.
    Set FALSE if answer fully resolves request.

    Returns:
        Formatted markdown plan if engineer_intervention_needed=True.
    """
    plan = {
        "engineer_intervention_needed": engineer_intervention_needed,
        "issue_summary": issue_summary,
        "steps_completed": steps_completed,
        "next_steps": next_steps,
        "outcome": outcome,
    }

    if runtime and hasattr(runtime, "context") and runtime.context is not None:
        try:
            runtime.context.resolution_plan = plan
        except Exception as exc:
            logger.warning("Failed to store resolution_plan into AgentContext: %s", exc)

    if not engineer_intervention_needed:
        return "No engineer intervention needed - issue resolved with KB answer."

    return _render_plan_markdown(plan)
