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


@tool(
    "generate_resolution_plan",
    args_schema=ResolutionPlanResult,
    description=ResolutionPlanResult.__doc__,
)
async def generate_resolution_plan(
    engineer_intervention_needed: bool = False,
    issue_summary: str = "",
    steps_completed: list[str] = None,
    next_steps: list[str] = None,
    outcome: ResolutionOutcome | None = None,
    runtime: ToolRuntime[AgentContext, None] | None = None,
) -> dict:
    plan = {
        "engineer_intervention_needed": engineer_intervention_needed,
        "issue_summary": issue_summary,
        "steps_completed": steps_completed or [],
        "next_steps": next_steps or [],
        "outcome": outcome,
    }

    if runtime and hasattr(runtime, "context") and runtime.context is not None:
        try:
            runtime.context.resolution_plan = plan
        except Exception as exc:
            logger.warning("Failed to store resolution_plan into AgentContext: %s", exc)

    return {
        "json": plan,
        "markdown": _render_plan_markdown(plan) if engineer_intervention_needed else "",
    }
