"""Pydantic schemas for structured agent outputs.

We keep these schemas lean and avoid duplicating existing runtime structures:
- Articles are represented as dicts serialized from existing `Article` objects.
- Query trace is represented as dicts captured during actual tool execution.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SGRPlanResult(BaseModel):
    """Request analyzer for Comindware Platform support.

    Produces:
    - spam score
    - spam score explanation
    - user intent
    - suggested subqueries (1-10)
    - optional action plan (0-10 steps)
    - optional clarification question for vague/off-topic inputs
    """

    spam_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Classify if the user request is spam (unrelated to Comindware Platform support or infrastructure). "
            "0.0-0.2: clearly relevant. "
            "0.3-0.5: ambiguous. "
            "0.6-0.8: likely irrelevant "
            "0.9-1.0: obviously spam. "
        ),
    )
    spam_reason: str = Field(
        ...,
        max_length=150,
        description="Briefly explain your spam classification (10-20 words, in Russian).",
    )
    user_intent: str = Field(
        ...,
        max_length=300,
        description="Compile short summary of what the user wants (1-2 sentences, in Russian).",
    )
    subqueries: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "Generate focused knowledge base search queries (1-10). "
            "In Russian, unique, specific, no semantic duplicates."
        ),
    )
    action_plan: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Plan concrete steps to resolve the request (up to 10 steps, in Russian).",
    )
    ask_for_clarification: bool = Field(
        default=False,
        description="Set True if spam_score >= 0.6 or request is vague/harmful/off-topic.",
    )
    clarification_question: str | None = Field(
        default=None,
        max_length=200,
        description="If ask_for_clarification is True, provide a helpful question in Russian.",
    )


class StructuredAgentResult(BaseModel):
    """Structured output from `ask_comindware_structured()`.

    Notes:
    - `per_query_results` is a list of dicts captured from actual tool calls:
      {query, confidence: {...}, articles: [...]}
    - `final_articles` is a list of dicts serialized from existing Article objects
      (same shape as retrieve_context tool output's article items).
    """

    plan: SGRPlanResult
    per_query_results: list[dict[str, Any]] = Field(default_factory=list)
    final_articles: list[dict[str, Any]] = Field(default_factory=list)
    answer_text: str = ""
    diagnostics: dict[str, Any] = Field(default_factory=dict)

