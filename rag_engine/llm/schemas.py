"""Pydantic schemas for structured agent outputs.

We keep these schemas lean and avoid duplicating existing runtime structures:
- Articles are represented as dicts serialized from existing `Article` objects.
- Query trace is represented as dicts captured during actual tool execution.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SGRAction(str, Enum):
    """Routing actions for request handling."""

    PROCEED = "proceed"
    ASK_CLARIFICATION = "ask_clarification"
    DECLINE = "decline"


class SGRCategory(str, Enum):
    """Request categories for classification."""

    SETUP_HELP = "setup_help"
    TROUBLESHOOTING = "troubleshooting"
    FEATURE_REQUEST = "feature_request"
    GENERAL_QUESTION = "general_question"
    ADMINISTRATION = "administration"
    INTEGRATION = "integration"
    TRAINING = "training"
    HOW_TO = "how_to"
    OTHER = "other"


class SGRPlanResult(BaseModel):
    """Request analyzer for Comindware Platform support.

    Sequential reasoning like a human would think:
    1. Understand what user wants
    2. Identify topic and category
    3. Assess confidence in understanding
    4. Check if request is relevant/safe
    5. Plan search strategy
    6. Decide action
    7. Prepare clarification if needed
    """

    user_intent: str = Field(
        ...,
        max_length=300,
        description=(
            "What does the user actually want to achieve? "
            "Think beyond keywords: What is their underlying goal? "
            "What business problem are they trying to solve? "
            "Write 1-2 clear sentences in Russian, as if explaining to support engineer."
        ),
    )

    topic: str = Field(
        default="",
        max_length=100,
        description=(
            "What is this request about? "
            "Example: 'Настройка SSO', 'Создание процесса', 'Интеграция с API'. "
            "Keep it concise (2-5 words). "
            "Write in Russian."
        ),
    )

    category: SGRCategory = Field(
        default=SGRCategory.GENERAL_QUESTION,
        description=(
            "What type of request is this? Choose the most appropriate category from the enum."
        ),
    )

    @field_validator("category", mode="before")
    @classmethod
    def _convert_category(cls, v: Any) -> SGRCategory:
        if isinstance(v, SGRCategory):
            return v
        if isinstance(v, str):
            for cat in SGRCategory:
                if cat.value.lower() == v.lower():
                    return cat
        return SGRCategory.GENERAL_QUESTION

    intent_confidence: float | None = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "How confident are you in understanding what the user wants? "
            "Think: Is the request clear? Do you understand the context? "
            "0.0-0.4: Very unclear, major uncertainties; "
            "0.5-0.7: Somewhat clear but some gaps; "
            "0.8-1.0: Clear and well-understood."
        ),
    )

    clarification_questions_to_ask: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "If intent_confidence < 0.7, what specific questions would help you understand better? "
            "Write in Russian, be polite and specific. "
            "These questions will be shown to the user to get clarification. "
            "Empty list if intent_confidence >= 0.7."
        ),
    )

    spam_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Is this request appropriate for Comindware Platform support? "
            "0.0-0.2: Clearly relevant; "
            "0.3-0.5: Ambiguous or partially related; "
            "0.6-0.8: Likely irrelevant; "
            "0.9-1.0: Obviously spam or malicious."
        ),
    )

    spam_reason: str = Field(
        default="",
        max_length=150,
        description=(
            "Briefly explain your spam_score in 10-20 words. "
            "Write in Russian. "
            "Leave empty if spam_score < 0.3 (clearly not spam)."
        ),
    )

    knowledge_base_search_queries: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "What specific terms should be used to search the articles the knowledge base? "
            "Include: feature names, technical terms, error messages, relevant keywords. "
            "Write in Russian, avoid duplicates. "
            "Leave EMPTY if no search needed (e.g., simple greetings, time/date questions, "
            "or direct answers not requiring documentation lookup)."
        ),
    )

    @field_validator("knowledge_base_search_queries", mode="before")
    @classmethod
    def _convert_queries(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v] if v else []
        if isinstance(v, list):
            return [str(item) for item in v]
        return []

    action_plan: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "How will you answer this request? "
            "Steps: search docs -> evaluate -> synthesize answer OR ask clarification. "
            "Write in Russian as actionable instructions to yourself."
        ),
    )

    @field_validator("action_plan", mode="before")
    @classmethod
    def _convert_action_plan(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v] if v else []
        if isinstance(v, list):
            return [str(item) for item in v]
        return []

    action: SGRAction = Field(
        default=SGRAction.PROCEED,
        description=(
            "REASONING STEP 10 - Routing Decision: "
            "Based on all previous reasoning, what action to take? "
            "'proceed': request is relevant and clear - assist normally; "
            "'ask_clarification': request needs clarification from user; "
            "'decline': request is off-topic or inappropriate - do not assist; "
            "Guideline: spam_score >= 0.7 suggests decline; "
            "intent_confidence < 0.6 suggests ask_clarification; "
            "Guardian safety levels are handled separately before this tool is called."
        ),
    )


class ResolutionOutcome(str, Enum):
    """Resolution status for support engineer plan."""

    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    ESCALATION_REQUIRED = "escalation_required"
    USER_FOLLOWUP_NEEDED = "user_followup_needed"
    NOT_APPLICABLE = "not_applicable"


class ResolutionPlanResult(BaseModel):
    """Support Resolution Plan - generates actionable plan for human engineers.

    Called at final answer generation to provide structured guidance.
    The LLM decides if a plan is actually needed via engineer_intervention_needed field.

    NOTE: Fill text fields in Russian for human readability.
    Enum values remain in English for token efficiency.
    """

    engineer_intervention_needed: bool = Field(
        ...,
        description=(
            "Is support engineer intervention or escalation needed for this issue? "
            "Set to FALSE for: version queries, simple how-tos with complete KB answers, "
            "factual lookups that are fully resolved, self-service queries. "
            "Set to TRUE for: errors, bugs, configuration issues, incomplete solutions, "
            "troubleshooting required, or any issue requiring human investigation/action. "
            "This field is REQUIRED - the LLM must explicitly decide."
        ),
    )

    issue_summary: str = Field(
        default="",
        max_length=500,
        description=(
            "Brief summary of the user's issue in 2-3 sentences. "
            "Write in Russian for the support engineer. "
            "Only meaningful when engineer_intervention_needed=True."
        ),
    )

    steps_completed: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "List of steps already taken by the system. "
            "Include: KB search, documentation analysis, solutions provided. "
            "Write in Russian, be concise and informative. "
            "Only meaningful when engineer_intervention_needed=True."
        ),
    )

    next_steps: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Recommended next steps for the support engineer. "
            "What does the human need to do after this response? "
            "Examples: 'Check user permissions', 'Update documentation', 'Create dev ticket'. "
            "Write in Russian. "
            "Only meaningful when engineer_intervention_needed=True."
        ),
    )

    outcome: ResolutionOutcome | None = Field(
        default=None,
        description=(
            "Resolution status based on the answer provided. "
            "Use English enum value: "
            "'resolved': Fully resolved; "
            "'partially_resolved': Partially resolved, additional actions needed; "
            "'escalation_required': Requires escalation; "
            "'user_followup_needed': User follow-up required; "
            "'not_applicable': Plan not needed (use when engineer_intervention_needed=False)."
        ),
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
