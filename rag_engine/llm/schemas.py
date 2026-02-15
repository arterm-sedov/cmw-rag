"""Pydantic schemas for structured agent outputs.

We keep these schemas lean and avoid duplicating existing runtime structures:
- Articles are represented as dicts serialized from existing `Article` objects.
- Query trace is represented as dicts captured during actual tool execution.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SGRAction(str, Enum):
    """Routing actions for request handling."""

    PROCEED = "proceed"
    ASK_CLARIFICATION = "ask_clarification"
    DECLINE = "decline"


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
            "REASONING STEP 1 - Intent Understanding: "
            "What does the user actually want to achieve? "
            "Think beyond keywords: What is their underlying goal? "
            "What business problem are they trying to solve? "
            "Write 1-2 clear sentences in Russian, as if explaining to a colleague."
        ),
    )

    topic: str = Field(
        ...,
        max_length=100,
        description=(
            "REASONING STEP 2 - Topic Identification: "
            "What is this request about? "
            "Example: 'Настройка SSO', 'Создание процесса', 'Интеграция с API'. "
            "Keep it concise (2-5 words). "
            "Write in Russian."
        ),
    )

    category: str = Field(
        ...,
        max_length=50,
        description=(
            "REASONING STEP 3 - Category Classification: "
            "What type of request is this? "
            "'Помощь в настройке', 'Устранение неполадок', 'Запрос функции', 'Общий вопрос'. "
            "Choose the most appropriate category. "
            "Write in Russian."
        ),
    )

    intent_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "REASONING STEP 4 - Confidence Assessment: "
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
            "REASONING STEP 5 - Clarification Questions to Ask: "
            "If intent_confidence < 0.7, what specific questions would help you understand better? "
            "Write in Russian, be polite and specific. "
            "These questions will be shown to the user to get clarification. "
            "Empty list if intent_confidence >= 0.7."
        ),
    )

    spam_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "REASONING STEP 6 - Validity Assessment: "
            "Is this request appropriate for Comindware Platform support? "
            "0.0-0.2: Clearly relevant; "
            "0.3-0.5: Ambiguous or partially related; "
            "0.6-0.8: Likely irrelevant; "
            "0.9-1.0: Obviously spam or malicious."
        ),
    )

    spam_reason: str = Field(
        ...,
        max_length=150,
        description=(
            "REASONING STEP 7 - Spam Justification: "
            "Briefly explain your spam_score in 10-20 words. "
            "Write in Russian."
        ),
    )

    knowledge_base_search_queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "REASONING STEP 8 - Knowledge Base Search Queries: "
            "What specific terms should be used to search the knowledge base? "
            "Include: feature names, technical terms, error messages, relevant keywords. "
            "Write in Russian, avoid duplicates. "
            "These queries will be used to retrieve relevant documentation."
        ),
    )

    action_plan: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "REASONING STEP 9 - Execution Plan: "
            "How will you answer this request? "
            "Steps: search docs -> evaluate -> synthesize answer OR ask clarification. "
            "Write in Russian as actionable instructions to yourself."
        ),
    )

    action: SGRAction = Field(
        ...,
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
