"""Tests for SGR planning tool and schema validation."""

import pytest

from rag_engine.llm.schemas import SGRAction, SGRCategory, SGRPlanResult
from rag_engine.tools.analyse_user_request import (
    analyse_user_request,
    render_sgr_template,
)


class TestSGRPlanResultSchema:
    """Test SGRPlanResult Pydantic schema validation."""

    def test_empty_args_returns_defaults(self):
        """Schema should return all defaults when no args provided."""
        result = SGRPlanResult()
        assert result.user_intent == ""
        assert result.topic == ""
        assert result.answer_language == "ru"
        expected_default = SGRCategory.OTHER if hasattr(SGRCategory, "OTHER") else list(SGRCategory)[0]
        assert result.category == expected_default
        assert result.intent_confidence == 0.0
        assert result.clarification_questions_to_ask == []
        assert result.spam_score == 0.0
        assert result.spam_reason == ""
        assert result.knowledge_base_search_queries == []
        assert result.action_plan == []
        assert result.action == SGRAction.PROCEED

    def test_none_converted_to_empty_list(self):
        """Validators should convert None to [] for list fields."""
        result = SGRPlanResult(
            clarification_questions_to_ask=None,
            knowledge_base_search_queries=None,
            action_plan=None,
        )
        assert result.clarification_questions_to_ask == []
        assert result.knowledge_base_search_queries == []
        assert result.action_plan == []

    def test_string_converted_to_list(self):
        """Validators should convert string to single-item list."""
        result = SGRPlanResult(
            knowledge_base_search_queries="single query",
            action_plan="single step",
        )
        assert result.knowledge_base_search_queries == ["single query"]
        assert result.action_plan == ["single step"]

    def test_empty_string_converted_to_empty_list(self):
        """Validators should convert empty string to empty list."""
        result = SGRPlanResult(
            knowledge_base_search_queries="",
            action_plan="",
        )
        assert result.knowledge_base_search_queries == []
        assert result.action_plan == []

    def test_category_string_converted_to_enum(self):
        """Category validator should convert string to enum."""
        # 'documentation' is a known code in cmw_platform.yaml
        result = SGRPlanResult(category="documentation")
        assert result.category == SGRCategory.DOCUMENTATION
        assert result.category.value == "documentation"

    def test_category_case_insensitive(self):
        """Category validator should be case-insensitive."""
        result = SGRPlanResult(category="DOCUMENTATION")
        assert result.category == SGRCategory.DOCUMENTATION

    def test_category_invalid_returns_default(self):
        """Invalid category should return default."""
        result = SGRPlanResult(category="invalid_category")
        expected_default = SGRCategory.OTHER if hasattr(SGRCategory, "OTHER") else list(SGRCategory)[0]
        assert result.category == expected_default


class TestAnalyseUserRequestTool:
    """Test analyse_user_request tool execution."""

    @pytest.mark.asyncio
    async def test_empty_args_returns_valid_result(self):
        """Tool should work with empty args."""
        result = await analyse_user_request.ainvoke({})
        assert "json" in result
        assert "markdown" in result
        assert result["json"]["user_intent"] == ""
        assert result["json"]["knowledge_base_search_queries"] == []

    @pytest.mark.asyncio
    async def test_partial_args_returns_valid_result(self):
        """Tool should work with partial args."""
        result = await analyse_user_request.ainvoke({"user_intent": "test query"})
        assert result["json"]["user_intent"] == "test query"
        assert result["json"]["knowledge_base_search_queries"] == []

    @pytest.mark.asyncio
    async def test_none_args_converted_to_empty_list(self):
        """Tool should handle None values gracefully."""
        result = await analyse_user_request.ainvoke(
            {
                "knowledge_base_search_queries": None,
                "clarification_questions_to_ask": None,
                "action_plan": None,
            }
        )
        assert result["json"]["knowledge_base_search_queries"] == []
        assert result["json"]["clarification_questions_to_ask"] == []
        assert result["json"]["action_plan"] == []

    @pytest.mark.asyncio
    async def test_returns_both_json_and_markdown(self):
        """Tool should return both structured data and formatted markdown."""
        result = await analyse_user_request.ainvoke(
            {
                "user_intent": "test",
                "knowledge_base_search_queries": ["query1", "query2"],
            }
        )
        assert isinstance(result["json"], dict)
        assert isinstance(result["markdown"], str)
        assert "query1" in result["markdown"] or "Query" in result["markdown"]

    @pytest.mark.asyncio
    async def test_markdown_contains_intent(self):
        """Markdown output should contain user intent."""
        result = await analyse_user_request.ainvoke({"user_intent": "How to configure SSO"})
        assert "How to configure SSO" in result["markdown"]

    def test_tool_description_from_schema(self):
        """Tool description should come from schema docstring."""
        assert analyse_user_request.description.strip() == SGRPlanResult.__doc__.strip()
        assert "Comindware Platform support" in analyse_user_request.description
        assert "Edge cases" in analyse_user_request.description


class TestRenderSGRTemplate:
    """Test SGR template rendering."""

    def _make_complete_plan(self, **overrides) -> dict:
        """Create a complete plan dict with all required fields."""
        base = {
            "user_intent": "test intent",
            "topic": "test topic",
            "answer_language": "ru",
            "category": SGRCategory.DOCUMENTATION,
            "intent_confidence": 0.9,

            "clarification_questions_to_ask": [],
            "spam_score": 0.0,
            "spam_reason": "",
            "knowledge_base_search_queries": [],
            "action_plan": ["step 1"],
            "action": SGRAction.PROCEED,
        }
        base.update(overrides)
        return base

    def test_proceed_template_renders(self):
        """Proceed template should render correctly."""
        plan = self._make_complete_plan(knowledge_base_search_queries=["query1"])
        result = render_sgr_template("proceed", plan)
        assert "test intent" in result
        assert "query1" in result

    def test_clarify_template_renders(self):
        """Clarify template should render correctly."""
        plan = self._make_complete_plan(
            clarification_questions_to_ask=["question1?"],
            action=SGRAction.ASK_CLARIFICATION,
        )
        result = render_sgr_template("ask_clarification", plan)
        assert "test intent" in result

    def test_decline_template_renders(self):
        """Decline template should render correctly."""
        plan = self._make_complete_plan(
            spam_reason="off-topic",
            action=SGRAction.DECLINE,
        )
        result = render_sgr_template("decline", plan)
        assert "off-topic" in result
        # Russian heading is expected by default (answer_language="ru")
        assert "Анализ запроса" in result

    def test_answer_language_controls_template_language(self):
        """Answer language in plan should control template localization."""
        plan_ru = self._make_complete_plan(
            knowledge_base_search_queries=["q1"],
            answer_language="ru",
        )
        result_ru = render_sgr_template("proceed", plan_ru)
        assert "Язык ответа: русский" in result_ru

        plan_en = self._make_complete_plan(
            knowledge_base_search_queries=["q1"],
            answer_language="en",
        )
        result_en = render_sgr_template("proceed", plan_en)
        assert "Answer language: English" in result_en

    def test_empty_queries_no_error(self):
        """Template should handle empty query list."""
        plan = self._make_complete_plan(
            knowledge_base_search_queries=[],
        )
        result = render_sgr_template("proceed", plan)
        assert "test intent" in result
