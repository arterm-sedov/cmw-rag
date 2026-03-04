"""Tests for SRP (Support Resolution Plan) tool and schema validation."""

import pytest

from rag_engine.llm.schemas import ResolutionPlanResult
from rag_engine.tools.generate_resolution_plan import (
    _render_plan_markdown,
    generate_resolution_plan,
)


class TestResolutionPlanResultSchema:
    """Test ResolutionPlanResult Pydantic schema validation."""

    def test_empty_args_returns_defaults(self):
        """Schema should return all defaults when no args provided."""
        result = ResolutionPlanResult()
        assert result.engineer_intervention_needed is False
        assert result.issue_summary == ""
        assert result.steps_completed == []
        assert result.next_steps == []
        assert result.outcome is None

    def test_none_converted_to_empty_list(self):
        """Validators should convert None to [] for list fields."""
        result = ResolutionPlanResult(
            steps_completed=None,
            next_steps=None,
        )
        assert result.steps_completed == []
        assert result.next_steps == []

    def test_string_converted_to_list(self):
        """Validators should convert string to single-item list."""
        result = ResolutionPlanResult(
            steps_completed="single step",
            next_steps="single next step",
        )
        assert result.steps_completed == ["single step"]
        assert result.next_steps == ["single next step"]

    def test_empty_string_converted_to_empty_list(self):
        """Validators should convert empty string to empty list."""
        result = ResolutionPlanResult(
            steps_completed="",
            next_steps="",
        )
        assert result.steps_completed == []
        assert result.next_steps == []

    def test_engineer_intervention_needed_defaults_to_false(self):
        """engineer_intervention_needed should default to False."""
        result = ResolutionPlanResult()
        assert result.engineer_intervention_needed is False

        result2 = ResolutionPlanResult(engineer_intervention_needed=True)
        assert result2.engineer_intervention_needed is True


class TestGenerateResolutionPlanTool:
    """Test generate_resolution_plan tool execution."""

    @pytest.mark.asyncio
    async def test_empty_args_returns_valid_result(self):
        """Tool should work with empty args."""
        result = await generate_resolution_plan.ainvoke({})
        assert "json" in result
        assert "markdown" in result
        assert result["json"]["engineer_intervention_needed"] is False
        assert result["json"]["issue_summary"] == ""
        assert result["json"]["steps_completed"] == []

    @pytest.mark.asyncio
    async def test_partial_args_returns_valid_result(self):
        """Tool should work with partial args."""
        result = await generate_resolution_plan.ainvoke({"issue_summary": "User could not login"})
        assert result["json"]["issue_summary"] == "User could not login"
        assert result["json"]["steps_completed"] == []

    @pytest.mark.asyncio
    async def test_none_args_converted_to_empty_list(self):
        """Tool should handle None values gracefully."""
        result = await generate_resolution_plan.ainvoke(
            {
                "steps_completed": None,
                "next_steps": None,
            }
        )
        assert result["json"]["steps_completed"] == []
        assert result["json"]["next_steps"] == []

    @pytest.mark.asyncio
    async def test_returns_both_json_and_markdown(self):
        """Tool should return both structured data and formatted markdown."""
        result = await generate_resolution_plan.ainvoke(
            {
                "engineer_intervention_needed": True,
                "issue_summary": "Test issue",
                "steps_completed": ["step 1"],
                "next_steps": ["next step"],
            }
        )
        assert isinstance(result["json"], dict)
        assert isinstance(result["markdown"], str)
        assert "Test issue" in result["markdown"]

    @pytest.mark.asyncio
    async def test_no_intervention_still_returns_rendered_markdown(self):
        """When engineer_intervention_needed=False, markdown is still rendered (for srp_always_render_plan)."""
        result = await generate_resolution_plan.ainvoke(
            {
                "engineer_intervention_needed": False,
                "issue_summary": "Resolved issue",
            }
        )
        assert result["json"]["engineer_intervention_needed"] is False
        assert result["markdown"]
        assert "Resolved issue" in result["markdown"]

    @pytest.mark.asyncio
    async def test_intervention_returns_rendered_markdown(self):
        """When engineer_intervention_needed=True, markdown should be rendered."""
        result = await generate_resolution_plan.ainvoke(
            {
                "engineer_intervention_needed": True,
                "issue_summary": "Test issue summary",
                "steps_completed": ["step 1", "step 2"],
                "next_steps": ["contact user"],
                "outcome": "partially_resolved",
            }
        )
        assert result["json"]["engineer_intervention_needed"] is True
        assert "Test issue summary" in result["markdown"]
        assert "step 1" in result["markdown"]

    def test_tool_description_from_schema(self):
        """Tool description should come from schema docstring."""
        assert generate_resolution_plan.description.strip() == ResolutionPlanResult.__doc__.strip()
        assert "Comindware Platform" in generate_resolution_plan.description
        assert "engineer_intervention_needed" in generate_resolution_plan.description.lower()


class TestRenderPlanMarkdown:
    """Test SRP markdown rendering."""

    def test_renders_basic_plan(self):
        """Should render a basic plan correctly."""
        plan = {
            "issue_summary": "User login issue",
            "steps_completed": ["checked credentials"],
            "next_steps": ["reset password"],
            "outcome": "partially_resolved",
        }
        result = _render_plan_markdown(plan)
        assert "User login issue" in result
        assert "checked credentials" in result
        assert "reset password" in result

    def test_handles_empty_lists(self):
        """Should skip sections for empty steps/next_steps (conditional headings)."""
        plan = {
            "issue_summary": "Test issue",
            "steps_completed": [],
            "next_steps": [],
            "outcome": None,
        }
        result = _render_plan_markdown(plan)
        assert "Test issue" in result
        # Empty lists: only issue_summary section (one ## heading besides title)
        assert result.count("## ") == 1

    def test_handles_none_outcome(self):
        """Should skip outcome section when outcome is None."""
        plan = {
            "issue_summary": "Test",
            "steps_completed": [],
            "next_steps": [],
            "outcome": None,
        }
        result = _render_plan_markdown(plan)
        assert "Test" in result

    def test_all_empty_returns_only_title(self):
        """When all fields empty, output is only the main section title."""
        plan = {
            "issue_summary": "",
            "steps_completed": [],
            "next_steps": [],
            "outcome": None,
        }
        result = _render_plan_markdown(plan)
        assert result.startswith("# ")
        assert result.count("## ") == 0
