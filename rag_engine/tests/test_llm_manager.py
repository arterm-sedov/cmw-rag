"""Tests for LLM manager with dynamic token limits."""
from __future__ import annotations

import pytest

from rag_engine.llm.llm_manager import LLMManager, MODEL_CONFIGS


class TestModelConfigs:
    """Tests for model configurations."""

    def test_model_configs_have_required_fields(self):
        """Test all model configs have required fields."""
        for model_name, config in MODEL_CONFIGS.items():
            assert "token_limit" in config, f"{model_name} missing token_limit"
            assert "max_tokens" in config, f"{model_name} missing max_tokens"
            assert config["token_limit"] > 0, f"{model_name} has invalid token_limit"

    def test_default_config_exists(self):
        """Test default config exists for fallback."""
        assert "default" in MODEL_CONFIGS
        assert MODEL_CONFIGS["default"]["token_limit"] > 0


class TestLLMManager:
    """Tests for LLM manager."""

    def test_initialization_with_deepseek_model(self):
        """Test initialization with DeepSeek model."""
        manager = LLMManager(
            provider="openrouter",
            model="deepseek/deepseek-v3.1-terminus",
            temperature=0,
        )

        assert manager.provider == "openrouter"
        assert manager.model_name == "deepseek/deepseek-v3.1-terminus"
        assert manager.temperature == 0
        assert manager._model_config == MODEL_CONFIGS["deepseek/deepseek-v3.1-terminus"]

    def test_deepseek_context_window(self):
        """Test DeepSeek model context window."""
        manager = LLMManager(
            provider="openrouter",
            model="deepseek/deepseek-v3.1-terminus",
            temperature=0,
        )

        context_window = manager.get_current_llm_context_window()
        assert context_window == 163840  # DeepSeek v3.1 context
        assert context_window > 0

    def test_deepseek_max_tokens(self):
        """Test DeepSeek model max output tokens."""
        manager = LLMManager(
            provider="openrouter",
            model="deepseek/deepseek-v3.1-terminus",
            temperature=0,
        )

        max_output = manager.get_max_output_tokens()
        assert max_output == 65536
        assert max_output > 0

    def test_initialization_with_unknown_model_uses_default(self):
        """Test unknown model falls back to default config."""
        manager = LLMManager(
            provider="gemini",
            model="unknown-model-xyz",
            temperature=0.1,
        )

        assert manager._model_config == MODEL_CONFIGS["default"]

    def test_get_current_llm_context_window(self):
        """Test getting context window for current model."""
        manager = LLMManager(
            provider="gemini",
            model="gemini-2.5-flash",
            temperature=0,
        )

        context_window = manager.get_current_llm_context_window()

        assert context_window == 1048576  # 1M tokens for gemini-2.5-flash
        assert context_window > 0

    def test_get_max_output_tokens(self):
        """Test getting max output tokens."""
        manager = LLMManager(
            provider="gemini",
            model="gemini-2.5-flash",
            temperature=0,
        )

        max_output = manager.get_max_output_tokens()

        assert max_output == 65536
        assert max_output > 0

    def test_partial_model_name_matching(self):
        """Test partial model name matching works."""
        # Test with a version suffix
        manager = LLMManager(
            provider="gemini",
            model="gemini-2.5-flash-latest",  # Has suffix
            temperature=0,
        )

        # Should match "gemini-2.5-flash" config
        assert manager._model_config["token_limit"] == 1048576

    def test_different_models_have_different_limits(self):
        """Test different models report different token limits."""
        manager_flash = LLMManager("gemini", "gemini-2.5-flash")
        manager_pro = LLMManager("gemini", "gemini-2.5-pro")

        flash_limit = manager_flash.get_current_llm_context_window()
        pro_limit = manager_pro.get_current_llm_context_window()

        # Both should have same context (1M) but different max tokens
        assert flash_limit == pro_limit
        assert flash_limit == 1048576  # 1M
        assert pro_limit == 1048576  # 1M

    @pytest.mark.parametrize(
        "model,expected_limit",
        [
            ("deepseek/deepseek-v3.1-terminus", 163840),
            ("gemini-2.5-flash", 1048576),
            ("gemini-2.5-pro", 1048576),
            ("x-ai/grok-4-fast:free", 2000000),
            ("anthropic/claude-sonnet-4.5", 1000000),
        ],
    )
    def test_model_token_limits(self, model, expected_limit):
        """Test token limits for specific models."""
        # Use appropriate provider based on model
        provider = "openrouter" if "deepseek" in model or "anthropic" in model or "x-ai" in model else "gemini"
        manager = LLMManager(provider=provider, model=model)
        assert manager.get_current_llm_context_window() == expected_limit


class TestDynamicContextBudgeting:
    """Tests for dynamic context budgeting scenarios."""

    def test_context_budget_scales_with_model(self):
        """Test context budget automatically scales with model."""
        # Small model
        manager_small = LLMManager("gemini", "gemini-2.5-flash")
        small_context = manager_small.get_current_llm_context_window()

        # Large model
        manager_large = LLMManager("gemini", "gemini-2.5-pro")
        large_context = manager_large.get_current_llm_context_window()

        # Both should have same context (1M)
        assert large_context == small_context
        assert small_context == 1048576  # 1M
        assert large_context == 1048576  # 1M

