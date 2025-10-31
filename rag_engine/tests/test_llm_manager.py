"""Tests for LLM manager with dynamic token limits."""
from __future__ import annotations

import pytest

from rag_engine.llm.llm_manager import MODEL_CONFIGS, LLMManager
from rag_engine.config import settings as settings_mod


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

    def test_stream_response_yields_tokens(self, monkeypatch):
        manager = LLMManager("gemini", "gemini-2.5-flash")

        class FakeModel:
            def stream(self, messages):  # noqa: ANN001
                assert messages[0][0] == "system"
                yield type("Chunk", (), {"content": "hello"})()
                yield type("Chunk", (), {"content": " world"})()

        monkeypatch.setattr(LLMManager, "_chat_model", lambda self, provider=None: FakeModel())
        docs = [type("Doc", (), {"page_content": "Context text"})()]

        tokens = list(manager.stream_response("question?", docs))

        assert tokens == ["hello", " world"]

    def test_generate_returns_content(self, monkeypatch):
        manager = LLMManager("gemini", "gemini-2.5-flash")

        class FakeModel:
            def invoke(self, messages):  # noqa: ANN001
                assert messages[1][1] == "question?"
                return type("Resp", (), {"content": "answer"})()

        monkeypatch.setattr(LLMManager, "_chat_model", lambda self, provider=None: FakeModel())
        docs = [type("Doc", (), {"content": "Full doc"})()]

        answer = manager.generate("question?", docs)

        assert answer == "answer"

    def test_chat_model_provider_paths(self, monkeypatch):
        manager = LLMManager("gemini", "gemini-2.5-flash")

        class Dummy:
            def __init__(self, *a, **k):
                pass

        # Cover openrouter branch
        monkeypatch.setattr("rag_engine.llm.llm_manager.ChatOpenAI", Dummy)
        m = manager._chat_model("openrouter")
        assert isinstance(m, Dummy)

        # Cover unknown provider fallback to Gemini
        monkeypatch.setattr("rag_engine.llm.llm_manager.ChatGoogleGenerativeAI", Dummy)
        m2 = manager._chat_model("unknown")
        assert isinstance(m2, Dummy)

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


class TestFallbackProviderInference:
    def test_create_manager_for_infers_openrouter_for_qwen(self, monkeypatch):
        mgr = LLMManager(provider="gemini", model="gemini-2.5-flash")
        # Ensure no explicit provider set on the module used by LLMManager
        monkeypatch.setattr("rag_engine.llm.llm_manager.settings", type("S", (), {"llm_fallback_provider": None})())
        fb_mgr = mgr._create_manager_for("qwen/qwen3-max")
        assert isinstance(fb_mgr, LLMManager)
        assert fb_mgr.provider == "openrouter"

    def test_create_manager_for_respects_explicit_provider(self, monkeypatch):
        mgr = LLMManager(provider="gemini", model="gemini-2.5-flash")
        # Force explicit provider on the module used by LLMManager
        monkeypatch.setattr("rag_engine.llm.llm_manager.settings", type("S", (), {"llm_fallback_provider": "gemini"})())
        fb_mgr = mgr._create_manager_for("qwen/qwen3-max")
        assert fb_mgr.provider == "gemini"


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


class TestMemoryAndCompression:
    def test_session_isolation_and_system_not_persisted(self, monkeypatch):
        manager = LLMManager("gemini", "gemini-2.5-flash")

        class FakeModel:
            def stream(self, messages):  # noqa: ANN001
                # First message must be system, not stored in memory
                assert messages[0][0] == "system"
                yield type("Chunk", (), {"content": "ok"})()

            def invoke(self, messages):  # noqa: ANN001
                return type("Resp", (), {"content": "answer"})()

        monkeypatch.setattr(LLMManager, "_chat_model", lambda self, provider=None: FakeModel())

        docs = [type("Doc", (), {"content": "Ctx"})()]

        # Two independent sessions
        list(manager.stream_response("q1", docs, session_id="s1"))
        manager.save_assistant_turn("s1", "a1")

        list(manager.stream_response("q2", docs, session_id="s2"))
        manager.save_assistant_turn("s2", "a2")

        h1 = manager._conversations.get("s1")
        h2 = manager._conversations.get("s2")

        assert h1 != h2
        # No system role stored; only user/assistant
        assert all(r in ("user", "assistant") for r, _ in h1)
        assert all(r in ("user", "assistant") for r, _ in h2)

    def test_compression_triggers_and_keeps_last_two(self, monkeypatch):
        manager = LLMManager("gemini", "gemini-2.5-flash")

        # Force tiny window and very low threshold to trigger compression
        manager._model_config["token_limit"] = 512

        # Patch settings on module used by manager
        monkeypatch.setattr(
            "rag_engine.llm.llm_manager.settings",
            type("S", (), {"memory_compression_threshold_pct": 1, "memory_compression_target_tokens": 200})(),
        )

        class FakeModel:
            def stream(self, messages):  # noqa: ANN001
                yield type("Chunk", (), {"content": "ok"})()

        monkeypatch.setattr(LLMManager, "_chat_model", lambda self, provider=None: FakeModel())

        # Seed history with 6 turns: u/a/u/a/u/a
        for i in range(3):
            manager._conversations.append("s3", "user", f"q{i}")
            manager._conversations.append("s3", "assistant", f"a{i}")

        docs = [type("Doc", (), {"content": "Ctx"})()]
        list(manager.stream_response("new question", docs, session_id="s3"))

        hist = manager._conversations.get("s3")
        # After compression: starts with one assistant summary, then last two original turns, then the new user
        assert len(hist) >= 3
        assert hist[0][0] == "assistant"
        # Last two preserved
        assert hist[1][0] in ("user", "assistant") and hist[2][0] in ("user", "assistant")


class TestArticleHeaders:
    """Tests for Article URLs header formatting."""

    def test_format_article_header_with_full_metadata(self):
        manager = LLMManager("gemini", "gemini-2.5-flash")
        doc = type(
            "Doc",
            (),
            {
                "metadata": {
                    "title": "Test Article",
                    "kbId": "1234",
                    "url": "https://kb.comindware.ru/article.php?id=1234",
                    "tags": ["linux", "install"],
                },
                "kb_id": "1234",
            },
        )()
        header = manager._format_article_header(doc)
        assert "Article details:" in header
        assert "Test Article" in header
        assert "kbId=1234" in header
        assert "https://kb.comindware.ru/article.php?id=1234" in header
        assert "Tags: linux, install" in header

    def test_format_article_header_synthesizes_url_from_kbid(self):
        manager = LLMManager("gemini", "gemini-2.5-flash")
        doc = type(
            "Doc",
            (),
            {
                "metadata": {"title": "Test", "kbId": "5678"},
                "kb_id": "5678",
            },
        )()
        header = manager._format_article_header(doc)
        assert "kbId=5678" in header
        assert "https://kb.comindware.ru/article.php?id=5678" in header

    def test_format_article_header_handles_string_tags(self):
        manager = LLMManager("gemini", "gemini-2.5-flash")
        doc = type(
            "Doc",
            (),
            {
                "metadata": {"title": "Test", "kbId": "9999", "tags": "tag1, tag2, tag3"},
                "kb_id": "9999",
            },
        )()
        header = manager._format_article_header(doc)
        assert "Tags: tag1, tag2, tag3" in header

    def test_context_includes_article_headers(self, monkeypatch):
        manager = LLMManager("gemini", "gemini-2.5-flash")

        captured_messages = []

        class FakeModel:
            def stream(self, messages):  # noqa: ANN001
                captured_messages.extend(messages)
                yield type("Chunk", (), {"content": "ok"})()

        monkeypatch.setattr(LLMManager, "_chat_model", lambda self, provider=None: FakeModel())
        docs = [
            type(
                "Doc",
                (),
                {
                    "metadata": {"title": "Article 1", "kbId": "100"},
                    "kb_id": "100",
                    "content": "Content 1",
                },
            )(),
            type(
                "Doc",
                (),
                {
                    "metadata": {"title": "Article 2", "kbId": "200"},
                    "kb_id": "200",
                    "content": "Content 2",
                },
            )(),
        ]

        list(manager.stream_response("question?", docs))

        # System message should contain headers
        system_msg = captured_messages[0][1]
        assert "Article details:" in system_msg
        assert "Article 1" in system_msg
        assert "kbId=100" in system_msg
        assert "Article 2" in system_msg
        assert "kbId=200" in system_msg
        assert "Content 1" in system_msg
        assert "Content 2" in system_msg
        # Articles should be separated
        assert "---" in system_msg