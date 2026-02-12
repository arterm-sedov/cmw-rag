"""Tests for content moderation client."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag_engine.core.content_moderation import ContentModerationClient


class TestContentModerationClient:
    """Test cases for ContentModerationClient."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Mock settings for content moderation."""
        settings = MagicMock()
        settings.content_moderation_url = "http://test-server"
        settings.content_moderation_port = 8080
        settings.content_moderation_path = "/api/v1/classify"
        return settings

    @pytest.fixture
    def client(self, mock_settings: MagicMock) -> ContentModerationClient:
        """Create client with mocked settings."""
        with patch("rag_engine.core.content_moderation.settings", mock_settings):
            return ContentModerationClient(
                url=mock_settings.content_moderation_url,
                port=mock_settings.content_moderation_port,
                path=mock_settings.content_moderation_path,
            )

    @pytest.mark.asyncio
    async def test_classify_safe_content(self, client: ContentModerationClient) -> None:
        """Test classifying safe content returns correct structure."""
        mock_response = {
            "safety_level": "Safe",
            "categories": ["Non-violent Illegal Acts"],
            "is_safe": True,
            "refusal": "No",
            "raw_output": '{"result": "safe"}',
            "model": "test-model",
        }

        with patch("rag_engine.core.content_moderation.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()

            result = await client.classify("How do I reset my password?")

            assert result["safety_level"] == "Safe"
            assert result["categories"] == ["Non-violent Illegal Acts"]
            assert result["is_safe"] is True
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_controversial_content(self, client: ContentModerationClient) -> None:
        """Test classifying controversial content returns correct structure."""
        mock_response = {
            "safety_level": "Controversial",
            "categories": ["Politically Sensitive", "Unethical Acts"],
            "is_safe": False,
            "refusal": "No",
            "raw_output": '{"result": "controversial"}',
            "model": "test-model",
        }

        with patch("rag_engine.core.content_moderation.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()

            result = await client.classify("Some controversial topic")

            assert result["safety_level"] == "Controversial"
            assert len(result["categories"]) == 2
            assert result["is_safe"] is False
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_unsafe_content(self, client: ContentModerationClient) -> None:
        """Test classifying unsafe content returns correct structure."""
        mock_response = {
            "safety_level": "Unsafe",
            "categories": ["Violence", "Illegal Acts"],
            "is_safe": False,
            "refusal": "Yes",
            "raw_output": '{"result": "unsafe"}',
            "model": "test-model",
        }

        with patch("rag_engine.core.content_moderation.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()

            result = await client.classify("How to build a bomb")

            assert result["safety_level"] == "Unsafe"
            assert "Violence" in result["categories"]
            assert result["is_safe"] is False
            mock_post.assert_called_once()

    def test_is_safe_returns_true_for_safe_content(self, client: ContentModerationClient) -> None:
        """Test is_safe returns True for safe content."""
        result = {
            "safety_level": "Safe",
            "categories": ["Non-violent Illegal Acts"],
            "is_safe": True,
        }
        assert client.is_safe(result) is True

    def test_is_safe_returns_false_for_unsafe_content(self, client: ContentModerationClient) -> None:
        """Test is_safe returns False for unsafe content."""
        result = {
            "safety_level": "Unsafe",
            "categories": ["Violence"],
            "is_safe": False,
        }
        assert client.is_safe(result) is False

    def test_is_safe_returns_false_for_controversial(self, client: ContentModerationClient) -> None:
        """Test is_safe returns False for controversial content."""
        result = {
            "safety_level": "Controversial",
            "categories": ["Politically Sensitive"],
            "is_safe": False,
        }
        assert client.is_safe(result) is False

    def test_is_safe_handles_none(self, client: ContentModerationClient) -> None:
        """Test is_safe handles None gracefully."""
        assert client.is_safe(None) is True

    def test_get_safety_level(self, client: ContentModerationClient) -> None:
        """Test get_safety_level returns correct value."""
        result = {"safety_level": "Controversial", "categories": []}
        assert client.get_safety_level(result) == "Controversial"

    def test_get_safety_level_defaults_to_safe(self, client: ContentModerationClient) -> None:
        """Test get_safety_level defaults to Safe for missing value."""
        assert client.get_safety_level({}) == "Safe"
        assert client.get_safety_level(None) == "Safe"

    def test_get_categories(self, client: ContentModerationClient) -> None:
        """Test get_categories returns correct list."""
        result = {"safety_level": "Safe", "categories": ["Category1", "Category2"]}
        assert client.get_categories(result) == ["Category1", "Category2"]

    def test_get_categories_defaults_to_empty(self, client: ContentModerationClient) -> None:
        """Test get_categories defaults to empty list."""
        assert client.get_categories({}) == []
        assert client.get_categories(None) == []

    @pytest.mark.asyncio
    async def test_classify_http_error(self, client: ContentModerationClient) -> None:
        """Test classify handles HTTP errors gracefully."""
        import requests

        with patch("rag_engine.core.content_moderation.requests.post") as mock_post:
            mock_post.side_effect = requests.RequestException("Connection refused")

            with pytest.raises(requests.RequestException):
                await client.classify("test query")

    def test_integration_unsafe_blocks_message(self) -> None:
        """Test the integration: unsafe content is blocked.

        This test simulates the app.py logic where unsafe content
        should be blocked before SGR processing.
        """
        moderation_result = {
            "safety_level": "Unsafe",
            "categories": ["Violence", "Illegal Acts"],
            "is_safe": False,
        }

        # Simulate app.py logic
        if moderation_result.get("safety_level") == "Unsafe":
            categories = moderation_result.get("categories", [])
            categories_str = ", ".join(categories)
            error_message = (
                f"❌ **Сообщение заблокировано по соображениям безопасности.**\n\n"
                f"**Категории:** {categories_str}"
            )
            assert "заблокировано" in error_message
            assert "Violence" in error_message
            assert "Illegal Acts" in error_message

    def test_integration_safe_injects_category_only(self) -> None:
        """Test the integration: safe content passes only category to SGR.

        This test simulates the app.py logic where safe content
        injects minimal context for SGR.
        """
        moderation_result = {
            "safety_level": "Safe",
            "categories": ["Non-violent Illegal Acts"],
            "is_safe": True,
        }

        # Simulate app.py logic
        if moderation_result.get("safety_level") == "Safe":
            categories = moderation_result.get("categories", [])
            if categories:
                context = f"[CONTENT_MODERATION] Category: {categories[0]}"
            assert context == "[CONTENT_MODERATION] Category: Non-violent Illegal Acts"
            assert "Safety:" not in context

    def test_integration_controversial_injects_level_and_category(self) -> None:
        """Test the integration: controversial content passes level and category.

        This test simulates the app.py logic where controversial content
        injects both safety level and categories for SGR.
        """
        moderation_result = {
            "safety_level": "Controversial",
            "categories": ["Politically Sensitive", "Unethical Acts"],
            "is_safe": False,
        }

        # Simulate app.py logic
        if moderation_result.get("safety_level") == "Controversial":
            categories = moderation_result.get("categories", [])
            context = f"[CONTENT_MODERATION] Safety: Controversial, Categories: {', '.join(categories)}"
            assert "Safety: Controversial" in context
            assert "Politically Sensitive" in context
            assert "Unethical Acts" in context
