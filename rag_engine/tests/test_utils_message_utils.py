"""Tests for message utilities."""
from __future__ import annotations

from rag_engine.utils.message_utils import (
    get_message_content,
    normalize_gradio_history_message,
)


class TestGetMessageContent:
    """Tests for get_message_content function."""

    def test_string_content(self):
        """Test extracting content from plain string format."""
        msg = {"role": "user", "content": "Hello"}
        assert get_message_content(msg) == "Hello"

    def test_gradio6_structured_content(self):
        """Test extracting content from structured format."""
        msg = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        assert get_message_content(msg) == "Hello"

    def test_gradio6_multiple_text_blocks(self):
        """Test extracting content from multiple text blocks."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }
        assert get_message_content(msg) == "Hello World"

    def test_gradio6_with_image_block(self):
        """Test extracting content with image block."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Check this"},
                {"type": "image", "path": "/path/to/image.png"},
            ],
        }
        content = get_message_content(msg)
        assert "Check this" in content
        assert "[Image:" in content

    def test_empty_content(self):
        """Test handling empty content."""
        msg = {"role": "user", "content": None}
        assert get_message_content(msg) is None

    def test_empty_content_list(self):
        """Test handling empty content list."""
        msg = {"role": "user", "content": []}
        assert get_message_content(msg) is None


class TestNormalizeGradioHistoryMessage:
    """Tests for normalize_gradio_history_message function."""

    def test_string_content_unchanged(self):
        """Test that string content is unchanged."""
        msg = {"role": "user", "content": "Hello"}
        normalized = normalize_gradio_history_message(msg)
        assert normalized == {"role": "user", "content": "Hello"}

    def test_gradio6_single_text_block(self):
        """Test normalizing structured content single text block."""
        msg = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        normalized = normalize_gradio_history_message(msg)
        assert normalized == {"role": "user", "content": "Hello"}

    def test_gradio6_multiple_text_blocks(self):
        """Test normalizing multiple text blocks."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }
        normalized = normalize_gradio_history_message(msg)
        assert normalized == {"role": "user", "content": "Hello World"}

    def test_gradio6_with_image(self):
        """Test normalizing message with image block."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Check this"},
                {"type": "image", "path": "/path/to/image.png"},
            ],
        }
        normalized = normalize_gradio_history_message(msg)
        assert normalized["content"] == "Check this [Image: /path/to/image.png]"

    def test_preserves_other_fields(self):
        """Test that other message fields are preserved."""
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Answer"}],
            "metadata": {"key": "value"},
        }
        normalized = normalize_gradio_history_message(msg)
        assert normalized["role"] == "assistant"
        assert normalized["content"] == "Answer"
        assert normalized["metadata"] == {"key": "value"}

    def test_empty_content_list(self):
        """Test handling empty content list."""
        msg = {"role": "user", "content": []}
        normalized = normalize_gradio_history_message(msg)
        assert normalized == {"role": "user", "content": ""}


