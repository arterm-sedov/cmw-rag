"""Streaming and UI metadata helpers for agent chat interface."""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def yield_search_started() -> dict:
    """Yield metadata message for search started.

    Returns:
        Gradio message dict with metadata for search started

    Example:
        >>> from rag_engine.api.stream_helpers import yield_search_started
        >>> msg = yield_search_started()
        >>> "Searching" in msg["metadata"]["title"]
        True
    """
    return {
        "role": "assistant",
        "content": "",
        "metadata": {
            "title": "🔍 Searching information in the knowledge base",
        },
    }


def yield_search_completed(count: int | None = None) -> dict:
    """Yield metadata message for search completed.

    Args:
        count: Optional article count to include in message

    Returns:
        Gradio message dict with metadata for search completed

    Example:
        >>> from rag_engine.api.stream_helpers import yield_search_completed
        >>> msg = yield_search_completed(5)
        >>> "Found 5" in msg["metadata"]["title"]
        True
    """
    if count is not None:
        article_word = "article" if count == 1 else "articles"
        title = f"✅ Found {count} {article_word}"
    else:
        title = "✅ Search completed"

    return {
        "role": "assistant",
        "content": "",
        "metadata": {"title": title},
    }


def yield_model_switch_notice(model: str) -> dict:
    """Yield metadata message for model switch.

    Args:
        model: Model name that was switched to

    Returns:
        Gradio message dict with metadata for model switch

    Example:
        >>> from rag_engine.api.stream_helpers import yield_model_switch_notice
        >>> msg = yield_model_switch_notice("gemini-2.5-pro")
        >>> model in msg["metadata"]["title"]
        True
    """
    return {
        "role": "assistant",
        "content": "",
        "metadata": {"title": f"⚡ Switching to {model} (larger context needed)"},
    }


def extract_article_count_from_tool_result(tool_result_content: str) -> int | None:
    """Extract article count from tool result JSON.

    Args:
        tool_result_content: JSON string from tool result

    Returns:
        Article count if found, None otherwise

    Example:
        >>> from rag_engine.api.stream_helpers import extract_article_count_from_tool_result
        >>> count = extract_article_count_from_tool_result('{"metadata": {"articles_count": 5}}')
        >>> count == 5
        True
    """
    try:
        result = json.loads(tool_result_content)
        return result.get("metadata", {}).get("articles_count")
    except (json.JSONDecodeError, KeyError):
        return None
