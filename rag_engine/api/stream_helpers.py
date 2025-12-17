"""Streaming and UI metadata helpers for agent chat interface."""
from __future__ import annotations

import json
import logging

from rag_engine.api.i18n import get_text

logger = logging.getLogger(__name__)


def yield_search_started(query: str | None = None) -> dict:
    """Yield metadata message for search started.

    Args:
        query: Optional user query being searched, for display in the bubble.

    Returns:
        Gradio message dict with metadata for search started.
        Content and title are resolved i18n strings (never i18n metadata objects).

    Example:
        >>> from rag_engine.api.stream_helpers import yield_search_started
        >>> msg = yield_search_started()
        >>> "Searching" in msg["metadata"]["title"] or "Поиск" in msg["metadata"]["title"]
        True
    """
    # Resolve i18n translations to plain strings before yielding
    # This ensures Chatbot receives strings, not __i18n__ metadata objects
    title = get_text("search_started_title")
    content = get_text("search_started_content", query=(query or "").strip())

    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
        },
    }


def yield_search_completed(count: int | None = None) -> dict:
    """Yield metadata message for search completed.

    Args:
        count: Optional article count to include in message.

    Returns:
        Gradio message dict with metadata for search completed.
        Content and title are resolved i18n strings (never i18n metadata objects).

    Example:
        >>> from rag_engine.api.stream_helpers import yield_search_completed
        >>> msg = yield_search_completed(5)
        >>> "Found" in msg["metadata"]["title"] or "завершен" in msg["metadata"]["title"]
        True
    """
    # Resolve i18n translations to plain strings
    title = get_text("search_completed_title_with_count")
    content = get_text(
        "search_completed_content_with_count",
        count=count if count is not None else 0,
    )

    return {
        "role": "assistant",
        "content": content,
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
        "metadata": {"title": f"⚡ Переключение на {model} (требуется больше контекста)"},
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
