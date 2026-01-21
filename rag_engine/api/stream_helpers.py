"""Streaming and UI metadata helpers for agent chat interface."""
from __future__ import annotations

import json
import logging

from rag_engine.api.i18n import get_text

logger = logging.getLogger(__name__)


class ToolCallAccumulator:
    """Accumulate tool call arguments from streaming chunks.

    In streaming mode, tool call arguments may arrive incrementally across multiple
    chunks. This class accumulates chunks by tool call index and extracts the query
    when complete.

    Example:
        >>> accumulator = ToolCallAccumulator()
        >>> # Process streaming chunks
        >>> for token in stream:
        ...     query = accumulator.process_token(token)
        ...     if query:
        ...         print(f"Query ready: {query}")
    """

    def __init__(self):
        """Initialize accumulator with empty state."""
        self._accumulated_calls: dict[int, dict[str, str]] = {}

    def get_tool_name(self, token) -> str | None:
        """Extract tool name from token (for any tool, not just retrieve_context).

        Args:
            token: LangChain token/message that may contain tool_call chunks

        Returns:
            Tool name if detected, None otherwise
        """
        # Try content_blocks for tool_call_chunk (streaming mode)
        content_blocks = getattr(token, "content_blocks", None)
        if content_blocks:
            for block in content_blocks:
                if block.get("type") == "tool_call_chunk":
                    chunk_name = block.get("name", "")
                    if chunk_name:
                        return chunk_name

        # Try tool_calls attribute (may be complete or partial)
        tool_calls = getattr(token, "tool_calls", None)
        if tool_calls:
            for tool_call in tool_calls:
                # Handle different formats: dict or object with attributes
                if isinstance(tool_call, dict):
                    name = tool_call.get("name", "")
                else:
                    name = getattr(tool_call, "name", "")
                if name:
                    return name

        return None

    def process_token(self, token) -> str | None:
        """Process a streaming token and accumulate tool call chunks.

        Args:
            token: LangChain token/message that may contain tool_call chunks

        Returns:
            Query string if a complete retrieve_context tool call is detected, None otherwise
        """
        # Try content_blocks for tool_call_chunk (streaming mode)
        content_blocks = getattr(token, "content_blocks", None)
        if content_blocks:
            for block in content_blocks:
                if block.get("type") == "tool_call_chunk":
                    chunk_index = block.get("index", 0)
                    chunk_name = block.get("name", "")
                    chunk_args = block.get("args", "")

                    # Initialize accumulator for this tool call index
                    if chunk_index not in self._accumulated_calls:
                        self._accumulated_calls[chunk_index] = {
                            "name": "",
                            "args": "",
                        }

                    # Accumulate tool call data
                    if chunk_name:
                        self._accumulated_calls[chunk_index]["name"] = chunk_name
                    if chunk_args:
                        # Args may come as string chunks - accumulate them
                        self._accumulated_calls[chunk_index]["args"] += str(chunk_args)

                    # Check if we have a complete retrieve_context call
                    call_data = self._accumulated_calls[chunk_index]
                    if call_data["name"] == "retrieve_context" and call_data["args"]:
                        query = self._extract_query_from_args(call_data["args"])
                        if query:
                            return query

        # Try tool_calls attribute (may be complete or partial)
        tool_calls = getattr(token, "tool_calls", None)
        if tool_calls:
            for tool_call in tool_calls:
                # Handle different formats: dict or object with attributes
                if isinstance(tool_call, dict):
                    args = tool_call.get("args", {}) or tool_call.get("arguments", {})
                    name = tool_call.get("name", "")
                else:
                    args = getattr(tool_call, "args", None) or getattr(tool_call, "arguments", None)
                    name = getattr(tool_call, "name", "")

                # Check if this is retrieve_context tool
                if name == "retrieve_context" and args:
                    query = self._extract_query_from_args(args)
                    if query:
                        return query

        return None

    def _extract_query_from_args(self, args: dict | str) -> str | None:
        """Extract query from tool call arguments.

        Args:
            args: Tool call arguments (dict or JSON string)

        Returns:
            Query string if found, None otherwise
        """
        if isinstance(args, dict):
            query = args.get("query")
            if query and isinstance(query, str):
                return query.strip()
        elif isinstance(args, str):
            # Args might be JSON string (from accumulated chunks)
            try:
                parsed_args = json.loads(args)
                if isinstance(parsed_args, dict):
                    query = parsed_args.get("query")
                    if query and isinstance(query, str):
                        return query.strip()
            except (json.JSONDecodeError, AttributeError):
                pass

        return None

    def reset(self) -> None:
        """Reset accumulator state (clear accumulated calls)."""
        self._accumulated_calls.clear()

    @staticmethod
    def extract_query_from_complete_tool_call(tool_call: dict | object) -> str | None:
        """Extract query from a complete tool call (non-streaming mode).

        Args:
            tool_call: Complete tool call object (dict or object with attributes)

        Returns:
            Query string if found, None otherwise
        """
        # Handle different formats: dict or object with attributes
        if isinstance(tool_call, dict):
            args = tool_call.get("args", {}) or tool_call.get("arguments", {})
            name = tool_call.get("name", "")
        else:
            args = getattr(tool_call, "args", None) or getattr(tool_call, "arguments", None)
            name = getattr(tool_call, "name", "")

        # Check if this is retrieve_context tool
        if name == "retrieve_context" and args:
            accumulator = ToolCallAccumulator()
            return accumulator._extract_query_from_args(args)

        return None


def yield_search_started(query: str | None = None) -> dict:
    """Yield metadata message for search started with pending spinner.

    Args:
        query: Optional user query being searched, for display in the bubble.

    Returns:
        Gradio message dict with metadata for search started.
        Content and title are resolved i18n strings (never i18n metadata objects).
        Includes status="pending" to show native Gradio spinner.

    Example:
        >>> from rag_engine.api.stream_helpers import yield_search_started
        >>> msg = yield_search_started()
        >>> "Searching" in msg["metadata"]["title"] or "Поиск" in msg["metadata"]["title"]
        True
        >>> msg["metadata"]["status"]
        'pending'
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
            # Explicit UI-only marker (used by _is_ui_only_message)
            "ui_type": "search_started",
            # Native Gradio spinner: "pending" shows spinner, "done" hides it
            "status": "pending",
        },
    }


def yield_thinking_block(tool_name: str) -> dict:
    """Yield metadata message for generic thinking block with pending spinner.

    Args:
        tool_name: Name of the tool being used (e.g., "add", "get_current_datetime")

    Returns:
        Gradio message dict with metadata for thinking block.
        Content and title are resolved i18n strings (never i18n metadata objects).
        Includes status="pending" to show native Gradio spinner.

    Example:
        >>> from rag_engine.api.stream_helpers import yield_thinking_block
        >>> msg = yield_thinking_block("add")
        >>> "Thinking" in msg["metadata"]["title"] or "Размышление" in msg["metadata"]["title"]
        True
        >>> msg["metadata"]["status"]
        'pending'
    """
    # Resolve i18n translations to plain strings before yielding
    # This ensures Chatbot receives strings, not __i18n__ metadata objects
    title = get_text("thinking_title")
    content = get_text("thinking_content", tool_name=tool_name)

    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
            # Explicit UI-only marker (used by _is_ui_only_message)
            "ui_type": "thinking",
            # Native Gradio spinner: "pending" shows spinner, "done" hides it
            "status": "pending",
        },
    }


def yield_sgr_planning_started() -> dict:
    """Yield metadata message for SGR planning started with pending spinner."""
    title = get_text("sgr_planning_title")
    content = get_text("sgr_planning_content")
    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
            "ui_type": "sgr_planning",
            "status": "pending",
        },
    }


def yield_search_completed(
    count: int | None = None,
    articles: list[dict] | None = None,
) -> dict:
    """Yield metadata message for search completed (no spinner, stays open).

    Args:
        count: Optional article count to include in message.
        articles: Optional list of article dicts with 'title' and 'url' keys to display as sources.

    Returns:
        Gradio message dict with metadata for search completed.
        Content and title are resolved i18n strings (never i18n metadata objects).
        No status field - accordion stays open to show clickable article links.

    Example:
        >>> from rag_engine.api.stream_helpers import yield_search_completed
        >>> msg = yield_search_completed(5)
        >>> "Found" in msg["metadata"]["title"] or "завершен" in msg["metadata"]["title"]
        True
    """
    # Resolve i18n translations to plain strings
    title = get_text("search_completed_title_with_count")
    base_content = get_text(
        "search_completed_content_with_count",
        count=count if count is not None else 0,
    )

    # Add article sources if provided
    content_parts = [base_content]
    if articles:
        sources_lines = []
        for i, article in enumerate(articles, start=1):
            title_text = article.get("title", "Untitled")
            url = article.get("url", "")
            if url:
                sources_lines.append(f"{i}. [{title_text}]({url})")
            else:
                sources_lines.append(f"{i}. {title_text}")

        if sources_lines:
            content_parts.append(f"\n\n{get_text('sources_header')}")
            content_parts.extend(sources_lines)

    content = "\n".join(content_parts)

    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
            # Explicit UI-only marker (used by _is_ui_only_message)
            "ui_type": "search_completed",
            # NO status field - accordion stays open to show clickable article links
            # Previous "search_started" spinner is stopped via update_message_status_in_history()
        },
    }


def yield_model_switch_notice(model: str) -> dict:
    """Yield metadata message for model switch (stays open for visibility).

    Args:
        model: Model name that was switched to

    Returns:
        Gradio message dict with metadata for model switch.
        Content and title are resolved i18n strings.
        No status field - accordion stays open so users see which model is being used.

    Example:
        >>> from rag_engine.api.stream_helpers import yield_model_switch_notice
        >>> msg = yield_model_switch_notice("gemini-2.5-pro")
        >>> model in msg["metadata"]["title"]
        True
    """
    # Resolve i18n translations to plain strings before yielding
    # This ensures Chatbot receives strings, not __i18n__ metadata objects
    title = get_text("model_switch_title", model=model)

    return {
        "role": "assistant",
        "content": "",
        "metadata": {
            "title": title,
            # Explicit UI-only marker (used by _is_ui_only_message)
            "ui_type": "model_switch",
            # NO status - accordion stays open for visibility (important info)
        },
    }


def yield_generating_answer() -> dict:
    """Yield metadata message for answer generation phase with spinner.

    Returns:
        Gradio message dict with metadata for answer generation.
        Content and title are resolved i18n strings (never i18n metadata objects).
        Includes status="pending" to show native Gradio spinner.

    Example:
        >>> from rag_engine.api.stream_helpers import yield_generating_answer
        >>> msg = yield_generating_answer()
        >>> "Generating" in msg["metadata"]["title"] or "Генерация" in msg["metadata"]["title"]
        True
        >>> msg["metadata"]["status"]
        'pending'
    """
    # Resolve i18n translations to plain strings before yielding
    title = get_text("generating_answer_title")
    content = get_text("generating_answer_content")

    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
            # Explicit UI-only marker (used by _is_ui_only_message)
            "ui_type": "generating_answer",
            # Native Gradio spinner: "pending" shows spinner during answer generation
            "status": "pending",
        },
    }


def yield_cancelled() -> dict:
    """Yield metadata message for cancelled response (stays open for visibility).

    Returns:
        Gradio message dict with metadata for cancellation.
        Content and title are resolved i18n strings (never i18n metadata objects).
        No status field - accordion stays open so users see the cancellation notice.

    Example:
        >>> from rag_engine.api.stream_helpers import yield_cancelled
        >>> msg = yield_cancelled()
        >>> "Cancelled" in msg["metadata"]["title"] or "Отменено" in msg["metadata"]["title"]
        True
    """
    # Resolve i18n translations to plain strings before yielding
    # This ensures Chatbot receives strings, not __i18n__ metadata objects
    title = get_text("cancelled_title")
    content = get_text("cancelled_message")

    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
            # Explicit UI-only marker (used by _is_ui_only_message)
            "ui_type": "cancelled",
            # NO status - accordion stays open for visibility (important notice)
        },
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


def update_search_started_in_history(gradio_history: list[dict], query: str) -> bool:
    """Update the last pending search_started message in Gradio history with new query.

    Only updates search_started messages that are still pending (not yet completed).
    This ensures each tool call's search_started message is updated with its own query,
    even when multiple tool calls occur in sequence.

    Args:
        gradio_history: List of Gradio message dictionaries
        query: Query string to update the search_started message with

    Returns:
        True if message was updated, False otherwise

    Example:
        >>> history = [{"role": "assistant", "content": "...", "metadata": {"title": "🧠 Поиск"}}]
        >>> update_search_started_in_history(history, "new query")
        True
    """
    if not query:
        return False

    # Find the last search_started message that is still pending
    # (i.e., doesn't have a search_completed message after it)
    last_pending_search_started_idx = None

    for i in range(len(gradio_history) - 1, -1, -1):
        msg = gradio_history[i]
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            metadata = msg.get("metadata", {})
            ui_type = metadata.get("ui_type")

            # If we find a search_completed, stop searching backwards
            # Any search_started before this point is already completed
            if ui_type == "search_completed":
                break

            # Track the most recent pending search_started
            if ui_type == "search_started":
                status = metadata.get("status", "pending")
                # Only update if still pending (not "done")
                if status == "pending":
                    last_pending_search_started_idx = i
                    # Use the first pending one we find (most recent)
                    break

    # Update the found pending search_started message
    if last_pending_search_started_idx is not None:
        updated_msg = yield_search_started(query)
        gradio_history[last_pending_search_started_idx] = updated_msg
        return True

    return False


def update_message_status_in_history(
    gradio_history: list[dict],
    ui_type: str,
    new_status: str,
) -> bool:
    """Update the status of the last message with given ui_type in Gradio history.

    This is useful for transitioning messages from "pending" to "done" state,
    which removes the spinner in Gradio's native UI.

    Args:
        gradio_history: List of Gradio message dictionaries
        ui_type: The ui_type to search for ("thinking", "search_started", etc.)
        new_status: New status value ("pending" or "done")

    Returns:
        True if message was updated, False otherwise

    Example:
        >>> history = [{
        ...     "role": "assistant",
        ...     "content": "Searching...",
        ...     "metadata": {"ui_type": "search_started", "status": "pending"}
        ... }]
        >>> update_message_status_in_history(history, "search_started", "done")
        True
        >>> history[0]["metadata"]["status"]
        'done'
    """
    # Search backwards to find the most recent message with matching ui_type
    for i in range(len(gradio_history) - 1, -1, -1):
        msg = gradio_history[i]
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            metadata = msg.get("metadata")
            # Handle None metadata (some messages may not have metadata)
            if metadata is None:
                continue
            if not isinstance(metadata, dict):
                continue
            if metadata.get("ui_type") == ui_type:
                # Update status in place
                metadata["status"] = new_status
                return True

    return False
