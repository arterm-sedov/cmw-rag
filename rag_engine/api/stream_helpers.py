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

    def _extract_query_from_args(self, args) -> str | None:
        """Extract query from accumulated args.

        Handles both string (JSON) and dict formats.

        Args:
            args: Tool call arguments (dict or JSON string)

        Returns:
            Query string if found, None otherwise
        """
        if isinstance(args, dict):
            return args.get("query", "")

        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                return parsed.get("query", "")
            except json.JSONDecodeError:
                # Try to extract from partial/accumulated JSON
                # Look for "query": "..." pattern
                import re
                match = re.search(r'"query"\s*:\s*"([^"]*)"', args)
                if match:
                    return match.group(1)

        return None


def yield_disclaimer() -> dict:
    """Yield disclaimer message at start of conversation (stays open for visibility).

    Returns:
        Gradio message dict with disclaimer content.
        The disclaimer stays visible to inform users about AI limitations.
    """
    from rag_engine.llm.prompts import AI_DISCLAIMER

    return {
        "role": "assistant",
        "content": AI_DISCLAIMER.strip(),
        # NO metadata - disclaimer is a regular message, not UI-only
        # This ensures it's included in conversation context
    }


def yield_thinking_spinner() -> dict:
    """Yield thinking spinner at start of agent processing.

    Returns:
        Gradio message dict with thinking spinner metadata.
        Includes status="pending" to show native Gradio spinner.
    """
    # Resolve i18n translations to plain strings before yielding
    # This ensures Chatbot receives strings, not __i18n__ metadata objects
    title = get_text("thinking_title")
    content = get_text("thinking_content_initial")

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


def yield_search_started(query: str | None = None, search_id: str | None = None) -> dict:
    """Yield metadata message for search started with pending spinner.

    Args:
        query: Optional search query string to display
        search_id: Unique ID to match this bubble with its result (for parallel execution)

    Returns:
        Gradio message dict with metadata for search started indicator.
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
    import uuid

    # Resolve i18n translations to plain strings before yielding
    # This ensures Chatbot receives strings, not __i18n__ metadata objects
    title = get_text("search_started_title")
    content = get_text("search_started_content", query=(query or "").strip())

    # Generate unique ID if not provided (for matching with result)
    if search_id is None:
        search_id = str(uuid.uuid4())[:8]  # Short UUID for readability

    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
            # Explicit UI-only marker (used by _is_ui_only_message)
            "ui_type": "search_started",
            # Native Gradio spinner: "pending" shows spinner, "done" hides it
            "status": "pending",
            # Unique ID for matching with result in parallel execution
            "search_id": search_id,
        },
    }


def yield_search_bubble(query: str, search_id: str | None = None) -> dict:
    """Create a unified search bubble that evolves from pending to complete.

    This is the single dynamic bubble approach that replaces the old two-bubble
    system (search_started + search_completed). The bubble is created with pending
    status and updated in-place when results arrive.

    Args:
        query: Search query string to display
        search_id: Unique ID for matching (auto-generated if not provided)

    Returns:
        Gradio message dict with metadata for search bubble.
        Content: "Ищу: {query}"
        Title: "🔄 Поиск в базе знаний"
        Status: "pending" (spinner visible)
    """
    import uuid

    if search_id is None:
        search_id = str(uuid.uuid4())[:8]

    title = get_text("search_started_title")
    content = get_text("search_started_content", query=query.strip())

    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
            "ui_type": "search_bubble",  # New unified type
            "status": "pending",
            "search_id": search_id,
            "query": query,  # Store original query for updates
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
    """Yield metadata message for SGR (Spam/Goal/Recipe) planning phase.

    This indicates the agent is analyzing the user request before executing tools.

    Returns:
        Gradio message dict with metadata for SGR planning indicator.
        Content and title are resolved i18n strings.
        Includes status="pending" to show native Gradio spinner.

    Example:
        >>> from rag_engine.api.stream_helpers import yield_sgr_planning_started
        >>> msg = yield_sgr_planning_started()
        >>> "Planning" in msg["metadata"]["title"] or "Планирование" in msg["metadata"]["title"]
        True
        >>> msg["metadata"]["status"]
        'pending'
    """
    # Resolve i18n translations to plain strings before yielding
    title = get_text("sgr_planning_title")
    content = get_text("sgr_planning_content")

    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
            # Explicit UI-only marker (used by _is_ui_only_message)
            "ui_type": "sgr_planning",
            # Native Gradio spinner: "pending" shows spinner, "done" hides it
            "status": "pending",
        },
    }


def yield_search_completed(count: int, articles: list[dict] | None = None) -> dict:
    """Yield metadata message for search completed (stays open for visibility).

    Args:
        count: Number of articles found
        articles: Optional list of article dicts with 'title' and 'url' keys

    Returns:
        Gradio message dict with metadata for search completed.
        Content and title are resolved i18n strings.
        No status field - accordion stays open to show clickable article links.

    Example:
        >>> from rag_engine.api.stream_helpers import yield_search_completed
        >>> msg = yield_search_completed(5)
        >>> "Found" in msg["metadata"]["title"] or "Найдено" in msg["metadata"]["title"]
        True
    """
    # Resolve i18n translations to plain strings before yielding
    # This ensures Chatbot receives strings, not __i18n__ metadata objects
    title = get_text("search_completed_title")

    # Format content based on count
    if count == 0:
        base_content = get_text("search_completed_content_no_results")
    elif count == 1:
        base_content = get_text("search_completed_content_single")
    else:
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
    # This ensures Chatbot receives strings, not __i18n__ metadata objects
    title = get_text("generating_answer_title")
    content = get_text("generating_answer_content")

    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
            # Explicit UI-only marker (used by _is_ui_only_message)
            "ui_type": "generating_answer",
            # Native Gradio spinner: "pending" shows spinner, "done" hides it
            "status": "pending",
        },
    }


def yield_cancelled() -> dict:
    """Yield metadata message for cancelled generation (stays open for visibility).

    Returns:
        Gradio message dict with metadata for cancelled indicator.
        Content and title are resolved i18n strings.
        No status field - accordion stays open so users see cancellation notice.

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
        logger.debug("update_search_started_in_history: query is empty, skipping update")
        return False

    # Find the last search_started message that is still pending
    # (i.e., doesn't have a search_completed message after it)
    last_pending_search_started_idx = None

    for i in range(len(gradio_history) - 1, -1, -1):
        msg = gradio_history[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        metadata = msg.get("metadata")
        if not metadata or not isinstance(metadata, dict):
            continue
        ui_type = metadata.get("ui_type")

        # If we find a search_completed, stop searching backwards
        # Any search_started before this point is already completed
        if ui_type == "search_completed":
            logger.debug("update_search_started_in_history: found search_completed at index %d, stopping search", i)
            break

        # Track the most recent pending search_started
        if ui_type == "search_started":
            # Check if this search_started already has the same query
            content = msg.get("content", "")
            empty_query_content = get_text("search_started_content", query="").strip()
            if content.strip() != empty_query_content:
                # This search_started already has a query, skip update (will append new bubble)
                logger.debug("update_search_started_in_history: pending search_started at index %d already has query, skip update (append new bubble)", i)
                break
            last_pending_search_started_idx = i
            logger.debug("update_search_started_in_history: found pending search_started at index %d", i)

    # Update the found pending search_started message
    if last_pending_search_started_idx is not None:
        updated_msg = yield_search_started(query)
        gradio_history[last_pending_search_started_idx] = updated_msg
        logger.info("update_search_started_in_history: updated search_started block at index %d with query='%s'", last_pending_search_started_idx, query[:50])
        return True

    logger.debug("update_search_started_in_history: no pending search_started block found to update")
    return False


def last_pending_search_started_has_query(gradio_history: list[dict], query: str) -> bool:
    """Return True if the most recent pending search_started message already displays this query.

    This prevents duplicate bubbles when multiple detection paths (accumulator vs tool_calls)
    try to create bubbles for the same query.

    Args:
        gradio_history: List of Gradio message dictionaries
        query: Query string to check

    Returns:
        True if the last pending search_started has the same query content, False otherwise
    """
    expected_content = get_text("search_started_content", query=query).strip()

    # Search backwards to find the most recent pending search_started
    for i in range(len(gradio_history) - 1, -1, -1):
        msg = gradio_history[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        metadata = msg.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if metadata.get("ui_type") != "search_started":
            continue

        # Check if content matches
        content = msg.get("content", "").strip()
        if content == expected_content:
            return True

        # Stop at first search_started found (most recent pending)
        break

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


def update_search_started_by_id(
    gradio_history: list[dict],
    search_id: str,
    new_status: str = "done",
) -> bool:
    """Update the status of a specific search_started message by its ID.

    This is crucial for parallel execution where multiple searches happen concurrently.
    Uses stable ID matching instead of content comparison for reliability.

    Args:
        gradio_history: List of Gradio message dictionaries
        search_id: The unique search_id to match
        new_status: New status value ("pending" or "done")

    Returns:
        True if message was updated, False otherwise
    """
    if not search_id:
        return False

    # Search backwards to find the search_started with matching ID
    for i in range(len(gradio_history) - 1, -1, -1):
        msg = gradio_history[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        metadata = msg.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if metadata.get("ui_type") != "search_started":
            continue

        # Match by search_id
        if metadata.get("search_id") == search_id:
            metadata["status"] = new_status
            logger.info(
                "Marked search_started as %s for search_id=%s (index %d)",
                new_status,
                search_id,
                i
            )
            return True

    logger.debug("No matching search_started found for search_id: %s", search_id)
    return False


def update_search_bubble_by_id(
    gradio_history: list[dict],
    search_id: str,
    count: int,
    articles: list[dict] | None = None,
) -> bool:
    """Update search bubble to show results.

    Transforms the bubble from pending state to complete state:
    - Title: "🔄 Поиск в базе знаний" → "✅ Поиск завершен"
    - Content: "Ищу: query" → "Запрос: query\nНайдено статей: count\n\nИсточники: [...]"
    - Status: "pending" → "done" (spinner stops)

    Args:
        gradio_history: List of Gradio message dictionaries
        search_id: The unique search_id to match
        count: Number of articles found
        articles: List of article dicts with 'title' and 'url' keys

    Returns:
        True if bubble was updated, False otherwise
    """
    if not search_id:
        return False

    # Find bubble by search_id
    for i in range(len(gradio_history) - 1, -1, -1):
        msg = gradio_history[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        metadata = msg.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if metadata.get("ui_type") != "search_bubble":
            continue
        if metadata.get("search_id") != search_id:
            continue

        # Found the bubble - update it
        query = metadata.get("query", "")

        # Build new content
        content_parts = [
            get_text("search_completed_query_prefix", query=query),
            get_text("search_completed_count", count=count),
        ]

        # Add sources if count > 0 and articles provided
        if count > 0 and articles:
            content_parts.append("")
            content_parts.append(get_text("sources_header"))
            for idx, article in enumerate(articles, start=1):
                title = article.get("title", "Untitled")
                url = article.get("url", "")
                if url:
                    content_parts.append(f"{idx}. [{title}]({url})")
                else:
                    content_parts.append(f"{idx}. {title}")

        # Update the bubble in place
        msg["content"] = "\n".join(content_parts)
        metadata["title"] = get_text("search_completed_title")
        metadata["status"] = "done"

        logger.info(
            "Updated search bubble to complete: search_id=%s, count=%d (index %d)",
            search_id,
            count,
            i
        )
        return True

    logger.debug("No matching search_bubble found for search_id: %s", search_id)
    return False
