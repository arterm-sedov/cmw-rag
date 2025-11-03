"""Context tracking utilities for agent execution.

These utilities help the agent track accumulated context during multi-tool execution,
enabling progressive budgeting and preventing context overflow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from rag_engine.retrieval.retriever import Article

logger = logging.getLogger(__name__)


class AgentContext(BaseModel):
    """Typed context passed from agent to tools for progressive budgeting.

    This context is passed via the `context` parameter in agent.invoke()
    and accessed by tools via `runtime.context` for clean, typed access.
    Follows LangChain 1.0 official pattern for context sharing.

    Attributes:
        conversation_tokens: Tokens used by conversation history
        accumulated_tool_tokens: Tokens from previous tool calls (deduplicated)
    """

    conversation_tokens: int = Field(
        default=0,
        description="Tokens used by conversation history (user/assistant messages)",
    )

    accumulated_tool_tokens: int = Field(
        default=0,
        description="Tokens accumulated from previous tool calls in this turn (deduplicated)",
    )


def estimate_accumulated_tokens(
    conversation_messages: list[dict],
    tool_results: list[str],
) -> tuple[int, int]:
    """Estimate tokens from conversation and accumulated tool results.

    This is the agent's responsibility - counting context as tool calls accumulate.
    The tool should just receive this information and pass it to the retriever.

    Args:
        conversation_messages: List of user/assistant messages from history
        tool_results: List of JSON strings from tool calls in THIS turn

    Returns:
        Tuple of (conversation_tokens, accumulated_tool_tokens)

    Example:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> tool_results = ['{"articles": [...]}', '{"articles": [...]}']
        >>> conv_tokens, tool_tokens = estimate_accumulated_tokens(messages, tool_results)
        >>> total = conv_tokens + tool_tokens  # Pass this to retriever
    """
    from rag_engine.llm.token_utils import count_tokens
    from rag_engine.tools.utils import parse_tool_result_to_articles

    conversation_tokens = 0
    accumulated_tool_tokens = 0

    # Count conversation history tokens
    for msg in conversation_messages:
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            conversation_tokens += count_tokens(content)

    # Count accumulated tool result tokens (deduplicated by kb_id)
    if tool_results:
        seen_kb_ids = set()

        for tool_result in tool_results:
            try:
                articles = parse_tool_result_to_articles(tool_result)
                for article in articles:
                    if article.kb_id and article.kb_id not in seen_kb_ids:
                        seen_kb_ids.add(article.kb_id)
                        # Count only unique articles
                        accumulated_tool_tokens += count_tokens(article.content)
            except Exception as exc:
                # If parsing fails, fall back to rough estimate
                logger.warning("Failed to parse tool result for token counting: %s", exc)
                accumulated_tool_tokens += count_tokens(tool_result) // 2

        logger.debug(
            "Accumulated context: conversation=%d tokens, tools=%d tokens (%d unique articles)",
            conversation_tokens,
            accumulated_tool_tokens,
            len(seen_kb_ids),
        )

    return conversation_tokens, accumulated_tool_tokens


def extract_articles_from_runtime_state(runtime_state: dict) -> list[Article]:
    """Extract articles from runtime state messages (for tool use).

    This is a lightweight helper for the tool to get already-accumulated
    article data from runtime.state without doing heavy parsing.

    Args:
        runtime_state: Runtime state dict with 'messages' key

    Returns:
        List of Article objects from tool result messages

    Example:
        >>> # In tool:
        >>> articles = extract_articles_from_runtime_state(runtime.state)
        >>> # Use for deduplication or other lightweight operations
    """
    from rag_engine.tools.utils import parse_tool_result_to_articles

    articles = []
    messages = runtime_state.get("messages", [])

    for msg in messages:
        # Get message content
        if hasattr(msg, "content"):
            content = msg.content
        else:
            content = msg.get("content", "") if isinstance(msg, dict) else ""

        if isinstance(content, str) and content:
            msg_type = getattr(msg, "type", None)

            if msg_type == "tool":
                # Tool result - parse articles
                try:
                    tool_articles = parse_tool_result_to_articles(content)
                    articles.extend(tool_articles)
                except Exception as exc:
                    logger.warning("Failed to parse articles from runtime state: %s", exc)

    return articles

