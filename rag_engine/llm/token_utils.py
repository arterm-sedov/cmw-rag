from __future__ import annotations

"""Shared token estimation utilities for consistent budgeting.

Single source of truth for counting tokens across system prompt, user
question, context, and reserved output/overhead.
"""


import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(content: str) -> int:
    """Count tokens in a string using exact tiktoken encoding.

    This is the centralized token counting utility used throughout the codebase.
    Uses tiktoken's cl100k_base encoding for accurate token counting
    across all languages, including Russian/Cyrillic text.

    Performance: Typically < 15ms for 200K chars, < 70ms for 1M chars.

    Args:
        content: Text to count tokens for

    Returns:
        Exact token count

    Example:
        >>> from rag_engine.llm.token_utils import count_tokens
        >>> tokens = count_tokens("Hello, world!")
        >>> tokens >= 3  # Exact count varies by encoding
        True
    """
    if not content:
        return 0

    content_str = str(content)
    return len(_ENCODING.encode(content_str))


def count_messages_tokens(messages: list) -> int:
    """Count tokens for a list of messages (dict or LangChain message objects).

    Handles both dict messages (from Gradio) and LangChain message objects.
    Uses exact tiktoken encoding for accurate token counting across all languages.

    Args:
        messages: List of message objects (dict or LangChain messages)

    Returns:
        Total token count across all messages

    Example:
        >>> from rag_engine.llm.token_utils import count_messages_tokens
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> tokens = count_messages_tokens(messages)
        >>> tokens >= 1
        True
    """
    total_tokens = 0
    for msg in messages:
        # Handle both dict (Gradio) and LangChain message objects
        if hasattr(msg, "content"):
            content = msg.content  # LangChain message object
        else:
            content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str) and content:
            total_tokens += count_tokens(content)
    return total_tokens


def estimate_tokens_for_request(
    *,
    system_prompt: str,
    question: str,
    context: str,
    max_output_tokens: int,
    overhead: int = 100,
) -> dict[str, int]:
    """Estimate token usage for a chat request.

    Returns a dict with input_tokens, output_tokens, and total_tokens.
    """
    # Coerce to strings defensively to avoid tests passing mocks/objects
    system_s = str(system_prompt or "")
    question_s = str(question or "")
    context_s = str(context or "")

    # Use centralized token counting utility
    system_tokens = count_tokens(system_s)
    question_tokens = count_tokens(question_s)
    context_tokens = count_tokens(context_s)
    input_tokens = system_tokens + question_tokens + context_tokens + int(overhead)
    output_tokens = int(max_output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


