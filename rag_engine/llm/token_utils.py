from __future__ import annotations

"""Shared token estimation utilities for consistent budgeting.

Single source of truth for counting tokens across system prompt, user
question, context, and reserved output/overhead.
"""


import tiktoken

from rag_engine.config.settings import settings

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Heuristic threshold to avoid expensive encodes on very large inputs
_FAST_PATH_CHAR_LEN = settings.retrieval_fast_token_char_threshold


def count_tokens(content: str) -> int:
    """Count tokens in a string using tiktoken with fast path for large content.

    This is the centralized token counting utility used throughout the codebase.
    For small to medium content (< threshold), uses exact tiktoken encoding.
    For very large content (>= threshold), uses fast approximation (chars // 4).

    Args:
        content: Text to count tokens for

    Returns:
        Estimated token count

    Example:
        >>> from rag_engine.llm.token_utils import count_tokens
        >>> tokens = count_tokens("Hello, world!")
        >>> tokens >= 3  # Exact count varies by encoding
        True

        >>> # For large content, automatically uses fast path
        >>> large_text = "x" * 100_000
        >>> tokens = count_tokens(large_text)  # Uses len(content) // 4
        >>> tokens == 25000
        True
    """
    if not content:
        return 0

    content_str = str(content)

    # Fast path for very large content to avoid performance issues
    # Matches the same logic used in estimate_tokens_for_request
    if len(content_str) > _FAST_PATH_CHAR_LEN:
        return len(content_str) // 4

    # Use exact tiktoken counting for smaller content
    return len(_ENCODING.encode(content_str))


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


