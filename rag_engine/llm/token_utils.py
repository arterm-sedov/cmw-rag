from __future__ import annotations

"""Shared token estimation utilities for consistent budgeting.

Single source of truth for counting tokens across system prompt, user
question, context, and reserved output/overhead.
"""

from typing import Dict

import tiktoken


_ENCODING = tiktoken.get_encoding("cl100k_base")


def estimate_tokens_for_request(
    *,
    system_prompt: str,
    question: str,
    context: str,
    max_output_tokens: int,
    overhead: int = 100,
) -> Dict[str, int]:
    """Estimate token usage for a chat request.

    Returns a dict with input_tokens, output_tokens, and total_tokens.
    """
    # Coerce to strings defensively to avoid tests passing mocks/objects
    system_tokens = len(_ENCODING.encode(str(system_prompt or "")))
    question_tokens = len(_ENCODING.encode(str(question or "")))
    context_tokens = len(_ENCODING.encode(str(context or "")))
    input_tokens = system_tokens + question_tokens + context_tokens + int(overhead)
    output_tokens = int(max_output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


