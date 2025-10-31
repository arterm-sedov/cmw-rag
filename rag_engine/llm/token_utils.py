from __future__ import annotations

"""Shared token estimation utilities for consistent budgeting.

Single source of truth for counting tokens across system prompt, user
question, context, and reserved output/overhead.
"""

from typing import Dict

import tiktoken
from rag_engine.config.settings import settings


_ENCODING = tiktoken.get_encoding("cl100k_base")

# Heuristic threshold to avoid expensive encodes on very large inputs
_FAST_PATH_CHAR_LEN = settings.retrieval_fast_token_char_threshold


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
    system_s = str(system_prompt or "")
    question_s = str(question or "")
    context_s = str(context or "")

    # Fast-path for very large strings: approximate as ~4 chars per token
    # This is sufficient for budgeting decisions and avoids long encode times
    system_tokens = (len(system_s) // 4) if len(system_s) > _FAST_PATH_CHAR_LEN else len(_ENCODING.encode(system_s))
    question_tokens = (len(question_s) // 4) if len(question_s) > _FAST_PATH_CHAR_LEN else len(_ENCODING.encode(question_s))
    context_tokens = (len(context_s) // 4) if len(context_s) > _FAST_PATH_CHAR_LEN else len(_ENCODING.encode(context_s))
    input_tokens = system_tokens + question_tokens + context_tokens + int(overhead)
    output_tokens = int(max_output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


