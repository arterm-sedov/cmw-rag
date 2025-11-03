"""Compression utilities for managing context limits dynamically."""
from __future__ import annotations

import json
import logging

from rag_engine.llm.summarization import summarize_to_tokens
from rag_engine.llm.token_utils import count_tokens
from rag_engine.utils.message_utils import (
    extract_user_question,
    get_message_content,
    is_tool_message,
    update_tool_message_content,
)

logger = logging.getLogger(__name__)


def compress_articles_to_target_tokens(
    articles: list[dict],
    target_ratio: float | None = None,
    min_tokens: int | None = None,
    guidance: str | None = None,
    llm_manager=None,
) -> tuple[list[dict], int]:
    """Compress articles to target token ratio.

    Compresses articles starting from the end (least relevant first) until
    target token reduction is achieved.

    Args:
        articles: List of article dicts with 'content', 'title', 'url', 'metadata'
        target_ratio: Target compression ratio (default: 0.30 = 30% of original)
        min_tokens: Minimum tokens per compressed article (default: 300)
        guidance: Optional user question for summarization guidance
        llm_manager: LLMManager instance for summarization

    Returns:
        Tuple of (compressed_articles, tokens_saved)

    Example:
        >>> from rag_engine.llm.compression import compress_articles_to_target_tokens
        >>> articles = [{"content": "..."}]
        >>> compressed, saved = compress_articles_to_target_tokens(articles)
        >>> saved >= 0
        True
    """
    from rag_engine.config.settings import settings

    if target_ratio is None:
        target_ratio = settings.llm_compression_article_ratio
    if min_tokens is None:
        min_tokens = settings.llm_compression_min_tokens

    if not articles or not llm_manager:
        return articles, 0

    tokens_saved = 0
    compressed_articles = list(articles)

    # Compress from the end (least relevant first)
    for i in range(len(compressed_articles) - 1, -1, -1):
        article = compressed_articles[i]
        original_content = article.get("content", "")
        if not original_content:
            continue

        original_tokens = count_tokens(original_content)
        article_target = max(min_tokens, int(original_tokens * target_ratio))

        try:
            compressed = summarize_to_tokens(
                title=article.get("title", "Article"),
                url=article.get("url", ""),
                matched_chunks=[original_content],
                full_body=None,
                target_tokens=article_target,
                guidance=guidance,
                llm=llm_manager,
                max_retries=1,  # Quick compression
            )

            compressed_tokens = count_tokens(compressed)
            compressed_articles[i]["content"] = compressed
            if "metadata" not in compressed_articles[i]:
                compressed_articles[i]["metadata"] = {}
            compressed_articles[i]["metadata"]["compressed"] = True

            saved = original_tokens - compressed_tokens
            tokens_saved += saved

            logger.info(
                "Compressed article '%s': %d → %d tokens (saved %d)",
                article.get("title", "")[:50],
                original_tokens,
                compressed_tokens,
                saved,
            )
        except Exception as exc:
            logger.warning("Failed to compress article at index %d: %s", i, exc)
            continue

    return compressed_articles, tokens_saved


def compress_tool_messages(
    messages: list,
    runtime,
    llm_manager,
    threshold_pct: float = 0.85,
    target_pct: float = 0.80,
) -> list | None:
    """Compress tool messages if context exceeds threshold.

    This function checks if the accumulated context exceeds the threshold,
    and if so, compresses tool messages to bring it below the target.

    Args:
        messages: List of message objects
        runtime: Runtime object with access to model config
        llm_manager: LLMManager instance for compression
        threshold_pct: Threshold percentage for triggering compression (default: 0.85)
        target_pct: Target percentage after compression (default: 0.80)

    Returns:
        Updated list of messages if compression occurred, None otherwise

    Example:
        >>> from rag_engine.llm.compression import compress_tool_messages
        >>> # ... setup messages, runtime, llm_manager ...
        >>> compressed = compress_tool_messages(messages, runtime, llm_manager)
    """
    from rag_engine.config.settings import settings
    from rag_engine.llm.llm_manager import get_context_window
    from rag_engine.llm.token_utils import count_messages_tokens

    if not messages:
        return None

    # Get current model's context window
    current_model = getattr(runtime, "model", None) or settings.default_model
    context_window = get_context_window(current_model)
    threshold = int(context_window * threshold_pct)

    # Count total tokens in messages and find tool message indices
    total_tokens = count_messages_tokens(messages)
    tool_message_indices = []

    for idx, msg in enumerate(messages):
        if is_tool_message(msg):
            tool_message_indices.append(idx)

    # Check if we need compression
    if total_tokens <= threshold:
        return None  # All good, no changes needed

    if not tool_message_indices:
        return None  # No tool messages to compress

    logger.warning(
        "Context at %d tokens (%.1f%% of %d window), compressing tool results",
        total_tokens,
        100 * total_tokens / context_window,
        context_window,
    )

    # Calculate how much to compress (target: get below target_pct)
    target_tokens = int(context_window * target_pct)
    tokens_to_save = total_tokens - target_tokens

    # Find the user's question for summarization guidance
    user_question = extract_user_question(messages)

    # Compress tool messages, starting from the last one (least relevant)
    tokens_saved = 0
    updated_messages = list(messages)  # Copy to avoid mutating original

    for idx in reversed(tool_message_indices):
        if tokens_saved >= tokens_to_save:
            break

        msg = updated_messages[idx]
        content = get_message_content(msg)

        if not content:
            continue

        try:
            # Parse tool result JSON
            result = json.loads(content)
            articles = result.get("articles", [])

            if not articles:
                continue

            # Compress articles
            compressed_articles, article_tokens_saved = compress_articles_to_target_tokens(
                articles=articles,
                target_ratio=None,
                min_tokens=None,
                guidance=user_question,
                llm_manager=llm_manager,
            )

            tokens_saved += article_tokens_saved

            # Update the message content with compressed articles
            result["articles"] = compressed_articles
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["compressed_articles_count"] = sum(
                1 for a in compressed_articles if a.get("metadata", {}).get("compressed")
            )
            result["metadata"]["tokens_saved_by_compression"] = tokens_saved

            # Create new compact JSON
            new_content = json.dumps(result, ensure_ascii=False, separators=(",", ":"))

            # Update message content
            updated_messages = update_tool_message_content(updated_messages, idx, new_content)

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Failed to compress tool message at index %d: %s", idx, exc)
            continue

    if tokens_saved > 0:
        logger.info(
            "Compression complete: saved %d tokens, new total ~%d (%.1f%% of window)",
            tokens_saved,
            total_tokens - tokens_saved,
            100 * (total_tokens - tokens_saved) / context_window,
        )
        return updated_messages

    return None  # No compression happened

