"""Utility functions for working with LangChain tools."""
from __future__ import annotations

import json
import logging
from typing import Any

from rag_engine.retrieval.retriever import Article

logger = logging.getLogger(__name__)


def parse_tool_result_to_articles(tool_result: str) -> list[Article]:
    """Parse retrieve_context tool JSON result into Article objects.

    Args:
        tool_result: JSON string from retrieve_context tool

    Returns:
        List of Article objects

    Example:
        >>> result_json = retrieve_context.invoke({"query": "test"})
        >>> articles = parse_tool_result_to_articles(result_json)
        >>> # Use articles with LLM or format_with_citations
    """
    try:
        result = json.loads(tool_result)
        articles = []

        for article_data in result.get("articles", []):
            article = Article(
                kb_id=article_data["kb_id"],
                content=article_data["content"],
                metadata=article_data["metadata"],
            )
            articles.append(article)

        return articles
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error("Failed to parse tool result: %s", exc)
        return []


def accumulate_articles_from_tool_results(tool_results: list[str]) -> list[Article]:
    """Accumulate articles from multiple retrieve_context tool calls.

    This function collects articles from multiple tool invocations and returns
    them as a single list. Deduplication by kbId/URL happens later in
    format_with_citations(), so all articles are preserved here.

    Args:
        tool_results: List of JSON strings from retrieve_context tool calls

    Returns:
        Combined list of Article objects from all tool calls

    Example:
        >>> # LLM makes multiple tool calls
        >>> result1 = retrieve_context.invoke({"query": "authentication"})
        >>> result2 = retrieve_context.invoke({"query": "permissions"})
        >>> 
        >>> # Accumulate all articles
        >>> all_articles = accumulate_articles_from_tool_results([result1, result2])
        >>> 
        >>> # Use with LLM - format_with_citations will deduplicate
        >>> answer = llm_manager.generate(question, all_articles)
        >>> final = format_with_citations(answer, all_articles)
    """
    accumulated = []

    for tool_result in tool_results:
        articles = parse_tool_result_to_articles(tool_result)
        accumulated.extend(articles)

    logger.info(
        "Accumulated %d articles from %d tool call(s)",
        len(accumulated),
        len(tool_results),
    )

    return accumulated


def extract_metadata_from_tool_result(tool_result: str) -> dict[str, Any]:
    """Extract metadata from retrieve_context tool result.

    Args:
        tool_result: JSON string from retrieve_context tool

    Returns:
        Metadata dict containing query, articles_count, has_results, etc.

    Example:
        >>> result = retrieve_context.invoke({"query": "test"})
        >>> meta = extract_metadata_from_tool_result(result)
        >>> print(f"Found {meta['articles_count']} articles")
    """
    try:
        result = json.loads(tool_result)
        return result.get("metadata", {})
    except json.JSONDecodeError:
        logger.error("Failed to parse tool result metadata")
        return {}

