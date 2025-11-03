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

    This function collects articles from multiple tool invocations and deduplicates
    them by kb_id to prevent the LLM from seeing the same article content multiple times.
    This is critical when the agent makes similar queries that return overlapping results.

    Deduplication strategy:
    - Primary key: kb_id (unique article identifier)
    - Preserves first occurrence of each article
    - Maintains original ordering from tool calls

    Args:
        tool_results: List of JSON strings from retrieve_context tool calls

    Returns:
        Deduplicated list of Article objects from all tool calls

    Example:
        >>> # LLM makes multiple tool calls with overlapping results
        >>> result1 = retrieve_context.invoke({"query": "authentication"})
        >>> # Returns: Article A, B, C
        >>> result2 = retrieve_context.invoke({"query": "user authentication"})
        >>> # Returns: Article A, D, E (A is duplicate!)
        >>>
        >>> # Accumulate and deduplicate
        >>> all_articles = accumulate_articles_from_tool_results([result1, result2])
        >>> # Returns: Article A, B, C, D, E (A only once!)
        >>>
        >>> # Use with LLM - no duplicate content in context
        >>> answer = llm_manager.generate(question, all_articles)
        >>> final = format_with_citations(answer, all_articles)
    """
    accumulated = []
    seen_kb_ids = set()

    for tool_result in tool_results:
        articles = parse_tool_result_to_articles(tool_result)

        for article in articles:
            # Deduplicate by kb_id (unique article identifier)
            if article.kb_id and article.kb_id not in seen_kb_ids:
                accumulated.append(article)
                seen_kb_ids.add(article.kb_id)
            elif not article.kb_id:
                # If no kb_id, preserve the article (rare edge case)
                accumulated.append(article)

    total_articles = sum(len(parse_tool_result_to_articles(r)) for r in tool_results)
    duplicates_removed = total_articles - len(accumulated)

    logger.info(
        "Accumulated %d unique articles from %d tool call(s) (removed %d duplicates)",
        len(accumulated),
        len(tool_results),
        duplicates_removed,
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

