"""Web search tool using Tavily.

Provides reusable web search functionality for any endpoint or agent.
"""

import json
import logging
import os

try:
    from langchain_tavily import TavilySearch

    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

logger = logging.getLogger(__name__)


class WebSearchResult:
    """Structured web search result."""

    def __init__(self, title: str, url: str, content: str):
        self.title = title
        self.url = url
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "content": self.content}


class WebSearchTool:
    """Reusable web search tool with configurable providers."""

    def __init__(self, provider: str = "tavily", max_results: int = 3):
        self.provider = provider
        self.max_results = max_results

    def search(self, query: str) -> list[WebSearchResult]:
        """Execute web search and return results."""
        if self.provider == "tavily":
            return self._search_tavily(query)
        logger.warning(f"Unknown search provider: {self.provider}")
        return []

    def _search_tavily(self, query: str) -> list[WebSearchResult]:
        """Search using Tavily API."""
        if not TAVILY_AVAILABLE:
            logger.debug("Tavily not available")
            return []
        if not os.getenv("TAVILY_API_KEY"):
            logger.debug("TAVILY_API_KEY not set")
            return []

        try:
            search = TavilySearch(max_results=self.max_results)
            result = search.invoke(query)

            if isinstance(result, dict):
                results = result.get("results", [])
            elif isinstance(result, list):
                results = result
            else:
                return []

            return [
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", "")[:500],
                )
                for item in results[: self.max_results]
                if isinstance(item, dict)
            ]
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

    def search_multiple(self, queries: list[str]) -> list[WebSearchResult]:
        """Execute multiple searches and merge results."""
        all_results = []
        for query in queries:
            all_results.extend(self.search(query))
        return all_results

    def to_json(self, results: list[WebSearchResult]) -> str:
        """Serialize results to JSON."""
        return json.dumps(
            {
                "type": "tool_response",
                "tool_name": "web_search",
                "results": [r.to_dict() for r in results],
            }
        )


_web_search_tool: WebSearchTool | None = None


def get_web_search_tool() -> WebSearchTool:
    """Get or create singleton web search tool."""
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool(max_results=3)
    return _web_search_tool


def web_search(query: str) -> str:
    """LangChain-compatible web search tool function.

    Args:
        query: Search query string

    Returns:
        JSON string with search results or error
    """
    tool = get_web_search_tool()
    results = tool.search(query)

    if not results:
        return json.dumps(
            {
                "type": "tool_response",
                "tool_name": "web_search",
                "error": "No results found or search unavailable",
            }
        )

    return tool.to_json(results)
