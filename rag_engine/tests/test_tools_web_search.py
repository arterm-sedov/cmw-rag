"""Tests for web search tool."""
from __future__ import annotations

import json

from rag_engine.tools.web_search import (
    WebSearchResult,
    WebSearchTool,
    get_web_search_tool,
    web_search,
)


class TestWebSearchResult:
    """Tests for WebSearchResult class."""

    def test_to_dict(self):
        """Test serialization to dict."""
        result = WebSearchResult(title="Test", url="https://example.com", content="Test content")
        d = result.to_dict()
        assert d["title"] == "Test"
        assert d["url"] == "https://example.com"
        assert d["content"] == "Test content"


class TestWebSearchTool:
    """Tests for WebSearchTool class."""

    def test_init_default_params(self):
        """Test default initialization."""
        tool = WebSearchTool()
        assert tool.provider == "tavily"
        assert tool.max_results == 3

    def test_init_custom_params(self):
        """Test custom initialization."""
        tool = WebSearchTool(provider="tavily", max_results=5)
        assert tool.provider == "tavily"
        assert tool.max_results == 5

    def test_search_unknown_provider(self, monkeypatch):
        """Test search with unknown provider returns empty."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tool = WebSearchTool(provider="unknown")
        results = tool.search("test query")
        assert results == []

    def test_search_tavily_no_api_key(self, monkeypatch):
        """Test Tavily search without API key returns empty."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tool = WebSearchTool(provider="tavily")
        results = tool.search("test query")
        assert results == []

    def test_to_json(self):
        """Test JSON serialization."""
        tool = WebSearchTool()
        results = [WebSearchResult(title="T", url="https://x.com", content="C")]
        json_str = tool.to_json(results)
        data = json.loads(json_str)
        assert data["type"] == "tool_response"
        assert data["tool_name"] == "web_search"
        assert len(data["results"]) == 1


class TestWebSearchFunctions:
    """Tests for module-level functions."""

    def test_get_web_search_tool_singleton(self):
        """Test singleton returns same instance."""
        t1 = get_web_search_tool()
        t2 = get_web_search_tool()
        assert t1 is t2

    def test_web_search_no_api_key(self, monkeypatch):
        """Test web_search function without API key returns error."""

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = web_search.invoke("test query")
        data = json.loads(result)
        assert data["type"] == "tool_response"
        assert data["tool_name"] == "web_search"
        assert "error" in data


class TestWebSearchToolDecorator:
    """Tests for LangChain @tool decorator."""

    def test_web_search_is_langchain_tool(self):
        """Test that web_search is a LangChain tool."""

        # Check it's a LangChain tool (has name attribute)
        assert hasattr(web_search, "name")
        assert web_search.name == "web_search"

    def test_web_search_has_description(self):
        """Test tool has description."""

        assert hasattr(web_search, "description")
        assert isinstance(web_search.description, str)
        assert len(web_search.description) > 0

    def test_web_search_args_schema(self):
        """Test tool has correct input schema."""
        from rag_engine.tools.web_search import WebSearchInput

        # Check schema has query field
        assert "query" in WebSearchInput.model_fields

    def test_web_search_no_api_key_returns_error(self, monkeypatch):
        """Test tool returns error when no API key."""

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = web_search.invoke("test query")
        data = json.loads(result)
        assert "error" in data
