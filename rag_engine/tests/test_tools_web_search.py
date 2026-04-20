"""Tests for web search tool."""
from __future__ import annotations

import json

import pytest

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
        result = web_search("test query")
        data = json.loads(result)
        assert data["type"] == "tool_response"
        assert data["tool_name"] == "web_search"
        assert "error" in data
