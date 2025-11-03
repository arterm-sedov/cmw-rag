"""Tests for RAG context retrieval LangChain tool."""
from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from rag_engine.retrieval.retriever import Article, RAGRetriever
from rag_engine.tools.retrieve_context import (
    RetrieveContextSchema,
    _format_articles_to_json,
    _get_or_create_retriever,
    retrieve_context,
)


class TestRetrieveContextSchema:
    """Tests for RetrieveContextSchema validation."""

    def test_valid_query(self):
        """Test valid query passes validation."""
        schema = RetrieveContextSchema(query="test query")
        assert schema.query == "test query"
        assert schema.top_k is None

    def test_query_with_whitespace(self):
        """Test query whitespace is stripped."""
        schema = RetrieveContextSchema(query="  test query  ")
        assert schema.query == "test query"

    def test_empty_query_raises_error(self):
        """Test empty query raises validation error."""
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            RetrieveContextSchema(query="")

    def test_whitespace_only_query_raises_error(self):
        """Test whitespace-only query raises validation error."""
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            RetrieveContextSchema(query="   ")

    def test_valid_top_k(self):
        """Test valid top_k passes validation."""
        schema = RetrieveContextSchema(query="test", top_k=5)
        assert schema.top_k == 5

    def test_zero_top_k_raises_error(self):
        """Test zero top_k raises validation error."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            RetrieveContextSchema(query="test", top_k=0)

    def test_negative_top_k_raises_error(self):
        """Test negative top_k raises validation error."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            RetrieveContextSchema(query="test", top_k=-1)


class TestFormatArticlesToJson:
    """Tests for _format_articles_to_json helper function."""

    def test_format_single_article(self):
        """Test formatting single article."""
        article = Article(
            kb_id="123",
            content="Test content",
            metadata={"title": "Test Article", "article_url": "https://example.com/123"},
        )
        result_str = _format_articles_to_json([article], "test query", None)
        result = json.loads(result_str)

        assert result["articles"] == [
            {
                "kb_id": "123",
                "title": "Test Article",
                "url": "https://example.com/123",
                "content": "Test content",
                "metadata": {"title": "Test Article", "article_url": "https://example.com/123"},
            }
        ]
        assert result["metadata"]["query"] == "test query"
        assert result["metadata"]["top_k_requested"] is None
        assert result["metadata"]["articles_count"] == 1
        assert result["metadata"]["has_results"] is True

    def test_format_multiple_articles(self):
        """Test formatting multiple articles."""
        articles = [
            Article(kb_id="1", content="Content 1", metadata={"title": "Article 1"}),
            Article(kb_id="2", content="Content 2", metadata={"title": "Article 2"}),
        ]
        result_str = _format_articles_to_json(articles, "query", 10)
        result = json.loads(result_str)

        assert len(result["articles"]) == 2
        assert result["metadata"]["articles_count"] == 2
        assert result["metadata"]["top_k_requested"] == 10
        assert result["metadata"]["has_results"] is True

    def test_format_empty_articles(self):
        """Test formatting empty article list."""
        result_str = _format_articles_to_json([], "query", None)
        result = json.loads(result_str)

        assert result["articles"] == []
        assert result["metadata"]["articles_count"] == 0
        assert result["metadata"]["has_results"] is False

    def test_format_url_fallback(self):
        """Test URL fallback when article_url not in metadata."""
        article = Article(kb_id="456", content="Content", metadata={"title": "Test"})
        result_str = _format_articles_to_json([article], "query", None)
        result = json.loads(result_str)

        assert result["articles"][0]["url"] == "https://kb.comindware.ru/article.php?id=456"

    def test_format_title_fallback(self):
        """Test title fallback when title not in metadata."""
        article = Article(kb_id="789", content="Content", metadata={})
        result_str = _format_articles_to_json([article], "query", None)
        result = json.loads(result_str)

        assert result["articles"][0]["title"] == "789"


class TestRetrieveContextTool:
    """Tests for retrieve_context tool function."""

    @patch("rag_engine.tools.retrieve_context._get_or_create_retriever")
    def test_retrieve_success(self, mock_get_retriever):
        """Test successful retrieval."""
        # Setup mock retriever
        mock_ret = Mock(spec=RAGRetriever)
        mock_articles = [
            Article(
                kb_id="1",
                content="Content 1",
                metadata={"title": "Article 1", "article_url": "https://example.com/1"},
            )
        ]
        mock_ret.retrieve.return_value = mock_articles
        mock_get_retriever.return_value = mock_ret

        # Invoke tool
        result_str = retrieve_context.invoke({"query": "test query", "top_k": 5})
        result = json.loads(result_str)

        # Verify retriever was called correctly (no reserved_tokens parameter)
        mock_ret.retrieve.assert_called_once_with("test query", top_k=5)

        # Verify result structure
        assert len(result["articles"]) == 1
        assert result["articles"][0]["kb_id"] == "1"
        assert result["articles"][0]["title"] == "Article 1"
        assert result["metadata"]["query"] == "test query"
        assert result["metadata"]["top_k_requested"] == 5
        assert result["metadata"]["has_results"] is True

    @patch("rag_engine.tools.retrieve_context._get_or_create_retriever")
    def test_retrieve_no_results(self, mock_get_retriever):
        """Test retrieval with no results."""
        mock_ret = Mock(spec=RAGRetriever)
        mock_ret.retrieve.return_value = []
        mock_get_retriever.return_value = mock_ret

        result_str = retrieve_context.invoke({"query": "no results query"})
        result = json.loads(result_str)

        assert result["articles"] == []
        assert result["metadata"]["has_results"] is False
        assert result["metadata"]["articles_count"] == 0

    @patch("rag_engine.tools.retrieve_context._get_or_create_retriever")
    def test_retrieve_error_handling(self, mock_get_retriever):
        """Test error handling during retrieval."""
        mock_ret = Mock(spec=RAGRetriever)
        mock_ret.retrieve.side_effect = Exception("Database connection failed")
        mock_get_retriever.return_value = mock_ret

        result_str = retrieve_context.invoke({"query": "error query"})
        result = json.loads(result_str)

        assert "error" in result
        assert "Retrieval failed" in result["error"]
        assert result["articles"] == []
        assert result["metadata"]["has_results"] is False

    @patch("rag_engine.tools.retrieve_context._get_or_create_retriever")
    def test_retrieve_same_as_direct_call(self, mock_get_retriever):
        """Test tool returns same articles as direct retriever.retrieve() call."""
        # Setup mock retriever with specific articles
        mock_ret = Mock(spec=RAGRetriever)
        mock_articles = [
            Article(
                kb_id="1",
                content="Full content 1",
                metadata={"title": "Article 1", "kbId": "1", "article_url": "https://example.com/1"},
            ),
            Article(
                kb_id="2",
                content="Full content 2",
                metadata={"title": "Article 2", "kbId": "2", "article_url": "https://example.com/2"},
            ),
        ]
        mock_ret.retrieve.return_value = mock_articles
        mock_get_retriever.return_value = mock_ret

        # Direct call
        direct_results = mock_ret.retrieve("test query", top_k=5)

        # Tool call
        tool_result_str = retrieve_context.invoke({"query": "test query", "top_k": 5})
        tool_result = json.loads(tool_result_str)

        # Verify same articles are returned (same count and kb_ids)
        assert len(tool_result["articles"]) == len(direct_results) == 2
        tool_kb_ids = {a["kb_id"] for a in tool_result["articles"]}
        direct_kb_ids = {a.kb_id for a in direct_results}
        assert tool_kb_ids == direct_kb_ids

        # Verify content matches
        for tool_article, direct_article in zip(tool_result["articles"], direct_results, strict=True):
            assert tool_article["kb_id"] == direct_article.kb_id
            assert tool_article["content"] == direct_article.content
            assert tool_article["title"] == direct_article.metadata.get("title", direct_article.kb_id)


