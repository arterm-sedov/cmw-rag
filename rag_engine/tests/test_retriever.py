"""Tests for RAG retriever with hybrid approach."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from rag_engine.retrieval.retriever import Article, RAGRetriever


class TestArticle:
    """Tests for Article class."""

    def test_article_initialization(self):
        article = Article(
            kb_id="test_article",
            content="Full article content here",
            metadata={"title": "Test", "source_file": "/path/to/file.md"},
        )
        assert article.kb_id == "test_article"
        assert article.content == "Full article content here"
        assert article.metadata["title"] == "Test"
        assert article.matched_chunks == []


class TestRAGRetriever:
    """Tests for RAG retriever with complete article loading."""

    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager with dynamic token limits."""
        manager = Mock()
        manager.get_current_llm_context_window.return_value = 100000  # 100K context
        return manager

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedder."""
        embedder = Mock()
        embedder.embed_query.return_value = [0.1] * 768
        embedder.embed_documents.return_value = [[0.1] * 768]
        return embedder

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        store = Mock()
        return store

    @pytest.fixture
    def retriever(self, mock_embedder, mock_vector_store, mock_llm_manager):
        """Create retriever with mocks."""
        return RAGRetriever(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            llm_manager=mock_llm_manager,
            top_k_retrieve=20,
            top_k_rerank=10,
            rerank_enabled=False,  # Disable for simpler tests
        )

    def test_initialization(self, retriever, mock_llm_manager):
        """Test retriever initializes correctly."""
        assert retriever.llm_manager == mock_llm_manager
        assert retriever.top_k_retrieve == 20
        assert retriever.top_k_rerank == 10
        assert retriever.reranker is None  # Disabled

    def test_read_article_success(self, retriever, tmp_path):
        """Test reading article from file."""
        # Create test file
        test_file = tmp_path / "test_article.md"
        test_file.write_text("---\ntitle: Test\n---\n\nArticle content here")

        # Read article
        content = retriever._read_article(str(test_file))

        assert content == "Article content here"
        assert "---" not in content  # Frontmatter removed

    def test_read_article_no_frontmatter(self, retriever, tmp_path):
        """Test reading article without frontmatter."""
        test_file = tmp_path / "simple.md"
        test_file.write_text("Just content")

        content = retriever._read_article(str(test_file))

        assert content == "Just content"

    def test_read_article_file_not_found(self, retriever):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            retriever._read_article("/nonexistent/file.md")

    @patch("rag_engine.retrieval.retriever.top_k_search")
    def test_retrieve_no_results(self, mock_search, retriever):
        """Test retrieve returns empty list when no chunks found."""
        mock_search.return_value = []

        articles = retriever.retrieve("test query")

        assert articles == []

    @patch("rag_engine.retrieval.retriever.top_k_search")
    def test_retrieve_returns_articles(self, mock_search, retriever, tmp_path):
        """Test retrieve returns Article objects (not chunks)."""
        # Create test file
        test_file = tmp_path / "article.md"
        test_file.write_text("Complete article content")

        # Mock chunk results
        mock_chunk = Mock()
        mock_chunk.metadata = {
            "kbId": "test_article",
            "source_file": str(test_file),
            "title": "Test Article",
        }
        mock_search.return_value = [mock_chunk]

        # Retrieve
        articles = retriever.retrieve("test query")

        # Verify
        assert len(articles) >= 1
        assert isinstance(articles[0], Article)
        assert articles[0].content == "Complete article content"
        assert articles[0].kb_id == "test_article"

    @patch("rag_engine.retrieval.retriever.top_k_search")
    def test_retrieve_groups_by_kbid(self, mock_search, retriever, tmp_path):
        """Test chunks from same article are grouped."""
        # Create test file
        test_file = tmp_path / "article.md"
        test_file.write_text("Article content")

        # Mock 3 chunks from same article
        chunks = []
        for i in range(3):
            chunk = Mock()
            chunk.metadata = {
                "kbId": "article_1",
                "source_file": str(test_file),
                "chunk_index": i,
            }
            chunks.append(chunk)

        mock_search.return_value = chunks

        # Retrieve
        articles = retriever.retrieve("test query")

        # Should return 1 article (not 3 chunks)
        assert len(articles) == 1
        assert isinstance(articles[0], Article)
        assert len(articles[0].matched_chunks) == 3

    def test_context_budgeting(self, retriever):
        """Test context budgeting respects token limits."""
        # Create articles with known sizes
        articles = [
            Article("article1", "x" * 10000, {}),  # ~2500 tokens
            Article("article2", "x" * 10000, {}),  # ~2500 tokens
            Article("article3", "x" * 10000, {}),  # ~2500 tokens
            Article("article4", "x" * 500000, {}),  # ~125K tokens (exceeds budget)
        ]

        # Context window is 100K, budget is 75K
        # Should fit articles 1, 2, 3 but not 4
        selected = retriever._apply_context_budget(articles)

        assert len(selected) <= 3  # Should not include article 4
        # Verify total is within budget
        total_tokens = sum(len(retriever._encoding.encode(a.content)) for a in selected)
        assert total_tokens <= 75000  # 75% of 100K

    def test_context_budgeting_logs_percentage(self, retriever, caplog):
        """Test context budgeting logs token usage percentage."""
        articles = [Article("test", "x" * 1000, {})]

        retriever._apply_context_budget(articles)

        # Check logs contain percentage
        assert "% of context window" in caplog.text

    @patch("rag_engine.retrieval.retriever.top_k_search")
    def test_retrieve_multiple_articles(self, mock_search, retriever, tmp_path):
        """Test retrieving multiple articles from different sources."""
        # Create 2 test files
        file1 = tmp_path / "article1.md"
        file1.write_text("Article 1 content")
        file2 = tmp_path / "article2.md"
        file2.write_text("Article 2 content")

        # Mock chunks from different articles
        chunk1 = Mock()
        chunk1.metadata = {
            "kbId": "article_1",
            "source_file": str(file1),
        }
        chunk2 = Mock()
        chunk2.metadata = {
            "kbId": "article_2",
            "source_file": str(file2),
        }

        mock_search.return_value = [chunk1, chunk2]

        # Retrieve
        articles = retriever.retrieve("test query")

        # Should have 2 distinct articles
        assert len(articles) == 2
        kb_ids = {a.kb_id for a in articles}
        assert kb_ids == {"article_1", "article_2"}


class TestIntegration:
    """Integration tests with real components."""

    @pytest.mark.skip(reason="Requires FRIDA model download")
    def test_end_to_end_with_real_embedder(self):
        """Test with real embedder (requires model)."""
        # This would test with actual FRIDA embedder
        # Skip in CI/CD unless models are cached
        pass

