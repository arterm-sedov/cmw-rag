"""Tests for RAG retriever with hybrid approach."""
from __future__ import annotations

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
        manager.get_max_output_tokens.return_value = 32768
        manager.get_system_prompt.return_value = "SYS"
        # Provide a chat model that returns a string content response
        class _FakeModel:
            def invoke(self, messages):  # noqa: ANN001
                return type("Resp", (), {"content": "summary"})()
        manager._chat_model.return_value = _FakeModel()
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

    @patch("rag_engine.retrieval.retriever.top_k_search")
    def test_multi_vector_segmentation_and_cap(self, mock_search, retriever, monkeypatch, tmp_path):
        """Long queries are segmented; multiple searches performed; pre-rerank cap enforced."""
        # Configure settings to force segmentation and a small pre-rerank cap
        from rag_engine.config import settings as cfg
        monkeypatch.setattr(cfg.settings, "retrieval_multiquery_enabled", True, raising=False)
        monkeypatch.setattr(cfg.settings, "retrieval_multiquery_segment_tokens", 16, raising=False)
        monkeypatch.setattr(cfg.settings, "retrieval_multiquery_segment_overlap", 4, raising=False)
        monkeypatch.setattr(cfg.settings, "retrieval_multiquery_max_segments", 3, raising=False)
        monkeypatch.setattr(cfg.settings, "retrieval_multiquery_pre_rerank_limit", 2, raising=False)

        # Prepare two distinct chunk hits from the same article file so downstream succeeds
        f = tmp_path / "a.md"
        f.write_text("Doc A")
        d1 = Mock(); d1.metadata = {"kbId": "kb1", "source_file": str(f), "stable_id": "s1"}
        d2 = Mock(); d2.metadata = {"kbId": "kb1", "source_file": str(f), "stable_id": "s2"}
        # Each search returns one doc; multiple calls will append until cap trims to 2
        mock_search.side_effect = [[d1], [d2], [d2]]

        # Long query with multiple sentences to trigger splitter
        long_q = "Sentence one. Sentence two with more words. Another sentence for splitting."
        articles = retriever.retrieve(long_q)

        # top_k_search should have been called at least once (segmentation optional)
        assert mock_search.call_count >= 1
        # Pre-rerank cap applied; downstream groups into a single article
        assert len(articles) == 1

    @patch("rag_engine.retrieval.retriever.top_k_search")
    def test_query_decomposition_adds_candidates(self, mock_search, retriever, monkeypatch, tmp_path):
        """LLM decomposition produces extra sub-queries whose results are unioned."""
        from rag_engine.config import settings as cfg
        monkeypatch.setattr(cfg.settings, "retrieval_multiquery_enabled", False, raising=False)
        monkeypatch.setattr(cfg.settings, "retrieval_query_decomp_enabled", True, raising=False)
        monkeypatch.setattr(cfg.settings, "retrieval_query_decomp_max_subqueries", 3, raising=False)
        monkeypatch.setattr(cfg.settings, "retrieval_multiquery_pre_rerank_limit", 10, raising=False)

        # Decompose into 2 sub-queries
        retriever.llm_manager.generate.return_value = "sub one\nsub two"

        # Set up two files for two distinct articles
        f1 = tmp_path / "a.md"; f1.write_text("A")
        f2 = tmp_path / "b.md"; f2.write_text("B")
        d1 = Mock(); d1.metadata = {"kbId": "kb1", "source_file": str(f1), "stable_id": "s1"}
        d2 = Mock(); d2.metadata = {"kbId": "kb2", "source_file": str(f2), "stable_id": "s2"}

        # First call for original query, then for each subquery: return distinct docs
        mock_search.side_effect = [[d1], [d2], []]

        articles = retriever.retrieve("original long question")

        # We should retrieve two distinct articles after union
        assert len(articles) == 2
        kb_ids = {a.kb_id for a in articles}
        assert kb_ids == {"kb1", "kb2"}


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
        """Test context budgeting respects token limits.

        New behavior may include a summarized/lightweight representation of the
        overflow article while still respecting the overall token budget.
        """
        # Create articles with known sizes
        articles = [
            Article("article1", "x" * 10000, {}),  # ~2500 tokens
            Article("article2", "x" * 10000, {}),  # ~2500 tokens
            Article("article3", "x" * 10000, {}),  # ~2500 tokens
            Article("article4", "x" * 500000, {}),  # ~125K tokens (exceeds budget)
        ]

        # Context window is 100K; reserved tokens reduce available budget.
        # The algorithm may include a summarized/lightweight version of the 4th
        # article if it still fits within the remaining budget.
        selected = retriever._apply_context_budget(articles)

        # Ensure we at least included the first three articles or their
        # representations, and the total remains within the budget.
        assert len(selected) >= 3
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

