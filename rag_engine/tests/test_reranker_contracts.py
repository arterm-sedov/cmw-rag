"""Test vLLM/Cohere contract compliance for reranker endpoints.

These tests verify the endpoint contracts defined in test_rerankers.yaml:
- /v1/score returns vLLM format: {data: [{index, object, score}, ...]}
- /v1/rerank returns Cohere format: {results: [{index, document, relevance_score}, ...]}

Tests use mocks to verify behavior, not implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_engine.config.schemas import RerankerFormatting, ServerRerankerConfig
from rag_engine.retrieval.reranker import RerankerAdapter


def dity_config() -> ServerRerankerConfig:
    """Create a DiTy cross-encoder config for testing."""
    return ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/rerank",
        reranker_type="cross_encoder",
    )


def qwen3_config() -> ServerRerankerConfig:
    """Create a Qwen3 LLM reranker config for testing."""
    return ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/rerank",
        reranker_type="llm_reranker",
        default_instruction="Given a web search query, retrieve relevant passages",
        formatting=RerankerFormatting(
            query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
            doc_template="<Document>: {doc}{suffix}",
            prefix="<|im_start|>system\nJudge...<|im_end|>\n<|im_start|>user\n",
            suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
        ),
    )


class MockDocument:
    """Mock document for testing."""

    def __init__(self, content: str, metadata: dict | None = None):
        self.page_content = content
        self.metadata = metadata or {}


class TestScoreEndpointContract:
    """Test /v1/score endpoint contract (vLLM format)."""

    def test_returns_data_array(self):
        """Response has 'data' key with array of scores."""
        config = dity_config()
        adapter = RerankerAdapter(config)

        with patch.object(adapter, "session") as mock_session:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "data": [
                    {"index": 0, "object": "score", "score": 0.95},
                    {"index": 1, "object": "score", "score": 0.05},
                ]
            }
            mock_session.post.return_value = mock_response

            scores = adapter.score("query", ["doc1", "doc2"])

            assert isinstance(scores, list)
            assert len(scores) == 2
            assert scores[0] == 0.95
            assert scores[1] == 0.05

    def test_preserves_original_order(self):
        """Scores returned in original document order (by index)."""
        config = dity_config()
        adapter = RerankerAdapter(config)

        with patch.object(adapter, "session") as mock_session:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "data": [
                    {"index": 1, "object": "score", "score": 0.9},
                    {"index": 0, "object": "score", "score": 0.1},
                ]
            }
            mock_session.post.return_value = mock_response

            scores = adapter.score("query", ["doc1", "doc2"])

            assert scores == [0.1, 0.9]

    def test_score_values_are_floats(self):
        """All score values are floats."""
        config = dity_config()
        adapter = RerankerAdapter(config)

        with patch.object(adapter, "session") as mock_session:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "data": [
                    {"index": 0, "object": "score", "score": 0.88},
                    {"index": 1, "object": "score", "score": 0.12},
                ]
            }
            mock_session.post.return_value = mock_response

            scores = adapter.score("query", ["doc1", "doc2"])

            assert all(isinstance(s, float) for s in scores)


class TestRerankEndpointContract:
    """Test /v1/rerank endpoint contract (Cohere format)."""

    def test_returns_results_array(self):
        """Response has 'results' key with array of ranked items."""
        config = dity_config()
        adapter = RerankerAdapter(config)

        with patch.object(adapter, "session") as mock_session:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "results": [
                    {"index": 1, "document": {"text": "doc2"}, "relevance_score": 0.95},
                    {"index": 0, "document": {"text": "doc1"}, "relevance_score": 0.05},
                ]
            }
            mock_session.post.return_value = mock_response

            candidates = [(MockDocument("doc1"), 0.0), (MockDocument("doc2"), 0.0)]
            results = adapter.rerank("query", candidates, top_k=2)

            assert len(results) == 2
            assert results[0][1] == 0.95

    def test_sorted_by_relevance(self):
        """Results sorted by relevance_score descending."""
        config = dity_config()
        adapter = RerankerAdapter(config)

        with patch.object(adapter, "session") as mock_session:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "results": [
                    {"index": 1, "document": {"text": "doc2"}, "relevance_score": 0.95},
                    {"index": 0, "document": {"text": "doc1"}, "relevance_score": 0.05},
                ]
            }
            mock_session.post.return_value = mock_response

            candidates = [(MockDocument("doc1"), 0.0), (MockDocument("doc2"), 0.0)]
            results = adapter.rerank("query", candidates, top_k=2)

            scores = [r[1] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_includes_document_text(self):
        """Each result maps back to original document."""
        config = dity_config()
        adapter = RerankerAdapter(config)

        with patch.object(adapter, "session") as mock_session:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "results": [
                    {"index": 1, "document": {"text": "doc2"}, "relevance_score": 0.95},
                    {"index": 0, "document": {"text": "doc1"}, "relevance_score": 0.05},
                ]
            }
            mock_session.post.return_value = mock_response

            candidates = [(MockDocument("doc1"), 0.0), (MockDocument("doc2"), 0.0)]
            results = adapter.rerank("query", candidates, top_k=2)

            assert results[0][0].page_content == "doc2"
            assert results[1][0].page_content == "doc1"

    def test_top_n_limits_results(self):
        """top_n parameter limits number of results."""
        config = dity_config()
        adapter = RerankerAdapter(config)

        with patch.object(adapter, "session") as mock_session:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "results": [
                    {"index": 0, "document": {"text": "doc1"}, "relevance_score": 0.95},
                ]
            }
            mock_session.post.return_value = mock_response

            candidates = [
                (MockDocument("doc1"), 0.0),
                (MockDocument("doc2"), 0.0),
                (MockDocument("doc3"), 0.0),
            ]
            results = adapter.rerank("query", candidates, top_k=1)

            assert len(results) == 1


class TestCrossEncoderFormatting:
    """Test cross-encoder behavior (no formatting needed)."""

    def test_no_query_formatting(self):
        """Cross-encoder passes raw query unchanged."""
        config = dity_config()
        adapter = RerankerAdapter(config)

        formatted = adapter.format_query("What is Python?", instruction="search")

        assert formatted == "What is Python?"

    def test_no_document_formatting(self):
        """Cross-encoder passes raw documents unchanged."""
        config = dity_config()
        adapter = RerankerAdapter(config)

        formatted = adapter.format_document("Python is a programming language")

        assert formatted == "Python is a programming language"

    def test_instruction_ignored_with_warning(self, caplog):
        """Cross-encoder logs warning if instruction provided."""
        config = dity_config()
        adapter = RerankerAdapter(config)

        with caplog.at_level("WARNING"):
            adapter.format_query("query", instruction="custom")

        assert "doesn't support instructions" in caplog.text


class TestLLMRerankerFormatting:
    """Test LLM reranker behavior (formatting required)."""

    def test_query_has_prefix_and_instruction(self):
        """LLM reranker query includes prefix and instruction."""
        config = qwen3_config()
        adapter = RerankerAdapter(config)

        formatted = adapter.format_query("What is AI?", instruction="search for AI info")

        assert "<|im_start|>system" in formatted
        assert "<Instruct>:" in formatted
        assert "search for AI info" in formatted
        assert "<Query>: What is AI?" in formatted

    def test_document_has_suffix(self):
        """LLM reranker document includes suffix."""
        config = qwen3_config()
        adapter = RerankerAdapter(config)

        formatted = adapter.format_document("Python is a language")

        assert "<Document>: Python is a language" in formatted
        assert "<|im_end|>" in formatted
        assert "<|im_start|>assistant" in formatted

    def test_uses_default_instruction_if_none_provided(self):
        """LLM reranker uses default instruction when none provided."""
        config = qwen3_config()
        adapter = RerankerAdapter(config)

        formatted = adapter.format_query("What is AI?")

        assert "Given a web search query" in formatted


class TestQwen3Formatting:
    """Test Qwen3-specific formatting."""

    def test_qwen3_query_format(self):
        """Qwen3 query format includes ChatML markers."""
        config = qwen3_config()
        adapter = RerankerAdapter(config)

        formatted = adapter.format_query("What is Python?")

        assert "<|im_start|>system" in formatted
        assert "<Instruct>:" in formatted
        assert "<Query>: What is Python?" in formatted

    def test_qwen3_document_format(self):
        """Qwen3 document format includes Document marker."""
        config = qwen3_config()
        adapter = RerankerAdapter(config)

        formatted = adapter.format_document("Python is a language")

        assert "<Document>: Python is a language" in formatted


class TestBgeGemmaFormatting:
    """Test BGE-Gemma-specific formatting."""

    def test_bge_gemma_query_format(self):
        """BGE-Gemma query format is 'A: {query}'."""
        config = ServerRerankerConfig(
            type="server",
            provider="mosec",
            endpoint="http://localhost:7998/v1/rerank",
            reranker_type="llm_reranker",
            formatting=RerankerFormatting(
                query_template="A: {query}",
                doc_template="B: {doc}\n{prompt}",
                prefix="",
                suffix="",
            ),
        )
        adapter = RerankerAdapter(config)

        formatted = adapter.format_query("What is Python?")

        assert formatted == "A: What is Python?"

    def test_bge_gemma_document_format(self):
        """BGE-Gemma document format includes prompt."""
        config = ServerRerankerConfig(
            type="server",
            provider="mosec",
            endpoint="http://localhost:7998/v1/rerank",
            reranker_type="llm_reranker",
            formatting=RerankerFormatting(
                query_template="A: {query}",
                doc_template="B: {doc}\n{prompt}",
                prefix="",
                suffix="",
            ),
        )
        adapter = RerankerAdapter(config)

        formatted = adapter.format_document("Python is a language")

        assert formatted.startswith("B: Python is a language")
