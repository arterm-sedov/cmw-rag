"""Tests for reranker factory and implementations."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

# Load .env file from project root if it exists
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from rag_engine.config.schemas import DirectRerankerConfig, ServerRerankerConfig
from rag_engine.retrieval.reranker import (
    CrossEncoderReranker,
    IdentityReranker,
    InfinityReranker,
    create_reranker,
)


class MockDocument:
    """Mock document for testing."""

    def __init__(self, content: str, metadata: dict | None = None):
        self.page_content = content
        self.metadata = metadata or {}


class TestCrossEncoderReranker:
    """Tests for direct CrossEncoder reranker (current implementation)."""

    def test_crossencoder_reranker_creation(self):
        """Test CrossEncoder reranker can be created."""
        pytest.importorskip("sentence_transformers")

        config = DirectRerankerConfig(
            type="direct",
            model="DiTy/cross-encoder-russian-msmarco",
            device="cpu",
            batch_size=16,
        )

        # Should not raise
        reranker = CrossEncoderReranker(
            model_name=config.model,
            batch_size=config.batch_size,
            device=config.device,
        )

        assert reranker is not None
        assert reranker.batch_size == 16

    @patch("rag_engine.retrieval.reranker.CrossEncoder")
    def test_crossencoder_rerank_basic(self, mock_crossencoder_class):
        """Test basic reranking functionality."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.5, 0.7]  # Scores
        mock_crossencoder_class.return_value = mock_model

        reranker = CrossEncoderReranker(
            model_name="DiTy/cross-encoder-russian-msmarco",
            batch_size=16,
            device="cpu",
        )

        # Create mock candidates
        candidates = [
            (MockDocument("Doc 1"), 0.0),
            (MockDocument("Doc 2"), 0.0),
            (MockDocument("Doc 3"), 0.0),
        ]

        result = reranker.rerank("query", candidates, top_k=2)

        # Verify sorted by score (0.9 first, 0.7 second)
        assert len(result) == 2
        assert result[0][1] == 0.9
        assert result[1][1] == 0.7

    @patch("rag_engine.retrieval.reranker.CrossEncoder")
    def test_crossencoder_instruction_warning(self, mock_crossencoder_class, caplog):
        """Test CrossEncoder ignores instructions with warning."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9]
        mock_crossencoder_class.return_value = mock_model

        reranker = CrossEncoderReranker(
            model_name="DiTy/cross-encoder-russian-msmarco",
            batch_size=16,
            device="cpu",
        )

        candidates = [(MockDocument("Doc 1"), 0.0)]

        with caplog.at_level("WARNING"):
            reranker.rerank("query", candidates, top_k=1, instruction="custom instruction")

        assert "doesn't support dynamic instructions" in caplog.text

    @patch("rag_engine.retrieval.reranker.CrossEncoder")
    def test_crossencoder_metadata_boost(self, mock_crossencoder_class):
        """Test metadata boost weights are applied."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]  # Base score
        mock_crossencoder_class.return_value = mock_model

        reranker = CrossEncoderReranker(
            model_name="DiTy/cross-encoder-russian-msmarco",
            batch_size=16,
            device="cpu",
        )

        # Create doc with tags
        doc = MockDocument("Doc with tags", metadata={"tags": ["important"]})
        candidates = [(doc, 0.0)]

        boost_weights = {"tag_match": 0.2}
        result = reranker.rerank("query", candidates, top_k=1, metadata_boost_weights=boost_weights)

        # Score should be boosted: 0.8 * (1 + 0.2) = 0.96
        assert result[0][1] == pytest.approx(0.96, 0.01)


class TestInfinityReranker:
    """Tests for Infinity server reranker."""

    @patch("rag_engine.retrieval.reranker.HTTPClientMixin._post")
    def test_infinity_dity_rerank(self, mock_post):
        """Test Infinity DiTy reranking (no instruction)."""
        mock_post.return_value = {"scores": [0.9, 0.5, 0.7]}

        config = ServerRerankerConfig(
            type="server",
            endpoint="http://localhost:8002",
            # No default_instruction for DiTy
        )

        reranker = InfinityReranker(config)

        candidates = [
            (MockDocument("Doc 1"), 0.0),
            (MockDocument("Doc 2"), 0.0),
            (MockDocument("Doc 3"), 0.0),
        ]

        result = reranker.rerank("test query", candidates, top_k=2)

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "/rerank"

        # For DiTy, query should NOT have instruction prefix
        assert call_args[0][1]["query"] == "test query"
        assert "documents" in call_args[0][1]
        assert call_args[0][1]["top_k"] == 2

        # Verify sorted results
        assert len(result) == 2
        assert result[0][1] == 0.9
        assert result[1][1] == 0.7

    @patch("rag_engine.retrieval.reranker.HTTPClientMixin._post")
    def test_infinity_qwen3_rerank_with_instruction(self, mock_post):
        """Test Infinity Qwen3 reranking with instruction."""
        mock_post.return_value = {"scores": [0.95, 0.85]}

        config = ServerRerankerConfig(
            type="server",
            endpoint="http://localhost:8005",
            default_instruction="Given a web search query, retrieve relevant passages that answer the query",
        )

        reranker = InfinityReranker(config)

        candidates = [
            (MockDocument("Relevant doc"), 0.0),
            (MockDocument("Another doc"), 0.0),
        ]

        result = reranker.rerank("What is AI?", candidates, top_k=2)

        # For Qwen3, query should have instruction format
        call_args = mock_post.call_args
        query_text = call_args[0][1]["query"]
        assert "Instruct:" in query_text
        assert "Query:" in query_text
        assert "What is AI?" in query_text

        assert len(result) == 2

    @patch("rag_engine.retrieval.reranker.HTTPClientMixin._post")
    def test_infinity_rerank_metadata_boost(self, mock_post):
        """Test Infinity reranking with metadata boost."""
        mock_post.return_value = {"scores": [0.8]}

        config = ServerRerankerConfig(
            type="server",
            endpoint="http://localhost:8002",
        )

        reranker = InfinityReranker(config)

        doc = MockDocument("Code example", metadata={"has_code": True})
        candidates = [(doc, 0.0)]

        boost_weights = {"code_presence": 0.3}
        result = reranker.rerank("query", candidates, top_k=1, metadata_boost_weights=boost_weights)

        # Score should be boosted: 0.8 * (1 + 0.3) = 1.04
        assert result[0][1] == pytest.approx(1.04, 0.01)

    @patch("rag_engine.retrieval.reranker.HTTPClientMixin._post")
    def test_infinity_custom_instruction_override(self, mock_post, caplog):
        """Test Infinity accepts custom instruction override for Qwen3."""
        mock_post.return_value = {"scores": [0.9]}

        config = ServerRerankerConfig(
            type="server",
            endpoint="http://localhost:8005",
            default_instruction="Default instruction",
        )

        reranker = InfinityReranker(config)

        candidates = [(MockDocument("Doc"), 0.0)]
        custom_instruction = "Custom search task"

        reranker.rerank("query", candidates, top_k=1, instruction=custom_instruction)

        # Verify custom instruction was used
        call_args = mock_post.call_args
        query_text = call_args[0][1]["query"]
        assert custom_instruction in query_text


class TestIdentityReranker:
    """Tests for identity (pass-through) reranker."""

    def test_identity_rerank_pass_through(self):
        """Test identity reranker returns candidates unchanged."""
        reranker = IdentityReranker()

        candidates = [
            (MockDocument("Doc A"), 0.5),
            (MockDocument("Doc B"), 0.8),
            (MockDocument("Doc C"), 0.3),
        ]

        result = reranker.rerank("query", candidates, top_k=2)

        # Should return first 2 unchanged
        assert len(result) == 2
        assert result[0][0].page_content == "Doc A"
        assert result[0][1] == 0.5
        assert result[1][0].page_content == "Doc B"
        assert result[1][1] == 0.8

    def test_identity_instruction_warning(self, caplog):
        """Test identity reranker warns about instructions."""
        reranker = IdentityReranker()

        with caplog.at_level("WARNING"):
            reranker.rerank("query", [(MockDocument("Doc"), 0.0)], top_k=1, instruction="test")

        assert "doesn't support instructions" in caplog.text


class TestCreateRerankerFactory:
    """Tests for reranker factory function with model-slug-based configuration."""

    @patch("rag_engine.retrieval.reranker.ModelRegistry")
    def test_factory_direct_crossencoder(self, mock_registry_cls):
        """Test factory creates CrossEncoder for direct provider."""
        mock_registry = MagicMock()
        mock_registry.get_model.return_value = {
            "canonical_slug": "DiTy/cross-encoder-russian-msmarco",
            "type": "reranker",
        }
        mock_registry.get_provider_config.return_value = {
            "device": "cpu",
            "batch_size": 16,
        }
        mock_registry_cls.return_value = mock_registry

        settings = MagicMock()
        settings.reranker_provider_type = "direct"
        settings.reranker_model = "DiTy/cross-encoder-russian-msmarco"

        with patch("rag_engine.retrieval.reranker.CrossEncoderReranker") as mock_crossencoder:
            mock_instance = MagicMock()
            mock_crossencoder.return_value = mock_instance

            reranker = create_reranker(settings)

            mock_crossencoder.assert_called_once_with(
                model_name="DiTy/cross-encoder-russian-msmarco",
                batch_size=16,
                device="cpu",
            )
            assert reranker == mock_instance

    @patch("rag_engine.retrieval.reranker.ModelRegistry")
    def test_factory_infinity_dity(self, mock_registry_cls):
        """Test factory creates Infinity reranker for server provider."""
        mock_registry = MagicMock()
        mock_registry.get_model.return_value = {
            "canonical_slug": "DiTy/cross-encoder-russian-msmarco",
            "type": "reranker",
        }
        mock_registry.get_provider_config.return_value = {
            # DiTy doesn't use instructions
        }
        mock_registry_cls.return_value = mock_registry

        settings = MagicMock()
        settings.reranker_provider_type = "infinity"
        settings.reranker_model = "DiTy/cross-encoder-russian-msmarco"
        settings.infinity_reranker_endpoint = "http://localhost:7998"

        with patch("rag_engine.retrieval.reranker.InfinityReranker") as mock_infinity:
            mock_instance = MagicMock()
            mock_infinity.return_value = mock_instance

            reranker = create_reranker(settings)

            mock_infinity.assert_called_once()
            assert reranker == mock_instance

    @patch("rag_engine.retrieval.reranker.ModelRegistry")
    def test_factory_infinity_qwen3(self, mock_registry_cls):
        """Test factory creates Infinity reranker for Qwen3."""
        mock_registry = MagicMock()
        mock_registry.get_model.return_value = {
            "canonical_slug": "Qwen/Qwen3-Reranker-8B",
            "type": "reranker",
        }
        mock_registry.get_provider_config.return_value = {
            "default_instruction": "Given a web search query, retrieve relevant passages that answer the query",
        }
        mock_registry_cls.return_value = mock_registry

        settings = MagicMock()
        settings.reranker_provider_type = "infinity"
        settings.reranker_model = "Qwen/Qwen3-Reranker-8B"
        settings.infinity_reranker_endpoint = "http://localhost:7998"

        with patch("rag_engine.retrieval.reranker.InfinityReranker") as mock_infinity:
            mock_instance = MagicMock()
            mock_infinity.return_value = mock_instance

            reranker = create_reranker(settings)

            mock_infinity.assert_called_once()
            assert reranker == mock_instance

    @patch("rag_engine.retrieval.reranker.ModelRegistry")
    def test_factory_case_insensitive_model_slug(self, mock_registry_cls):
        """Test factory handles case-insensitive model slugs."""
        mock_registry = MagicMock()
        mock_registry.get_model.return_value = {
            "canonical_slug": "DiTy/cross-encoder-russian-msmarco",
            "type": "reranker",
        }
        mock_registry.get_provider_config.return_value = {
            "device": "cpu",
            "batch_size": 16,
        }
        mock_registry_cls.return_value = mock_registry

        # Use lowercase model slug
        settings = MagicMock()
        settings.reranker_provider_type = "direct"
        settings.reranker_model = "dity/cross-encoder-russian-msmarco"

        with patch("rag_engine.retrieval.reranker.CrossEncoderReranker") as mock_crossencoder:
            mock_instance = MagicMock()
            mock_crossencoder.return_value = mock_instance

            reranker = create_reranker(settings)

            # Should use canonical slug
            mock_crossencoder.assert_called_once_with(
                model_name="DiTy/cross-encoder-russian-msmarco",
                batch_size=16,
                device="cpu",
            )
            assert reranker == mock_instance

    @patch("rag_engine.retrieval.reranker.ModelRegistry")
    def test_factory_unknown_provider(self, mock_registry_cls):
        """Test factory raises error for unknown provider."""
        mock_registry = MagicMock()
        mock_registry.get_model.return_value = {
            "canonical_slug": "DiTy/cross-encoder-russian-msmarco",
            "type": "reranker",
        }
        mock_registry.get_provider_config.side_effect = ValueError(
            "Provider unknown not supported for model DiTy/cross-encoder-russian-msmarco"
        )
        mock_registry_cls.return_value = mock_registry

        settings = MagicMock()
        settings.reranker_provider_type = "unknown"
        settings.reranker_model = "DiTy/cross-encoder-russian-msmarco"

        with pytest.raises(ValueError, match="Provider unknown not supported"):
            create_reranker(settings)


class TestBuildRerankerLegacy:
    """Tests for legacy build_reranker function (backward compatibility)."""

    @patch("rag_engine.retrieval.reranker.CrossEncoder")
    def test_legacy_build_reranker_first_available(self, mock_crossencoder_class):
        """Test legacy function tries models in order."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9]
        mock_crossencoder_class.return_value = mock_model

        from rag_engine.retrieval.reranker import build_reranker

        models = [
            {"model_name": "DiTy/cross-encoder-russian-msmarco", "batch_size": 16},
        ]

        reranker = build_reranker(models, device="cpu")

        # Should use first (and only) model
        assert reranker is not None
        mock_crossencoder_class.assert_called_once_with(
            "DiTy/cross-encoder-russian-msmarco", device="cpu"
        )

    @patch("rag_engine.retrieval.reranker.CrossEncoder")
    def test_legacy_build_reranker_fallback_to_identity(self, mock_crossencoder_class):
        """Test legacy function falls back to identity on failure."""
        mock_crossencoder_class.side_effect = Exception("Model load failed")

        from rag_engine.retrieval.reranker import build_reranker

        models = [
            {"model_name": "DiTy/cross-encoder-russian-msmarco", "batch_size": 16},
        ]

        reranker = build_reranker(models, device="cpu")

        # Should return IdentityReranker on failure
        assert isinstance(reranker, IdentityReranker)


@pytest.mark.integration
@pytest.mark.external
class TestRerankerIntegration:
    """Integration tests for rerankers - requires real models or servers.

    Run with: pytest -m integration -v
    Uses .env configuration for Infinity endpoints.
    """

    def test_crossencoder_real_model(self):
        """Test with real CrossEncoder model (requires download)."""
        pytest.importorskip("sentence_transformers")

        try:
            reranker = CrossEncoderReranker(
                model_name="DiTy/cross-encoder-russian-msmarco",
                batch_size=16,
                device="cpu",
            )

            candidates = [
                (MockDocument("This is a test document about Python"), 0.0),
                (MockDocument("Another document about JavaScript"), 0.0),
            ]

            result = reranker.rerank("What is Python?", candidates, top_k=2)

            assert len(result) == 2
            assert all(isinstance(score, float) for _, score in result)
            print(f"\n✓ CrossEncoder reranker working")
            print(f"  Scores: {[score for _, score in result]}")

        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    def test_infinity_dity_real_server(self):
        """Test Infinity DiTy reranker against real server.

        Requires: Infinity server running on INFINITY_RERANKER_ENDPOINT (default: http://localhost:8002)
        Start with: cmw-infinity start dity-reranker
        """
        import requests

        endpoint = os.getenv("INFINITY_RERANKER_ENDPOINT", "http://localhost:8002")

        # Check if server is running
        try:
            response = requests.get(f"{endpoint}/health", timeout=2)
            if response.status_code != 200:
                pytest.skip(f"Infinity server not healthy at {endpoint}")
        except requests.ConnectionError:
            pytest.skip(
                f"Infinity server not running at {endpoint}. Start with: cmw-infinity start dity-reranker"
            )

        from rag_engine.config.loader import load_reranker_config

        config = load_reranker_config("infinity_dity")
        reranker = InfinityReranker(config)

        candidates = [
            (MockDocument("Python is a programming language for data science"), 0.0),
            (MockDocument("JavaScript is used for web development"), 0.0),
            (MockDocument("Machine learning uses algorithms to learn from data"), 0.0),
        ]

        result = reranker.rerank("What is Python programming?", candidates, top_k=2)

        assert len(result) == 2
        assert all(isinstance(score, float) for _, score in result)
        assert all(0 <= score <= 1 for _, score in result)  # Scores are typically 0-1

        print(f"\n✓ Infinity DiTy reranker working")
        print(f"  Endpoint: {endpoint}")
        print(f"  Top scores: {[score for _, score in result]}")

    def test_infinity_bge_reranker_real_server(self):
        """Test Infinity BGE reranker against real server.

        Requires: Infinity server running on port 8001
        Start with: cmw-infinity start bge-reranker
        """
        import requests

        endpoint = "http://localhost:8001"

        # Check if server is running
        try:
            response = requests.get(f"{endpoint}/health", timeout=2)
            if response.status_code != 200:
                pytest.skip(f"BGE reranker server not healthy at {endpoint}")
        except requests.ConnectionError:
            pytest.skip(
                f"BGE reranker server not running at {endpoint}. Start with: cmw-infinity start bge-reranker"
            )

        from rag_engine.config.loader import load_reranker_config

        config = load_reranker_config("infinity_bge_reranker")
        reranker = InfinityReranker(config)

        candidates = [
            (MockDocument("First document about programming"), 0.0),
            (MockDocument("Second document about databases"), 0.0),
        ]

        result = reranker.rerank("programming languages", candidates, top_k=2)

        assert len(result) == 2
        print(f"\n✓ Infinity BGE reranker working")
        print(f"  Scores: {[score for _, score in result]}")
