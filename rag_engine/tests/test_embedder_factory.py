"""Tests for embedder factory and implementations."""

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

from rag_engine.config.schemas import (
    ApiEmbeddingConfig,
    DirectEmbeddingConfig,
    ServerEmbeddingConfig,
)
from rag_engine.retrieval.embedder import (
    FRIDAEmbedder,
    InfinityEmbedder,
    OpenRouterEmbedder,
    create_embedder,
)


class TestDirectFRIDAEmbedder:
    """Tests for direct FRIDA embedder (current implementation)."""

    def test_frida_embedder_creation(self):
        """Test FRIDA embedder can be created."""
        # Note: This requires sentence-transformers and the model
        # In CI/mock environment, this may be skipped
        pytest.importorskip("sentence_transformers")

        config = DirectEmbeddingConfig(
            type="direct",
            model="ai-forever/FRIDA",
            device="cpu",
            max_seq_length=512,
        )

        # Should not raise
        embedder = FRIDAEmbedder(
            model_name=config.model,
            device=config.device,
            max_seq_length=config.max_seq_length,
            check_disk_space=False,  # Skip disk check for tests
        )

        assert embedder is not None
        # FRIDA produces 1536-dim embeddings (not 1024 as previously documented)
        assert embedder.get_embedding_dim() == 1536

    def test_frida_instruction_warning(self, caplog):
        """Test FRIDA ignores dynamic instructions with warning."""
        pytest.importorskip("sentence_transformers")

        config = DirectEmbeddingConfig(
            type="direct",
            model="ai-forever/FRIDA",
            device="cpu",
            max_seq_length=512,
        )

        embedder = FRIDAEmbedder(
            model_name=config.model,
            device=config.device,
            max_seq_length=config.max_seq_length,
            check_disk_space=False,
        )

        # FRIDA should ignore instruction and log warning
        with caplog.at_level("WARNING"):
            embedding = embedder.embed_query("test query", instruction="custom instruction")

        assert "doesn't support dynamic instructions" in caplog.text
        assert len(embedding) == 1536


class TestOpenRouterEmbedder:
    """Tests for OpenRouter Qwen3 embedder."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_openrouter_embedder_creation(self):
        """Test OpenRouter embedder can be created with API key."""
        config = ApiEmbeddingConfig(
            type="api",
            endpoint="https://openrouter.ai/api/v1",
            model="qwen/qwen3-embedding-8b",
            default_instruction="Given a web search query, retrieve relevant passages that answer the query",
            timeout=60.0,
            max_retries=3,
        )

        embedder = OpenRouterEmbedder(config)

        assert embedder is not None
        assert embedder.model == config.model
        assert embedder.default_instruction == config.default_instruction

    @patch.dict(os.environ, {}, clear=True)
    def test_openrouter_embedder_no_api_key(self):
        """Test OpenRouter embedder works without env var (uses empty string)."""
        config = ApiEmbeddingConfig(
            type="api",
            endpoint="https://openrouter.ai/api/v1",
            model="qwen/qwen3-embedding-8b",
            default_instruction="Given a web search query, retrieve relevant passages that answer the query",
        )

        # Should not raise, uses empty string as default
        embedder = OpenRouterEmbedder(config)
        assert embedder is not None

    @patch("rag_engine.retrieval.embedder.OpenAI")
    def test_openrouter_embed_query_default_instruction(self, mock_openai_class):
        """Test OpenRouter uses default instruction for queries."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3, 0.4] * 1024)]  # 4096 dims
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = ApiEmbeddingConfig(
            type="api",
            endpoint="https://openrouter.ai/api/v1",
            model="qwen/qwen3-embedding-8b",
            default_instruction="Given a web search query, retrieve relevant passages that answer the query",
        )

        embedder = OpenRouterEmbedder(config)
        embedding = embedder.embed_query("What is Python?")

        # Verify API call
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args

        assert call_args.kwargs["model"] == "qwen/qwen3-embedding-8b"
        assert "Instruct:" in call_args.kwargs["input"]
        assert "Query:" in call_args.kwargs["input"]
        assert "What is Python?" in call_args.kwargs["input"]
        assert len(embedding) == 4096

    @patch("rag_engine.retrieval.embedder.OpenAI")
    def test_openrouter_embed_query_custom_instruction(self, mock_openai_class):
        """Test OpenRouter accepts custom instruction."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1024)]  # 1024 dims for 0.6B
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = ApiEmbeddingConfig(
            type="api",
            endpoint="https://openrouter.ai/api/v1",
            model="qwen/qwen3-embedding-0.6b",
            default_instruction="Given a web search query, retrieve relevant passages that answer the query",
        )

        embedder = OpenRouterEmbedder(config)

        # Use custom instruction
        custom_instruction = "Given a code search query, retrieve relevant code snippets"
        embedder.embed_query("sort function", instruction=custom_instruction)

        # Verify custom instruction was used
        call_args = mock_client.embeddings.create.call_args
        assert custom_instruction in call_args.kwargs["input"]

    @patch("rag_engine.retrieval.embedder.OpenAI")
    def test_openrouter_embed_documents_no_instruction(self, mock_openai_class):
        """Test OpenRouter doesn't add instruction to documents."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1024),
            MagicMock(embedding=[0.2] * 1024),
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = ApiEmbeddingConfig(
            type="api",
            endpoint="https://openrouter.ai/api/v1",
            model="qwen/qwen3-embedding-8b",
            default_instruction="Given a web search query, retrieve relevant passages that answer the query",
        )

        embedder = OpenRouterEmbedder(config)
        docs = ["Document 1", "Document 2"]
        embeddings = embedder.embed_documents(docs)

        # Verify documents don't get instruction prefix
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["input"] == docs  # No modification
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1024


class TestInfinityEmbedder:
    """Tests for Infinity server embedder."""

    @patch("rag_engine.retrieval.embedder.HTTPClientMixin._post")
    def test_infinity_frida_embed_query(self, mock_post):
        """Test Infinity FRIDA embed query with prefix."""
        mock_post.return_value = {"data": [{"embedding": [0.1] * 1024}]}

        config = ServerEmbeddingConfig(
            type="server",
            endpoint="http://localhost:7997/v1",
            query_prefix="search_query: ",
            doc_prefix="search_document: ",
        )

        embedder = InfinityEmbedder(config)
        embedding = embedder.embed_query("test query")

        # Verify prefix was added and request was made
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        # Verify endpoint path contains 'embeddings'
        assert "embeddings" in call_args[0][0]
        assert "search_query: test query" in call_args[0][1]["input"]
        assert len(embedding) == 1024

    @patch("rag_engine.retrieval.embedder.HTTPClientMixin._post")
    def test_infinity_frida_embed_documents(self, mock_post):
        """Test Infinity FRIDA embed documents with prefix."""
        mock_post.return_value = {
            "data": [
                {"embedding": [0.1] * 1024},
                {"embedding": [0.2] * 1024},
            ]
        }

        config = ServerEmbeddingConfig(
            type="server",
            endpoint="http://localhost:7997/v1",
            query_prefix="search_query: ",
            doc_prefix="search_document: ",
        )

        embedder = InfinityEmbedder(config)
        docs = ["Doc 1", "Doc 2"]
        embeddings = embedder.embed_documents(docs)

        # Verify prefixes were added
        call_args = mock_post.call_args
        # Verify endpoint path contains 'embeddings' (not testing exact path)
        assert "embeddings" in call_args[0][0]
        formatted_docs = call_args[0][1]["input"]
        assert all("search_document: " in doc for doc in formatted_docs)
        assert len(embeddings) == 2

    @patch("rag_engine.retrieval.embedder.HTTPClientMixin._post")
    def test_infinity_qwen3_embed_query_with_instruction(self, mock_post):
        """Test Infinity Qwen3 embed query with instruction format."""
        mock_post.return_value = {"data": [{"embedding": [0.1] * 4096}]}

        config = ServerEmbeddingConfig(
            type="server",
            endpoint="http://localhost:8000/v1",
            default_instruction="Given a web search query, retrieve relevant passages that answer the query",
        )

        embedder = InfinityEmbedder(config)
        embedding = embedder.embed_query("What is AI?")

        # Verify instruction format
        call_args = mock_post.call_args
        assert "Instruct:" in call_args[0][1]["input"][0]
        assert "Query:" in call_args[0][1]["input"][0]
        assert len(embedding) == 4096


class TestCreateEmbedderFactory:
    """Tests for embedder factory function with model-slug-based configuration."""

    @patch("rag_engine.retrieval.embedder.ModelRegistry")
    def test_factory_direct_frida(self, mock_registry_cls):
        """Test factory creates FRIDA embedder for direct provider."""
        mock_registry = MagicMock()
        mock_registry.get_model.return_value = {
            "canonical_slug": "ai-forever/FRIDA",
            "type": "embedding",
        }
        mock_registry.get_provider_config.return_value = {
            "device": "cpu",
            "max_seq_length": 512,
        }
        mock_registry_cls.return_value = mock_registry

        settings = MagicMock()
        settings.embedding_provider_type = "direct"
        settings.embedding_model = "ai-forever/FRIDA"

        with patch("rag_engine.retrieval.embedder.FRIDAEmbedder") as mock_frida:
            mock_instance = MagicMock()
            mock_frida.return_value = mock_instance

            embedder = create_embedder(settings)

            mock_frida.assert_called_once_with(
                model_name="ai-forever/FRIDA",
                device="cpu",
                max_seq_length=512,
            )
            assert embedder == mock_instance

    @patch("rag_engine.retrieval.embedder.ModelRegistry")
    def test_factory_openrouter_qwen3(self, mock_registry_cls):
        """Test factory creates OpenRouter embedder."""
        mock_registry = MagicMock()
        mock_registry.get_model.return_value = {
            "canonical_slug": "Qwen/Qwen3-Embedding-8B",
            "type": "embedding",
        }
        mock_registry.get_provider_config.return_value = {
            "model_id": "qwen/qwen3-embedding-8b",
            "default_instruction": "Given a web search query, retrieve relevant passages that answer the query",
        }
        mock_registry_cls.return_value = mock_registry

        settings = MagicMock()
        settings.embedding_provider_type = "openrouter"
        settings.embedding_model = "Qwen/Qwen3-Embedding-8B"
        settings.openrouter_endpoint = "https://openrouter.ai/api/v1"
        settings.embedding_timeout = 60.0
        settings.embedding_max_retries = 3

        with patch("rag_engine.retrieval.embedder.OpenRouterEmbedder") as mock_openrouter:
            mock_instance = MagicMock()
            mock_openrouter.return_value = mock_instance

            embedder = create_embedder(settings)

            mock_openrouter.assert_called_once()
            assert embedder == mock_instance

    @patch("rag_engine.retrieval.embedder.ModelRegistry")
    def test_factory_infinity_frida(self, mock_registry_cls):
        """Test factory creates Infinity embedder for server provider."""
        mock_registry = MagicMock()
        mock_registry.get_model.return_value = {
            "canonical_slug": "ai-forever/FRIDA",
            "type": "embedding",
        }
        mock_registry.get_provider_config.return_value = {
            "query_prefix": "search_query: ",
            "doc_prefix": "search_document: ",
        }
        mock_registry_cls.return_value = mock_registry

        settings = MagicMock()
        settings.embedding_provider_type = "infinity"
        settings.embedding_model = "ai-forever/FRIDA"
        settings.infinity_embedding_endpoint = "http://localhost:7997"

        with patch("rag_engine.retrieval.embedder.InfinityEmbedder") as mock_infinity:
            mock_instance = MagicMock()
            mock_infinity.return_value = mock_instance

            embedder = create_embedder(settings)

            mock_infinity.assert_called_once()
            assert embedder == mock_instance

    @patch("rag_engine.retrieval.embedder.ModelRegistry")
    def test_factory_case_insensitive_model_slug(self, mock_registry_cls):
        """Test factory handles case-insensitive model slugs."""
        mock_registry = MagicMock()
        mock_registry.get_model.return_value = {
            "canonical_slug": "ai-forever/FRIDA",
            "type": "embedding",
        }
        mock_registry.get_provider_config.return_value = {
            "device": "cpu",
            "max_seq_length": 512,
        }
        mock_registry_cls.return_value = mock_registry

        # Use lowercase model slug
        settings = MagicMock()
        settings.embedding_provider_type = "direct"
        settings.embedding_model = "ai-forever/frida"

        with patch("rag_engine.retrieval.embedder.FRIDAEmbedder") as mock_frida:
            mock_instance = MagicMock()
            mock_frida.return_value = mock_instance

            embedder = create_embedder(settings)

            # Should use canonical slug
            mock_frida.assert_called_once_with(
                model_name="ai-forever/FRIDA",
                device="cpu",
                max_seq_length=512,
            )
            assert embedder == mock_instance

    @patch("rag_engine.retrieval.embedder.ModelRegistry")
    def test_factory_unknown_provider(self, mock_registry_cls):
        """Test factory raises error for unknown provider."""
        mock_registry = MagicMock()
        mock_registry.get_model.return_value = {
            "canonical_slug": "ai-forever/FRIDA",
            "type": "embedding",
        }
        mock_registry.get_provider_config.side_effect = ValueError(
            "Provider unknown not supported for model ai-forever/FRIDA"
        )
        mock_registry_cls.return_value = mock_registry

        settings = MagicMock()
        settings.embedding_provider_type = "unknown"
        settings.embedding_model = "ai-forever/FRIDA"

        with pytest.raises(ValueError, match="Provider unknown not supported"):
            create_embedder(settings)


@pytest.mark.integration
@pytest.mark.external
class TestOpenRouterIntegration:
    """Integration tests for OpenRouter - uses real API key from .env.

    Run with: pytest -m integration -v
    Requires: OPENROUTER_API_KEY in .env file or environment
    """

    def test_openrouter_real_api_call_query(self):
        """Test actual API call to OpenRouter for query embedding.

        Uses OPENROUTER_API_KEY from .env file.
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or api_key == "your-openrouter-key":
            pytest.skip("OPENROUTER_API_KEY not set in .env (found placeholder or missing)")

        from rag_engine.config.loader import load_embedding_config

        config = load_embedding_config("openrouter_qwen3")
        embedder = OpenRouterEmbedder(config)

        # Real API call with default instruction
        embedding = embedder.embed_query("What is machine learning?")

        assert embedding is not None
        assert len(embedding) == 4096  # Qwen3-8B dimension
        assert all(isinstance(x, float) for x in embedding)
        assert all(-1 <= x <= 1 for x in embedding)  # Normalized embeddings

        print(f"\n✓ Query embedding successful")
        print(f"  Dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")

    def test_openrouter_real_api_call_documents(self):
        """Test actual API call to OpenRouter for document embeddings.

        Uses OPENROUTER_API_KEY from .env file.
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or api_key == "your-openrouter-key":
            pytest.skip("OPENROUTER_API_KEY not set in .env (found placeholder or missing)")

        from rag_engine.config.loader import load_embedding_config

        config = load_embedding_config("openrouter_qwen3")
        embedder = OpenRouterEmbedder(config)

        # Real API call for documents (no instruction added)
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
        ]
        embeddings = embedder.embed_documents(docs)

        assert len(embeddings) == 2
        assert all(len(e) == 4096 for e in embeddings)
        assert all(isinstance(x, float) for e in embeddings for x in e)

        print(f"\n✓ Document embeddings successful")
        print(f"  Documents: {len(embeddings)}")
        print(f"  Dimension per doc: {len(embeddings[0])}")

    def test_openrouter_real_api_custom_instruction(self):
        """Test OpenRouter with custom instruction override.

        Uses OPENROUTER_API_KEY from .env file.
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or api_key == "your-openrouter-key":
            pytest.skip("OPENROUTER_API_KEY not set in .env (found placeholder or missing)")

        from rag_engine.config.loader import load_embedding_config

        config = load_embedding_config("openrouter_qwen3")
        embedder = OpenRouterEmbedder(config)

        # Test with custom code search instruction
        custom_instruction = "Given a code search query, retrieve relevant code implementations"
        embedding = embedder.embed_query("sort list in python", instruction=custom_instruction)

        assert embedding is not None
        assert len(embedding) == 4096

        print(f"\n✓ Custom instruction embedding successful")
        print(f"  Instruction: {custom_instruction}")
        print(f"  Dimension: {len(embedding)}")
