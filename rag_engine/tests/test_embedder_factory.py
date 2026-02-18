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
    DirectEmbeddingConfig,
    OpenAIEmbeddingConfig,
)
from rag_engine.retrieval.embedder import (
    FRIDAEmbedder,
    OpenAICompatibleEmbedder,
    create_embedder,
)


class TestDirectFRIDAEmbedder:
    """Tests for direct FRIDA embedder (current implementation)."""

    def test_frida_embedder_creation(self):
        """Test FRIDA embedder can be created."""
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

        assert embedder is not None
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

        with caplog.at_level("WARNING"):
            embedding = embedder.embed_query("test query", instruction="custom instruction")

        assert "doesn't support dynamic instructions" in caplog.text
        assert len(embedding) == 1536


class TestOpenAICompatibleEmbedder:
    """Tests for OpenAI-compatible embedder (Infinity, Mosec, OpenRouter)."""

    @patch("rag_engine.retrieval.embedder.OpenAI")
    def test_frida_query_prefix(self, mock_openai_class):
        """Test FRIDA query with search_query prefix."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            endpoint="http://localhost:7997",
            model="ai-forever/FRIDA",
            query_prefix="search_query: ",
            doc_prefix="search_document: ",
        )

        embedder = OpenAICompatibleEmbedder(config)
        embedding = embedder.embed_query("test query")

        call_args = mock_client.embeddings.create.call_args
        assert "search_query: test query" in call_args.kwargs["input"]
        assert len(embedding) == 1536

    @patch("rag_engine.retrieval.embedder.OpenAI")
    def test_frida_doc_prefix(self, mock_openai_class):
        """Test FRIDA documents with search_document prefix."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            endpoint="http://localhost:7997",
            model="ai-forever/FRIDA",
            query_prefix="search_query: ",
            doc_prefix="search_document: ",
        )

        embedder = OpenAICompatibleEmbedder(config)
        docs = ["Doc 1", "Doc 2"]
        embeddings = embedder.embed_documents(docs)

        call_args = mock_client.embeddings.create.call_args
        formatted_docs = call_args.kwargs["input"]
        assert all("search_document: " in doc for doc in formatted_docs)
        assert len(embeddings) == 2

    @patch("rag_engine.retrieval.embedder.OpenAI")
    def test_frida_instruction_warning(self, mock_openai_class, caplog):
        """Test FRIDA warns when instruction is passed."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            endpoint="http://localhost:7997",
            model="ai-forever/FRIDA",
            query_prefix="search_query: ",
        )

        embedder = OpenAICompatibleEmbedder(config)

        with caplog.at_level("WARNING"):
            embedder.embed_query("test query", instruction="custom instruction")

        assert "doesn't support dynamic instructions" in caplog.text

    @patch("rag_engine.retrieval.embedder.OpenAI")
    def test_qwen3_instruction_format(self, mock_openai_class):
        """Test Qwen3 with Instruct/Query format."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 4096)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            endpoint="https://openrouter.ai/api/v1",
            model="qwen/qwen3-embedding-8b",
            default_instruction="Given a web search query, retrieve relevant passages",
        )

        embedder = OpenAICompatibleEmbedder(config)
        embedding = embedder.embed_query("What is AI?")

        call_args = mock_client.embeddings.create.call_args
        assert "Instruct:" in call_args.kwargs["input"]
        assert "Query:" in call_args.kwargs["input"]
        assert "What is AI?" in call_args.kwargs["input"]
        assert len(embedding) == 4096

    @patch("rag_engine.retrieval.embedder.OpenAI")
    def test_qwen3_custom_instruction(self, mock_openai_class):
        """Test Qwen3 accepts custom instruction."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1024)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            endpoint="https://openrouter.ai/api/v1",
            model="qwen/qwen3-embedding-0.6b",
            default_instruction="Default instruction",
        )

        embedder = OpenAICompatibleEmbedder(config)
        custom_instruction = "Given a code search query, retrieve relevant code snippets"
        embedder.embed_query("sort function", instruction=custom_instruction)

        call_args = mock_client.embeddings.create.call_args
        assert custom_instruction in call_args.kwargs["input"]

    @patch("rag_engine.retrieval.embedder.OpenAI")
    def test_qwen3_docs_no_formatting(self, mock_openai_class):
        """Test Qwen3 documents are not formatted."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1024),
            MagicMock(embedding=[0.2] * 1024),
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            endpoint="https://openrouter.ai/api/v1",
            model="qwen/qwen3-embedding-8b",
            default_instruction="Given a web search query, retrieve relevant passages",
        )

        embedder = OpenAICompatibleEmbedder(config)
        docs = ["Document 1", "Document 2"]
        embeddings = embedder.embed_documents(docs)

        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["input"] == docs
        assert len(embeddings) == 2

    @patch("rag_engine.retrieval.embedder.OpenAI")
    def test_error_traceability(self, mock_openai_class):
        """Test error logging includes endpoint and model context."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("Connection failed")
        mock_openai_class.return_value = mock_client

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            endpoint="http://localhost:7997",
            model="ai-forever/FRIDA",
        )

        embedder = OpenAICompatibleEmbedder(config)

        with pytest.raises(Exception):
            embedder.embed_query("test")


class TestCreateEmbedderFactory:
    """Tests for embedder factory function."""

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
    def test_factory_infinity_frida(self, mock_registry_cls):
        """Test factory creates embedder for Infinity provider with model=auto."""
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
        settings.embedding_timeout = 60.0
        settings.embedding_max_retries = 3

        with patch("rag_engine.retrieval.embedder.OpenAICompatibleEmbedder") as mock_embedder:
            mock_instance = MagicMock()
            mock_embedder.return_value = mock_instance

            embedder = create_embedder(settings)

            mock_embedder.assert_called_once()
            config_arg = mock_embedder.call_args[0][0]
            assert config_arg.model == "auto"
            assert config_arg.endpoint == "http://localhost:7997"
            assert embedder == mock_instance

    @patch("rag_engine.retrieval.embedder.ModelRegistry")
    def test_factory_mosec_frida(self, mock_registry_cls):
        """Test factory creates embedder for Mosec provider with exact model name."""
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
        settings.embedding_provider_type = "mosec"
        settings.embedding_model = "ai-forever/FRIDA"
        settings.mosec_embedding_endpoint = "http://localhost:7997"
        settings.embedding_timeout = 60.0
        settings.embedding_max_retries = 3

        with patch("rag_engine.retrieval.embedder.OpenAICompatibleEmbedder") as mock_embedder:
            mock_instance = MagicMock()
            mock_embedder.return_value = mock_instance

            embedder = create_embedder(settings)

            mock_embedder.assert_called_once()
            config_arg = mock_embedder.call_args[0][0]
            assert config_arg.model == "ai-forever/FRIDA"
            assert config_arg.endpoint == "http://localhost:7997"
            assert config_arg.api_key is None
            assert embedder == mock_instance

    @patch("rag_engine.retrieval.embedder.ModelRegistry")
    def test_factory_openrouter_qwen3(self, mock_registry_cls):
        """Test factory creates embedder for OpenRouter provider with API key."""
        mock_registry = MagicMock()
        mock_registry.get_model.return_value = {
            "canonical_slug": "Qwen/Qwen3-Embedding-8B",
            "type": "embedding",
        }
        mock_registry.get_provider_config.return_value = {
            "model_id": "qwen/qwen3-embedding-8b",
            "default_instruction": "Given a web search query, retrieve relevant passages",
        }
        mock_registry_cls.return_value = mock_registry

        settings = MagicMock()
        settings.embedding_provider_type = "openrouter"
        settings.embedding_model = "Qwen/Qwen3-Embedding-8B"
        settings.openrouter_endpoint = "https://openrouter.ai/api/v1"
        settings.openrouter_api_key = "test-api-key"
        settings.embedding_timeout = 60.0
        settings.embedding_max_retries = 3

        with patch("rag_engine.retrieval.embedder.OpenAICompatibleEmbedder") as mock_embedder:
            mock_instance = MagicMock()
            mock_embedder.return_value = mock_instance

            embedder = create_embedder(settings)

            mock_embedder.assert_called_once()
            config_arg = mock_embedder.call_args[0][0]
            assert config_arg.model == "qwen/qwen3-embedding-8b"
            assert config_arg.api_key == "test-api-key"
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

        settings = MagicMock()
        settings.embedding_provider_type = "direct"
        settings.embedding_model = "ai-forever/frida"

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
    """Integration tests for OpenRouter - uses real API key from .env."""

    def test_openrouter_real_api_call_query(self):
        """Test actual API call to OpenRouter for query embedding."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or api_key == "your-openrouter-key":
            pytest.skip("OPENROUTER_API_KEY not set in .env")

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            endpoint="https://openrouter.ai/api/v1",
            model="qwen/qwen3-embedding-8b",
            api_key=api_key,
            default_instruction="Given a web search query, retrieve relevant passages that answer the query",
        )

        embedder = OpenAICompatibleEmbedder(config)
        embedding = embedder.embed_query("What is machine learning?")

        assert embedding is not None
        assert len(embedding) == 4096
        assert all(isinstance(x, float) for x in embedding)

        print(f"\n✓ Query embedding successful")
        print(f"  Dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")

    def test_openrouter_real_api_call_documents(self):
        """Test actual API call to OpenRouter for document embeddings."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or api_key == "your-openrouter-key":
            pytest.skip("OPENROUTER_API_KEY not set in .env")

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            endpoint="https://openrouter.ai/api/v1",
            model="qwen/qwen3-embedding-8b",
            api_key=api_key,
            default_instruction="Given a web search query, retrieve relevant passages that answer the query",
        )

        embedder = OpenAICompatibleEmbedder(config)
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
        ]
        embeddings = embedder.embed_documents(docs)

        assert len(embeddings) == 2
        assert all(len(e) == 4096 for e in embeddings)

        print(f"\n✓ Document embeddings successful")
        print(f"  Documents: {len(embeddings)}")
        print(f"  Dimension per doc: {len(embeddings[0])}")
