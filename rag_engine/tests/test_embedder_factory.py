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

    @patch("rag_engine.retrieval.embedder.requests.post")
    def test_frida_query_prefix(self, mock_post):
        """Test FRIDA query with search_query prefix (local provider uses direct HTTP)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 1536}]}
        mock_post.return_value = mock_response

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            provider="mosec",
            dimensions=1536,
            local=True,
            endpoint="http://localhost:7997",
            model="ai-forever/FRIDA",
            query_prefix="search_query: ",
            doc_prefix="search_document: ",
        )

        embedder = OpenAICompatibleEmbedder(config)
        embedding = embedder.embed_query("test query")

        call_args = mock_post.call_args
        assert "search_query: test query" in call_args.kwargs["json"]["input"]
        assert len(embedding) == 1536

    @patch("rag_engine.retrieval.embedder.requests.post")
    def test_frida_doc_prefix(self, mock_post):
        """Test FRIDA documents with search_document prefix."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1] * 1536},
                {"embedding": [0.2] * 1536},
            ]
        }
        mock_post.return_value = mock_response

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            provider="mosec",
            dimensions=1536,
            local=True,
            endpoint="http://localhost:7997",
            model="ai-forever/FRIDA",
            query_prefix="search_query: ",
            doc_prefix="search_document: ",
        )

        embedder = OpenAICompatibleEmbedder(config)
        docs = ["Doc 1", "Doc 2"]
        embeddings = embedder.embed_documents(docs)

        call_args = mock_post.call_args
        formatted_docs = call_args.kwargs["json"]["input"]
        assert all("search_document: " in doc for doc in formatted_docs)
        assert len(embeddings) == 2

    @patch("rag_engine.retrieval.embedder.requests.post")
    def test_frida_instruction_warning(self, mock_post, caplog):
        """Test FRIDA warns when instruction is passed."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1] * 1536}]}
        mock_post.return_value = mock_response

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            provider="mosec",
            dimensions=1536,
            local=True,
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
        """Test Qwen3 with Instruct/Query format (cloud provider uses SDK)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 4096)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            provider="openrouter",
            dimensions=4096,
            local=False,
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
            provider="openrouter",
            dimensions=4096,
            local=False,
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
            provider="openrouter",
            dimensions=4096,
            local=False,
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

    @patch("rag_engine.retrieval.embedder.requests.post")
    def test_error_traceability(self, mock_post):
        """Test error logging includes endpoint and model context."""
        mock_post.side_effect = Exception("Connection failed")

        config = OpenAIEmbeddingConfig(
            type="openai_compatible",
            provider="mosec",
            dimensions=1536,
            local=True,
            endpoint="http://localhost:7997",
            model="ai-forever/FRIDA",
        )

        embedder = OpenAICompatibleEmbedder(config)

        with pytest.raises(Exception, match="Connection failed"):
            embedder.embed_query("test query")




class TestCreateEmbedderFactory:
    """Tests for embedder factory function."""
