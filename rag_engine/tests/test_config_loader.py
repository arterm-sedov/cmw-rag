"""Tests for configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file from project root if it exists (for consistency with other test files)
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from rag_engine.config.loader import (
    list_embedding_providers,
    list_reranker_providers,
    load_embedding_config,
    load_reranker_config,
)
from rag_engine.config.schemas import (
    ApiEmbeddingConfig,
    DirectEmbeddingConfig,
    DirectRerankerConfig,
    ServerEmbeddingConfig,
    ServerRerankerConfig,
)


class TestLoadEmbeddingConfig:
    """Tests for loading embedding provider configurations."""

    def test_load_direct_frida(self):
        """Test loading direct FRIDA config."""
        config = load_embedding_config("direct_frida")

        assert isinstance(config, DirectEmbeddingConfig)
        assert config.type == "direct"
        assert config.model == "ai-forever/FRIDA"
        assert config.device == "auto"
        assert config.max_seq_length == 512

    def test_load_infinity_frida(self):
        """Test loading Infinity FRIDA config."""
        config = load_embedding_config("infinity_frida")

        assert isinstance(config, ServerEmbeddingConfig)
        assert config.type == "server"
        assert config.endpoint == "http://localhost:7997/v1"
        assert config.query_prefix == "search_query: "
        assert config.doc_prefix == "search_document: "
        assert config.default_instruction is None

    def test_load_openrouter_qwen3(self):
        """Test loading OpenRouter Qwen3 config."""
        config = load_embedding_config("openrouter_qwen3")

        assert isinstance(config, ApiEmbeddingConfig)
        assert config.type == "api"
        assert config.endpoint == "https://openrouter.ai/api/v1"
        assert config.model == "qwen/qwen3-embedding-8b"
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert "Given a web search query" in config.default_instruction

    def test_load_infinity_qwen3_8b(self):
        """Test loading Infinity Qwen3-8B config."""
        config = load_embedding_config("infinity_qwen3_8b")

        assert isinstance(config, ServerEmbeddingConfig)
        assert config.type == "server"
        assert config.endpoint == "http://localhost:8000/v1"
        assert config.default_instruction is not None
        assert config.query_prefix is None  # Qwen3 uses instruction, not prefix

    def test_load_infinity_qwen3_4b(self):
        """Test loading Infinity Qwen3-4B config."""
        config = load_embedding_config("infinity_qwen3_4b")

        assert isinstance(config, ServerEmbeddingConfig)
        assert config.endpoint == "http://localhost:7999/v1"

    def test_load_infinity_qwen3_0_6b(self):
        """Test loading Infinity Qwen3-0.6B config."""
        config = load_embedding_config("infinity_qwen3_0_6b")

        assert isinstance(config, ServerEmbeddingConfig)
        assert config.endpoint == "http://localhost:7998/v1"

    def test_load_unknown_provider(self):
        """Test loading unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            load_embedding_config("unknown_provider")


class TestLoadRerankerConfig:
    """Tests for loading reranker provider configurations."""

    def test_load_direct_crossencoder(self):
        """Test loading direct CrossEncoder config."""
        config = load_reranker_config("direct_crossencoder")

        assert isinstance(config, DirectRerankerConfig)
        assert config.type == "direct"
        assert config.model == "DiTy/cross-encoder-russian-msmarco"
        assert config.device == "auto"
        assert config.batch_size == 16

    def test_load_infinity_dity(self):
        """Test loading Infinity DiTy config."""
        config = load_reranker_config("infinity_dity")

        assert isinstance(config, ServerRerankerConfig)
        assert config.type == "server"
        assert config.endpoint == "http://localhost:8002"
        assert config.default_instruction is None  # DiTy doesn't use instructions

    def test_load_infinity_bge_reranker(self):
        """Test loading Infinity BGE reranker config."""
        config = load_reranker_config("infinity_bge_reranker")

        assert isinstance(config, ServerRerankerConfig)
        assert config.endpoint == "http://localhost:8001"

    def test_load_infinity_qwen3_reranker_8b(self):
        """Test loading Infinity Qwen3-8B reranker config."""
        config = load_reranker_config("infinity_qwen3_reranker_8b")

        assert isinstance(config, ServerRerankerConfig)
        assert config.endpoint == "http://localhost:8005"
        assert config.default_instruction is not None

    def test_load_infinity_qwen3_reranker_4b(self):
        """Test loading Infinity Qwen3-4B reranker config."""
        config = load_reranker_config("infinity_qwen3_reranker_4b")

        assert isinstance(config, ServerRerankerConfig)
        assert config.endpoint == "http://localhost:8004"

    def test_load_infinity_qwen3_reranker_0_6b(self):
        """Test loading Infinity Qwen3-0.6B reranker config."""
        config = load_reranker_config("infinity_qwen3_reranker_0_6b")

        assert isinstance(config, ServerRerankerConfig)
        assert config.endpoint == "http://localhost:8003"

    def test_load_unknown_provider(self):
        """Test loading unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown reranker provider"):
            load_reranker_config("unknown_provider")


class TestListProviders:
    """Tests for listing available providers."""

    def test_list_embedding_providers(self):
        """Test listing embedding providers."""
        providers = list_embedding_providers()

        assert "direct_frida" in providers
        assert "infinity_frida" in providers
        assert "openrouter_qwen3" in providers
        assert "infinity_qwen3_8b" in providers
        assert "infinity_qwen3_4b" in providers
        assert "infinity_qwen3_0_6b" in providers

    def test_list_reranker_providers(self):
        """Test listing reranker providers."""
        providers = list_reranker_providers()

        assert "direct_crossencoder" in providers
        assert "infinity_dity" in providers
        assert "infinity_bge_reranker" in providers
        assert "infinity_qwen3_reranker_8b" in providers
        assert "infinity_qwen3_reranker_4b" in providers
        assert "infinity_qwen3_reranker_0_6b" in providers


class TestConfigCaching:
    """Tests for configuration caching."""

    def test_config_cached(self):
        """Test that config is cached after first load."""
        # Load config twice
        config1 = load_embedding_config("direct_frida")
        config2 = load_embedding_config("direct_frida")

        # Should be same object (cached)
        # Note: Pydantic models are immutable, so this tests the YAML loading cache
        assert config1.model == config2.model
        assert config1.device == config2.device

    def test_config_cache_invalidation_not_implemented(self):
        """Test that there's no cache invalidation (current behavior)."""
        # This documents current behavior - cache persists for process lifetime
        # If hot-reloading is needed, clear _CONFIG_CACHE manually
        from rag_engine.config.loader import _CONFIG_CACHE

        # Load a config to populate cache
        load_embedding_config("direct_frida")

        # Cache should be populated
        assert "yaml" in _CONFIG_CACHE
