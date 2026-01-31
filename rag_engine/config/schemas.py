"""Pydantic schemas for embedding and reranker provider configurations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field
from typing_extensions import Annotated

logger = logging.getLogger(__name__)


# ============ EMBEDDING CONFIGS ============


class DirectEmbeddingConfig(BaseModel):
    """Direct sentence-transformers embedder (current implementation)."""

    type: Literal["direct"]
    model: str = Field(..., description="Model name for sentence-transformers")
    device: str = Field(default="auto")
    max_seq_length: int = Field(default=512)


class ServerEmbeddingConfig(BaseModel):
    """HTTP server embedder (Infinity)."""

    type: Literal["server"]
    endpoint: str = Field(..., description="HTTP endpoint (e.g., http://localhost:7997/v1)")

    # Model-specific formatting
    query_prefix: Optional[str] = Field(None)  # FRIDA: "search_query: "
    doc_prefix: Optional[str] = Field(None)  # FRIDA: "search_document: "
    default_instruction: Optional[str] = Field(None)  # Qwen3: instruction template


class ApiEmbeddingConfig(BaseModel):
    """Cloud API embedder (OpenRouter)."""

    type: Literal["api"]
    endpoint: str = Field(..., description="API endpoint URL")
    model: str = Field(..., description="Model identifier (e.g., qwen/qwen3-embedding-8b)")
    default_instruction: str = Field(..., description="Default instruction template")
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Max retries on failure")


# Discriminated union for type-safe config loading
EmbeddingProviderConfig = Annotated[
    Union[DirectEmbeddingConfig, ServerEmbeddingConfig, ApiEmbeddingConfig],
    Field(discriminator="type"),
]


# ============ RERANKER CONFIGS ============


class DirectRerankerConfig(BaseModel):
    """Direct CrossEncoder reranker (current implementation)."""

    type: Literal["direct"]
    model: str = Field(..., description="Model name for CrossEncoder")
    device: str = Field(default="auto")
    batch_size: int = Field(default=16)


class ServerRerankerConfig(BaseModel):
    """HTTP server reranker (Infinity)."""

    type: Literal["server"]
    endpoint: str = Field(..., description="HTTP endpoint (e.g., http://localhost:7998)")
    default_instruction: Optional[str] = Field(None)  # Qwen3 only


RerankerProviderConfig = Annotated[
    Union[DirectRerankerConfig, ServerRerankerConfig],
    Field(discriminator="type"),
]


# ============ MODEL REGISTRY ============


class ModelRegistry:
    """Registry for model metadata loaded from YAML.

    Supports case-insensitive model slug lookup with canonical normalization.
    All model slugs are stored in HuggingFace format (e.g., "Qwen/Qwen3-Embedding-8B")
    but can be looked up with any case variation (e.g., "qwen/qwen3-embedding-8b").
    """

    _instance = None
    _models: dict[str, dict[str, Any]] = {}
    _defaults: dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_registry()
        return cls._instance

    def _load_registry(self) -> None:
        """Load model registry from YAML file."""
        config_path = Path(__file__).parent / "models.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Model registry not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Build case-insensitive lookup table
        for model_slug, model_data in data.get("models", {}).items():
            # Store with normalized (lowercase) key for lookup
            normalized_key = model_slug.lower()
            self._models[normalized_key] = {
                "canonical_slug": model_slug,
                **model_data,
            }

        self._defaults = data.get("defaults", {})
        logger.info(f"Loaded {len(self._models)} models from registry")

    def _normalize_slug(self, model_slug: str) -> str:
        """Normalize model slug to lowercase for case-insensitive lookup."""
        return model_slug.lower().strip()

    def get_model(self, model_slug: str) -> dict[str, Any]:
        """Get model metadata by slug (case-insensitive).

        Args:
            model_slug: Model identifier (e.g., "Qwen/Qwen3-Embedding-8B" or "qwen/qwen3-embedding-8b")

        Returns:
            Model metadata dict with canonical_slug, type, dimensions, provider_formats

        Raises:
            ValueError: If model not found in registry
        """
        normalized = self._normalize_slug(model_slug)
        if normalized not in self._models:
            available = [m["canonical_slug"] for m in self._models.values()]
            raise ValueError(f"Unknown model: {model_slug}. Available: {available}")
        return self._models[normalized]

    def get_canonical_slug(self, model_slug: str) -> str:
        """Get canonical (HuggingFace format) slug from any case variation.

        Args:
            model_slug: Model identifier in any case

        Returns:
            Canonical slug in HuggingFace format (e.g., "Qwen/Qwen3-Embedding-8B")
        """
        return self.get_model(model_slug)["canonical_slug"]

    def get_dimension(self, model_slug: str) -> int:
        """Get embedding dimension for a model.

        Args:
            model_slug: Model identifier

        Returns:
            Embedding dimension (int)

        Raises:
            ValueError: If model not found or not an embedding model
        """
        model = self.get_model(model_slug)
        if model.get("type") != "embedding":
            raise ValueError(f"Model {model_slug} is not an embedding model")
        return model["dimensions"]

    def get_provider_config(self, model_slug: str, provider: str) -> dict[str, Any]:
        """Get provider-specific configuration for a model.

        Args:
            model_slug: Model identifier
            provider: Provider type ("direct", "infinity", "openrouter")

        Returns:
            Provider configuration dict

        Raises:
            ValueError: If provider not supported for this model
        """
        model = self.get_model(model_slug)
        formats = model.get("provider_formats", {})

        if provider not in formats:
            raise ValueError(f"Provider {provider} not supported for model {model_slug}")

        config = formats[provider]
        if not config.get("supported", True):
            raise ValueError(f"Provider {provider} is not supported for model {model_slug}")

        return config

    def get_default_endpoint(self, provider: str, model_type: str) -> str:
        """Get default endpoint for a provider.

        Args:
            provider: Provider type ("infinity", "openrouter")
            model_type: "embedding" or "reranker"

        Returns:
            Default endpoint URL
        """
        endpoints = self._defaults.get("endpoints", {})

        if provider == "infinity":
            return endpoints.get("infinity", {}).get(
                model_type, f"http://localhost:{7997 if model_type == 'embedding' else 7998}"
            )
        elif provider == "openrouter":
            return endpoints.get("openrouter", "https://openrouter.ai/api/v1")

        raise ValueError(f"Unknown provider: {provider}")

    def get_timeout(self, provider: str) -> float:
        """Get default timeout for a provider."""
        return self._defaults.get("timeouts", {}).get(provider, 60.0)

    def get_retries(self, provider: str) -> int:
        """Get default retries for a provider."""
        return self._defaults.get("retries", {}).get(provider, 3)

    def list_models(self, model_type: Optional[str] = None) -> list[str]:
        """List available models.

        Args:
            model_type: Filter by type ("embedding" or "reranker"), or None for all

        Returns:
            List of canonical model slugs
        """
        models = self._models.values()
        if model_type:
            models = [m for m in models if m.get("type") == model_type]
        return [m["canonical_slug"] for m in models]


# Convenience function for backward compatibility
def get_model_dimension(model_slug: str) -> int:
    """Get embedding dimension for a model (convenience function)."""
    return ModelRegistry().get_dimension(model_slug)
