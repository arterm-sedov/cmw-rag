"""Unified reranker provider with factory pattern.

Supports two provider types:
1. Direct: sentence-transformers CrossEncoder (DiTy, BGE)
2. Server: Infinity HTTP API (DiTy, BGE, Qwen3)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Optional, Protocol

from rag_engine.config.schemas import DirectRerankerConfig, ModelRegistry, ServerRerankerConfig
from rag_engine.retrieval.embedder import HTTPClientMixin
from rag_engine.utils.device_utils import detect_device

logger = logging.getLogger(__name__)

try:  # Provide a module-level alias for monkeypatch-friendly tests
    from sentence_transformers import CrossEncoder as CrossEncoder  # type: ignore
except Exception:  # noqa: BLE001
    CrossEncoder = None  # type: ignore


class Reranker(Protocol):
    """Unified interface for all reranker providers."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Any, float]],
        top_k: int,
        metadata_boost_weights: Optional[dict[str, float]] = None,
        instruction: Optional[str] = None,  # Qwen3 only
    ) -> list[tuple[Any, float]]:
        """
        Rerank candidates based on query relevance.

        Args:
            query: Search query
            candidates: List of (document, initial_score) tuples
            top_k: Number of top results to return
            metadata_boost_weights: Optional metadata-based score boosts
            instruction: Optional custom instruction (Qwen3 reranker only)

        Returns:
            Sorted list of (document, reranker_score) tuples
        """
        ...


class IdentityReranker:
    """Pass-through reranker used when cross-encoder is unavailable."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Any, float]],
        top_k: int,
        metadata_boost_weights: Optional[dict[str, float]] = None,
        instruction: Optional[str] = None,
    ) -> list[tuple[Any, float]]:
        if instruction:
            logger.warning("IdentityReranker doesn't support instructions, ignoring")
        return list(candidates)[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers CrossEncoder (direct)."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        device: str = "auto",
        model: Any | None = None,
    ):
        """Initialize cross-encoder reranker.

        Args:
            model_name: Name of the CrossEncoder model
            batch_size: Batch size for reranking
            device: Device to run the model on ('auto', 'cpu', or 'cuda').
                    'auto' will detect and use GPU if available, else CPU.
            model: Optional pre-initialized model (for testing)
        """
        # Auto-detect device if "auto" is specified
        if device == "auto":
            device = detect_device("auto")

        # Allow injecting a fake model for tests; otherwise construct via module-level alias
        global CrossEncoder  # use the module-level symbol (monkeypatchable)
        if model is not None:
            self.model = model
        else:
            if CrossEncoder is None:
                # Import here if alias was not resolved at import time
                from sentence_transformers import CrossEncoder as _CE  # type: ignore

                CrossEncoder = _CE  # type: ignore
            self.model = CrossEncoder(model_name, device=device)  # type: ignore[operator]
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Any, float]],
        top_k: int,
        metadata_boost_weights: Optional[dict[str, float]] = None,
        instruction: Optional[str] = None,
    ) -> list[tuple[Any, float]]:
        """Rerank candidates using cross-encoder."""
        if instruction:
            logger.warning("CrossEncoder doesn't support dynamic instructions, ignoring")

        pairs = [
            (query, doc.page_content if hasattr(doc, "page_content") else str(doc))
            for doc, _ in candidates
        ]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Combine with optional metadata boosts
        scored: list[tuple[Any, float]] = []
        for (doc, _), score in zip(candidates, scores, strict=False):
            boost = 0.0
            if metadata_boost_weights and hasattr(doc, "metadata"):
                meta = getattr(doc, "metadata", {})
                if meta.get("tags") and metadata_boost_weights.get("tag_match"):
                    boost += metadata_boost_weights["tag_match"]
                if meta.get("has_code") and metadata_boost_weights.get("code_presence"):
                    boost += metadata_boost_weights["code_presence"]
                if meta.get("section_heading") and metadata_boost_weights.get("section_match"):
                    boost += metadata_boost_weights["section_match"]
            final_score = float(score) * (1.0 + boost)
            scored.append((doc, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class InfinityReranker(HTTPClientMixin):
    """DiTy/BGE/Qwen3 via Infinity HTTP server."""

    def __init__(self, config: ServerRerankerConfig):
        super().__init__(
            endpoint=config.endpoint,
            timeout=60.0,
            max_retries=3,
        )
        self.default_instruction = config.default_instruction

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Any, float]],
        top_k: int,
        metadata_boost_weights: Optional[dict[str, float]] = None,
        instruction: Optional[str] = None,
    ) -> list[tuple[Any, float]]:
        """Rerank candidates via Infinity server."""
        if self.default_instruction:
            # Qwen3 format: "Instruct: {task}\nQuery: {query}"
            task = instruction or self.default_instruction
            formatted_query = f"Instruct: {task}\nQuery: {query}"
        else:
            # DiTy/BGE format: raw query
            if instruction:
                logger.warning("This reranker doesn't support dynamic instructions, ignoring")
            formatted_query = query

        documents = [
            doc.page_content if hasattr(doc, "page_content") else str(doc) for doc, _ in candidates
        ]

        response = self._post(
            "/rerank",
            {"query": formatted_query, "documents": documents, "top_k": top_k},
        )

        scores = response["scores"]

        # Apply metadata boosts if provided
        scored: list[tuple[Any, float]] = []
        for (doc, _), score in zip(candidates, scores, strict=False):
            boost = 0.0
            if metadata_boost_weights and hasattr(doc, "metadata"):
                meta = getattr(doc, "metadata", {})
                if meta.get("tags") and metadata_boost_weights.get("tag_match"):
                    boost += metadata_boost_weights["tag_match"]
                if meta.get("has_code") and metadata_boost_weights.get("code_presence"):
                    boost += metadata_boost_weights["code_presence"]
                if meta.get("section_heading") and metadata_boost_weights.get("section_match"):
                    boost += metadata_boost_weights["section_match"]
            final_score = float(score) * (1.0 + boost)
            scored.append((doc, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def create_reranker(settings) -> Reranker:
    """Factory creates appropriate reranker based on model slug and provider type.

    Args:
        settings: Application settings with reranker_provider_type and reranker_model fields

    Returns:
        Configured reranker instance

    Raises:
        ValueError: If unknown provider or model specified
    """
    provider = settings.reranker_provider_type.lower()
    model_slug = settings.reranker_model

    logger.info(f"Creating reranker: provider={provider}, model={model_slug}")

    # Get model metadata from registry (case-insensitive lookup)
    registry = ModelRegistry()
    model_data = registry.get_model(model_slug)
    canonical_slug = model_data["canonical_slug"]

    # Get provider-specific configuration
    provider_config = registry.get_provider_config(canonical_slug, provider)

    if provider == "direct":
        # Direct CrossEncoder; device from model registry (YAML)
        device = provider_config.get("device", "auto")
        batch_size = provider_config.get("batch_size", 16)
        return CrossEncoderReranker(
            model_name=canonical_slug,
            batch_size=batch_size,
            device=device,
        )

    elif provider == "infinity":
        # Infinity HTTP server - use endpoint from settings or default
        endpoint = settings.infinity_reranker_endpoint

        config = ServerRerankerConfig(
            type="server",
            endpoint=endpoint,
            default_instruction=provider_config.get("default_instruction"),
        )
        return InfinityReranker(config)

    elif provider == "openrouter":
        # OpenRouter reranker API (when available)
        raise NotImplementedError(
            "OpenRouter reranker support is not yet implemented. Use infinity provider for now."
        )

    else:
        raise ValueError(f"Unknown reranker provider: {provider}. Supported: direct, infinity")


# Legacy function for backward compatibility
def build_reranker(prioritized_models: list[dict[str, Any]], device: str = "auto") -> Any:
    """Return first available cross-encoder; fallback to identity reranker.

    Args:
        prioritized_models: List of model configs to try in order
        device: Device to run the model on ('auto', 'cpu', or 'cuda').
                'auto' will detect and use GPU if available, else CPU.

    Returns:
        Reranker instance
    """
    for cfg in prioritized_models:
        try:
            return CrossEncoderReranker(
                cfg["model_name"],
                batch_size=cfg.get("batch_size", 16),
                device=cfg.get("device", device),
            )
        except Exception:  # noqa: BLE001
            continue
    return IdentityReranker()
