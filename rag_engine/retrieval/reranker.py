"""Unified reranker provider with factory pattern.

Supports two provider types:
1. Direct: sentence-transformers CrossEncoder (DiTy, BGE)
2. Server: Infinity/Mosec HTTP API (DiTy, BGE, Qwen3)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Optional, Protocol

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rag_engine.config.schemas import ModelRegistry, RerankerFormatting, ServerRerankerConfig
from rag_engine.utils.device_utils import detect_device

logger = logging.getLogger(__name__)

try:  # Provide a module-level alias for monkeypatch-friendly tests
    from sentence_transformers import CrossEncoder as CrossEncoder  # type: ignore
except Exception:  # noqa: BLE001
    CrossEncoder = None  # type: ignore


class HTTPClientMixin:
    """Mixin providing resilient HTTP client with retries and timeouts."""

    def __init__(self, endpoint: str, timeout: float = 60.0, max_retries: int = 3):
        self.endpoint = endpoint
        self.timeout = timeout

        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _post(self, json_data: dict) -> dict:
        """Make POST request with error handling."""
        url = self.endpoint
        try:
            response = self.session.post(url, json=json_data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Request to {url} timed out after {self.timeout}s")
            raise RuntimeError(f"Server at {url} not responding")
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to {url}")
            raise RuntimeError(f"Server at {url} is not running")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from {url}: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Server returned error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected error calling {url}: {e}")
            raise


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
    """DiTy/BGE/Qwen3 via Infinity/Mosec HTTP server.

    DEPRECATED: Use RerankerAdapter instead for vLLM/Cohere compatible endpoints.
    This class uses old /v1/rerank format with {scores: [...]} response.
    """

    def __init__(self, config: ServerRerankerConfig):
        super().__init__(
            endpoint=config.endpoint,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        self.config = config

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Any, float]],
        top_k: int,
        metadata_boost_weights: Optional[dict[str, float]] = None,
        instruction: Optional[str] = None,
    ) -> list[tuple[Any, float]]:
        """Rerank candidates via HTTP server (old format)."""
        if self.config.default_instruction:
            task = instruction or self.config.default_instruction
            formatted_query = f"Instruct: {task}\nQuery: {query}"
        else:
            if instruction:
                logger.warning("This reranker doesn't support dynamic instructions, ignoring")
            formatted_query = query

        documents = [
            doc.page_content if hasattr(doc, "page_content") else str(doc) for doc, _ in candidates
        ]

        response = self._post(
            {"query": formatted_query, "documents": documents, "top_k": top_k},
        )

        scores = response["scores"]

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


class RerankerAdapter(HTTPClientMixin):
    """Client-side formatting adapter for server rerankers.

    Handles both cross-encoder (no formatting) and LLM reranker (formatting required).
    Uses industry-standard vLLM/Cohere contracts:
    - /v1/score: returns {data: [{index, object, score}, ...]} (vLLM format)
    - /v1/rerank: returns {results: [{index, document, relevance_score}, ...]} (Cohere format)
    """

    def __init__(self, config: ServerRerankerConfig):
        # Derive base URL from endpoint (e.g., http://localhost:7998/v1/rerank -> http://localhost:7998)
        base_url = (
            config.endpoint.rsplit("/v1/", 1)[0] if "/v1/" in config.endpoint else config.endpoint
        )
        super().__init__(
            endpoint=base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        self.config = config

    def format_query(self, query: str, instruction: str | None = None) -> str:
        """Format query based on reranker type.

        Cross-encoder: raw query (no transformation)
        LLM reranker: apply template with prefix/instruction
        """
        if self.config.reranker_type == "cross_encoder":
            if instruction:
                logger.warning("Cross-encoder doesn't support instructions, ignoring")
            return query

        if not self.config.formatting:
            return query

        fmt = self.config.formatting
        task = instruction or self.config.default_instruction or ""

        return fmt.query_template.format(
            prefix=fmt.prefix,
            instruction=task,
            query=query,
        )

    def format_document(self, doc: str) -> str:
        """Format document based on reranker type.

        Cross-encoder: raw document
        LLM reranker: apply template with suffix/prompt
        """
        if self.config.reranker_type == "cross_encoder":
            return doc

        if not self.config.formatting:
            return doc

        return self.config.formatting.doc_template.format(
            doc=doc,
            suffix=self.config.formatting.suffix,
            prompt=self.config.formatting.prompt or "",
        )

    def score(
        self, query: str, documents: list[str], instruction: str | None = None
    ) -> list[float]:
        """Get raw scores via /v1/score endpoint.

        Returns scores in original document order (vLLM format).
        """
        formatted_query = self.format_query(query, instruction)
        formatted_docs = [self.format_document(d) for d in documents]

        # Use /v1/score endpoint
        url = f"{self.endpoint}/v1/score"
        try:
            response = self.session.post(
                url,
                json={"query": formatted_query, "documents": formatted_docs},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()["data"]
        except requests.exceptions.Timeout:
            logger.error(f"Request to {url} timed out after {self.timeout}s")
            raise RuntimeError(f"Server at {url} not responding")
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to {url}")
            raise RuntimeError(f"Server at {url} is not running")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from {url}: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Server returned error: {e.response.status_code}")

        sorted_data = sorted(data, key=lambda x: x["index"])
        return [item["score"] for item in sorted_data]

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Any, float]],
        top_k: int,
        metadata_boost_weights: Optional[dict[str, float]] = None,
        instruction: Optional[str] = None,
    ) -> list[tuple[Any, float]]:
        """Rerank candidates using /v1/rerank endpoint.

        Returns (document, score) tuples sorted by relevance.
        """
        documents = [
            doc.page_content if hasattr(doc, "page_content") else str(doc) for doc, _ in candidates
        ]

        formatted_query = self.format_query(query, instruction)
        formatted_docs = [self.format_document(d) for d in documents]

        payload: dict = {"query": formatted_query, "documents": formatted_docs}
        if top_k:
            payload["top_n"] = top_k

        # Use /v1/rerank endpoint
        url = f"{self.endpoint}/v1/rerank"
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            results = response.json()["results"]
        except requests.exceptions.Timeout:
            logger.error(f"Request to {url} timed out after {self.timeout}s")
            raise RuntimeError(f"Server at {url} not responding")
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to {url}")
            raise RuntimeError(f"Server at {url} is not running")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from {url}: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Server returned error: {e.response.status_code}")

        scored: list[tuple[Any, float]] = []
        for result in results:
            idx = result["index"]
            doc, _ = candidates[idx]
            score = result["relevance_score"]

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

    # Get model-level configuration
    model_instruction = registry.get_default_instruction(canonical_slug)
    reranker_type = model_data.get("reranker_type", "cross_encoder")

    # Get formatting config for LLM rerankers
    formatting_data = model_data.get("formatting")
    formatting = None
    if formatting_data:
        formatting = RerankerFormatting(**formatting_data)

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

    # Map provider to endpoint
    if provider == "infinity":
        endpoint = settings.infinity_reranker_endpoint
    elif provider == "mosec":
        endpoint = settings.mosec_reranker_endpoint
    elif provider == "openrouter":
        raise NotImplementedError(
            "OpenRouter reranker support is not yet implemented. Use infinity or mosec provider for now."
        )
    else:
        raise ValueError(
            f"Unknown reranker provider: {provider}. Supported: direct, infinity, mosec"
        )

    config = ServerRerankerConfig(
        type="server",
        provider=provider,
        endpoint=endpoint,
        reranker_type=reranker_type,
        formatting=formatting,
        default_instruction=model_instruction,
        timeout=settings.reranker_timeout,
        max_retries=settings.reranker_max_retries,
    )
    return RerankerAdapter(config)


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
