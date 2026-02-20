"""Unified embedding provider with factory pattern.

Supports three provider types:
1. Direct: sentence-transformers (FRIDA)
2. OpenAI-compatible: Infinity, Mosec, OpenRouter HTTP APIs
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import requests
from openai import OpenAI

from rag_engine.config.schemas import (
    ModelRegistry,
    OpenAIEmbeddingConfig,
)
from rag_engine.utils.device_utils import detect_device
from rag_engine.utils.disk_space import check_disk_space_available, get_huggingface_cache_dir

logger = logging.getLogger(__name__)

# FRIDA model size estimate in GB (conservative estimate including temporary download space)
FRIDA_MODEL_SIZE_GB = 4.0

# Serialize model initialization to avoid meta-tensor race across threads
_frida_init_lock = threading.Lock()


class Embedder(Protocol):
    """Unified interface for all embedding providers."""

    def embed_query(self, query: str, instruction: str | None = None) -> list[float]:
        """
        Embed a single query.

        Args:
            query: The search query text
            instruction: Optional custom instruction (Qwen3 only, overrides default)

        Returns:
            Embedding vector as list of floats
        """
        ...

    def embed_documents(
        self, texts: list[str], batch_size: int = 8, show_progress: bool = True
    ) -> list[list[float]]:
        """Embed multiple documents."""
        ...

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        ...


class FRIDAEmbedder:
    """FRIDA embeddings via sentence-transformers (direct)."""

    def __init__(
        self,
        model_name: str = "ai-forever/FRIDA",
        device: str = "auto",
        max_seq_length: int = 512,
        check_disk_space: bool = True,
    ):
        """Initialize FRIDA embedder.

        Args:
            model_name: Name of the SentenceTransformer model
            device: Device to run the model on ('auto', 'cpu', or 'cuda').
                    'auto' will detect and use GPU if available, else CPU.
            max_seq_length: Maximum sequence length
            check_disk_space: Whether to check disk space before downloading model

        Raises:
            OSError: If insufficient disk space for model download
        """
        # Lazy import to avoid heavy dependencies when not needed
        from sentence_transformers import SentenceTransformer

        # Auto-detect device if "auto" is specified
        if device == "auto":
            device = detect_device("auto")
        if check_disk_space and model_name == "ai-forever/FRIDA":
            cache_dir = get_huggingface_cache_dir()
            available, free_gb, message = check_disk_space_available(
                required_gb=FRIDA_MODEL_SIZE_GB,
                cache_dir=cache_dir,
            )
            logger.info(message)
            if not available:
                raise OSError(
                    f"[Errno 28] No space left on device. {message}\n"
                    f"Solutions:\n"
                    f"1. Free up disk space (need at least {FRIDA_MODEL_SIZE_GB * 1.2:.2f} GB)\n"
                    f'2. Clear HuggingFace cache: python -c "from huggingface_hub import scan_cache_dir; '
                    f'cache_info = scan_cache_dir(); print(cache_info.delete_revisions([], min_size=1024**3*2))"\n'
                    f"3. Set cache directory to a different drive with more space:\n"
                    f"   export HF_HOME=/path/to/larger/drive/.cache/huggingface\n"
                    f"   or on Windows: set HF_HOME=D:\\path\\to\\larger\\drive\\.cache\\huggingface"
                )

        try:
            with _frida_init_lock:
                logger.info(f"Loading embedder: {model_name} on {device}")
                self.model = SentenceTransformer(model_name, device=device)
                self.model.max_seq_length = max_seq_length
                logger.info(f"Embedder loaded. Dimension: {self.get_embedding_dim()}")
        except OSError as e:
            if "No space left on device" in str(e) or "[Errno 28]" in str(e):
                cache_dir = get_huggingface_cache_dir()
                available, free_gb, _ = check_disk_space_available(
                    FRIDA_MODEL_SIZE_GB, cache_dir=cache_dir
                )
                raise OSError(
                    f"Disk space error during model download: {e}\n"
                    f"Required space: ~{FRIDA_MODEL_SIZE_GB} GB\n"
                    f"Free space: {free_gb:.2f} GB\n"
                    f"See error message above for solutions."
                ) from e
            raise
        except NotImplementedError as e:
            # Workaround for PyTorch meta tensor device move errors on some installs
            if "Cannot copy out of meta tensor" in str(e):
                logger.warning(
                    "Encountered meta tensor move error; reloading FRIDA on CPU fallback"
                )
                with _frida_init_lock:
                    self.model = SentenceTransformer(model_name, device="cpu")
                    self.model.max_seq_length = max_seq_length
                    logger.info(f"Embedder loaded on CPU. Dimension: {self.get_embedding_dim()}")
            else:
                raise

    def embed_query(self, query: str, instruction: str | None = None) -> list[float]:
        """Embed a search query using search_query prefix."""
        if instruction:
            logger.warning("FRIDA doesn't support dynamic instructions, ignoring")
        return self.model.encode(
            query,
            prompt_name="search_query",
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    def embed_documents(
        self, texts: list[str], batch_size: int = 8, show_progress: bool = True
    ) -> list[list[float]]:
        """Embed documents using search_document prefix."""
        embeddings = self.model.encode(
            texts,
            prompt_name="search_document",
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        dim: int = self.model.get_sentence_embedding_dimension() or 0
        if dim == 0:
            raise RuntimeError("Could not determine embedding dimension")
        return dim


class Qwen3DirectEmbedder:
    """Qwen3 embeddings via HuggingFace transformers (Direct GPU).

    Uses AutoModel/AutoTokenizer with last-token pooling.
    Supports instruction-based query formatting.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_seq_length: int = 8192,
        default_instruction: str | None = None,
    ):
        """Initialize Qwen3 embedder.

        Args:
            model_name: HuggingFace model path (e.g., "Qwen/Qwen3-Embedding-0.6B")
            device: Device to run on ('auto', 'cpu', 'cuda')
            max_seq_length: Maximum sequence length
            default_instruction: Default instruction for query formatting
        """
        # Lazy imports to avoid heavy dependencies when not needed
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.max_seq_length = max_seq_length
        self.default_instruction = default_instruction

        logger.info(f"Loading Qwen3 embedder: {model_name} on {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()

        logger.info(f"Qwen3 embedder loaded. Dimension: {self.get_embedding_dim()}")

    def _format_query(self, query: str, instruction: str | None = None) -> str:
        """Format query with instruction."""
        if self.default_instruction or instruction:
            task = instruction or self.default_instruction
            return f"Instruct: {task}\nQuery: {query}"
        return query

    def _compute_embedding(self, text: str) -> list[float]:
        """Compute embedding for single text."""
        import torch

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state

            # Last-token pooling (Qwen3 uses this)
            attention_mask = inputs["attention_mask"]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            last_idx = sequence_lengths[0].item()
            embedding = hidden_states[0, last_idx, :]

            # Normalize
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

        return embedding.cpu().numpy().tolist()

    def embed_query(self, query: str, instruction: str | None = None) -> list[float]:
        """Embed a search query with instruction formatting."""
        formatted = self._format_query(query, instruction)
        return self._compute_embedding(formatted)

    def embed_documents(
        self, texts: list[str], batch_size: int = 8, show_progress: bool = True
    ) -> list[list[float]]:
        """Embed multiple documents (no instruction formatting)."""
        embeddings = []
        for i, text in enumerate(texts):
            embeddings.append(self._compute_embedding(text))
            if show_progress and (i + 1) % 10 == 0:
                logger.debug(f"Embedded {i + 1}/{len(texts)} documents")
        return embeddings

    def get_embedding_dim(self) -> int:
        """Get embedding dimension from model config."""
        return self.model.config.hidden_size


class OpenAICompatibleEmbedder:
    """Embedder for OpenAI-compatible APIs.

    Uses direct HTTP for local providers (3x faster).
    Uses OpenAI SDK for remote providers (auth + retries).
    """

    def __init__(self, config: OpenAIEmbeddingConfig):
        self.config = config
        self._client: OpenAI | None = None  # Lazy load for remote only

    @property
    def client(self) -> OpenAI:
        """Lazy-load OpenAI client for remote providers."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.config.endpoint,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def _format_query(self, query: str, instruction: str | None = None) -> str:
        """Format query based on model type."""
        if self.config.query_prefix:
            if instruction:
                logger.warning("FRIDA doesn't support dynamic instructions, ignoring")
            return f"{self.config.query_prefix}{query}"
        elif self.config.default_instruction:
            task = instruction or self.config.default_instruction
            return f"Instruct: {task}\nQuery: {query}"
        return query

    def _format_documents(self, texts: list[str]) -> list[str]:
        """Format documents based on model type."""
        if self.config.doc_prefix:
            return [f"{self.config.doc_prefix}{t}" for t in texts]
        return texts

    def embed_query(self, query: str, instruction: str | None = None) -> list[float]:
        formatted = self._format_query(query, instruction)
        try:
            if self.config.local:
                return self._embed_local(formatted)
            else:
                return self._embed_remote(formatted)
        except Exception as e:
            logger.error(
                f"Embedding error from {self.config.endpoint} (model={self.config.model}): {e}"
            )
            raise

    def embed_documents(
        self, texts: list[str], batch_size: int = 8, show_progress: bool = True
    ) -> list[list[float]]:
        formatted = self._format_documents(texts)
        try:
            if self.config.local:
                return self._embed_documents_local(formatted)
            else:
                return self._embed_documents_remote(formatted)
        except Exception as e:
            logger.error(
                f"Embedding error from {self.config.endpoint} (model={self.config.model}): {e}"
            )
            raise

    def _embed_local(self, text: str) -> list[float]:
        """Direct HTTP request for local providers."""
        resp = requests.post(
            self.config.endpoint,
            json={"input": text, "model": self.config.model},
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def _embed_remote(self, text: str) -> list[float]:
        """OpenAI SDK for remote providers."""
        response = self.client.embeddings.create(model=self.config.model, input=text)
        return response.data[0].embedding

    def _embed_documents_local(self, texts: list[str]) -> list[list[float]]:
        """Direct HTTP for local providers."""
        resp = requests.post(
            self.config.endpoint,
            json={"input": texts, "model": self.config.model},
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]

    def _embed_documents_remote(self, texts: list[str]) -> list[list[float]]:
        """OpenAI SDK for remote providers."""
        response = self.client.embeddings.create(model=self.config.model, input=texts)
        return [d.embedding for d in response.data]

    def get_embedding_dim(self) -> int:
        """Return dimension from config (no test request)."""
        return self.config.dimensions


def create_embedder(settings) -> Embedder:
    """Factory creates embedder from .env + models.yaml.

    Args:
        settings: Application settings with embedding_provider_type and embedding_model fields

    Returns:
        Configured embedder instance

    Raises:
        ValueError: If unknown provider or model specified
    """
    provider = settings.embedding_provider_type.lower()
    model_slug = settings.embedding_model

    logger.info(f"Creating embedder: provider={provider}, model={model_slug}")

    # Get model metadata from registry (case-insensitive lookup)
    registry = ModelRegistry()
    model_data = registry.get_model(model_slug)
    canonical_slug = model_data["canonical_slug"]
    dimensions_raw = model_data.get("dimensions")
    assert dimensions_raw is not None, f"Missing dimensions for model: {model_slug}"
    dimensions: int = int(dimensions_raw)

    # Get provider-specific configuration
    provider_config = registry.get_provider_config(canonical_slug, provider)

    if provider == "direct":
        # Direct inference: FRIDA (sentence-transformers) or Qwen3 (transformers)
        device = provider_config.get("device", "auto")
        max_seq_length = provider_config.get("max_seq_length", 512)

        # Detect Qwen3 models by slug pattern
        is_qwen3 = "qwen3" in canonical_slug.lower() and "embedding" in canonical_slug.lower()

        if is_qwen3:
            # Qwen3 uses transformers with last-token pooling
            return Qwen3DirectEmbedder(
                model_name=canonical_slug,
                device=device,
                max_seq_length=max_seq_length,
                default_instruction=provider_config.get("default_instruction"),
            )
        else:
            # FRIDA and other sentence-transformers models
            return FRIDAEmbedder(
                model_name=canonical_slug,
                device=device,
                max_seq_length=max_seq_length,
            )

    # Map provider to endpoint/model/api_key
    if provider == "mosec":
        endpoint = settings.mosec_embedding_endpoint
        model = canonical_slug
        api_key = None
    elif provider == "infinity":
        endpoint = settings.infinity_embedding_endpoint
        model = "auto"
        api_key = None
    elif provider == "vllm":
        endpoint = settings.vllm_embedding_endpoint
        model = canonical_slug
        api_key = None
    elif provider == "openrouter":
        endpoint = settings.openrouter_endpoint
        model = provider_config.get("model_id", canonical_slug.lower())
        api_key = settings.openrouter_api_key
    else:
        raise ValueError(
            f"Unknown embedder provider: {provider}. Supported: direct, infinity, mosec, vllm, openrouter"
        )

    config = OpenAIEmbeddingConfig(
        type="openai_compatible",
        provider=provider,
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        dimensions=dimensions,
        local=settings.embedding_local,
        query_prefix=provider_config.get("query_prefix"),
        doc_prefix=provider_config.get("doc_prefix"),
        default_instruction=provider_config.get("default_instruction"),
        timeout=settings.embedding_timeout,
        max_retries=settings.embedding_max_retries,
    )
    return OpenAICompatibleEmbedder(config)
