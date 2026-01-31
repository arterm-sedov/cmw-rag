"""Unified embedding provider with factory pattern.

Supports three provider types:
1. Direct: sentence-transformers (FRIDA)
2. Server: Infinity HTTP API (FRIDA, Qwen3)
3. API: OpenRouter cloud API (Qwen3)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Protocol

from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rag_engine.config.schemas import (
    ApiEmbeddingConfig,
    DirectEmbeddingConfig,
    ModelRegistry,
    ServerEmbeddingConfig,
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

    def embed_query(self, query: str, instruction: Optional[str] = None) -> list[float]:
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

    def embed_query(self, query: str, instruction: Optional[str] = None) -> list[float]:
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
        return self.model.get_sentence_embedding_dimension()


class HTTPClientMixin:
    """Mixin providing resilient HTTP client with retries and timeouts."""

    def __init__(self, endpoint: str, timeout: float = 60.0, max_retries: int = 3):
        self.endpoint = endpoint
        self.timeout = timeout

        # Setup session with retry strategy
        import requests

        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _post(self, path: str, json_data: dict) -> dict:
        """Make POST request with error handling."""
        import requests

        url = f"{self.endpoint}{path}"
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


class InfinityEmbedder(HTTPClientMixin):
    """FRIDA/Qwen3 via Infinity HTTP server."""

    def __init__(self, config: ServerEmbeddingConfig):
        super().__init__(
            endpoint=config.endpoint,
            timeout=60.0,
            max_retries=3,
        )
        self.query_prefix = config.query_prefix
        self.doc_prefix = config.doc_prefix
        self.default_instruction = config.default_instruction

    def embed_query(self, query: str, instruction: Optional[str] = None) -> list[float]:
        if self.default_instruction:
            # Qwen3 format
            task = instruction or self.default_instruction
            formatted = f"Instruct: {task}\nQuery: {query}"
        else:
            # FRIDA format
            if instruction:
                logger.warning("FRIDA doesn't support dynamic instructions, ignoring")
            formatted = f"{self.query_prefix}{query}"

        response = self._post(
            "/embeddings",
            {"input": [formatted], "model": "auto"},  # Infinity ignores model, uses loaded model
        )
        return response["data"][0]["embedding"]

    def embed_documents(
        self, texts: list[str], batch_size: int = 8, show_progress: bool = True
    ) -> list[list[float]]:
        if self.default_instruction:
            # Qwen3 - documents don't get instruction
            formatted = texts
        else:
            # FRIDA - add prefixes
            formatted = [f"{self.doc_prefix}{t}" for t in texts]

        response = self._post(
            "/embeddings",
            {"input": formatted, "model": "auto"},
        )
        return [d["embedding"] for d in response["data"]]

    def get_embedding_dim(self) -> int:
        """Get embedding dimension from server."""
        # Make a test request to get dimension
        test_embedding = self.embed_query("test")
        return len(test_embedding)


class OpenRouterEmbedder:
    """Qwen3 via OpenRouter API."""

    def __init__(self, config: ApiEmbeddingConfig):
        self.client = OpenAI(
            base_url=config.endpoint,
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        )
        self.model = config.model
        self.default_instruction = config.default_instruction

    def embed_query(self, query: str, instruction: Optional[str] = None) -> list[float]:
        # Dynamic instruction support!
        task = instruction or self.default_instruction
        formatted = f"Instruct: {task}\nQuery: {query}"
        response = self.client.embeddings.create(model=self.model, input=formatted)
        return response.data[0].embedding

    def embed_documents(
        self, texts: list[str], batch_size: int = 8, show_progress: bool = True
    ) -> list[list[float]]:
        # Documents don't get instruction
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in response.data]

    def get_embedding_dim(self) -> int:
        """Get embedding dimension from API."""
        # Make a test request to get dimension
        test_embedding = self.embed_query("test")
        return len(test_embedding)


def create_embedder(settings) -> Embedder:
    """Factory creates appropriate embedder based on model slug and provider type.

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

    # Get provider-specific configuration
    provider_config = registry.get_provider_config(canonical_slug, provider)

    if provider == "direct":
        # Direct sentence-transformers; device from model registry (YAML)
        device = provider_config.get("device", "auto")
        max_seq_length = provider_config.get("max_seq_length", 512)
        return FRIDAEmbedder(
            model_name=canonical_slug,
            device=device,
            max_seq_length=max_seq_length,
        )

    elif provider == "infinity":
        # Infinity HTTP server - use endpoint from settings
        endpoint = settings.infinity_embedding_endpoint

        config = ServerEmbeddingConfig(
            type="server",
            endpoint=endpoint,
            query_prefix=provider_config.get("query_prefix"),
            doc_prefix=provider_config.get("doc_prefix"),
            default_instruction=provider_config.get("default_instruction"),
        )
        return InfinityEmbedder(config)

    elif provider == "openrouter":
        # OpenRouter API
        endpoint = settings.openrouter_endpoint
        model_id = provider_config.get("model_id", canonical_slug.lower())
        default_instruction = provider_config.get(
            "default_instruction",
            "Given a web search query, retrieve relevant passages that answer the query",
        )

        config = ApiEmbeddingConfig(
            type="api",
            endpoint=endpoint,
            model=model_id,
            default_instruction=default_instruction,
            timeout=settings.embedding_timeout,
            max_retries=settings.embedding_max_retries,
        )
        return OpenRouterEmbedder(config)

    else:
        raise ValueError(
            f"Unknown embedder provider: {provider}. Supported: direct, infinity, openrouter"
        )
