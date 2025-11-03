"""Direct sentence-transformers FRIDA embedder with prefixes."""
from __future__ import annotations

import logging
import threading

from sentence_transformers import SentenceTransformer

from rag_engine.utils.disk_space import check_disk_space_available, get_huggingface_cache_dir

logger = logging.getLogger(__name__)

# FRIDA model size estimate in GB (conservative estimate including temporary download space)
FRIDA_MODEL_SIZE_GB = 4.0

# Serialize model initialization to avoid meta-tensor race across threads
_frida_init_lock = threading.Lock()


class FRIDAEmbedder:
    """FRIDA embeddings with explicit prefix support for RAG."""

    def __init__(
        self,
        model_name: str = "ai-forever/FRIDA",
        device: str = "cpu",
        max_seq_length: int = 512,
        check_disk_space: bool = True,
    ):
        """Initialize FRIDA embedder.

        Args:
            model_name: Name of the SentenceTransformer model
            device: Device to run the model on ('cpu' or 'cuda')
            max_seq_length: Maximum sequence length
            check_disk_space: Whether to check disk space before downloading model

        Raises:
            OSError: If insufficient disk space for model download
        """
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
                    f"2. Clear HuggingFace cache: python -c \"from huggingface_hub import scan_cache_dir; "
                    f"cache_info = scan_cache_dir(); print(cache_info.delete_revisions([], min_size=1024**3*2))\"\n"
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
                    logger.info(
                        f"Embedder loaded on CPU. Dimension: {self.get_embedding_dim()}"
                    )
            else:
                raise

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query using search_query prefix."""
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
