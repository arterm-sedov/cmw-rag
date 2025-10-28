"""Direct sentence-transformers FRIDA embedder with prefixes."""
from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer


class FRIDAEmbedder:
    """FRIDA embeddings with explicit prefix support for RAG."""

    def __init__(
        self,
        model_name: str = "ai-forever/FRIDA",
        device: str = "cpu",
        max_seq_length: int = 512,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(
            query,
            prompt_name="search_query",
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    def embed_documents(
        self, texts: List[str], batch_size: int = 8, show_progress: bool = True
    ) -> List[List[float]]:
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
        return self.model.get_sentence_embedding_dimension()


