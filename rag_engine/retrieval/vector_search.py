"""Vector search wrapper."""
from __future__ import annotations

from typing import Any, List


def top_k_search(store, embedding: List[float], k: int) -> List[Any]:
    """Return top-k results from the vector store by query embedding."""
    return store.similarity_search(query_embedding=embedding, k=k)


