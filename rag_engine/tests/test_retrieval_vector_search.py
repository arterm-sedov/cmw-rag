from __future__ import annotations

from unittest.mock import Mock

from rag_engine.retrieval.vector_search import top_k_search


def test_top_k_search_delegates_to_store():
    store = Mock()
    store.similarity_search.return_value = ["result"]

    results = top_k_search(store, embedding=[0.1, 0.2], k=3)

    store.similarity_search.assert_called_once_with(query_embedding=[0.1, 0.2], k=3)
    assert results == ["result"]

