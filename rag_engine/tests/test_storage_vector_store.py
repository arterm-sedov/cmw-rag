from __future__ import annotations

from pathlib import Path

from rag_engine.storage.vector_store import ChromaStore


def test_chroma_store_add_and_query(tmp_path):
    persist_dir = tmp_path / "chroma"
    store = ChromaStore(persist_dir=str(persist_dir), collection_name="test_collection")

    texts = ["First document", "Second document"]
    metadatas = [{"kbId": "doc1"}, {"kbId": "doc2"}]
    embeddings = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]

    store.add(texts=texts, metadatas=metadatas, ids=["1", "2"], embeddings=embeddings)

    results = store.similarity_search(query_embedding=[0.1, 0.0, 0.0], k=1)

    assert len(results) == 1
    assert results[0].metadata["kbId"] == "doc1"
    assert Path(store.persist_dir).exists()

