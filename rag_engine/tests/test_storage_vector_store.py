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


def test_chroma_store_get_any_and_delete_where(tmp_path):
    persist_dir = tmp_path / "chroma"
    store = ChromaStore(persist_dir=str(persist_dir), collection_name="test_collection")

    texts = ["Doc A", "Doc B"]
    metas = [
        {"kbId": "a", "doc_stable_id": "docA", "file_mtime_epoch": 100},
        {"kbId": "b", "doc_stable_id": "docB", "file_mtime_epoch": 200},
    ]
    ids = ["idA", "idB"]
    embs = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
    store.add(texts=texts, metadatas=metas, ids=ids, embeddings=embs)

    any_meta = store.get_any_doc_meta({"doc_stable_id": "docA"})
    assert any_meta is not None
    assert any_meta.get("kbId") == "a"

    store.delete_where({"doc_stable_id": "docA"})
    any_meta_after = store.get_any_doc_meta({"doc_stable_id": "docA"})
    assert any_meta_after is None
