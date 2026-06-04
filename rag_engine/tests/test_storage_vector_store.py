"""Tests for async ChromaStore."""

from __future__ import annotations

import pytest

from rag_engine.storage.vector_store import ChromaStore


@pytest.mark.asyncio
async def test_chroma_store_add_and_query():
    """Test async add and similarity search."""
    store = ChromaStore(collection_name="test_collection")

    texts = ["First document", "Second document"]
    metadatas = [{"kbId": "doc1"}, {"kbId": "doc2"}]
    embeddings = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]

    await store.add_async(texts=texts, metadatas=metadatas, ids=["1", "2"], embeddings=embeddings)

    results = await store.similarity_search_async(query_embedding=[0.1, 0.0, 0.0], k=1)

    assert len(results) == 1
    assert results[0].metadata["kbId"] == "doc1"


@pytest.mark.asyncio
async def test_chroma_store_get_any_and_delete_where():
    """Test async get_any_doc_meta and delete_where."""
    store = ChromaStore(collection_name="test_collection")

    texts = ["Doc A", "Doc B"]
    metas = [
        {"kbId": "a", "doc_stable_id": "docA", "file_mtime_epoch": 100},
        {"kbId": "b", "doc_stable_id": "docB", "file_mtime_epoch": 200},
    ]
    ids = ["idA", "idB"]
    embs = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
    await store.add_async(texts=texts, metadatas=metas, ids=ids, embeddings=embs)

    any_meta = await store.get_any_doc_meta_async({"doc_stable_id": "docA"})
    assert any_meta is not None
    assert any_meta.get("kbId") == "a"

    await store.delete_where_async({"doc_stable_id": "docA"})
    any_meta_after = await store.get_any_doc_meta_async({"doc_stable_id": "docA"})
    assert any_meta_after is None


@pytest.mark.asyncio
async def test_get_by_kb_id_returns_metadata():
    """Test get_by_kb_id_async returns metadata for a known kbId."""
    store = ChromaStore(collection_name="test_get_by_kbid")

    texts = ["Article content here"]
    metas = [{"kbId": "1234", "source_file": "/corpus/v6/1234-article.md", "title": "Test"}]
    await store.add_async(texts=texts, metadatas=metas, ids=["id1234"], embeddings=[[0.5, 0.0, 0.0]])

    result = await store.get_by_kb_id_async("1234")
    assert result is not None
    assert result["kbId"] == "1234"
    assert result["source_file"] == "/corpus/v6/1234-article.md"


@pytest.mark.asyncio
async def test_get_by_kb_id_returns_none_for_missing():
    """Test get_by_kb_id_async returns None for unknown kbId."""
    store = ChromaStore(collection_name="test_get_by_kbid_missing")
    result = await store.get_by_kb_id_async("nonexistent")
    assert result is None
