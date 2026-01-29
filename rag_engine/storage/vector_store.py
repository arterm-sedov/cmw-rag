"""Chroma vector store wrapper (direct chromadb client)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb


@dataclass
class RetrievedDoc:
    page_content: str
    metadata: dict[str, Any]


class ChromaStore:
    """Thin wrapper around Chroma persistent client for add/query."""

    def __init__(self, persist_dir: str, collection_name: str):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client: chromadb.PersistentClient | None = None
        self._collection = None

    @property
    def collection(self):
        if self._client is None:
            # Use PersistentClient from latest ChromaDB (1.x)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def add(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def similarity_search(self, query_embedding: list[float], k: int = 5) -> list[RetrievedDoc]:
        res = self.collection.query(
            query_embeddings=[query_embedding], n_results=k, include=["documents", "metadatas"]
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [RetrievedDoc(page_content=d, metadata=m) for d, m in zip(docs, metas)]

    # Incremental reindexing helpers
    def get_any_doc_meta(self, where: dict[str, Any]) -> dict[str, Any] | None:
        """Return metadata of any one document matching a metadata filter."""
        res = self.collection.get(where=where, include=["metadatas"], limit=1)
        metas = res.get("metadatas", [])
        if metas:
            return metas[0]
        return None

    def delete_where(self, where: dict[str, Any]) -> None:
        """Delete all records matching a metadata filter."""
        self.collection.delete(where=where)
