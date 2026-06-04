"""Tests for direct kbId article fetch tool."""

from __future__ import annotations

import json
import sys

# Trigger module import so sys.modules has the real module (not the __init__.py tool re-export).
from rag_engine.tools import retrieve_context as _trigger  # noqa: F401

MODULE = "rag_engine.tools.retrieve_context"


def _tc():
    return sys.modules[MODULE]


class TestFetchArticlesByKbIds:
    def test_returns_single_article(self, monkeypatch):
        """Direct fetch returns one article in standard JSON format."""
        tc = _tc()
        from rag_engine.storage.vector_store import ChromaStore

        async def fake_get_by_kb_id(self, kb_id: str):  # noqa: ARG001
            return {
                "kbId": kb_id,
                "source_file": f"/corpus/{kb_id}.md",
                "title": f"Title {kb_id}",
            }

        def fake_read_article(source_file: str):  # noqa: ARG001
            return f"Content of {source_file}"

        monkeypatch.setattr(ChromaStore, "get_by_kb_id_async", fake_get_by_kb_id)
        monkeypatch.setattr(tc, "_read_article", fake_read_article)
        monkeypatch.setattr(tc, "get_collection_name", lambda v: "mkdocs_kb")

        result = asyncio_run(tc._fetch_articles_by_kb_ids_core(["123"], "v6"))
        data = json.loads(result)

        assert len(data["articles"]) == 1
        assert data["articles"][0]["kb_id"] == "123"
        assert data["articles"][0]["title"] == "Title 123"
        assert "Content of" in data["articles"][0]["content"]

    def test_returns_multiple_articles(self, monkeypatch):
        """Direct fetch returns multiple articles in order."""
        tc = _tc()
        from rag_engine.storage.vector_store import ChromaStore

        async def fake_get_by_kb_id(self, kb_id: str):  # noqa: ARG001
            return {
                "kbId": kb_id,
                "source_file": f"/corpus/{kb_id}.md",
                "title": f"T{kb_id}",
            }

        def fake_read_article(source_file: str):  # noqa: ARG001
            return f"Body of {source_file}"

        monkeypatch.setattr(ChromaStore, "get_by_kb_id_async", fake_get_by_kb_id)
        monkeypatch.setattr(tc, "_read_article", fake_read_article)
        monkeypatch.setattr(tc, "get_collection_name", lambda v: "mkdocs_kb")

        result = asyncio_run(tc._fetch_articles_by_kb_ids_core(["111", "222"], "v6"))
        data = json.loads(result)

        assert len(data["articles"]) == 2
        assert data["articles"][0]["kb_id"] == "111"
        assert data["articles"][1]["kb_id"] == "222"

    def test_skips_missing_kb_ids(self, monkeypatch):
        """Missing kbIds are skipped gracefully, returning only found articles."""
        tc = _tc()
        from rag_engine.storage.vector_store import ChromaStore

        async def fake_get_by_kb_id(self, kb_id: str):  # noqa: ARG001
            if kb_id == "missing":
                return None
            return {
                "kbId": kb_id,
                "source_file": f"/corpus/{kb_id}.md",
                "title": f"T{kb_id}",
            }

        def fake_read_article(source_file: str):  # noqa: ARG001
            return "Body"

        monkeypatch.setattr(ChromaStore, "get_by_kb_id_async", fake_get_by_kb_id)
        monkeypatch.setattr(tc, "_read_article", fake_read_article)
        monkeypatch.setattr(tc, "get_collection_name", lambda v: "mkdocs_kb")

        result = asyncio_run(
            tc._fetch_articles_by_kb_ids_core(["exists", "missing"], "v6")
        )
        data = json.loads(result)

        assert len(data["articles"]) == 1
        assert data["articles"][0]["kb_id"] == "exists"

    def test_uses_version_collection(self, monkeypatch):
        """v5 fetch creates a store with the v5 collection name."""
        tc = _tc()
        captured_collection: list[str] = []

        class FakeStore:
            def __init__(self, *, collection_name: str, host=None, port=None):  # noqa: ARG002
                captured_collection.append(collection_name)

            async def get_by_kb_id_async(self, kb_id: str) -> dict | None:
                return {"kbId": "123", "source_file": "/f.md", "title": "T"}

        def fake_read_article(source_file: str):  # noqa: ARG001
            return "Body"

        monkeypatch.setattr(tc, "ChromaStore", FakeStore)
        monkeypatch.setattr(tc, "_read_article", fake_read_article)
        monkeypatch.setattr(tc, "get_collection_name", lambda v: f"coll_{v}")

        asyncio_run(tc._fetch_articles_by_kb_ids_core(["123"], "v5"))
        assert "coll_v5" in captured_collection

        captured_collection.clear()
        asyncio_run(tc._fetch_articles_by_kb_ids_core(["123"], "v6"))
        assert "coll_v6" in captured_collection


def asyncio_run(coro):
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
