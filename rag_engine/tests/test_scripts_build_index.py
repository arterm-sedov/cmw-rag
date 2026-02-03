"""Tests for build_index script (async version)."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest


def test_build_index_help(monkeypatch):
    """Test build_index --help works with async entry point."""
    module = importlib.import_module("rag_engine.scripts.build_index")
    argv = sys.argv
    monkeypatch.setattr(sys, "argv", ["build_index.py", "--help"])

    with pytest.raises(SystemExit) as exc:
        # Use the async entry point
        asyncio.run(module.run_async())

    assert exc.value.code == 0
    monkeypatch.setattr(sys, "argv", argv)


@pytest.mark.asyncio
async def test_build_index_runs_with_fakes(monkeypatch, docs_fixture_path):
    """Test async build_index runs with mocked components."""
    recorded = {}

    class FakeEmbedder:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

    class FakeIndexer:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        async def index_documents_async(self, docs, chunk_size, chunk_overlap, **kwargs):  # noqa: ANN001
            recorded["docs"] = docs
            recorded["chunk_size"] = chunk_size
            recorded["chunk_overlap"] = chunk_overlap
            recorded["kwargs"] = kwargs
            return {
                "total_docs": len(docs),
                "processed_docs": len(docs),
                "new_docs": len(docs),
                "reindexed_docs": 0,
                "skipped_docs": 0,
                "empty_docs": 0,
                "no_chunk_docs": 0,
                "total_chunks_indexed": len(docs),
            }

    module = importlib.reload(importlib.import_module("rag_engine.scripts.build_index"))

    def fake_document_processor(mode):
        def process(source, max_files=None):  # noqa: ARG001
            return [SimpleNamespace(content="text", metadata={})]

        return SimpleNamespace(process=process)

    monkeypatch.setattr(module, "FRIDAEmbedder", FakeEmbedder)
    monkeypatch.setattr(module, "ChromaStore", FakeStore)
    monkeypatch.setattr(module, "RAGIndexer", FakeIndexer)
    monkeypatch.setattr(module, "DocumentProcessor", fake_document_processor)

    monkeypatch.setattr(module.settings, "chromadb_persist_dir", "./tmp")
    monkeypatch.setattr(module.settings, "chromadb_collection", "test")
    monkeypatch.setattr(module.settings, "chunk_size", 500)
    monkeypatch.setattr(module.settings, "chunk_overlap", 150)

    argv = sys.argv
    monkeypatch.setattr(
        sys, "argv", ["build_index.py", "--source", str(docs_fixture_path), "--mode", "folder"]
    )

    await module.run_async()

    assert recorded["chunk_size"] == 500
    assert recorded["chunk_overlap"] == 150
    sys.argv = argv


@pytest.mark.asyncio
async def test_build_index_respects_max_files(monkeypatch, docs_fixture_path):
    """Test async build_index respects max_files parameter."""
    recorded = {}

    class FakeEmbedder:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

    class FakeIndexer:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        async def index_documents_async(self, docs, chunk_size, chunk_overlap, **kwargs):  # noqa: ANN001
            recorded["kwargs"] = kwargs
            recorded["docs_count"] = len(docs)
            return {
                "total_docs": len(docs),
                "processed_docs": len(docs),
                "new_docs": len(docs),
                "reindexed_docs": 0,
                "skipped_docs": 0,
                "empty_docs": 0,
                "no_chunk_docs": 0,
                "total_chunks_indexed": len(docs),
            }

    def fake_document_processor(mode):
        def process(source, max_files=None):  # noqa: ARG001
            recorded["max_files_to_processor"] = max_files
            # Return limited docs based on max_files
            all_docs = [
                SimpleNamespace(content="text", metadata={"kbId": "1"}),
                SimpleNamespace(content="text2", metadata={"kbId": "2"}),
                SimpleNamespace(content="text3", metadata={"kbId": "3"}),
            ]
            if max_files is not None:
                return all_docs[:max_files]
            return all_docs

        return SimpleNamespace(process=process)

    module = importlib.reload(importlib.import_module("rag_engine.scripts.build_index"))

    monkeypatch.setattr(module, "FRIDAEmbedder", FakeEmbedder)
    monkeypatch.setattr(module, "ChromaStore", FakeStore)
    monkeypatch.setattr(module, "RAGIndexer", FakeIndexer)
    monkeypatch.setattr(module, "DocumentProcessor", fake_document_processor)

    argv = sys.argv
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_index.py",
            "--source",
            str(docs_fixture_path),
            "--mode",
            "folder",
            "--max-files",
            "1",
        ],
    )

    await module.run_async()

    # Verify max_files is passed to DocumentProcessor
    assert recorded["max_files_to_processor"] == 1
    # Verify that only 1 doc was passed to indexer (DocumentProcessor limited it)
    assert recorded["docs_count"] == 1
    # Verify max_files is still passed to indexer (as safety check)
    assert recorded["kwargs"].get("max_files") == 1
    sys.argv = argv
