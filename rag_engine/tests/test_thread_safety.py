from __future__ import annotations

import importlib
import threading
import time


def test_retriever_singleton_serialized(monkeypatch):
    """Concurrent first calls to _get_or_create_retriever must create exactly one instance."""
    # Import module fresh to reset globals
    tool_mod = importlib.import_module("rag_engine.tools.retrieve_context")

    # Ensure starting from a clean state (no app-injected retriever, no lazy instance)
    if getattr(tool_mod, "_app_retriever", None) is not None:
        tool_mod._app_retriever = None  # type: ignore[attr-defined]
    if getattr(tool_mod, "_retriever", None) is not None:
        tool_mod._retriever = None  # type: ignore[attr-defined]

    # Count how many times RAGRetriever is constructed
    construct_count = {"n": 0}

    class FakeRetriever:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            # Simulate slow construction to magnify race window
            time.sleep(0.05)
            construct_count["n"] += 1

    # Minimal fakes for dependencies used by _get_or_create_retriever
    # Patch the symbol imported into the tool module (not the source module)
    monkeypatch.setattr(tool_mod, "RAGRetriever", FakeRetriever)
    monkeypatch.setattr("rag_engine.retrieval.embedder.FRIDAEmbedder", lambda *a, **k: object())
    monkeypatch.setattr("rag_engine.storage.vector_store.ChromaStore", lambda *a, **k: object())
    monkeypatch.setattr("rag_engine.llm.llm_manager.LLMManager", lambda *a, **k: object())

    # Call _get_or_create_retriever concurrently
    results: list[object] = []

    def worker() -> None:
        r = tool_mod._get_or_create_retriever()
        results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one construction should have happened
    assert construct_count["n"] == 1
    # All calls returned the same instance
    assert len(set(map(id, results))) == 1


