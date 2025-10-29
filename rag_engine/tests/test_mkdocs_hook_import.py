from __future__ import annotations


def test_import_rag_indexing_hook_module():
    # Importing the compatibility stub should execute its top-level lines
    mod = __import__("rag_engine.rag_indexing_hook", fromlist=["*"])
    assert mod is not None

