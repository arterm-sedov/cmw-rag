from __future__ import annotations

import pytest

from rag_engine.retrieval.embedder import FRIDAEmbedder


@pytest.mark.external
def test_frida_embedder_roundtrip():
    try:
        embedder = FRIDAEmbedder(model_name="ai-forever/FRIDA", device="cpu")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"FRIDA model unavailable: {exc}")

    query_vec = embedder.embed_query("test query about workflows")
    doc_vecs = embedder.embed_documents(["A short document about workflows."], show_progress=False)

    assert len(query_vec) == embedder.get_embedding_dim()
    assert len(doc_vecs) == 1
    assert len(doc_vecs[0]) == len(query_vec)

