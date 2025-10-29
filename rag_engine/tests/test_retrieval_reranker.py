from __future__ import annotations

from unittest.mock import MagicMock

from rag_engine.retrieval.reranker import CrossEncoderReranker, IdentityReranker, build_reranker


def test_identity_reranker_returns_top_k():
    reranker = IdentityReranker()
    candidates = [("doc1", 0.1), ("doc2", 0.2), ("doc3", 0.3)]

    top = reranker.rerank("query", candidates, top_k=2)

    assert top == candidates[:2]


def test_cross_encoder_reranker_uses_model(monkeypatch):
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.9, 0.1]

    monkeypatch.setattr("rag_engine.retrieval.reranker.CrossEncoder", lambda name: fake_model)

    reranker = CrossEncoderReranker(model_name="fake/model")
    candidates = [(MagicMock(page_content="doc1", metadata={}), 0.0), (MagicMock(page_content="doc2", metadata={}), 0.0)]

    ranked = reranker.rerank("query", candidates, top_k=2)

    assert ranked[0][0].page_content == "doc1"
    fake_model.predict.assert_called_once()


def test_build_reranker_falls_back_to_identity(monkeypatch):
    def raise_error(name):  # noqa: ANN001
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr("rag_engine.retrieval.reranker.CrossEncoder", raise_error)

    reranker = build_reranker([{"model_name": "missing/model"}])

    assert isinstance(reranker, IdentityReranker)

