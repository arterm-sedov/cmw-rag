from __future__ import annotations

from rag_engine.retrieval.confidence import compute_retrieval_confidence


def test_compute_retrieval_confidence_empty():
    conf = compute_retrieval_confidence([])
    assert conf["likely_relevant"] is False
    assert conf["top_score"] == 0.0


def test_compute_retrieval_confidence_basic():
    scored = [(object(), 0.9), (object(), 0.2), (object(), 0.1)]
    conf = compute_retrieval_confidence(scored, relevance_threshold=0.5, mean_top_k=2)
    assert conf["top_score"] == 0.9
    assert conf["n_above_threshold"] == 1
    assert conf["likely_relevant"] is True

