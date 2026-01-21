"""Retrieval confidence metrics from reranker scores.

This module is intentionally lightweight: it returns plain dicts (no extra schemas)
so we can attach small numeric signals to trace outputs without bloating payloads.
"""

from __future__ import annotations

from statistics import median
from typing import Any


def compute_retrieval_confidence(
    scored_chunks: list[tuple[Any, float]],
    *,
    relevance_threshold: float | None = None,
    mean_top_k: int = 5,
) -> dict[str, Any]:
    """Compute query-level confidence metrics from reranker scores.

    Args:
        scored_chunks: List of (chunk, rerank_score) tuples (typically already top-k).
        relevance_threshold: Score above which a chunk is considered likely relevant.
            If None, defaults to 0.5.
        mean_top_k: How many top scores to average for mean_top_k metric.

    Returns:
        Dict with:
        - top_score: max score
        - mean_top_k: average of top-N scores
        - score_gap: top_score - median_score
        - n_above_threshold: count of scores >= relevance_threshold
        - likely_relevant: heuristic boolean
    """
    threshold = 0.5 if relevance_threshold is None else float(relevance_threshold)

    scores = [float(s) for _, s in scored_chunks if s is not None]
    if not scores:
        return {
            "top_score": 0.0,
            "mean_top_k": 0.0,
            "score_gap": 0.0,
            "n_above_threshold": 0,
            "likely_relevant": False,
        }

    scores_sorted = sorted(scores, reverse=True)
    top_score = scores_sorted[0]
    top_n = scores_sorted[: max(1, int(mean_top_k))]
    mean_top = sum(top_n) / len(top_n) if top_n else 0.0
    med = float(median(scores_sorted))
    gap = top_score - med
    n_above = sum(1 for s in scores_sorted if s >= threshold)

    # Conservative heuristic: require at least one above threshold and some separation.
    likely = bool(top_score >= threshold and (gap >= 0.05 or n_above >= 2))

    return {
        "top_score": top_score,
        "mean_top_k": mean_top,
        "score_gap": gap,
        "n_above_threshold": n_above,
        "likely_relevant": likely,
    }

