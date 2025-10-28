"""Lean CrossEncoder reranker with identity fallback."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


class IdentityReranker:
    """Pass-through reranker used when cross-encoder is unavailable."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[Tuple[Any, float]],
        top_k: int,
        metadata_boost_weights: Dict[str, float] | None = None,
    ) -> List[Tuple[Any, float]]:
        return list(candidates)[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str, batch_size: int = 16):
        from sentence_transformers import CrossEncoder  # lazy import

        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: Sequence[Tuple[Any, float]],
        top_k: int,
        metadata_boost_weights: Dict[str, float] | None = None,
    ) -> List[Tuple[Any, float]]:
        pairs = [(query, doc.page_content if hasattr(doc, "page_content") else str(doc)) for doc, _ in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Combine with optional metadata boosts
        scored: List[Tuple[Any, float]] = []
        for (doc, _), score in zip(candidates, scores):
            boost = 0.0
            if metadata_boost_weights and hasattr(doc, "metadata"):
                meta = getattr(doc, "metadata", {})
                if meta.get("tags") and metadata_boost_weights.get("tag_match"):
                    boost += metadata_boost_weights["tag_match"]
                if meta.get("has_code") and metadata_boost_weights.get("code_presence"):
                    boost += metadata_boost_weights["code_presence"]
                if meta.get("section_heading") and metadata_boost_weights.get("section_match"):
                    boost += metadata_boost_weights["section_match"]
            final_score = float(score) * (1.0 + boost)
            scored.append((doc, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def build_reranker(prioritized_models: List[Dict[str, Any]]) -> Any:
    """Return first available cross-encoder; fallback to identity reranker."""
    for cfg in prioritized_models:
        try:
            return CrossEncoderReranker(cfg["model_name"], batch_size=cfg.get("batch_size", 16))
        except Exception:  # noqa: BLE001
            continue
    return IdentityReranker()


