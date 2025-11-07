"""Lean CrossEncoder reranker with identity fallback."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag_engine.utils.device_utils import detect_device

try:  # Provide a module-level alias for monkeypatch-friendly tests
    from sentence_transformers import CrossEncoder as CrossEncoder  # type: ignore
except Exception:  # noqa: BLE001
    CrossEncoder = None  # type: ignore

class IdentityReranker:
    """Pass-through reranker used when cross-encoder is unavailable."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Any, float]],
        top_k: int,
        metadata_boost_weights: dict[str, float] | None = None,
    ) -> list[tuple[Any, float]]:
        return list(candidates)[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        device: str = "auto",
        model: Any | None = None,
    ):
        """Initialize cross-encoder reranker.

        Args:
            model_name: Name of the CrossEncoder model
            batch_size: Batch size for reranking
            device: Device to run the model on ('auto', 'cpu', or 'cuda').
                    'auto' will detect and use GPU if available, else CPU.
            model: Optional pre-initialized model (for testing)
        """
        # Auto-detect device if "auto" is specified
        if device == "auto":
            device = detect_device("auto")

        # Allow injecting a fake model for tests; otherwise construct via module-level alias
        global CrossEncoder  # use the module-level symbol (monkeypatchable)
        if model is not None:
            self.model = model
        else:
            if CrossEncoder is None:
                # Import here if alias was not resolved at import time
                from sentence_transformers import CrossEncoder as _CE  # type: ignore
                CrossEncoder = _CE  # type: ignore
            self.model = CrossEncoder(model_name, device=device)  # type: ignore[operator]
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Any, float]],
        top_k: int,
        metadata_boost_weights: dict[str, float] | None = None,
    ) -> list[tuple[Any, float]]:
        pairs = [(query, doc.page_content if hasattr(doc, "page_content") else str(doc)) for doc, _ in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Combine with optional metadata boosts
        scored: list[tuple[Any, float]] = []
        for (doc, _), score in zip(candidates, scores, strict=False):
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


def build_reranker(
    prioritized_models: list[dict[str, Any]], device: str = "auto"
) -> Any:
    """Return first available cross-encoder; fallback to identity reranker.

    Args:
        prioritized_models: List of model configs to try in order
        device: Device to run the model on ('auto', 'cpu', or 'cuda').
                'auto' will detect and use GPU if available, else CPU.
    """
    for cfg in prioritized_models:
        try:
            return CrossEncoderReranker(
                cfg["model_name"],
                batch_size=cfg.get("batch_size", 16),
                device=cfg.get("device", device),
            )
        except Exception:  # noqa: BLE001
            continue
    return IdentityReranker()


