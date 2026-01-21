"""RAG retriever: search, rerank chunks, load complete articles."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import tiktoken

from rag_engine.config.settings import settings
from rag_engine.core.chunker import split_text
from rag_engine.retrieval.reranker import build_reranker
from rag_engine.retrieval.vector_search import top_k_search
from rag_engine.utils.metadata_utils import extract_numeric_kbid
from rag_engine.utils.path_utils import normalize_path

logger = logging.getLogger(__name__)


class Article:
    """Complete article with metadata (loaded from source_file)."""

    def __init__(self, kb_id: str, content: str, metadata: dict[str, Any]):
        self.kb_id = kb_id
        self.content = content
        self.metadata = metadata
        self.matched_chunks: list[Any] = []  # Store matched chunks for reference
        self._is_lightweight: bool = False


class RAGRetriever:
    def __init__(
        self,
        embedder,
        vector_store,
        llm_manager,  # NEW: For dynamic context window
        top_k_retrieve: int,
        top_k_rerank: int,
        rerank_enabled: bool = True,
        rerankers: list[dict[str, Any]] | None = None,
        metadata_boost_weights: dict[str, float] | None = None,
    ):
        self.embedder = embedder
        self.store = vector_store
        self.llm_manager = llm_manager  # For dynamic context budgeting
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.reranker = (
            build_reranker(
                rerankers
                or [
                    {"model_name": "DiTy/cross-encoder-russian-msmarco", "batch_size": 16},
                    {"model_name": "BAAI/bge-reranker-v2-m3", "batch_size": 16},
                    {"model_name": "jinaai/jina-reranker-v2-base-multilingual", "batch_size": 16},
                ],
                device=settings.embedding_device,  # Reuse embedding device setting
            )
            if rerank_enabled
            else None
        )
        # Log reranker type for debugging
        if self.reranker is not None:
            reranker_type = type(self.reranker).__name__
            logger.info("Reranker type: %s", reranker_type)
            if reranker_type == "IdentityReranker":
                logger.warning(
                    "IdentityReranker is being used (all reranker models failed to load). "
                    "Scores will be based on vector similarity only."
                )
        self.metadata_boost_weights = metadata_boost_weights or {
            "tag_match": 1.2,
            "code_presence": 1.15,
            "section_match": 1.1,
        }
        self._encoding = tiktoken.get_encoding("cl100k_base")
        logger.info(
            "RAGRetriever initialized (top_k_retrieve=%d, top_k_rerank=%d, reranker=%s)",
            top_k_retrieve,
            top_k_rerank,
            "enabled" if self.reranker else "disabled",
        )

    # --- Query segmentation helpers (multi-vector retrieval) ---
    def _toklen(self, s: str) -> int:
        return len(self._encoding.encode(s or ""))

    def _split_query_segments(
        self,
        query: str,
        max_seg_tokens: int,
        overlap: int,
        max_segments: int,
    ) -> list[str]:
        # Reuse code-safe, token-aware splitter used by indexer
        segments = list(split_text(query, chunk_size=max_seg_tokens, chunk_overlap=overlap))
        out: list[str] = []
        for seg in segments:
            seg = (seg or "").strip()
            if not seg:
                continue
            ids = self._encoding.encode(seg)
            if len(ids) > max_seg_tokens:
                seg = self._encoding.decode(ids[:max_seg_tokens])
            out.append(seg)
            if len(out) >= max_segments:
                break
        # Edge guard: if we ended with a single segment equal to original, return original only
        if len(out) == 1 and out[0] == query.strip():
            return [query]
        return out or [query]

    def _llm_decompose_query(self, query: str, max_subq: int) -> list[str]:
        """Optionally decompose query via LLM into concise sub-queries.

        Uses current LLM with no context to avoid window pressure. On any
        exception, returns an empty list.
        """
        try:
            max_n = max(1, int(max_subq))
            from rag_engine.llm.prompts import QUERY_DECOMPOSITION_PROMPT
            prompt = QUERY_DECOMPOSITION_PROMPT.format(max_n=max_n, question=query)
            resp = self.llm_manager.generate(question=prompt, context_docs=[])
            if not resp:
                return []
            lines = [ln.strip("- ").strip() for ln in str(resp).splitlines()]
            subqs = [ln for ln in lines if ln]
            return subqs[:max_n]
        except Exception:
            return []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        include_confidence: bool = True,
    ) -> list[Article]:
        """Retrieve complete articles for query using hybrid approach.

        Hybrid approach:
        1. Vector search on chunks (top-20)
        2. Rerank chunks (top-10)
        3. Group chunks by kbId and preserve max reranker score
        4. Read complete articles from source_file
        5. Sort by rank and normalize ranks (0.0 = best, 1.0 = worst)
        6. Return ALL articles uncompressed with ranking information

        **Returns uncompressed articles with ranking metadata.**
        **Compression happens in agent middleware (before_model), not here.**

        Args:
            query: User query string
            top_k: Override top_k_rerank if provided

        Returns:
            List of complete Article objects (uncompressed, sorted by rank)
            Each article has metadata["rerank_score"] and metadata["normalized_rank"]
        """
        qk = top_k or self.top_k_rerank

        # 1. Build candidate set via multi-vector query segmentation
        candidates: list[Any] = []
        seen_ids: set[str] = set()

        def _add_candidates_for(text: str) -> None:
            qv = self.embedder.embed_query(text)
            seg_hits = top_k_search(self.store, qv, k=self.top_k_retrieve)
            for doc in seg_hits:
                # Handle None metadata gracefully
                metadata = getattr(doc, "metadata", None) or {}
                sid = (
                    metadata.get("stable_id")
                    or getattr(doc, "id", None)
                    or str(id(doc))
                )
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)
                candidates.append(doc)

        use_multi = (
            settings.retrieval_multiquery_enabled
            and self._toklen(query) > int(settings.retrieval_multiquery_segment_tokens)
        )
        if use_multi:
            segs = self._split_query_segments(
                query,
                max_seg_tokens=int(settings.retrieval_multiquery_segment_tokens),
                overlap=int(settings.retrieval_multiquery_segment_overlap),
                max_segments=int(settings.retrieval_multiquery_max_segments),
            )
            # Edge guard: if segmentation collapses to original, do single path
            if len(segs) == 1 and segs[0].strip() == query.strip():
                _add_candidates_for(query)
            else:
                for seg in segs:
                    _add_candidates_for(seg)
        else:
            _add_candidates_for(query)

        # Optional: LLM query decomposition (additive)
        if settings.retrieval_query_decomp_enabled:
            subqs = self._llm_decompose_query(query, int(settings.retrieval_query_decomp_max_subqueries))
            for sq in subqs:
                _add_candidates_for(sq)

        # If nothing found, retry single-query once
        if not candidates:
            _add_candidates_for(query)

        # Pre-rerank cap to bound latency
        prl = int(getattr(settings, "retrieval_multiquery_pre_rerank_limit", 0) or 0)
        if prl > 0 and len(candidates) > prl:
            candidates = candidates[:prl]

        if not candidates:
            logger.warning("No chunks found for query")
            return []

        logger.info("Retrieved %d chunks from vector store", len(candidates))

        # 2. Rerank chunks (more efficient than reranking complete articles)
        # Prepare candidates with placeholder scores (reranker will replace them)
        scored_candidates: list[tuple[Any, float]] = [(doc, 0.0) for doc in candidates]
        
        if self.reranker is not None and candidates:
            # Reranker computes new scores (replaces placeholder 0.0 scores)
            scored_candidates = self.reranker.rerank(
                query,
                scored_candidates,
                top_k=qk,
                metadata_boost_weights=self.metadata_boost_weights,
            )
            # Log sample scores to verify reranker is producing non-zero values
            if scored_candidates:
                sample_scores = [s for _, s in scored_candidates[:3]]
                logger.info(
                    "Reranked to top-%d chunks (sample scores: %s)",
                    len(scored_candidates),
                    sample_scores,
                )
        else:
            # No reranker: just take top-k (scores remain 0.0)
            scored_candidates = scored_candidates[:qk]
            logger.info("No reranking, using top-%d chunks", len(scored_candidates))

        retrieval_confidence: dict[str, Any] | None = None
        if include_confidence:
            from rag_engine.retrieval.confidence import compute_retrieval_confidence

            retrieval_confidence = compute_retrieval_confidence(scored_candidates)

        # 3. Group chunks by kb_id and preserve MAX reranker score as article rank
        articles_map: dict[str, tuple[list[Any], float]] = defaultdict(lambda: ([], -float('inf')))
        for doc, score in scored_candidates:
            # Handle None metadata gracefully
            metadata = getattr(doc, "metadata", None) or {}
            # Preserve raw rerank score for trace/debugging
            if isinstance(metadata, dict):
                metadata["rerank_score_raw"] = score
            raw_kb_id = metadata.get("kbId", "")
            if raw_kb_id:
                # Normalize kbId for consistent grouping (handles suffixed kbIds)
                kb_id = extract_numeric_kbid(raw_kb_id) or str(raw_kb_id)
                chunks, best_score = articles_map[kb_id]
                chunks.append(doc)
                # Keep the highest reranker score for this article
                articles_map[kb_id] = (chunks, max(best_score, score))

        logger.info("Top chunks belong to %d unique articles", len(articles_map))

        # 4. Read complete articles and attach ranking information
        articles: list[Article] = []
        for kb_id, (chunks, max_score) in articles_map.items():
            # Use first chunk's metadata to get source file
            first_chunk_meta = getattr(chunks[0], "metadata", None) or {}
            source_file = first_chunk_meta.get("source_file")
            if not source_file:
                logger.warning("No source_file for kbId=%s", kb_id)
                continue

            try:
                article_content = self._read_article(source_file)
                # Preserve original metadata for internal ops; add a clean URL for citations
                article_metadata = dict(first_chunk_meta)
                if "article_url" not in article_metadata:
                    # Prefer explicit frontmatter URL
                    frontmatter_url = article_metadata.get("url")
                    if frontmatter_url:
                        article_metadata["article_url"] = str(frontmatter_url)
                    else:
                        # Normalize kbId for URL construction (handles edge cases)
                        raw_kbid = article_metadata.get("kbId") or kb_id
                        kbid = extract_numeric_kbid(raw_kbid) or str(raw_kbid) if raw_kbid else None
                        if kbid is not None:
                            article_metadata["article_url"] = f"https://kb.comindware.ru/article.php?id={kbid}"

                article = Article(
                    kb_id=kb_id,
                    content=article_content,
                    metadata=article_metadata,
                )
                article.matched_chunks = chunks
                # Store reranker score in metadata for proportional compression
                article.metadata["rerank_score"] = max_score
                if retrieval_confidence is not None:
                    article.metadata["retrieval_confidence"] = retrieval_confidence
                logger.debug(
                    "Article %s: rerank_score=%.4f (from %d chunks, max_score=%.4f)",
                    kb_id,
                    max_score,
                    len(chunks),
                    max_score,
                )
                articles.append(article)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to read article %s: %s", source_file, exc)
                continue

        logger.info("Loaded %d complete articles", len(articles))

        # 5. Sort articles by reranker score (highest first = best rank)
        articles.sort(key=lambda a: a.metadata.get("rerank_score", -float('inf')), reverse=True)

        # 6. Normalize ranks: 0.0 = best rank, 1.0 = worst rank (for proportional compression)
        if len(articles) > 1:
            for idx, article in enumerate(articles):
                # Normalized rank: 0.0 = first (best), 1.0 = last (worst)
                article.metadata["normalized_rank"] = idx / (len(articles) - 1)
                article.metadata["article_rank"] = idx  # Position-based rank (0-based)
        else:
            if articles:
                articles[0].metadata["normalized_rank"] = 0.0
                articles[0].metadata["article_rank"] = 0

        logger.info("Loaded %d complete articles (uncompressed, sorted by rank)", len(articles))

        # NO _apply_context_budget call - return all uncompressed articles with ranks
        return articles

    def _read_article(self, source_file: str) -> str:
        """Read complete article from filesystem.

        Args:
            source_file: Absolute or relative path to article file
                         (may contain Windows-style backslashes)

        Returns:
            Complete article content (without frontmatter)
        """
        # Normalize path to handle Windows-style backslashes on POSIX systems
        file_path = normalize_path(source_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Article file not found: {source_file}")

        content = file_path.read_text(encoding="utf-8")

        # Remove YAML frontmatter if present
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2].strip()

        return content
