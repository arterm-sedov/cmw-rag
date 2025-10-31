"""RAG retriever: embed, search, rerank chunks, load complete articles."""
from __future__ import annotations

import logging
import re
from collections import defaultdict
import json
from hashlib import sha1
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import tiktoken

from rag_engine.core.chunker import split_text
from rag_engine.config.settings import settings
from rag_engine.core.metadata_enricher import enrich_metadata
from rag_engine.retrieval.reranker import build_reranker
from rag_engine.retrieval.vector_search import top_k_search

logger = logging.getLogger(__name__)


def _stable_id(kb_id: str, chunk_index: int, content: str, source_file: str | None = None) -> str:
    """Generate a globally unique but stable ID for a chunk.

    Uses a document seed derived from source_file (preferred) or kb_id to
    disambiguate chunks from different documents that might otherwise share
    kbId and chunk content at the same index.
    """
    doc_seed = (source_file or kb_id).encode("utf-8")
    doc_hash = sha1(doc_seed).hexdigest()[:8]
    text_hash = sha1(content.encode("utf-8")).hexdigest()[:10]
    return f"{doc_hash}:{chunk_index}:{text_hash}"


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
                ]
            )
            if rerank_enabled
            else None
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

    def index_documents(
        self,
        documents: list[Any],
        chunk_size: int,
        chunk_overlap: int,
        max_files: int | None = None,
    ) -> None:
        """Index documents incrementally with immediate database writes.
        
        Processes documents one at a time: chunk → embed → write to ChromaDB.
        This enables partial progress recovery and reduces memory usage.
        ChromaDB automatically handles deduplication via stable IDs.
        
        Args:
            documents: List of document objects with content and metadata
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        total_docs = len(documents)
        processed_docs = 0
        total_chunks_indexed = 0
        
        for doc_idx, doc in enumerate(documents, start=1):
            base_meta = dict(getattr(doc, "metadata", {}))
            kb_id = base_meta.get("kbId", base_meta.get("source_file", "doc"))
            content = getattr(doc, "content", "")
            
            # Skip empty documents
            if not content or not content.strip():
                logger.warning("Skipping empty document %d/%d (kbId: %s)", doc_idx, total_docs, kb_id)
                continue
            
            # Respect max_files limit (counts only documents we actually index)
            if max_files is not None and processed_docs >= max_files:
                logger.info("Max files limit reached: %d", max_files)
                break

            # Generate chunks for this document
            chunks = list(split_text(content, chunk_size, chunk_overlap))
            if not chunks:
                logger.warning("No chunks generated for document %d/%d (kbId: %s)", doc_idx, total_docs, kb_id)
                continue
            
            # Prepare chunk data
            texts: list[str] = []
            metadatas: list[dict[str, Any]] = []
            ids: list[str] = []
            
            # Compute per-document invariants
            doc_stable_id = sha1(kb_id.encode("utf-8")).hexdigest()[:12]
            source_file = base_meta.get("source_file")
            file_modified_at_iso = None
            file_mtime_epoch = None
            if source_file:
                try:
                    p = Path(source_file)
                    if p.exists():
                        stat = p.stat()
                        # Use UTC ISO string for readability and epoch for comparisons
                        file_mtime_epoch = int(stat.st_mtime)
                        file_modified_at_iso = datetime.utcfromtimestamp(stat.st_mtime).replace(tzinfo=timezone.utc).isoformat()
                except Exception:  # noqa: BLE001
                    pass

            # Incremental reindexing: skip unchanged docs, replace outdated
            if file_mtime_epoch is not None:
                existing = self.store.get_any_doc_meta({"doc_stable_id": doc_stable_id})
                if existing is not None:
                    existing_epoch = existing.get("file_mtime_epoch")
                    if isinstance(existing_epoch, int) and existing_epoch >= file_mtime_epoch:
                        logger.info("Skipping unchanged document %d/%d (kbId: %s)", doc_idx, total_docs, kb_id)
                        continue
                    # Delete outdated chunks for this document before re-adding
                    try:
                        self.store.delete_where({"doc_stable_id": doc_stable_id})
                        logger.info("Deleted outdated chunks for kbId=%s (doc_stable_id=%s)", kb_id, doc_stable_id)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to delete outdated chunks for kbId=%s: %s", kb_id, exc)

            for idx, chunk in enumerate(chunks):
                # Generate stable ID first so we can store it in metadata
                stable_id = _stable_id(kb_id, idx, chunk, base_meta.get("source_file"))

                meta = enrich_metadata(base_meta, chunk, idx)
                # Ensure critical identifiers and timestamps are present
                meta.setdefault("kbId", kb_id)
                meta["stable_id"] = stable_id
                meta["doc_stable_id"] = doc_stable_id
                if file_modified_at_iso is not None:
                    meta["file_modified_at"] = file_modified_at_iso
                if file_mtime_epoch is not None:
                    meta["file_mtime_epoch"] = file_mtime_epoch

                texts.append(chunk)
                metadatas.append(meta)
                ids.append(stable_id)

            # Sanitize metadata values for Chroma (drop None, stringify lists/dicts)
            def _sanitize_metadata(m: dict[str, Any]) -> dict[str, Any]:
                out: dict[str, Any] = {}
                for k, v in (m or {}).items():
                    if v is None:
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        out[k] = v
                    elif isinstance(v, list):
                        if not v:
                            continue
                        out[k] = ", ".join(str(x) for x in v)
                    else:
                        try:
                            out[k] = json.dumps(v, ensure_ascii=False)
                        except Exception:  # noqa: BLE001
                            out[k] = str(v)
                return out

            # Deduplicate by ID within this document batch before embedding
            seen_ids: set[str] = set()
            dedup_texts: list[str] = []
            dedup_metadatas: list[dict[str, Any]] = []
            dedup_ids: list[str] = []
            for t, m, i in zip(texts, metadatas, ids):
                if i in seen_ids:
                    continue
                seen_ids.add(i)
                dedup_texts.append(t)
                dedup_metadatas.append(_sanitize_metadata(m))
                dedup_ids.append(i)

            # Embed chunks for this (deduplicated) document
            embeddings = self.embedder.embed_documents(dedup_texts, show_progress=False)

            # Write immediately to ChromaDB
            self.store.add(texts=dedup_texts, metadatas=dedup_metadatas, ids=dedup_ids, embeddings=embeddings)
            
            total_chunks_indexed += len(chunks)
            processed_docs += 1
            logger.info(
                "Indexed document %d/%d: %d chunks (kbId: %s, total chunks: %d)",
                doc_idx,
                total_docs,
                len(chunks),
                kb_id,
                total_chunks_indexed,
            )

    def retrieve(self, query: str, top_k: int | None = None) -> list[Article]:
        """Retrieve complete articles for query using hybrid approach.

        Hybrid approach:
        1. Vector search on chunks (top-20)
        2. Rerank chunks (top-10)
        3. Group chunks by kbId
        4. Read complete articles from source_file
        5. Apply context budgeting with dynamic token limits

        Args:
            query: User query string
            top_k: Override top_k_rerank if provided

        Returns:
            List of complete Article objects within context budget
        """
        qk = top_k or self.top_k_rerank

        # 1. Build candidate set via multi-vector query segmentation
        candidates: list[Any] = []
        seen_ids: set[str] = set()

        def _add_candidates_for(text: str) -> None:
            qv = self.embedder.embed_query(text)
            seg_hits = top_k_search(self.store, qv, k=self.top_k_retrieve)
            for doc in seg_hits:
                sid = (
                    getattr(doc, "metadata", {}).get("stable_id")
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
        scored_candidates: list[tuple[Any, float]] = [(doc, 0.0) for doc in candidates]
        if self.reranker is not None and candidates:
            scored_candidates = self.reranker.rerank(
                query,
                scored_candidates,
                top_k=qk,
                metadata_boost_weights=self.metadata_boost_weights,
            )
            logger.info("Reranked to top-%d chunks", len(scored_candidates))
        else:
            scored_candidates = scored_candidates[:qk]
            logger.info("No reranking, using top-%d chunks", len(scored_candidates))

        # 3. Group top-ranked chunks by kbId (article identifier)
        articles_map: dict[str, list[Any]] = defaultdict(list)
        for doc, _score in scored_candidates:
            kb_id = getattr(doc, "metadata", {}).get("kbId", "")
            if kb_id:
                articles_map[kb_id].append(doc)

        logger.info("Top chunks belong to %d unique articles", len(articles_map))

        # 4. Read complete articles from filesystem
        articles: list[Article] = []
        for kb_id, chunks in articles_map.items():
            # Use first chunk's metadata to get source file
            source_file = chunks[0].metadata.get("source_file")
            if not source_file:
                logger.warning("No source_file for kbId=%s", kb_id)
                continue

            try:
                article_content = self._read_article(source_file)
                # Preserve original metadata for internal ops; add a clean URL for citations
                article_metadata = dict(chunks[0].metadata)
                if "article_url" not in article_metadata:
                    # Prefer explicit frontmatter URL
                    frontmatter_url = article_metadata.get("url")
                    if frontmatter_url:
                        article_metadata["article_url"] = str(frontmatter_url)
                    else:
                        # kbId is guaranteed numeric by indexer; construct URL directly
                        kbid = article_metadata.get("kbId") or kb_id
                        if kbid is not None:
                            article_metadata["article_url"] = f"https://kb.comindware.ru/article.php?id={kbid}"

                article = Article(
                    kb_id=kb_id,
                    content=article_content,
                    metadata=article_metadata,
                )
                article.matched_chunks = chunks
                articles.append(article)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to read article %s: %s", source_file, exc)
                continue

        logger.info("Loaded %d complete articles", len(articles))

        # 5. Apply context budgeting with dynamic token limits (pass question)
        articles = self._apply_context_budget(articles, question=query)

        return articles

    def _read_article(self, source_file: str) -> str:
        """Read complete article from filesystem.

        Args:
            source_file: Absolute or relative path to article file

        Returns:
            Complete article content (without frontmatter)
        """
        file_path = Path(source_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Article file not found: {source_file}")

        content = file_path.read_text(encoding="utf-8")

        # Remove YAML frontmatter if present
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2].strip()

        return content

    def _apply_context_budget(self, articles: list[Article], question: str = "", system_prompt: str = "") -> list[Article]:
        """Select articles within context budget using dynamic token limits.

        Uses LLM manager to get model-specific context window and reserves
        25% for prompt overhead and output tokens.

        For articles that don't fit with full content, creates lightweight
        representations with title, URL, and relevant matched chunks so the
        LLM can still cite them.

        Args:
            articles: Sorted list of articles

        Returns:
            Articles that fit within context budget (full content) plus
            lightweight representations (title, URL, chunks) for remaining articles
        """
        # Get dynamic context window from LLM manager
        context_window = self.llm_manager.get_current_llm_context_window()
        max_output_tokens = self.llm_manager.get_max_output_tokens()

        # Derive system prompt
        sys_prompt = system_prompt or self.llm_manager.get_system_prompt()

        # Reserve accurately using shared util
        from rag_engine.llm.token_utils import estimate_tokens_for_request
        reserved_est = estimate_tokens_for_request(
            system_prompt=sys_prompt,
            question=question or "",
            context="",
            max_output_tokens=max_output_tokens,
            overhead=200,
        )
        max_context_tokens = max(0, context_window - reserved_est["total_tokens"])

        logger.info(
            "Context window: %d tokens, using %d (75%%) for articles",
            context_window,
            max_context_tokens,
        )

        selected: list[Article] = []
        total_tokens = 0
        # By default assume all articles fit; adjusted only if we break early
        full_content_idx = len(articles)

        # First pass: try to fit full article content
        # Articles added here will NOT be processed again in the second pass
        for idx, article in enumerate(articles):
            # Count tokens in article (use conservative estimate to avoid undercount)
            # Fast path for very large bodies: approximate to avoid slow encodes
            if len(article.content) > settings.retrieval_fast_token_char_threshold:
                tokens_by_encoder = len(article.content) // 4
            else:
                tokens_by_encoder = len(self._encoding.encode(article.content))
            tokens_by_chars = len(article.content) // 4
            article_tokens = max(tokens_by_encoder, tokens_by_chars)

            if total_tokens + article_tokens > max_context_tokens:
                logger.info(
                    "Context budget reached for full content: %d/%d tokens (%.1f%% of window)",
                    total_tokens,
                    max_context_tokens,
                    (total_tokens / context_window * 100),
                )
                full_content_idx = idx  # Articles from this index onward weren't included
                break

            selected.append(article)
            total_tokens += article_tokens

        # Second pass: summarization-first for remaining articles
        # Only processes articles[full_content_idx:] - articles already included as full
        # (indices 0 to full_content_idx-1) are skipped, ensuring each article appears once
        lightweight_count = 0

        from rag_engine.llm.summarization import summarize_to_tokens

        overflow = list(articles[full_content_idx:])
        while overflow and total_tokens < max_context_tokens:
            remaining_budget = max_context_tokens - total_tokens
            remaining_overflow = len(overflow)
            per_target = max(300, min(2000, remaining_budget // max(1, remaining_overflow)))

            article = overflow.pop(0)
            chunk_texts = []
            for chunk in article.matched_chunks:
                chunk_content = getattr(chunk, "page_content", None) or getattr(chunk, "content", "")
                if chunk_content:
                    chunk_texts.append(str(chunk_content))

            title = article.metadata.get("title", article.kb_id)
            article_url = article.metadata.get("article_url") or article.metadata.get("url")
            if not article_url:
                kbid = article.metadata.get("kbId") or article.kb_id
                if kbid:
                    article_url = f"https://kb.comindware.ru/article.php?id={kbid}"

            summary = summarize_to_tokens(
                title=title,
                url=article_url or "",
                matched_chunks=chunk_texts,
                full_body=article.content,
                target_tokens=per_target,
                guidance=question or "",
                llm=self.llm_manager,
                max_retries=2,
            )

            # Fast path for very large summaries (rare): approximate
            if len(summary) > settings.retrieval_fast_token_char_threshold:
                tokens_by_encoder = len(summary) // 4
            else:
                tokens_by_encoder = len(self._encoding.encode(summary))
            tokens_by_chars = len(summary) // 4
            summary_tokens = max(tokens_by_encoder, tokens_by_chars)

            if total_tokens + summary_tokens > max_context_tokens:
                # Not enough space even for summary; stop summarization loop
                overflow.insert(0, article)
                break

            summarized_article = Article(kb_id=article.kb_id, content=summary, metadata=article.metadata)
            summarized_article.matched_chunks = article.matched_chunks
            selected.append(summarized_article)
            total_tokens += summary_tokens

        # If still space and overflow remains, convert remaining to lightweight stitching
        def _create_lightweight_article(src: Article) -> tuple[Article, int]:
            chunk_texts = []
            for chunk in src.matched_chunks:
                chunk_content = getattr(chunk, "page_content", None) or getattr(chunk, "content", "")
                if chunk_content:
                    chunk_texts.append(str(chunk_content))
            title = src.metadata.get("title", src.kb_id)
            article_url = src.metadata.get("article_url") or src.metadata.get("url")
            if not article_url:
                kbid = src.metadata.get("kbId") or src.kb_id
                if kbid:
                    article_url = f"https://kb.comindware.ru/article.php?id={kbid}"
            content = f"# {title}\n\nURL: {article_url}\n\n" + "\n\n---\n\n".join(chunk_texts)
            lw = Article(kb_id=src.kb_id, content=content, metadata=src.metadata)
            lw.matched_chunks = src.matched_chunks
            lw._is_lightweight = True
            enc = len(self._encoding.encode(content))
            by_chars = len(content) // 4
            return lw, max(enc, by_chars)

        for article in overflow:
            lw_article, lw_tokens = _create_lightweight_article(article)
            if total_tokens + lw_tokens > max_context_tokens:
                continue
            selected.append(lw_article)
            total_tokens += lw_tokens
            lightweight_count += 1

        logger.info(
            "Selected %d full articles + %d lightweight articles (%d tokens, %.1f%% of context window)",
            len(selected) - lightweight_count,
            lightweight_count,
            total_tokens,
            (total_tokens / context_window * 100),
        )
        # Also emit at warning level to ensure capture in default caplog
        logger.warning(
            "Selected %d articles (%.1f%% of context window, %d full + %d lightweight)",
            len(selected),
            (total_tokens / context_window * 100),
            len(selected) - lightweight_count,
            lightweight_count,
        )

        return selected


