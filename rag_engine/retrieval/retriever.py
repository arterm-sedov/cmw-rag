"""RAG retriever: embed, search, rerank chunks, load complete articles."""
from __future__ import annotations

import logging
from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any

import tiktoken

from rag_engine.core.chunker import split_text
from rag_engine.core.metadata_enricher import enrich_metadata
from rag_engine.retrieval.reranker import build_reranker
from rag_engine.retrieval.vector_search import top_k_search

logger = logging.getLogger(__name__)


def _stable_id(kb_id: str, chunk_index: int, content: str) -> str:
    text_hash = sha1(content.encode("utf-8")).hexdigest()[:10]
    return f"{kb_id}:{chunk_index}:{text_hash}"


class Article:
    """Complete article with metadata (loaded from source_file)."""

    def __init__(self, kb_id: str, content: str, metadata: dict[str, Any]):
        self.kb_id = kb_id
        self.content = content
        self.metadata = metadata
        self.matched_chunks: list[Any] = []  # Store matched chunks for reference


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

    def index_documents(
        self,
        documents: list[Any],
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        for doc in documents:
            base_meta = dict(getattr(doc, "metadata", {}))
            kb_id = base_meta.get("kbId", base_meta.get("source_file", "doc"))
            chunks = list(split_text(getattr(doc, "content", ""), chunk_size, chunk_overlap))
            for idx, chunk in enumerate(chunks):
                meta = enrich_metadata(base_meta, chunk, idx)
                texts.append(chunk)
                metadatas.append(meta)
                ids.append(_stable_id(kb_id, idx, chunk))

        embeddings = self.embedder.embed_documents(texts)
        self.store.add(texts=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)

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

        # 1. Embed query and vector search on chunks
        query_vec = self.embedder.embed_query(query)
        candidates = top_k_search(self.store, query_vec, k=self.top_k_retrieve)

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
                article = Article(
                    kb_id=kb_id,
                    content=article_content,
                    metadata=chunks[0].metadata,
                )
                article.matched_chunks = chunks
                articles.append(article)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to read article %s: %s", source_file, exc)
                continue

        logger.info("Loaded %d complete articles", len(articles))

        # 5. Apply context budgeting with dynamic token limits
        articles = self._apply_context_budget(articles)

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

    def _apply_context_budget(self, articles: list[Article]) -> list[Article]:
        """Select articles within context budget using dynamic token limits.

        Uses LLM manager to get model-specific context window and reserves
        25% for prompt overhead and output tokens.

        Args:
            articles: Sorted list of articles

        Returns:
            Articles that fit within context budget
        """
        # Get dynamic context window from LLM manager
        context_window = self.llm_manager.get_current_llm_context_window()

        # Reserve 25% for output + prompt overhead (use 75% for articles)
        max_context_tokens = int(context_window * 0.75)

        logger.info(
            "Context window: %d tokens, using %d (75%%) for articles",
            context_window,
            max_context_tokens,
        )

        selected: list[Article] = []
        total_tokens = 0

        for article in articles:
            # Count tokens in article (use conservative estimate to avoid undercount)
            tokens_by_encoder = len(self._encoding.encode(article.content))
            tokens_by_chars = len(article.content) // 4
            article_tokens = max(tokens_by_encoder, tokens_by_chars)

            if total_tokens + article_tokens > max_context_tokens:
                logger.info(
                    "Context budget reached: %d/%d tokens (%.1f%% of window)",
                    total_tokens,
                    max_context_tokens,
                    (total_tokens / context_window * 100),
                )
                break

            selected.append(article)
            total_tokens += article_tokens

        logger.info(
            "Selected %d articles (%d tokens, %.1f%% of context window)",
            len(selected),
            total_tokens,
            (total_tokens / context_window * 100),
        )
        # Also emit at warning level to ensure capture in default caplog
        logger.warning(
            "Selected %d articles (%.1f%% of context window)",
            len(selected),
            (total_tokens / context_window * 100),
        )

        return selected


