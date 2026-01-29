"""LangChain 1.0 tool for RAG context retrieval.

This tool is self-sufficient and handles all retrieval mechanics internally.
The agent only needs to import and use the tool - no setup required.

Now supports async execution to prevent blocking the event loop during
concurrent requests.
"""
from __future__ import annotations

import json
import logging
import threading

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field, field_validator

from rag_engine.config.settings import settings
from rag_engine.retrieval.retriever import Article, RAGRetriever
from rag_engine.utils.context_tracker import AgentContext
from rag_engine.utils.thread_pool import run_in_thread_pool

logger = logging.getLogger(__name__)

# Module-level retriever instance (lazy singleton)
_retriever: RAGRetriever | None = None
_retriever_init_lock = threading.Lock()


def _get_or_create_retriever() -> RAGRetriever:
    """Get or create the retriever instance (lazy singleton).

    This creates the retriever on first use and reuses it for subsequent calls.
    The retriever is stateless, so it's safe to share across sessions.

    Returns:
        RAGRetriever instance
    """
    global _retriever
    if _retriever is None:
        # Serialize first-time initialization across threads
        with _retriever_init_lock:
            if _retriever is None:
                from rag_engine.llm.llm_manager import LLMManager
                from rag_engine.retrieval.embedder import FRIDAEmbedder
                from rag_engine.storage.vector_store import ChromaStore

                logger.info("Initializing retriever for retrieve_context tool (first use)")

                # Initialize infrastructure
                embedder = FRIDAEmbedder(
                    model_name=settings.embedding_model,
                    device=settings.embedding_device,
                )
                vector_store = ChromaStore(
                    persist_dir=settings.chromadb_persist_dir,
                    collection_name=settings.chromadb_collection,
                )
                llm_manager = LLMManager(
                    provider=settings.default_llm_provider,
                    model=settings.default_model,
                    temperature=settings.llm_temperature,
                )

                # Create retriever
                _retriever = RAGRetriever(
                    embedder=embedder,
                    vector_store=vector_store,
                    llm_manager=llm_manager,
                    top_k_retrieve=settings.top_k_retrieve,
                    top_k_rerank=settings.top_k_rerank,
                    rerank_enabled=settings.rerank_enabled,
                )
                logger.info("Retriever initialized successfully")

    return _retriever


class RetrieveContextSchema(BaseModel):
    """
    Schema for retrieving context documents from the knowledge base.

    This schema defines the input parameters for the retrieve_context tool,
    following LangChain 1.0 and Pydantic best practices. Field descriptions
    are written for LLM understanding and MCP server compatibility.
    """

    query: str = Field(
        ...,
        description="Ыearch query to find relevant articles from the knowledge base. "
        "This should be unique, clear, specific, focused. ",
        min_length=1,
    )
    top_k: int | None = Field(
        default=None,
        description="Maximum number of articles to retrieve. "
        f"Default: {settings.top_k_rerank} articles. "
        "Use a smaller value (e.g., 3) for focused retrieval. "
        "Use larger value (e.g., 10) for comprehensive coverage. "
    )
    exclude_kb_ids: list[str] | None = Field(
        default=None,
        description="Optional list of article kb_ids to exclude from results (for deduplication). "
        "Use this to prevent retrieving articles you've already fetched in previous tool calls. ",
    )

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate that query is not empty."""
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("query must be a non-empty string")
        return v.strip() if isinstance(v, str) else v

    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k(cls, v: int | str | None) -> int | None:
        """Validate that top_k is positive if provided. Coerce string to int (e.g. from JSON/LLM)."""
        if v is None:
            return None
        if isinstance(v, str):
            try:
                v = int(v)
            except (ValueError, TypeError) as e:
                raise ValueError(f"top_k must be a valid integer, got: {v!r}") from e
        if not isinstance(v, int):
            raise ValueError(f"top_k must be an integer or None, got: {type(v).__name__}")
        if v <= 0:
            raise ValueError("top_k must be a positive integer")
        return v


def _format_articles_to_json(articles: list[Article], query: str, top_k: int | None) -> str:
    """Convert Article objects to JSON format with ranking information."""
    articles_data = []
    for article in articles:
        title = article.metadata.get("title", article.kb_id)
        url = (
            article.metadata.get("article_url")
            or article.metadata.get("url")
            or f"https://kb.comindware.ru/article.php?id={article.kb_id}"
        )

        # Include all metadata including rank information for proportional compression
        article_metadata = dict(article.metadata)
        # Metadata already contains rerank_score and normalized_rank from retriever

        articles_data.append({
            "kb_id": article.kb_id,
            "title": title,
            "url": url,
            "content": article.content,  # Uncompressed content
            "metadata": article_metadata,  # Includes rerank_score, normalized_rank
        })

    result = {
        "articles": articles_data,
        "metadata": {
            "query": query,
            "top_k_requested": top_k,
            "articles_count": len(articles_data),
            "has_results": len(articles_data) > 0,
        },
    }
    return json.dumps(result, ensure_ascii=False, separators=(',', ':'))


def _build_query_trace_entry(query: str, articles: list[Article]) -> dict:
    """Build per-query trace data from retrieved Article objects.

    This is stored in AgentContext.query_traces (excluded from LLM context) to avoid
    bloating the tool JSON payload while still allowing batch/UI inspection.
    """
    per_article: list[dict] = []
    confidence = None
    if articles:
        confidence = (articles[0].metadata or {}).get("retrieval_confidence")

    for article in articles:
        meta = article.metadata or {}
        title = meta.get("title", article.kb_id)
        url = (
            meta.get("article_url")
            or meta.get("url")
            or f"https://kb.comindware.ru/article.php?id={article.kb_id}"
        )

        chunks = []
        matched = getattr(article, "matched_chunks", []) or []
        # Prefer chunks sorted by preserved rerank_score_raw (desc)
        def _chunk_score(doc: object) -> float:
            md = getattr(doc, "metadata", None) or {}
            try:
                return float(md.get("rerank_score_raw", 0.0) or 0.0)
            except Exception:
                return 0.0

        matched_sorted = sorted(matched, key=_chunk_score, reverse=True)
        for idx, doc in enumerate(matched_sorted, start=1):
            md = getattr(doc, "metadata", None) or {}
            text = getattr(doc, "page_content", "") or ""
            snippet = (text[:200] + "…") if isinstance(text, str) and len(text) > 200 else str(text)
            chunks.append(
                {
                    "snippet": snippet,
                    "rerank_score_raw": md.get("rerank_score_raw"),
                    "rerank_rank": idx,
                }
            )

        per_article.append(
            {
                "kb_id": article.kb_id,
                "title": title,
                "url": url,
                "article_rank": meta.get("article_rank"),
                "normalized_rank": meta.get("normalized_rank"),
                "rerank_score": meta.get("rerank_score"),
                "chunks": chunks,
            }
        )

    return {"query": query, "confidence": confidence, "articles": per_article}


@tool("retrieve_context", args_schema=RetrieveContextSchema)
async def retrieve_context(
    query: str,
    top_k: int | None = None,
    exclude_kb_ids: list[str] | None = None,
    runtime: ToolRuntime[AgentContext, None] | None = None,
) -> str:
    """Retrieve relevant context articles from the knowledge base using semantic search.

    This tool searches the Comindware knowledge base for articles relevant to your query.

    It returns formatted context with article titles, URLs, and content ready for consumption.

    **Use this tool:**
    - When you need information from the knowledge base to answer a user's question
    - When the user's request is vague or ambiguous call this tool multiple times
      with unique queries to find comprehensive information
    - When you need to explore different aspects of a topic
    - When initial results are insufficient and you want to refine your search

    **Query best practices:**
    - Always query the knowledge base in Russian, even if the question is in English.
    - Use specific, focused queries for better results (e.g., "настройка аутентификации" not "аутентификация").
    - Combine results from multiple queries to build a comprehensive understanding.
    - Usually 1-3 quality search queries (with reasonable top_k) are enough to answer the question.
    - If no results found, try broader, alternative phrasings, synonyms or related terms.
    - Break down vague or complex questions into multiple focused queries with different query angles.
    - The knowledge base is focused to the Comindware Platform and its use cases, hence:
      - DO NOT include the "Comindware Platform" term in search queries unless really needed.
      - For general, business or industry-specific questions extract technical and platform-relevant search queries (excluding industry/business keywords).

    **Query decomposition:**
    - For better search results, paraphrase and split the user question into several unique queries, using different phrases and keywords.
    - Call retrieve_context multiple times with unique queries. Do not search for semantically similar queries more than once.
    - If you have suggested subqueries from the analyse_user_request plan, use them as starting points but feel free to rephrase or add more queries as needed to comprehensively cover the topic.

    **Examples:**
    - User question: Как всё настроить?
      - Unique search queries:
        * настройка и запуск ПО
        * подготовка к установке
        * системные требования
    - User question: Как настроить взаимодействие между подразделениями
      - Unique search queries:
        * настройка почты
        * получение и отправка почты
        * подключения и пути передачи данных
        * SMTP/IMAP/Exchange
        * межпроцессное взаимодействие
        * сообщения
        * HTTP/HTTPS
        * REST API
    - User question: "Как писать тройки"
      - Unique search queries:
        * написание выражений на N3
        * синтаксис N3
        * примеры N3
        * справочник по N3
        * язык N3
    - User question: "Как провести отпуск"
      - Unique search queries:
        * бизнес-приложения
        * шаблоны
        * атрибуты
        * записи
        * формы

    Returns:
        JSON string containing structured article data. Format:
        {
          "articles": [
            {
              "kb_id": "string",
              "title": "string",
              "url": "string",
              "content": "string",
              "metadata": {...}
            }
          ],
          "metadata": {
            "query": "string",
            "top_k_requested": "int | null",
            "articles_count": "int",
            "has_results": "bool"
          }
        }

        Each article includes title, URL, content (uncompressed), and complete metadata
        including ranking information (rerank_score, normalized_rank) for citations.
        If no documents found, returns JSON with empty articles array and has_results: false.

    **Note**: You can call this tool multiple times in the same conversation turn to gather
    information from different angles or aspects of a topic. Each call is independent.
    """
    try:
        # Get retriever (initialization is fast, but wrap in thread pool for safety)
        retriever = await run_in_thread_pool(_get_or_create_retriever)

        # Run blocking retrieval in thread pool to avoid blocking event loop
        # Always compute confidence for trace/debug; this is a small dict payload.
        docs = await run_in_thread_pool(
            lambda: retriever.retrieve(query, top_k=top_k, include_confidence=True)
        )

        # Determine which kb_ids to exclude (explicit argument takes precedence, then context)
        excluded_set: set[str] = set()
        if exclude_kb_ids:
            excluded_set = set(exclude_kb_ids)
        elif runtime and hasattr(runtime, "context") and runtime.context:
            excluded_set = getattr(runtime.context, "fetched_kb_ids", set())

        # Filter out excluded articles
        if excluded_set:
            original_count = len(docs)
            docs = [doc for doc in docs if doc.kb_id not in excluded_set]
            filtered_count = original_count - len(docs)
            if filtered_count > 0:
                logger.info(
                    "Filtered out %d already-fetched articles, returning %d new articles for query: %s",
                    filtered_count,
                    len(docs),
                    query,
                )

        logger.info("Retrieved %d articles for query: %s", len(docs), query)

        # Store per-query trace in AgentContext (excluded from LLM context)
        # Workaround for LangChain streaming bug: try runtime.context first, fallback to thread-local
        context_to_use = None
        if runtime and hasattr(runtime, "context") and runtime.context:
            context_to_use = runtime.context
            logger.debug("Using runtime.context for query trace storage")
        else:
            # Fallback: get context from thread-local storage (workaround for streaming bug)
            from rag_engine.utils.context_tracker import get_current_context
            context_to_use = get_current_context()
            if context_to_use is None:
                logger.warning(
                    "Cannot store query trace: runtime=%s, has_context=%s, context=%s, thread_local=%s",
                    runtime is not None,
                    hasattr(runtime, "context") if runtime else False,
                    getattr(runtime, "context", None) if runtime else None,
                    context_to_use is not None,
                )
            else:
                logger.debug("Using thread-local context for query trace storage")

        if context_to_use:
            try:
                trace_entry = _build_query_trace_entry(query, docs)
                context_to_use.query_traces.append(trace_entry)
                has_confidence = trace_entry.get("confidence") is not None
                logger.info(
                    "Stored query trace: query=%r, articles=%d, has_confidence=%s, chunks_total=%d",
                    query,
                    len(trace_entry.get("articles", [])),
                    has_confidence,
                    sum(len(a.get("chunks", [])) for a in trace_entry.get("articles", [])),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to build/store query trace: %s", exc, exc_info=True)
        else:
            logger.warning("No context available to store query trace for query: %r", query)

        # Formatting is fast and CPU-bound, can stay in event loop
        return _format_articles_to_json(docs, query, top_k)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error during retrieval: %s", exc, exc_info=True)
        return json.dumps(
            {
                "error": f"Retrieval failed: {str(exc)}",
                "articles": [],
                "metadata": {"has_results": False, "query": query, "top_k_requested": top_k, "articles_count": 0},
            },
            ensure_ascii=False,
        )

