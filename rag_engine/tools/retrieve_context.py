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
            if _retriever is not None:
                return _retriever
        from rag_engine.config.settings import settings
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
        description="The search query or question to find relevant documents from the knowledge base. "
        "This should be a clear, specific question or search phrase. ",
        min_length=1,
    )
    top_k: int | None = Field(
        default=None,
        description="Maximum number of articles to retrieve. If not specified, uses the system's "
        "default top_k_rerank setting (typically 5-10 articles). "
        "Use a smaller value (e.g., 3) for focused retrieval, or larger (e.g., 10) "
        "for comprehensive coverage. "
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
    def validate_top_k(cls, v: int | None) -> int | None:
        """Validate that top_k is positive if provided."""
        if v is not None and v <= 0:
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


@tool("retrieve_context", args_schema=RetrieveContextSchema)
async def retrieve_context(
    query: str,
    top_k: int | None = None,
    runtime: ToolRuntime[AgentContext, None] | None = None,
) -> str:
    """Retrieve relevant context articles from the knowledge base using semantic search.

    This tool searches the Comindware knowledge base for articles relevant to your query using.

    It returns formatted context with article titles, URLs, and content ready for consumption.

    **Use this tool:**
    - When you need information from the knowledge base to answer a user's question
    - When the user's request is vague or ambiguous call this tool multiple times
      with unique queries to find comprehensive information
    - When you need to explore different aspects of a topic
    - When initial results are insufficient and you want to refine your search

    **Iterative search strategy for vague requests:**

    **Query best practices:**
    - Always query the knowledge base in Russian, even if the question is in English
    - Use specific, focused queries for better results (e.g., "настройка аутентификации" vs "аутентификация")
    - Combine results from multiple queries to build a comprehensive understanding
    - Usually 1-3 quality search queries (with reasonable top_k) are enough to answer the question
    - If no results found, try broader, alternative phrasings, synonyms or related terms
    - Break down vague or complex questions into multiple focused queries with different query angles
    - The knowledge base is focused to the Comindware Platform and its use cases, hence:
      - Avoid including the term "Comindware Platform" in search queries unless really needed.
      - For general, business or industry-specific questions extract technical and platform-relevant search queries (excluding industry/business keywords).

    **Query decomposition examples:**
    For better search results, paraphrase and split the user question into several unique queries, using different phrases and keywords.
    Call retrieve_context multiple times with unique queries. Do not search for similar queries more than once.

    Examples:
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
        * тройки
        * написание троек
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

    Args:
        query: Search query to find relevant articles. Be specific and focused.
        top_k: Optional limit on number of articles (default uses system setting, typically 5-10)

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
        docs = await run_in_thread_pool(retriever.retrieve, query, top_k=top_k)
        logger.info("Retrieved %d articles for query: %s", len(docs), query)

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

