"""LangChain 1.0 tool for RAG context retrieval.

This tool is self-sufficient and handles all retrieval mechanics internally.
The agent only needs to import and use the tool - no setup required.
"""
from __future__ import annotations

import json
import logging

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field, field_validator

from rag_engine.retrieval.retriever import Article, RAGRetriever

logger = logging.getLogger(__name__)

# Module-level retriever instance (lazy singleton)
_retriever: RAGRetriever | None = None


def _get_or_create_retriever() -> RAGRetriever:
    """Get or create the retriever instance (lazy singleton).

    This creates the retriever on first use and reuses it for subsequent calls.
    The retriever is stateless, so it's safe to share across sessions.

    Returns:
        RAGRetriever instance
    """
    global _retriever
    if _retriever is None:
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
        "This should be a clear, specific question or search phrase. "
        "Examples: 'How to configure authentication?', 'What is RAG?', 'user permissions setup'. "
        "RU: Поисковый запрос или вопрос для поиска релевантных документов из базы знаний. "
        "Должен быть четким и конкретным.",
        min_length=1,
    )
    top_k: int | None = Field(
        default=None,
        description="Maximum number of articles to retrieve. If not specified, uses the system's "
        "default top_k_rerank setting (typically 5-10 articles). "
        "Use a smaller value (e.g., 3) for focused retrieval, or larger (e.g., 10) "
        "for comprehensive coverage. "
        "RU: Максимальное количество статей для получения. Если не указано, используется "
        "настроенное значение top_k_rerank (обычно 5-10 статей).",
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
    """Convert Article objects to JSON format."""
    articles_data = []
    for article in articles:
        # Extract title with fallback
        title = article.metadata.get("title", article.kb_id)

        # Extract URL with fallback chain
        url = (
            article.metadata.get("article_url")
            or article.metadata.get("url")
            or f"https://kb.comindware.ru/article.php?id={article.kb_id}"
        )

        articles_data.append({
            "kb_id": article.kb_id,
            "title": title,
            "url": url,
            "content": article.content,
            "metadata": dict(article.metadata),
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
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool("retrieve_context", args_schema=RetrieveContextSchema)
def retrieve_context(
    query: str,
    top_k: int | None = None,
    runtime: ToolRuntime | None = None,
) -> str:
    """
    Retrieve relevant context documents from the knowledge base using semantic search.

    This tool searches the indexed knowledge base for articles relevant to your query using
    vector search, reranking, and intelligent context budgeting. It returns formatted context
    with article titles, URLs, and content ready for consumption.

    **When to use this tool:**
    - When you need information from the knowledge base to answer a user's question
    - When the user's request is vague or ambiguous - you can call this tool multiple times
      with different query variations to find comprehensive information
    - When you need to explore different aspects of a topic
    - When initial results are insufficient and you want to refine your search

    **Iterative search strategy for vague requests:**
    - For vague or complex user requests, call this tool multiple times with different query angles
    - Example: If user asks "how do I set things up?", try queries like:
     * "initial setup configuration"
     * "getting started guide"
     * "installation requirements"
    - Combine results from multiple queries to build a comprehensive understanding
    - If no results found, try broader or alternative phrasings

    **Query best practices:**
    - Use specific, focused queries for better results (e.g., "authentication setup" vs "setup")
    - Break down complex questions into multiple focused queries
    - Use synonyms or related terms if initial query returns no results
    - Consider different aspects of the topic (e.g., "configuration", "troubleshooting", "examples")

    Args:
        query: Search query or question to find relevant documents. Be specific and focused.
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

        Each article includes title, URL, content (full or summarized), and complete metadata
        for citations. If no documents found, returns JSON with empty articles array and
        has_results: false.

    **Note**: You can call this tool multiple times in the same conversation turn to gather
    information from different angles or aspects of a topic. Each call is independent.
    """
    try:
        # Get or create retriever (lazy initialization)
        retriever = _get_or_create_retriever()

        # Estimate TOTAL reserved tokens for accurate context budgeting
        # This includes:
        # 1. Conversation history (user/assistant messages)
        # 2. Tool results from previous calls IN THIS TURN (critical for multi-tool scenarios)
        conversation_tokens = 0
        tool_result_tokens = 0

        if runtime and hasattr(runtime, 'state'):
            messages = runtime.state.get("messages", [])
            for msg in messages:
                # Get message content (handle both dict and LangChain objects)
                if hasattr(msg, "content"):
                    content = msg.content
                else:
                    content = msg.get("content", "") if isinstance(msg, dict) else ""

                if isinstance(content, str) and content:
                    msg_tokens = len(content) // 4  # Fast approximation

                    # Classify: is this a tool result or conversation?
                    # Tool results are JSON with "articles" key, much larger than normal messages
                    msg_type = getattr(msg, "type", None)
                    is_tool_result = (
                        msg_type == "tool" or
                        (isinstance(content, str) and '"articles"' in content and len(content) > 5000)
                    )

                    if is_tool_result:
                        # This is a tool result from a previous retrieve_context call
                        # Tool results are JSON-heavy and bloat context significantly
                        tool_result_tokens += msg_tokens
                    else:
                        # Regular conversation message
                        conversation_tokens += msg_tokens

        # Total reserved includes BOTH conversation AND accumulated tool results
        total_reserved_tokens = conversation_tokens + tool_result_tokens

        logger.info(
            "Retrieving articles: query=%s, top_k=%s, reserved_tokens=%d "
            "(conversation: %d, tool_results: %d)",
            query[:100],
            top_k,
            total_reserved_tokens,
            conversation_tokens,
            tool_result_tokens,
        )

        # Retrieve articles with accurate context budgeting
        # Retriever will reduce article count/size if reserved tokens are high
        docs = retriever.retrieve(query, top_k=top_k, reserved_tokens=total_reserved_tokens)
        logger.info("Retrieved %d articles for query: %s", len(docs), query)
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

