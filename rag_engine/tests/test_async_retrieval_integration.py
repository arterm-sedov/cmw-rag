"""Integration test for async ChromaDB retrieval."""

from __future__ import annotations

import asyncio
import logging

from rag_engine.config.settings import settings
from rag_engine.retrieval.embedder import FRIDAEmbedder
from rag_engine.storage.vector_store import ChromaStore
from rag_engine.retrieval.retriever import RAGRetriever
from rag_engine.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)


async def test_async_retrieval():
    """Test async retrieval against real ChromaDB server."""

    print("Setting up async ChromaDB retrieval test...")

    # Initialize components
    embedder = FRIDAEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )

    store = ChromaStore(collection_name=settings.chromadb_collection)

    llm_manager = LLMManager(
        provider=settings.default_llm_provider,
        model=settings.default_model,
        temperature=settings.llm_temperature,
    )

    retriever = RAGRetriever(
        embedder=embedder,
        vector_store=store,
        llm_manager=llm_manager,
        top_k_retrieve=settings.top_k_retrieve,
        top_k_rerank=settings.top_k_rerank,
        rerank_enabled=settings.rerank_enabled,
    )

    # Test basic query
    test_query = "настройка аутентификации"
    print(f"Testing query: {test_query}")

    try:
        # Measure retrieval time
        import time

        start_time = time.time()

        articles = await retriever.retrieve_async(test_query, top_k=5)

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ Retrieved {len(articles)} articles in {duration:.3f}s")

        # Display top results
        for i, article in enumerate(articles[:3], 1):
            print(
                f"  {i}. {article.metadata.get('title', article.kb_id)} (score: {article.metadata.get('rerank_score', 0):.3f})"
            )
            print(f"     URL: {article.metadata.get('article_url', 'N/A')}")
            print(f"     Content preview: {article.content[:100]}...")

        # Test parallel multi-query retrieval
        print("\nTesting parallel multi-query retrieval...")
        complex_query = "настройка почты и SMTP настройки"

        start_time = time.time()
        articles_complex = await retriever.retrieve_async(complex_query, top_k=3)
        end_time = time.time()

        print(
            f"✅ Complex query retrieved {len(articles_complex)} articles in {end_time - start_time:.3f}s"
        )

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Async retrieval test failed")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = asyncio.run(test_async_retrieval())
    if success:
        print("\n🎉 Async ChromaDB retrieval test completed successfully!")
    else:
        print("\n💥 Async ChromaDB retrieval test failed!")
        exit(1)
