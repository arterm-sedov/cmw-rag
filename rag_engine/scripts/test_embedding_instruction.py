"""Test if embedding instruction is being used."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_engine.config.settings import settings
from rag_engine.retrieval.embedder import create_embedder


def test_embedding_instruction():
    print("Testing embedding instruction usage...")

    # Create embedder
    embedder = create_embedder(settings)
    print(f"Embedder type: {type(embedder).__name__}")
    print(f"Provider: {settings.embedding_provider_type}")
    print(f"Model: {settings.embedding_model}")

    # Check if it has default_instruction
    if hasattr(embedder, "default_instruction"):
        print(f"Default instruction: {embedder.default_instruction}")
    elif hasattr(embedder, "config") and hasattr(embedder.config, "default_instruction"):
        print(f"Config default instruction: {embedder.config.default_instruction}")

    # Test embedding a query
    test_query = "Как настроить интеграцию с 1С?"
    print(f"\nTest query: {test_query}")

    try:
        # This should use the default instruction
        embedding = embedder.embed_query(test_query)
        print(f"Embedding generated successfully, length: {len(embedding)}")
        print("First 5 values:", embedding[:5])
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_embedding_instruction()
