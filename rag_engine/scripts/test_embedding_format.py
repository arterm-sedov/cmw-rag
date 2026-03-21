"""Test what query is actually sent to embedding API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
from rag_engine.config.settings import settings
from rag_engine.retrieval.embedder import create_embedder


def test_actual_embedding_call():
    print("Testing what gets sent to embedding API...")

    embedder = create_embedder(settings)
    print(f"Embedder type: {type(embedder).__name__}")
    print(f"Provider: {settings.embedding_provider_type}")
    print(f"Model: {settings.embedding_model}")

    if hasattr(embedder, "config"):
        print(f"Config default_instruction: {repr(embedder.config.default_instruction)}")

    test_query = "Как настроить интеграцию с 1С?"
    print(f"\nTest query: {test_query}")

    # Check the _format_query method
    formatted = embedder._format_query(test_query, None)
    print(f"_format_query result: {repr(formatted)}")

    expected_with_instruction = (
        f"Instruct: {embedder.config.default_instruction}\nQuery: {test_query}"
    )
    print(f"Expected with instruction: {repr(expected_with_instruction)}")

    if formatted == expected_with_instruction:
        print("✅ CORRECT: Instruction is being applied!")
    else:
        print("❌ ISSUE: Instruction is NOT being applied correctly")
        print(f"  Got:      {repr(formatted)}")
        print(f"  Expected: {repr(expected_with_instruction)}")

    # Also test with explicit instruction
    explicit_inst = "Найди релевантную техническую документацию"
    formatted_explicit = embedder._format_query(test_query, explicit_inst)
    print(f"\nWith explicit instruction '{explicit_inst}':")
    print(f"_format_query result: {repr(formatted_explicit)}")

    expected_explicit = f"Instruct: {explicit_inst}\nQuery: {test_query}"
    print(f"Expected: {repr(expected_explicit)}")

    if formatted_explicit == expected_explicit:
        print("✅ CORRECT: Explicit instruction works!")
    else:
        print("❌ ISSUE: Explicit instruction not working")


if __name__ == "__main__":
    test_actual_embedding_call()
