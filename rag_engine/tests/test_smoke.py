from __future__ import annotations

from rag_engine.core.chunker import split_text


def test_split_text_basic():
    text = "# Title\n\nSome content\n\n```python\nprint('x')\n```\n"
    chunks = list(split_text(text, chunk_size=50, chunk_overlap=10))
    assert chunks, "Expected non-empty chunks"


