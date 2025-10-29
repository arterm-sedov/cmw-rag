from __future__ import annotations

from rag_engine.core.metadata_enricher import detect_code_languages, enrich_metadata


def test_detect_code_languages_extracts_unique_sorted():
    text = """```python\nprint('hi')\n```\n```SQL\nSELECT 1;\n```"""
    languages = detect_code_languages(text)

    assert languages == ["python", "sql"]


def test_enrich_metadata_populates_required_fields():
    base_meta = {"kbId": "doc1", "title": "Doc"}
    content = """```
code fence without language
```"""

    enriched = enrich_metadata(base_meta, content, chunk_index=3)

    assert enriched["chunk_index"] == 3
    assert enriched["has_code"] is True
    assert enriched["code_languages"] == []
    assert enriched["char_count"] == len(content)

