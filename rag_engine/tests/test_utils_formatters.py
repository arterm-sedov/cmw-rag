from __future__ import annotations

from types import SimpleNamespace

from rag_engine.utils.formatters import format_with_citations


def test_format_with_citations_adds_section_links():
    docs = [
        SimpleNamespace(
            metadata={
                "title": "Intro",
                "url": "https://example.com/intro",
                "section_anchor": "#overview",
            }
        )
    ]

    result = format_with_citations("Answer", docs)

    assert "## Sources" in result
    assert "[Intro](https://example.com/intro#overview)" in result


def test_format_with_citations_handles_missing_url():
    docs = [SimpleNamespace(metadata={"title": "Doc without URL"})]
    result = format_with_citations("Answer", docs)

    assert "Doc without URL" in result


def test_format_with_citations_no_docs_returns_answer():
    result = format_with_citations("Answer only", [])
    assert result == "Answer only"

