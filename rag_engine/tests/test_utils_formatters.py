from __future__ import annotations

from types import SimpleNamespace

from rag_engine.utils.formatters import format_with_citations


def test_format_with_citations_adds_section_links():
    docs = [
        SimpleNamespace(
            metadata={
                "kbId": "42",
                "title": "Intro",
                "url": "https://example.com/intro",
                "section_anchor": "#overview",
            }
        )
    ]

    result = format_with_citations("Answer", docs)

    # Accept both Russian and English headings for sources
    assert any(h in result for h in ("## Источники:", "## Sources:"))
    assert "[Intro](https://example.com/intro#overview)" in result


def test_format_with_citations_handles_missing_url():
    docs = [SimpleNamespace(metadata={"kbId": "99", "title": "Doc without URL"})]
    result = format_with_citations("Answer", docs)

    assert "Doc without URL" in result


def test_format_with_citations_no_docs_returns_answer():
    result = format_with_citations("Answer only", [])
    assert result == "Answer only"

