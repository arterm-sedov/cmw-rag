from __future__ import annotations

from rag_engine.api.app import format_articles_dataframe, format_confidence_badge, format_spam_badge


def test_format_spam_badge_returns_html():
    html = format_spam_badge(0.1)
    assert "Spam" in html or "Спам" in html


def test_format_confidence_badge_handles_empty():
    html = format_confidence_badge([])
    assert "Confidence" in html or "Уверенность" in html


def test_format_articles_dataframe_shape():
    rows = format_articles_dataframe(
        [
            {"kb_id": "5000", "title": "T", "url": "u", "metadata": {"title": "T", "rerank_score": 0.9, "url": "u"}}
        ]
    )
    assert rows and rows[0][0] == 1

