from __future__ import annotations

from rag_engine.api.app import format_articles_dataframe


def test_format_articles_dataframe_shape():
    rows = format_articles_dataframe(
        [
            {
                "kb_id": "5000",
                "title": "T",
                "url": "u",
                "metadata": {
                    "title": "T",
                    "rerank_score": 0.92,
                    "normalized_rank": 0.123,
                    "url": "u",
                },
            }
        ]
    )
    assert rows and rows[0][0] == 1  # Rank
    assert rows[0][1] == "T"  # Title
    assert rows[0][2] == "0.92"  # Confidence (rerank_score)
    assert rows[0][3] == "0.123"  # Normalized rank
    assert rows[0][4] == "u"  # URL
    assert len(rows[0]) == 5  # 5 columns total


def test_format_articles_dataframe_handles_missing_normalized_rank():
    rows = format_articles_dataframe(
        [{"kb_id": "5001", "title": "T2", "url": "u2", "metadata": {"rerank_score": 0.8}}]
    )
    assert rows and rows[0][3] == ""  # Normalized rank missing = empty string
