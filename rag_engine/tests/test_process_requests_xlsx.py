from __future__ import annotations

from types import SimpleNamespace

from rag_engine.scripts.process_requests_xlsx import (
    _build_answer_column,
    _safe_json_extract,
    build_markdown_request,
    format_articles_column,
    format_chunks_column,
)


def test_build_markdown_request_h1_and_body():
    md = build_markdown_request("Тема", "<p>Привет</p>")
    assert md.startswith("# Тема")
    assert "Привет" in md


def test_safe_json_extract_handles_code_fences():
    data = _safe_json_extract(
        "```json\n"
        '{ "spam_score": 0.12, "reason": "ok" }\n'
        "```"
    )
    assert data["spam_score"] == 0.12
    assert data["reason"] == "ok"


def test_format_articles_column_matches_template_shape():
    articles = [
        SimpleNamespace(
            kb_id="5000",
            metadata={"title": "Заголовок 1", "article_url": "https://kb.comindware.ru/article.php?id=5000"},
            matched_chunks=[],
        ),
        SimpleNamespace(
            kb_id="6000",
            metadata={"title": "Заголовок 2", "article_url": "https://kb.comindware.ru/article.php?id=6000"},
            matched_chunks=[],
        ),
    ]
    out = format_articles_column(articles=articles, top_k=5)
    assert "Гиперссылки (top_k=5):" in out
    assert "Запрос 1:" in out
    assert "Итоговый набор статей:" in out
    assert "[5000 - Заголовок 1](https://kb.comindware.ru/article.php?id=5000)" in out


def test_format_chunks_column_includes_ellipsis_for_many_chunks():
    chunks = [
        SimpleNamespace(page_content="A" * 120),
        SimpleNamespace(page_content="B" * 120),
        SimpleNamespace(page_content="C" * 120),
        SimpleNamespace(page_content="D" * 120),
    ]
    articles = [
        SimpleNamespace(
            kb_id="5000",
            metadata={"title": "Заголовок 1", "article_url": "https://kb.comindware.ru/article.php?id=5000"},
            matched_chunks=chunks,
        )
    ]
    out = format_chunks_column(articles=articles, max_chars=100)
    assert "Найденные чанки:" in out
    assert "Запрос 1:" in out
    assert "- kbId 5000:" in out
    assert "   …" in out
    assert "[chunk4:100]" in out


def test_build_answer_column_injects_disclaimer_and_articles_list():
    articles = [
        SimpleNamespace(
            kb_id="5000",
            metadata={"title": "Заголовок 1", "article_url": "https://kb.comindware.ru/article.php?id=5000"},
            matched_chunks=[],
        )
    ]

    # Monkeypatch the heavy answer generation helper by importing it and replacing
    # at runtime (works because _build_answer_column calls _build_answer_from_articles).
    import rag_engine.scripts.process_requests_xlsx as mod

    mod._build_answer_from_articles = lambda *, question_md, articles: "Ответ агента"  # type: ignore[assignment]

    out = _build_answer_column(question_md="# Тема\n\nТекст", articles=articles, top_k=5)
    assert "Сгенерированный ИИ контент" in out
    assert "Рекомендация обратиться" in out
    assert "Перечень рекомендованных статей:" in out
    assert "Заголовок 1" in out
    assert "Ответ агента" in out

