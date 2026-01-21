"""Formatting helpers for batch (Excel) outputs and UI debug panels."""

from __future__ import annotations

from typing import Any

from rag_engine.llm.prompts import AI_DISCLAIMER


def _md_link(label: str, url: str | None) -> str:
    if url:
        return f"[{label}]({url})"
    return label


def format_articles_column_from_trace(
    query_results: list[dict], top_k: int, final_articles: list[dict] | None = None
) -> str:
    """Format 'Статьи' column from per-query trace dicts.

    Args:
        query_results: List of per-query trace dicts with 'query' and 'articles' keys.
        top_k: Maximum number of articles to show per query and in final list.
        final_articles: Optional final deduplicated articles list (from StructuredAgentResult.final_articles).
    """
    lines: list[str] = []
    for qi, qr in enumerate(query_results or [], start=1):
        q = qr.get("query", "")
        lines.append(f"Запрос {qi}: {q}".rstrip())
        arts = (qr.get("articles") or [])[: max(0, int(top_k))]
        for a in arts:
            kb_id = a.get("kb_id", "")
            title = a.get("title", kb_id) or kb_id or "Untitled"
            url = a.get("url", "")
            label = f"{kb_id} - {title}".strip(" -")
            lines.append(f"- {_md_link(label, url)}")
        lines.append("")

    # Add final deduplicated articles list if provided
    if final_articles:
        lines.extend(["", "Итоговый набор статей:"])
        for a in (final_articles or [])[: max(0, int(top_k))]:
            kb_id = a.get("kb_id", "")
            title = a.get("title", kb_id) or kb_id or "Untitled"
            url = a.get("url") or (a.get("metadata", {}) or {}).get("article_url") or (a.get("metadata", {}) or {}).get("url")
            label = f"{kb_id} - {title}".strip(" -")
            lines.append(f"- {_md_link(label, url)}")
    return "\n".join(lines).strip() + "\n"


def format_chunks_column_from_trace(
    query_results: list[dict], max_chars: int = 100, final_articles: list[dict] | None = None
) -> str:
    """Format 'Найденные чанки' column from per-query trace dicts.

    Args:
        query_results: List of per-query trace dicts with 'query', 'articles', and 'chunks' keys.
        max_chars: Maximum characters to show per chunk snippet.
        final_articles: Optional final articles list (used for fallback when query_results is empty).
    """
    #lines: list[str] = ["Найденные чанки:", ""]
    lines: list[str] = []
    
    if not query_results:
        # Fallback: if no query traces, show a diagnostic message
        if final_articles:
            lines.append("Запросы не зафиксированы в trace (query_traces пуст).")
            lines.append("")
            lines.append("Найденные статьи (без деталей чанков):")
            for a in (final_articles or [])[:10]:  # Show up to 10 articles
                kb_id = a.get("kb_id", "") or ""
                title = a.get("title", kb_id) or kb_id or "Untitled"
                lines.append(f"- kbId {kb_id}: {title}")
        else:
            lines.append("Данные о чанках недоступны (query_traces пуст, final_articles пуст).")
        return "\n".join(lines).strip() + "\n"
    
    for qi, qr in enumerate(query_results or [], start=1):
        q = qr.get("query", "")
        lines.append(f"Запрос {qi}: {q}".rstrip())

        for a in qr.get("articles", []) or []:
            kb_id = a.get("kb_id", "") or ""
            lines.append(f"- kbId {kb_id}:")
            chunks = a.get("chunks") or []
            if not chunks:
                lines.append("   - [chunk1:0] ...")
                continue

            def _emit(idx: int, ch: dict) -> None:
                snippet = str(ch.get("snippet", "") or "")
                norm = " ".join(snippet.split())
                trimmed = norm[:max_chars]
                suffix = "..." if len(norm) > max_chars else ""
                score = ch.get("rerank_score_raw")
                score_str = f" (score: {score:.3f})" if score is not None else ""
                lines.append(f"   - [chunk{idx}:{max_chars}]{score_str} {trimmed}{suffix}")

            if len(chunks) <= 3:
                for i, ch in enumerate(chunks, 1):
                    _emit(i, ch)
            else:
                for i, ch in enumerate(chunks[:2], 1):
                    _emit(i, ch)
                lines.append("   …")
                _emit(len(chunks), chunks[-1])
        lines.append("")

    #lines.append("Чанки тоже ранжировать")
    return "\n".join(lines).strip() + "\n"


def build_answer_column_from_result(result: Any, top_k: int = 5) -> str:
    """Build 'Ответ на обращение' column from StructuredAgentResult-like object."""
    # result.answer_text is already grounded with citations; we prepend disclaimer and helpful boilerplate.
    articles = getattr(result, "final_articles", []) or []
    lines: list[str] = [AI_DISCLAIMER.strip(), ""]
    lines.append("Если ответ не помог — обратитесь в техподдержку.")
    lines.append("")
    lines.append("Перечень рекомендованных статей:")
    for a in articles[: max(0, int(top_k))]:
        kb_id = a.get("kb_id", "")
        title = a.get("title", kb_id) or kb_id or "Untitled"
        url = a.get("url") or (a.get("metadata", {}) or {}).get("article_url") or (a.get("metadata", {}) or {}).get("url")
        label = f"{kb_id} - {title}".strip(" -")
        lines.append(f"- {_md_link(label, url)}")
    lines.append("")
    lines.append(getattr(result, "answer_text", "") or "")
    return "\n".join(lines).strip() + "\n"

