#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RowResult:
    articles_text: str
    chunks_text: str
    answer_text: str
    spam_score: float


def build_markdown_request(subject: str, html_description: str) -> str:
    subject = (subject or "").strip()
    html_description = html_description or ""

    # Local import so unit tests don’t require heavy optional deps at import time.
    from markdownify import markdownify as html_to_md

    body_md = html_to_md(html_description, heading_style="ATX").strip()
    parts = []
    if subject:
        parts.append(f"# {subject}")
    if body_md:
        parts.append(body_md)
    return "\n\n".join(parts).strip()


def _get_api_app():
    # Imported lazily to avoid importing heavy app singletons during unit tests.
    from rag_engine.api import app as api_app

    return api_app


def _safe_json_extract(text: str) -> dict[str, Any]:
    """Parse JSON from a model response, tolerating code fences and extra text."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty JSON response")

    # Strip ```json ... ``` fences if present
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        # Some models return 'json\n{...}'
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract the first {...} block
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class SpamScoreResult(BaseModel):
    """Structured output schema for spam score classification."""

    spam_score: float = Field(
        ...,
        description="Number in [0,1] where 1 = definitely NOT a Comindware support/dev ticket, "
        "0 = definitely a relevant Comindware support/dev ticket",
    )
    reason: str = Field(..., description="Short string (<= 20 words) explaining the score")


def format_articles_column(*, articles: Iterable[Any], top_k: int) -> str:
    lines: list[str] = [f"Гиперссылки (top_k={top_k}):", "", "Запрос 1:"]

    ranked = list(articles)[: max(0, int(top_k))]
    for a in ranked:
        title = (getattr(a, "metadata", {}) or {}).get("title") or getattr(a, "kb_id", "") or "Untitled"
        url = (getattr(a, "metadata", {}) or {}).get("article_url") or (getattr(a, "metadata", {}) or {}).get("url")
        kb_id = getattr(a, "kb_id", "") or ""
        label = f"{kb_id} - {title}".strip(" -")
        link = f"[{label}]({url})" if url else label
        lines.append(f"- {link}")

    lines.extend(["", "Ранжировать по релевантности", "", "Итоговый набор статей:"])
    for a in ranked:
        title = (getattr(a, "metadata", {}) or {}).get("title") or getattr(a, "kb_id", "") or "Untitled"
        url = (getattr(a, "metadata", {}) or {}).get("article_url") or (getattr(a, "metadata", {}) or {}).get("url")
        kb_id = getattr(a, "kb_id", "") or ""
        label = f"{kb_id} - {title}".strip(" -")
        link = f"[{label}]({url})" if url else label
        lines.append(f"- {link}")

    return "\n".join(lines).strip() + "\n"


def _normalize_chunk_text(text: str) -> str:
    return " ".join((text or "").split())


def format_chunks_column(*, articles: Iterable[Any], max_chars: int = 100) -> str:
    lines: list[str] = ["Найденные чанки:", "", "Запрос 1:"]

    for a in articles:
        kb_id = getattr(a, "kb_id", "") or ""
        lines.append(f"- kbId {kb_id}:")

        chunks = list(getattr(a, "matched_chunks", []) or [])
        if not chunks:
            lines.append("   - [chunk1:0] ...")
            continue

        def _chunk_text(c: Any) -> str:
            return getattr(c, "page_content", None) or getattr(c, "content", "") or ""

        def _emit(idx: int, c: Any) -> None:
            snippet = _normalize_chunk_text(_chunk_text(c))[:max_chars]
            suffix = "..." if len(_normalize_chunk_text(_chunk_text(c))) > max_chars else ""
            lines.append(f"   - [chunk{idx}:{max_chars}] {snippet}{suffix}")

        if len(chunks) <= 3:
            for i, c in enumerate(chunks, 1):
                _emit(i, c)
        else:
            # Show a small head, an ellipsis line, and the last chunk (matches template intent)
            for i, c in enumerate(chunks[:2], 1):
                _emit(i, c)
            lines.append("   …")
            _emit(len(chunks), chunks[-1])

    lines.extend(["", "Чанки тоже ранжировать"])
    return "\n".join(lines).strip() + "\n"


def _article_title_and_url(article: Any) -> tuple[str, str]:
    meta = getattr(article, "metadata", {}) or {}
    title = meta.get("title") or getattr(article, "kb_id", "") or "Untitled"
    url = meta.get("article_url") or meta.get("url") or ""
    return str(title), str(url)


def _render_article_link_line(*, article: Any) -> str:
    kb_id = getattr(article, "kb_id", "") or ""
    title, url = _article_title_and_url(article)
    label = f"{kb_id} - {title}".strip(" -")
    return f"- [{label}]({url})" if url else f"- {label}"


def _article_to_dict(article: Any) -> dict[str, Any]:
    """Convert Article object to dict format expected by compression functions."""
    meta = dict(getattr(article, "metadata", {}) or {})
    return {
        "content": getattr(article, "content", ""),
        "metadata": meta,
        "title": meta.get("title") or getattr(article, "kb_id", "") or "Untitled",
        "url": meta.get("article_url") or meta.get("url") or "",
        "_original_article": article,  # Preserve reference for conversion back
    }


def _dict_to_article_like(article_dict: dict[str, Any]) -> Any:
    """Convert compressed dict back to Article-like object for generate() and format_with_citations."""
    original = article_dict.get("_original_article")
    if original:
        # Update content in-place to preserve all other attributes (kb_id, matched_chunks, etc.)
        original.content = article_dict["content"]
        # Update metadata if compression flag was added
        if article_dict.get("metadata", {}).get("compressed"):
            original.metadata = dict(article_dict["metadata"])
        return original

    # Fallback: create minimal Article-like object
    from types import SimpleNamespace

    return SimpleNamespace(
        content=article_dict["content"],
        metadata=article_dict.get("metadata", {}),
        kb_id=article_dict.get("metadata", {}).get("kbId") or article_dict.get("metadata", {}).get("kb_id", ""),
        matched_chunks=getattr(original, "matched_chunks", []) if original else [],
    )


def _build_answer_from_articles(*, question_md: str, articles: list[Any]) -> str:
    """Generate final answer text with citations, applying compression if needed.

    Reuses the agent's compression logic to ensure articles fit within context window.
    """
    api_app = _get_api_app()

    from rag_engine.config.settings import settings
    from rag_engine.llm.compression import compress_all_articles_proportionally_by_rank
    from rag_engine.llm.llm_manager import get_context_window
    from rag_engine.llm.prompts import USER_QUESTION_TEMPLATE_FIRST, get_system_prompt
    from rag_engine.llm.token_utils import estimate_tokens_for_request
    from rag_engine.utils.formatters import format_with_citations

    if not articles:
        return ""

    wrapped = USER_QUESTION_TEMPLATE_FIRST.format(question=question_md)

    # Convert Article objects to dicts for compression
    articles_dicts = [_article_to_dict(a) for a in articles]

    # Estimate tokens BEFORE compression (same logic as agent middleware)
    system_prompt = get_system_prompt()
    context_preview = "\n\n---\n\n".join([a["content"] for a in articles_dicts])
    from rag_engine.llm.token_utils import count_tokens

    context_tokens = count_tokens(context_preview)
    est = estimate_tokens_for_request(
        system_prompt=system_prompt,
        question=wrapped,
        context=context_preview,
        reserved_output_tokens=None,  # Will derive from settings
        overhead=100,
    )

    # Get compression thresholds from settings (same as agent)
    context_window = get_context_window(api_app.llm_manager._model_config["model"])
    threshold_pct = float(getattr(settings, "llm_compression_threshold_pct", 0.85))
    target_pct = float(getattr(settings, "llm_compression_target_pct", 0.80))
    threshold = int(context_window * threshold_pct)
    target_tokens = int(context_window * target_pct)

    # Compress if needed (same logic as compress_tool_results middleware)
    if est["total_tokens"] > threshold:
        logger.info(
            "Compressing %d articles: %d tokens exceeds threshold %d (%.1f%% of %d window)",
            len(articles_dicts),
            est["total_tokens"],
            threshold,
            threshold_pct * 100,
            context_window,
        )

        # Reserve space for system + question + overhead
        # est["input_tokens"] = system + question + context + overhead(100)
        # Since generate() doesn't bind tools, we don't need tool schema overhead
        non_article_tokens = est["input_tokens"] - context_tokens  # system + question + overhead(100)
        available_for_articles = max(0, target_tokens - non_article_tokens)

        if available_for_articles <= 0:
            logger.warning(
                "Available budget is zero or negative (%d). Using aggressive fallback: 10%% of context window",
                available_for_articles,
            )
            available_for_articles = max(300 * len(articles_dicts), int(context_window * 0.10))

        # Reuse agent's compression function
        compressed_dicts, tokens_saved = compress_all_articles_proportionally_by_rank(
            articles=articles_dicts,
            target_tokens=available_for_articles,
            guidance=question_md,
            llm_manager=api_app.llm_manager,
        )
        logger.info("Compression saved %d tokens", tokens_saved)
        articles_dicts = compressed_dicts

    # Convert back to Article-like objects for generate() and format_with_citations
    compressed_articles = [_dict_to_article_like(a) for a in articles_dicts]

    # Generate answer using compressed articles
    answer = api_app.llm_manager.generate(wrapped, compressed_articles)
    return format_with_citations(answer, compressed_articles) if compressed_articles else answer


def _build_answer_column(
    *,
    question_md: str,
    articles: list[Any],
    top_k: int,
) -> str:
    """Build the `Ответ на обращение` column matching the CSV template shape."""
    from rag_engine.llm.prompts import AI_DISCLAIMER

    answer_with_citations = _build_answer_from_articles(question_md=question_md, articles=articles).strip()

    lines: list[str] = [
        AI_DISCLAIMER.rstrip(),
        "Рекомендация обратиться в службу поддержки за разъяснениями",
        "",
        "Рекомендация ознакомиться с базой знаний и указанными статьями",
        "",
        "Перечень рекомендованных статей:",
        "",
    ]

    for a in articles[: max(0, int(top_k))]:
        title, url = _article_title_and_url(a)
        lines.append(f"- [{title}]({url})." if url else f"- {title}.")

    lines.extend(["", answer_with_citations])
    return "\n".join(lines).strip()


def spam_score(markdown_request: str) -> float:
    # IMPORTANT: do not use LLMManager.generate() here, because it injects the main
    # assistant system prompt (including off-topic paraphrasing). We want a strict
    # classifier with no retrieval and no answer-style constraints.
    system = (
        "You are a strict classifier.\n\n"
        "Task: Given a support request (subject + body), estimate whether it is a Comindware Platform "
        "support/development ticket.\n\n"
        "Rules:\n"
        "- If it’s obviously unrelated (e.g., marketing, personal, random email thread, other product), spam_score >= 0.9\n"
        "- If it’s ambiguous but could be IT/support-ish, spam_score 0.4–0.7\n"
        "- If it clearly references Comindware Platform/KB/processes/apps/configuration/integrations, spam_score <= 0.2\n"
    )
    user = f"Request:\n<<<\n{markdown_request}\n>>>"

    api_app = _get_api_app()
    model = api_app.llm_manager._chat_model(provider=None, structured_output_schema=SpamScoreResult)
    resp = model.invoke([("system", system), ("user", user)])

    # With structured output, resp is a Pydantic model instance
    if isinstance(resp, SpamScoreResult):
        score = resp.spam_score
    else:
        # Fallback: extract from content if structured output failed
        content = getattr(resp, "content", "") or ""
        data = _safe_json_extract(content)
        score = float(data.get("spam_score", 1.0))

    return _clamp01(score)


async def process_one(*, subject: str, html_description: str, top_k: int) -> RowResult:
    api_app = _get_api_app()
    md_request = build_markdown_request(subject, html_description)

    # Deterministic retrieval for columns
    retrieved_articles = list(api_app.retriever.retrieve(md_request, top_k=top_k))
    articles_text = format_articles_column(articles=retrieved_articles, top_k=top_k)
    chunks_text = format_chunks_column(articles=retrieved_articles, max_chars=100)
    answer_text = _build_answer_column(question_md=md_request, articles=retrieved_articles, top_k=top_k)

    # Spam score (LLM-only, no retrieval)
    score = spam_score(md_request)

    return RowResult(
        articles_text=articles_text,
        chunks_text=chunks_text,
        answer_text=answer_text,
        spam_score=score,
    )


async def process_file(
    *,
    input_path: Path,
    output_path: Path,
    id_col: str,
    subject_col: str,
    html_col: str,
    top_k: int,
) -> None:
    df = pd.read_excel(input_path, engine="openpyxl")

    for col in (id_col, subject_col, html_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col!r}. Available: {list(df.columns)}")

    out_articles: list[str] = []
    out_chunks: list[str] = []
    out_answers: list[str] = []
    out_spam: list[float] = []

    for idx, row in df.iterrows():
        req_id = row.get(id_col, "")
        try:
            subject = str(row.get(subject_col, "") or "")
            html_desc = str(row.get(html_col, "") or "")

            res = await process_one(subject=subject, html_description=html_desc, top_k=top_k)
            out_articles.append(res.articles_text)
            out_chunks.append(res.chunks_text)
            out_answers.append(res.answer_text)
            out_spam.append(res.spam_score)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Row failed (idx=%s, %s=%s): %s", idx, id_col, req_id, exc)
            out_articles.append("")
            out_chunks.append("")
            out_answers.append("")
            out_spam.append(1.0)

        if (idx + 1) % 25 == 0:
            logger.info("Processed %d rows", idx + 1)

    df["Статьи"] = out_articles
    df["Столбец1"] = out_chunks
    df["Ответ на обращение"] = out_answers
    df["SpamScore"] = out_spam

    if output_path.suffix.lower() in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        df.to_excel(output_path, engine="openpyxl", index=False)
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process support requests from XLSX via CMW RAG agent")
    parser.add_argument("--input", required=True, help="Path to input .xlsx")
    parser.add_argument("--output", required=True, help="Path to output .xlsx or .csv")
    parser.add_argument("--id-col", default="ID", help="Request ID column name (default: ID)")
    parser.add_argument(
        "--subject-col",
        default="Название",
        help="Request subject column name (default: Название)",
    )
    parser.add_argument(
        "--html-col",
        default="Описание",
        help="Request HTML description column name (default: Описание)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K articles to retrieve (default: 5)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    asyncio.run(
        process_file(
            input_path=input_path,
            output_path=output_path,
            id_col=str(args.id_col),
            subject_col=str(args.subject_col),
            html_col=str(args.html_col),
            top_k=int(args.top_k),
        )
    )


if __name__ == "__main__":
    main()

