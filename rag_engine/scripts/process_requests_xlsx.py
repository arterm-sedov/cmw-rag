#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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


async def process_one(*, subject: str, html_description: str, top_k: int) -> RowResult:
    api_app = _get_api_app()
    md_request = build_markdown_request(subject, html_description)

    # Structured agent call (same behavior as interactive agent, but returns trace + spam score)
    structured = await api_app.ask_comindware_structured(
        md_request,
        include_per_query_trace=True,
    )

    from rag_engine.utils.trace_formatters import (
        build_answer_column_from_result,
        format_articles_column_from_trace,
        format_chunks_column_from_trace,
    )

    articles_text = format_articles_column_from_trace(structured.per_query_results, top_k=top_k)
    chunks_text = format_chunks_column_from_trace(structured.per_query_results, max_chars=100)
    answer_text = build_answer_column_from_result(structured, top_k=top_k)
    score = structured.plan.spam_score

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
    max_rows: int | None = None,
) -> None:
    df = pd.read_excel(input_path, engine="openpyxl")

    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
        logger.info("Limited processing to first %d rows", max_rows)

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
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output .xlsx or .csv (default: auto-generated from input filename with timestamp)",
    )
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to process (default: process all rows)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    input_path = Path(args.input).resolve()

    # Generate output path if not provided, otherwise use provided path
    if args.output is None:
        # Use same directory and extension as input, add timestamp suffix to stem
        output_path = input_path
    else:
        output_path = Path(args.output).resolve()

    # Add timestamp suffix to output filename: _processed_YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    output_path = output_path.with_stem(f"{output_path.stem}_processed{timestamp}")
    asyncio.run(
        process_file(
            input_path=input_path,
            output_path=output_path,
            id_col=str(args.id_col),
            subject_col=str(args.subject_col),
            html_col=str(args.html_col),
            top_k=int(args.top_k),
            max_rows=args.limit,
        )
    )


if __name__ == "__main__":
    main()

