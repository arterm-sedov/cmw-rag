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
    confidence_score: float | None
    num_requests: int


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

    # Debug: log what we received
    logger.debug(
        "Structured result: per_query_results=%d items, final_articles=%d items",
        len(structured.per_query_results),
        len(structured.final_articles),
    )
    if structured.per_query_results:
        first_trace = structured.per_query_results[0]
        logger.debug(
            "First query trace: query=%r, articles=%d, has_chunks=%s",
            first_trace.get("query", ""),
            len(first_trace.get("articles", [])),
            any(len(a.get("chunks", [])) > 0 for a in first_trace.get("articles", [])),
        )

    articles_text = format_articles_column_from_trace(
        structured.per_query_results, top_k=top_k, final_articles=structured.final_articles
    )
    chunks_text = format_chunks_column_from_trace(
        structured.per_query_results, max_chars=100, final_articles=structured.final_articles
    )

    # If chunks_text is empty but we have final_articles, log a warning
    if not chunks_text.strip() and structured.final_articles:
        logger.warning(
            "Chunks column is empty but final_articles has %d items. "
            "This suggests query_traces were not captured during retrieval.",
            len(structured.final_articles),
        )

    answer_text = build_answer_column_from_result(structured, top_k=top_k)
    score = structured.plan.spam_score

    # Calculate confidence score (average top_score from all query traces)
    confidence_score: float | None = None
    if structured.per_query_results:
        scores = []
        for trace in structured.per_query_results:
            conf = trace.get("confidence") if isinstance(trace, dict) else None
            if isinstance(conf, dict):
                top_score = conf.get("top_score")
                if isinstance(top_score, (int, float)):
                    scores.append(float(top_score))
        if scores:
            confidence_score = sum(scores) / len(scores)

    # Number of requests = number of tool calls (query traces)
    num_requests = len(structured.per_query_results)

    return RowResult(
        articles_text=articles_text,
        chunks_text=chunks_text,
        answer_text=answer_text,
        spam_score=score,
        confidence_score=confidence_score,
        num_requests=num_requests,
    )


def _write_output_file(
    *,
    df: pd.DataFrame,
    output_path: Path,
    out_articles: list[str],
    out_chunks: list[str],
    out_answers: list[str],
    out_spam: list[float | None],
    out_confidence: list[float | None],
    out_num_requests: list[int | None],
) -> None:
    """Write current state to output file."""
    df_output = df.copy()
    df_output["Статьи"] = out_articles
    df_output["Найденные чанки"] = out_chunks
    df_output["Ответ на обращение"] = out_answers
    df_output["Оценка спама"] = out_spam
    df_output["Уверенность"] = out_confidence
    df_output["Количество запросов"] = out_num_requests

    if output_path.suffix.lower() in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        df_output.to_excel(output_path, engine="openpyxl", index=False)
    else:
        df_output.to_csv(output_path, index=False, encoding="utf-8-sig")


async def process_file(
    *,
    input_path: Path,
    output_path: Path,
    id_col: str,
    subject_col: str,
    html_col: str,
    top_k: int,
    max_rows: int | None = None,
    start_row: int | None = None,
) -> None:
    df_full = pd.read_excel(input_path, engine="openpyxl")

    # Store original length for output initialization
    original_total_rows = len(df_full)

    # Apply start_row offset if specified (1-indexed, so row 5 means skip rows 1-4)
    df = df_full
    if start_row is not None and start_row > 1:
        df = df.iloc[start_row - 1 :]
        logger.info("Starting processing from row %d", start_row)

    # Apply max_rows limit if specified (after start_row offset)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
        logger.info("Limited processing to %d rows", max_rows)

    for col in (id_col, subject_col, html_col):
        if col not in df_full.columns:
            raise ValueError(f"Missing required column: {col!r}. Available: {list(df_full.columns)}")

    # Initialize output file with empty results (pad to match original dataframe length)
    # This ensures output indices match the original Excel row positions
    out_articles: list[str] = [""] * original_total_rows
    out_chunks: list[str] = [""] * original_total_rows
    out_answers: list[str] = [""] * original_total_rows
    out_spam: list[float | None] = [None] * original_total_rows  # Empty for unprocessed rows
    out_confidence: list[float | None] = [None] * original_total_rows  # Empty for unprocessed rows
    out_num_requests: list[int | None] = [None] * original_total_rows  # Empty for unprocessed rows

    # Write initial empty file (use full dataframe to preserve all original rows)
    _write_output_file(
        df=df_full,
        output_path=output_path,
        out_articles=out_articles,
        out_chunks=out_chunks,
        out_answers=out_answers,
        out_spam=out_spam,
        out_confidence=out_confidence,
        out_num_requests=out_num_requests,
    )
    logger.info("Initialized output file: %s", output_path)

    rows_to_process = len(df)
    for row_idx, (idx, row) in enumerate(df.iterrows(), start=1):
        req_id = row.get(id_col, "")
        try:
            subject = str(row.get(subject_col, "") or "")
            html_desc = str(row.get(html_col, "") or "")

            res = await process_one(subject=subject, html_description=html_desc, top_k=top_k)
            # Use original dataframe index (idx) to write to correct position in output
            # idx is the original row index from the Excel file (0-indexed)
            out_articles[idx] = res.articles_text
            out_chunks[idx] = res.chunks_text
            out_answers[idx] = res.answer_text
            out_spam[idx] = res.spam_score
            out_confidence[idx] = res.confidence_score
            out_num_requests[idx] = res.num_requests
        except Exception as exc:  # noqa: BLE001
            logger.exception("Row failed (idx=%s, %s=%s): %s", idx, id_col, req_id, exc)
            # Keep default values (already set to empty/None during initialization)

        # Write output file after each row (use full dataframe to preserve all original rows)
        _write_output_file(
            df=df_full,
            output_path=output_path,
            out_articles=out_articles,
            out_chunks=out_chunks,
            out_answers=out_answers,
            out_spam=out_spam,
            out_confidence=out_confidence,
            out_num_requests=out_num_requests,
        )
        logger.info("Processed row %d/%d (ID=%s, Excel row %d), output file updated", row_idx, rows_to_process, req_id, idx + 1)

        if row_idx % 25 == 0:
            logger.info("Progress: %d/%d rows processed", row_idx, rows_to_process)

    logger.info("Completed processing %d rows. Final output: %s", rows_to_process, output_path)


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
        "--start",
        type=int,
        default=None,
        help="Starting row number (1-indexed, default: 1). Rows before this will be skipped.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to process (default: process all rows from start)",
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
            start_row=args.start,
        )
    )


if __name__ == "__main__":
    main()

