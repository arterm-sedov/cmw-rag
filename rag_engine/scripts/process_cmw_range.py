#!/usr/bin/env python
# ruff: noqa: E402
"""Process a range of CMW Platform records via the process-support-request API with throttling.

Calls POST /api/v1/cmw/process-support-request for each record ID. Does not abort on
per-record failures; logs and continues.

Uses existing .env: GRADIO_SERVER_NAME, GRADIO_SERVER_PORT (base URL), CMW_API_KEY.
Override with --base-url, --throttle-sec, --timeout. App must be running.
Writes the same log output to stdout and to cmw_range.log in the current directory.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import requests

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after each record so the log file is visible immediately."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _default_base_url() -> str:
    """RAG app base URL from .env: GRADIO_SERVER_NAME + GRADIO_SERVER_PORT."""
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1").strip()
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = os.environ.get("GRADIO_SERVER_PORT", "7860").strip()
    return f"http://{host}:{port}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process CMW records via API with throttle (app must be running)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="RAG app base URL (default: from GRADIO_SERVER_NAME + GRADIO_SERVER_PORT)",
    )
    parser.add_argument("--from-id", type=int, help="Start record ID (inclusive)")
    parser.add_argument("--to-id", type=int, help="End record ID (inclusive)")
    parser.add_argument(
        "--ids",
        type=str,
        help="Comma-separated list of IDs (overrides from-id/to-id)",
    )
    parser.add_argument(
        "--throttle-sec",
        type=float,
        default=30.0,
        help="Seconds between requests (default: 30)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("CMW_API_KEY"),
        help="X-API-Key (default: CMW_API_KEY from .env)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print IDs only, do not call API",
    )
    args = parser.parse_args()

    # Log to both stdout (via basicConfig) and a simple file in the current directory
    log_file = Path.cwd() / "cmw_range.log"
    file_handler = _FlushingFileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to %s", log_file)

    base_url = (args.base_url or _default_base_url()).rstrip("/")

    if args.ids:
        record_ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    elif args.from_id is not None and args.to_id is not None:
        record_ids = [str(i) for i in range(args.from_id, args.to_id + 1)]
    else:
        parser.error("Provide either --from-id and --to-id or --ids")
        return

    total = len(record_ids)

    url = f"{base_url}/api/v1/cmw/process-support-request"
    headers = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    ok = 0
    fail = 0
    for idx, rid in enumerate(record_ids, start=1):
        prefix = f"[{idx}/{total}]"
        if args.dry_run:
            logger.info("%s Would process record %s", prefix, rid)
            continue
        try:
            r = requests.post(
                url, json={"request_id": rid}, headers=headers or None, timeout=args.timeout
            )
            ct = r.headers.get("content-type", "")
            data = r.json() if "application/json" in ct else {}
            if data.get("success"):
                ok += 1
                logger.info("%s Record %s: started", prefix, rid)
            else:
                fail += 1
                logger.warning("%s Record %s: %s", prefix, rid, data.get("error", r.text))
        except Exception as e:
            fail += 1
            logger.exception("%s Record %s: %s", prefix, rid, e)
        if not args.dry_run and idx < total:
            time.sleep(args.throttle_sec)

    if not args.dry_run:
        logger.info(
            "Summary: processed %d records (%d started, %d failed)",
            total,
            ok,
            fail,
        )
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
