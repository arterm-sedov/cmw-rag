#!/usr/bin/env python
# ruff: noqa: E402
"""Terminal TUI showing RAG app process load: thread count (agent load proxy), CPU, RAM.

Finds the Python process listening on GRADIO_SERVER_PORT (from .env) and refreshes
a simple dashboard. Run in a separate terminal while the app and/or process_cmw_range
are running. Press Ctrl+C to exit.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

import psutil


def _get_port() -> int:
    port = os.environ.get("GRADIO_SERVER_PORT", "7860").strip()
    try:
        return int(port)
    except ValueError:
        return 7860


def _find_process_by_port(port: int) -> psutil.Process | None:
    for conn in psutil.net_connections(kind="inet"):
        if conn.status != "LISTEN":
            continue
        if getattr(conn.laddr, "port", None) == port:
            try:
                return psutil.Process(conn.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return None


def _bar(value: int, max_val: int, width: int = 40, fill: str = "█", empty: str = "░") -> str:
    if max_val <= 0:
        return empty * width
    n = min(width, max(0, int(width * value / max_val)))
    return fill * n + empty * (width - n)


def main() -> None:
    port = _get_port()
    refresh_sec = 2
    peak_threads = 1

    print(f"Finding RAG app (port {port})... Ctrl+C to exit.\n")

    while True:
        try:
            proc = _find_process_by_port(port)
        except Exception:
            proc = None

        if proc is None:
            if sys.platform == "win32":
                os.system("cls")
            else:
                os.system("clear")
            print("  RAG App Agent Load")
            print("  " + "=" * 50)
            print(f"  Process not found (no listener on port {port}).")
            print("  Start the app with: python rag_engine/api/app.py")
            print("  " + "=" * 50)
            time.sleep(refresh_sec)
            continue

        try:
            thread_count = proc.num_threads()
            peak_threads = max(peak_threads, thread_count)
            bar_scale = max(peak_threads, 20)
            cpu = proc.cpu_percent(interval=refresh_sec)
            mem = proc.memory_info()
            mem_mb = mem.rss / (1024 * 1024)
            cmdline = " ".join(proc.cmdline()[-2:]) if proc.cmdline() else "?"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            if sys.platform == "win32":
                os.system("cls")
            else:
                os.system("clear")
            print("  RAG App Agent Load")
            print("  " + "=" * 50)
            print("  Process exited or access denied.")
            print("  " + "=" * 50)
            time.sleep(refresh_sec)
            continue

        if sys.platform == "win32":
            os.system("cls")
        else:
            os.system("clear")
        print("  RAG App Agent Load")
        print("  " + "=" * 50)
        bar = _bar(thread_count, bar_scale)
        print(f"  PID:        {proc.pid}")
        print(f"  Threads:    {thread_count:4d}  [{bar}] peak={peak_threads}")
        print(f"  CPU:        {cpu:5.1f}%")
        print(f"  RAM:        {mem_mb:5.1f} MB")
        print(f"  Command:    {cmdline}")
        print("  " + "=" * 50)
        print(f"  Refreshing every {refresh_sec}s...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
