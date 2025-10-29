from __future__ import annotations

import logging

from rag_engine.utils.logging_manager import setup_logging


def test_setup_logging_idempotent():
    root = logging.getLogger()
    root.handlers.clear()

    setup_logging()
    first_count = len(root.handlers)

    setup_logging()
    second_count = len(root.handlers)

    assert first_count == second_count == 1

