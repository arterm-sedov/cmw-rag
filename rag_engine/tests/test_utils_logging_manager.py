"""Tests for logging manager."""

from __future__ import annotations

import logging

from rag_engine.utils.logging_manager import setup_logging


def test_setup_logging_idempotent():
    """Test that setup_logging can be called multiple times without creating duplicate handlers.

    Note: The agent may have already set up logging, so we test idempotency
    (handler count doesn't increase), not exact count.
    """
    root = logging.getLogger()
    initial_count = len(root.handlers)

    # First call
    setup_logging()
    first_count = len(root.handlers)

    # Second call should not add more handlers
    setup_logging()
    second_count = len(root.handlers)

    # Handler count should not increase after first call
    assert first_count == second_count, (
        f"Handler count increased from {first_count} to {second_count}"
    )
    # Should have at least the handlers we expect
    assert second_count >= initial_count
