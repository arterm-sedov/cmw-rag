"""Metadata utilities for document processing."""
from __future__ import annotations

import re


def extract_numeric_kbid(kb_id: str | int | None) -> str | None:
    """Extract the first numeric sequence from kbId, handling suffixes like '4578-toc'.

    Args:
        kb_id: kbId value (can be string, int, or None)

    Returns:
        String with just the numeric part, or None if no numeric sequence found
    """
    if kb_id is None:
        return None
    match = re.match(r'^(\d+)', str(kb_id))
    return match.group(1) if match else None

