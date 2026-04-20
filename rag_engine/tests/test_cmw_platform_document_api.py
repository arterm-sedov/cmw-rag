"""Tests for CMW Document API.

TDD: Write tests BEFORE implementation.
"""

import inspect

import pytest


def test_get_document_content_exists():
    """get_document_content function should exist."""
    from rag_engine.cmw_platform.document_api import get_document_content

    assert callable(get_document_content)


def test_get_document_content_accepts_platform_param():
    """get_document_content should accept platform parameter."""
    from rag_engine.cmw_platform.document_api import get_document_content

    sig = inspect.signature(get_document_content)
    params = list(sig.parameters.keys())
    assert "platform" in params, f"Expected 'platform' in {params}"


def test_get_document_content_signature():
    """get_document_content should have correct signature."""
    from rag_engine.cmw_platform.document_api import get_document_content

    sig = inspect.signature(get_document_content)
    params = list(sig.parameters.keys())

    assert "document_id" in params
    assert "platform" in params


def test_default_platform_is_primary():
    """Default platform should be 'primary'."""
    from rag_engine.cmw_platform.document_api import DEFAULT_PLATFORM

    assert DEFAULT_PLATFORM == "primary"
