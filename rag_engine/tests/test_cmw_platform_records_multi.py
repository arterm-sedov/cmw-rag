"""Tests for multi-platform CMW records and connector.

TDD: Write tests BEFORE implementation.
"""

import inspect
from unittest.mock import patch

import pytest


def test_read_record_accepts_platform_param():
    """read_record should accept platform parameter."""
    from rag_engine.cmw_platform.records import read_record

    sig = inspect.signature(read_record)
    params = list(sig.parameters.keys())
    assert "platform" in params, f"Expected 'platform' in {params}"


def test_create_record_accepts_platform_param():
    """create_record should accept platform parameter."""
    from rag_engine.cmw_platform.records import create_record

    sig = inspect.signature(create_record)
    params = list(sig.parameters.keys())
    assert "platform" in params, f"Expected 'platform' in {params}"


def test_update_record_accepts_platform_param():
    """update_record should accept platform parameter."""
    from rag_engine.cmw_platform.records import update_record

    sig = inspect.signature(update_record)
    params = list(sig.parameters.keys())
    assert "platform" in params, f"Expected 'platform' in {params}"


def test_platform_connector_init_accepts_platform():
    """PlatformConnector.__init__ should accept platform parameter."""
    from rag_engine.cmw_platform.connector import PlatformConnector

    sig = inspect.signature(PlatformConnector.__init__)
    params = list(sig.parameters.keys())
    assert "platform" in params, f"Expected 'platform' in {params}"


def test_platform_connector_default_is_primary():
    """PlatformConnector default platform should be 'primary'."""
    from rag_engine.cmw_platform.connector import PlatformConnector

    conn = PlatformConnector()
    assert conn.platform == "primary"


def test_platform_connector_secondary_is_set():
    """PlatformConnector with platform='secondary' should work."""
    from rag_engine.cmw_platform.connector import PlatformConnector

    conn = PlatformConnector(platform="secondary")
    assert conn.platform == "secondary"


def test_platform_connector_start_request_uses_platform():
    """start_request should use the platform config."""
    from rag_engine.cmw_platform.connector import PlatformConnector

    with patch("rag_engine.cmw_platform.records.read_record") as mock_read:
        mock_read.return_value = {
            "success": False,
            "error": "Test error",
            "data": {},
        }

        conn = PlatformConnector(platform="secondary")
        conn.start_request("test-id-123")

        # Verify read_record was called with platform parameter
        mock_read.assert_called_once()
        _, kwargs = mock_read.call_args
        assert kwargs.get("platform") == "secondary"
