"""Tests for CMW Document Summary Connector.

TDD: Write tests BEFORE implementation.
"""

import inspect
from unittest.mock import MagicMock, patch


def test_document_summary_connector_exists():
    """DocumentSummaryConnector class should exist."""
    from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

    assert callable(DocumentSummaryConnector)


def test_document_summary_connector_init_accepts_platform():
    """DocumentSummaryConnector.__init__ should accept platform parameter."""
    from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

    sig = inspect.signature(DocumentSummaryConnector.__init__)
    params = list(sig.parameters.keys())
    assert "platform" in params


def test_document_summary_connector_default_is_secondary():
    """DocumentSummaryConnector default platform should be 'secondary'."""
    from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

    conn = DocumentSummaryConnector()
    assert conn.platform == "secondary"


def test_document_summary_connector_process_signature():
    """process method should exist and accept record_id."""
    from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

    conn = DocumentSummaryConnector()
    assert hasattr(conn, "process")

    sig = inspect.signature(conn.process)
    params = list(sig.parameters.keys())
    assert "record_id" in params


def test_document_summary_connector_returns_process_result():
    """process should return ProcessResult."""
    from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector, ProcessResult

    with patch("rag_engine.cmw_platform.summary_connector.records.read_record") as mock_read:
        mock_read.return_value = {
            "success": False,
            "error": "Test error",
            "data": {},
        }

        conn = DocumentSummaryConnector()
        result = conn.process("test-id")

        assert isinstance(result, ProcessResult)


def test_document_summary_connector_no_document_returns_error():
    """process should return error if no document attached."""
    from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

    with patch("rag_engine.cmw_platform.summary_connector.records.read_record") as mock_read:
        mock_read.return_value = {
            "success": True,
            "data": {
                "test-id": {
                    "Commerpredloshenie": None,  # No document
                    "prompt": "Summarize this",
                }
            },
        }

        conn = DocumentSummaryConnector()
        result = conn.process("test-id")

        assert result.success is False
        assert "No document" in result.error


def test_process_result_has_required_fields():
    """ProcessResult should have success, message, error fields."""
    from rag_engine.cmw_platform.summary_connector import ProcessResult

    result = ProcessResult(success=True, message="OK", error=None)
    assert result.success is True
    assert result.message == "OK"
    assert result.error is None


class TestAgenticSummarization:
    """Tests for agentic document summarization using LangChain agent."""

    def test_no_hardcoded_keyword_matching(self):
        """Connector should NOT have hardcoded trigger keywords for web search."""
        from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

        conn = DocumentSummaryConnector()

        assert not hasattr(conn, "_fetch_search_context"), \
            "Hardcoded _fetch_search_context should be removed"
        assert not hasattr(conn, "_build_search_queries"), \
            "Hardcoded _build_search_queries should be removed"

    def test_summarize_removes_hardcoded_methods(self):
        """After refactor, _fetch_search_context and _build_search_queries should be removed."""
        from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

        conn = DocumentSummaryConnector()

        assert not hasattr(conn, "_fetch_search_context"), \
            "Hardcoded _fetch_search_context should be removed"
        assert not hasattr(conn, "_build_search_queries"), \
            "Hardcoded _build_search_queries should be removed"
