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


class TestAsyncStart:
    """Tests for start() async fire-and-forget method."""

    def _make_pipeline_config(self):
        return {
            "input": {"attributes": {"document_file": "doc", "user_prompt": "p"}},
            "output": {},
        }

    def test_start_exists(self):
        """start() method should exist on DocumentSummaryConnector."""
        from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

        conn = DocumentSummaryConnector()
        assert hasattr(conn, "start")

    def test_start_returns_process_result_on_read_failure(self):
        """start() should return ProcessResult when record is unreadable."""
        from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector, ProcessResult

        with patch("rag_engine.cmw_platform.summary_connector.records.read_record") as mock_read:
            mock_read.return_value = {"success": False, "error": "Not found", "data": {}}
            with patch(
                "rag_engine.cmw_platform.summary_connector.config.load_pipeline_config"
            ) as mock_cfg:
                mock_cfg.return_value = self._make_pipeline_config()
                conn = DocumentSummaryConnector()
                result = conn.start("test-id")
                assert isinstance(result, ProcessResult)
                assert result.success is False

    def test_start_success_returns_ack_without_summary(self):
        """start() should return ACK without summary when record is readable."""
        from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

        with patch("rag_engine.cmw_platform.summary_connector.records.read_record") as mock_read:
            mock_read.return_value = {
                "success": True,
                "data": {"test-id": {"doc": {"id": "file-1"}, "p": ""}},
            }
            with patch(
                "rag_engine.cmw_platform.summary_connector.config.load_pipeline_config"
            ) as mock_cfg:
                mock_cfg.return_value = self._make_pipeline_config()
                with patch("rag_engine.cmw_platform.summary_connector.threading.Thread") as mock_thread:
                    conn = DocumentSummaryConnector()
                    result = conn.start("test-id")
                    assert result.success is True
                    assert result.summary is None
                    mock_thread.assert_called_once()

    def test_start_spawns_process_in_daemon_thread(self):
        """start() should spawn process() in a daemon thread."""
        from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

        with patch("rag_engine.cmw_platform.summary_connector.records.read_record") as mock_read:
            mock_read.return_value = {
                "success": True,
                "data": {"test-id": {"doc": {"id": "file-1"}, "p": ""}},
            }
            with patch(
                "rag_engine.cmw_platform.summary_connector.config.load_pipeline_config"
            ) as mock_cfg:
                mock_cfg.return_value = self._make_pipeline_config()
                with patch("rag_engine.cmw_platform.summary_connector.threading.Thread") as mock_thread:
                    conn = DocumentSummaryConnector()
                    conn.start("test-id")
                    call_kwargs = mock_thread.call_args[1]
                    assert call_kwargs["target"] == conn.process
                    assert call_kwargs["daemon"] is True

    def test_start_no_document_attr_returns_error(self):
        """start() should return error if no document attribute configured."""
        from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

        with patch(
            "rag_engine.cmw_platform.summary_connector.config.load_pipeline_config"
        ) as mock_cfg:
            mock_cfg.return_value = {"input": {"attributes": {}}, "output": {}}
            conn = DocumentSummaryConnector()
            result = conn.start("test-id")
            assert result.success is False
            assert "No document attribute" in result.error


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
