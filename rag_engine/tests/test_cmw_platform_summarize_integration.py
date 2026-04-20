"""Integration tests for CMW document summarization.

Tests use fixtures and mocks to verify document processing flow.
Note: Full agent testing is done in unit tests (test_cmw_platform_summary_connector.py).
"""

import base64
from unittest.mock import patch

SAMPLE_DOCUMENT_TEXT = """Коммерческое предложение

От: ООО "СтройПоставка"
Кому: ООО "НефтьТранс"
Дата: 15.04.2026

Тема: Поставка строительных материалов

Итого: 875000 руб. (включая НДС)
"""


class TestDocumentSummaryFlow:
    """Tests for document summary flow through CMW platform."""

    def test_no_document_returns_error(self):
        """Test processing record without document returns error."""
        from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

        mock_record = {
            "success": True,
            "data": {
                "record-123": {
                    "Commerpredloshenie": None,
                    "promt": "Составь резюме",
                }
            },
        }

        with patch("rag_engine.cmw_platform.summary_connector.records.read_record", return_value=mock_record):
            conn = DocumentSummaryConnector()
            result = conn.process("record-123")

        assert result.success is False
        assert "No document" in result.error

    def test_get_system_prompt_returns_config_value(self):
        """Test _get_system_prompt reads from config."""
        from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

        conn = DocumentSummaryConnector(platform="secondary")
        prompt = conn._get_system_prompt()

        assert prompt
        assert len(prompt) > 0
        assert "бизнес" in prompt.lower() or "русскому" in prompt.lower()

    def test_process_with_tools_handles_base64_content(self):
        """Test _process_with_tools accepts base64 content."""
        from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

        conn = DocumentSummaryConnector()

        content = base64.b64encode(b"Test content").decode()
        doc_result = {
            "content": content,
            "filename": "doc.txt",
            "mime_type": "text/plain",
        }

        text = conn._process_with_tools(doc_result)

        assert text == ""


class TestProcessResult:
    """Tests for ProcessResult dataclass."""

    def test_process_result_success(self):
        """Test successful result structure."""
        from rag_engine.cmw_platform.summary_connector import ProcessResult

        result = ProcessResult(success=True, message="OK", summary="Summary text")
        assert result.success is True
        assert result.message == "OK"
        assert result.summary == "Summary text"
        assert result.error is None

    def test_process_result_failure(self):
        """Test failure result structure."""
        from rag_engine.cmw_platform.summary_connector import ProcessResult

        result = ProcessResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
