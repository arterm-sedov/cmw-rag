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
                    "document_file": None,
                    "user_prompt": "Составь резюме",
                }
            },
        }

        with patch("rag_engine.cmw_platform.summary_connector.records.read_record", return_value=mock_record):
            with patch("rag_engine.cmw_platform.summary_connector.config.load_pipeline_config") as mock_cfg:
                mock_cfg.return_value = {
                    "input": {"attributes": {"document_file": "Commerpredloshenie", "user_prompt": "promt"}},
                    "output": {"summary_attribute": "summary"},
                }
                conn = DocumentSummaryConnector()
                result = conn.process("record-123")

        assert result.success is False
        assert "No document" in result.error

    def test_extract_text_handles_base64_content(self):
        """Test _extract_text accepts base64 content."""
        from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

        content = base64.b64encode(b"Test content").decode()
        doc_result = {
            "content": content,
            "filename": "doc.txt",
            "mime_type": "text/plain",
        }

        text = DocumentSummaryConnector._extract_text(doc_result)

        assert text == ""

    def test_case_insensitive_lookup(self):
        """Test to_api_alias handles CMW API key normalization."""
        from rag_engine.cmw_platform.attribute_types import to_api_alias

        assert to_api_alias("Commerpredloshenie") == "commerpredloshenie"
        assert to_api_alias("promt") == "promt"
        assert to_api_alias("Name") == "name"
        assert to_api_alias("currentBuild") == "currentBuild"


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
