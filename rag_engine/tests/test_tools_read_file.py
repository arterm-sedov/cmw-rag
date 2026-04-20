"""Test read_file tool - TDD approach.

Tests the unified read_file tool that handles both text files and PDFs.
"""

import json
import os
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestReadFileTool:
    """Test suite for read_file tool."""

    def test_read_text_file_txt(self):
        """Test reading a .txt file."""
        from rag_engine.tools.read_file import read_file

        file_path = str(FIXTURES_DIR / "test.txt")
        result = read_file.invoke({"file_reference": file_path})

        assert result is not None
        result_data = json.loads(result)
        assert result_data["type"] == "tool_response"
        assert result_data["tool_name"] == "read_file"
        assert "success" in result_data or "result" in result_data
        assert "test.txt" in result_data.get("result", "").lower() or "test" in result_data.get("file_info", {}).get("name", "").lower()

    def test_read_markdown_file_md(self):
        """Test reading a .md file."""
        from rag_engine.tools.read_file import read_file

        file_path = str(FIXTURES_DIR / "test.md")
        result = read_file.invoke({"file_reference": file_path})

        assert result is not None
        result_data = json.loads(result)
        assert result_data["type"] == "tool_response"
        assert "result" in result_data

    def test_read_json_file(self):
        """Test reading a .json file."""
        from rag_engine.tools.read_file import read_file

        file_path = str(FIXTURES_DIR / "test.json")
        result = read_file.invoke({"file_reference": file_path})

        assert result is not None
        result_data = json.loads(result)
        assert result_data["type"] == "tool_response"
        assert "result" in result_data
        assert "test" in result_data.get("result", "").lower()

    def test_read_pdf_file(self):
        """Test reading a .pdf file."""
        from rag_engine.tools.read_file import read_file

        file_path = str(FIXTURES_DIR / "sample.pdf")
        assert os.path.exists(file_path), f"PDF fixture not found: {file_path}"

        result = read_file.invoke({"file_reference": file_path})

        assert result is not None
        result_data = json.loads(result)
        assert result_data["type"] == "tool_response"
        assert result_data["tool_name"] == "read_file"
        assert "sample.pdf" in result

    def test_read_missing_file(self):
        """Test error handling for missing file."""
        from rag_engine.tools.read_file import read_file

        result = read_file.invoke({"file_reference": "nonexistent_file.txt"})

        assert result is not None
        result_data = json.loads(result)
        assert result_data["type"] == "tool_response"
        assert "error" in result_data or result_data.get("file_info", {}).get("exists") == False

    def test_read_unsupported_file(self):
        """Test error handling for unsupported file type."""
        from rag_engine.tools.read_file import read_file

        unsupported_path = str(FIXTURES_DIR / "test.exe")
        with open(unsupported_path, "wb") as f:
            f.write(b"\x00\x01\x02\x03")

        try:
            result = read_file.invoke({"file_reference": unsupported_path})

            assert result is not None
            result_data = json.loads(result)
            assert result_data["type"] == "tool_response"
        finally:
            os.remove(unsupported_path)

    def test_file_utils_text_file_detection(self):
        """Test FileUtils text file detection."""
        from rag_engine.tools.file_utils import FileUtils

        assert FileUtils.is_text_file("test.txt") == True
        assert FileUtils.is_text_file("test.md") == True
        assert FileUtils.is_text_file("test.json") == True
        assert FileUtils.is_text_file("test.py") == True
        assert FileUtils.is_text_file("test.pdf") == False
        assert FileUtils.is_text_file("test.exe") == False

    def test_file_utils_encoding_fallback(self):
        """Test FileUtils encoding fallback for text files."""
        from rag_engine.tools.file_utils import FileUtils

        result = FileUtils.read_text_file(str(FIXTURES_DIR / "test.txt"))
        assert result.success == True
        assert result.content is not None
        assert "test" in result.content.lower()

    def test_pdf_utils_is_available(self):
        """Test PDFUtils availability check."""
        from rag_engine.tools.pdf_utils import PDFUtils

        assert PDFUtils.is_available() == True

    def test_pdf_utils_is_pdf_file(self):
        """Test PDFUtils PDF file detection."""
        from rag_engine.tools.pdf_utils import PDFUtils

        pdf_path = str(FIXTURES_DIR / "sample.pdf")
        assert PDFUtils.is_pdf_file(pdf_path) == True
        txt_path = str(FIXTURES_DIR / "test.txt")
        assert PDFUtils.is_pdf_file(txt_path) == False

    def test_pdf_utils_extract_text(self):
        """Test PDF text extraction."""
        from rag_engine.tools.pdf_utils import PDFUtils

        pdf_path = str(FIXTURES_DIR / "sample.pdf")
        result = PDFUtils.extract_text_from_pdf(pdf_path)

        assert result.success == True
        assert result.text_content is not None
        assert len(result.text_content) > 0