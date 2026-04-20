"""Tests for CMW Document Processor.

TDD: Write tests BEFORE implementation.
"""

import base64
import inspect


def test_decode_base64_content():
    """decode_base64_content should decode base64 to bytes."""
    from rag_engine.cmw_platform.document_processor import decode_base64_content

    b64 = base64.b64encode(b"Hello World").decode()
    result = decode_base64_content(b64)
    assert result["success"] is True
    assert result["data"] == b"Hello World"


def test_decode_base64_content_invalid():
    """decode_base64_content should handle invalid base64."""
    from rag_engine.cmw_platform.document_processor import decode_base64_content

    result = decode_base64_content("not-valid-base64!!!")
    assert result["success"] is False
    assert "error" in result


def test_detect_mime_type_pdf():
    """detect_mime_type should detect PDF files."""
    from rag_engine.cmw_platform.document_processor import detect_mime_type

    data = b"%PDF-1.4 test content"
    mime = detect_mime_type(data)
    assert mime == "application/pdf"


def test_detect_mime_type_docx():
    """detect_mime_type should detect DOCX files."""
    from rag_engine.cmw_platform.document_processor import detect_mime_type

    data = b"PK\x03\x04"  # ZIP magic bytes for DOCX
    mime = detect_mime_type(data)
    assert "openxmlformats" in mime or mime == "application/zip"


def test_process_document_signature():
    """process_document should have correct signature."""
    from rag_engine.cmw_platform.document_processor import process_document

    sig = inspect.signature(process_document)
    params = list(sig.parameters.keys())

    assert "base64_content" in params
    assert "mime_type" in params
    assert "filename" in params


def test_process_document_returns_dict():
    """process_document should return a dictionary."""
    from rag_engine.cmw_platform.document_processor import process_document

    # Invalid base64 should return error dict
    result = process_document("invalid")
    assert isinstance(result, dict)
    assert "success" in result


def test_process_document_pdf():
    """process_document should handle PDF content."""
    from rag_engine.cmw_platform.document_processor import process_document

    with open("/tmp/test_minimal.pdf", "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    result = process_document(b64, mime_type="application/pdf")

    assert "success" in result
    assert "text" in result or "error" in result
    assert "file_type" in result
    assert result["file_type"] == "pdf"


def test_process_document_unsupported_type():
    """process_document should handle unsupported types."""
    from rag_engine.cmw_platform.document_processor import process_document

    # Create content that won't match any known type
    data = b"RAWWWWWWWWW"
    b64 = base64.b64encode(data).decode()

    result = process_document(b64, mime_type="application/unknown")

    assert result["success"] is False
    assert "Unsupported" in result.get("error", "")


def test_process_document_docx():
    """process_document should handle DOCX content."""
    from rag_engine.cmw_platform.document_processor import process_document

    with open("/tmp/test_minimal.docx", "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    result = process_document(b64, mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    assert "success" in result
    assert "text" in result or "error" in result
    assert "file_type" in result
    assert result["file_type"] == "docx"
