"""CMW Platform Document Summary Connector.

Orchestrates document fetch → process → summarize → write back workflow.
"""

import logging
import os
import tempfile
from dataclasses import dataclass

from rag_engine.cmw_platform import config, records
from rag_engine.cmw_platform.document_api import get_document_content

logger = logging.getLogger(__name__)

DEFAULT_PLATFORM = "secondary"


@dataclass
class ProcessResult:
    """Result of a document summarization operation."""

    success: bool
    message: str | None = None
    error: str | None = None
    summary: str | None = None


class DocumentSummaryConnector:
    """Process document through LLM for summarization.

    Workflow:
        1. Read record → get document_id from "Commerpredloshenie"
        2. GET /webapi/Document/{documentId}/Content → base64 content
        3. Save to temp file → process using tools
        4. LLM: summarize with {prompt}
        5. Write summary → "summary" attribute

    Args:
        platform: Platform name (e.g., "secondary").
                 Defaults to "secondary".
    """

    def __init__(self, platform: str = DEFAULT_PLATFORM):
        self.platform = platform or DEFAULT_PLATFORM

    def _get_model(self) -> str:
        """Get LLM model from platform config or environment."""
        cfg = config.load_cmw_config(self.platform)
        return cfg.get("pipeline", {}).get("model") or os.getenv("DEFAULT_MODEL", "qwen/qwen3.5-27b")

    def process(self, record_id: str) -> ProcessResult:
        """Process document: fetch → extract → summarize → write back."""
        try:
            # 1. Read record to get document_id and prompt
            record = records.read_record(
                record_id,
                fields=["Commerpredloshenie", "prompt"],
                platform=self.platform,
            )

            if not record.get("success"):
                return ProcessResult(
                    success=False,
                    error=f"Failed to read record: {record.get('error')}",
                )

            record_data = record.get("data", {}).get(record_id, {})
            document_ref = record_data.get("commerpredloshenie") or record_data.get("Commerpredloshenie")
            user_prompt = record_data.get("prompt", "") or ""

            # Extract document ID from reference
            document_id = None
            if isinstance(document_ref, dict):
                document_id = document_ref.get("id")
            elif isinstance(document_ref, str):
                document_id = document_ref

            if not document_id:
                return ProcessResult(success=False, error="No document attached to record")

            # 2. Fetch document content
            doc_result = get_document_content(document_id, platform=self.platform)
            if not doc_result.get("success"):
                return ProcessResult(success=False, error=f"Failed to fetch document: {doc_result.get('error')}")

            # 3. Save to temp file and process using tools
            document_text = self._process_with_tools(doc_result)

            if not document_text:
                return ProcessResult(success=False, error="Failed to extract text from document")

            # 4. Summarize with LLM
            summary = self._summarize(document_text, user_prompt)

            # 5. Write summary back to record
            write_result = records.update_record(
                record_id=record_id,
                values={"summary": summary},
                platform=self.platform,
            )

            if not write_result.get("success"):
                return ProcessResult(
                    success=False,
                    error=f"Failed to write summary: {write_result.get('error')}",
                    summary=summary,
                )

            return ProcessResult(
                success=True,
                message=f"Summary generated for {doc_result.get('filename')}",
                summary=summary,
            )

        except Exception as e:
            logger.exception("Document summarization failed")
            return ProcessResult(success=False, error=str(e))

    def _process_with_tools(self, doc_result: dict) -> str:
        """Process document using tools (read_file, pdf_utils)."""
        import base64

        from rag_engine.tools.pdf_utils import PDFUtils

        content = doc_result.get("content", "")
        mime_type = doc_result.get("mime_type", "")
        filename = doc_result.get("filename", "document")

        # Decode base64 to bytes
        try:
            data = base64.b64decode(content)
        except Exception as e:
            logger.error(f"Failed to decode base64: {e}")
            return ""

        # Determine file extension
        ext = ""
        if mime_type == "application/pdf" or filename.endswith(".pdf"):
            ext = ".pdf"
        elif "wordprocessingml" in mime_type or filename.endswith(".docx"):
            ext = ".docx"
        elif "spreadsheetml" in mime_type or filename.endswith(".xlsx"):
            ext = ".xlsx"
        elif "zip" in mime_type or filename.endswith(".zip"):
            ext = ".zip"
        elif "image" in mime_type:
            ext = os.path.splitext(filename)[1] or ".img"

        # Save to temp file
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(data)
                temp_path = f.name

            # Process based on file type
            if ext == ".pdf":
                if not PDFUtils.is_available():
                    logger.error("PyMuPDF4LLM not available for PDF processing")
                    return ""
                pdf_result = PDFUtils.extract_text_from_pdf(temp_path)
                if pdf_result.success:
                    return pdf_result.text_content
                logger.error(f"PDF extraction failed: {pdf_result.error_message}")
                return ""
            else:
                # Use read_file tool for text extraction
                from rag_engine.tools.read_file import read_file

                result = read_file(temp_path)
                # read_file returns JSON string, parse it
                import json

                try:
                    parsed = json.loads(result)
                    if parsed.get("result"):
                        return parsed["result"]
                    elif parsed.get("error"):
                        logger.error(f"File processing error: {parsed.get('error')}")
                        return ""
                except json.JSONDecodeError:
                    # If not JSON, treat as raw text
                    return result

                return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

        return ""

    def _summarize(self, text: str, user_prompt: str) -> str:
        """Call LLM to summarize text - no system prompt, just direct summarization."""
        from rag_engine.llm.llm_manager import LLMManager

        model_name = self._get_model()
        llm = LLMManager(provider="openrouter", model=model_name)
        model = llm._chat_model()

        prompt = f"""{user_prompt}

Document to summarize:
{text[:50000]}

Provide a concise summary following the user's instructions."""

        resp = model.invoke([("user", prompt)])
        return getattr(resp, "content", "")
