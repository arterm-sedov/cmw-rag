"""CMW Platform Document Summary Connector.

Orchestrates document fetch → process → summarize → write back workflow.
"""

import logging
from dataclasses import dataclass

from rag_engine.cmw_platform import records
from rag_engine.cmw_platform.document_api import get_document_content
from rag_engine.cmw_platform.document_processor import process_document

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
        3. Decode base64 → detect file type → process
        4. LLM: summarize with {prompt}
        5. Write summary → "summary" attribute

    Args:
        platform: Platform name (e.g., "secondary").
                 Defaults to "secondary".
    """

    def __init__(self, platform: str = DEFAULT_PLATFORM):
        self.platform = platform or DEFAULT_PLATFORM

    def process(self, record_id: str) -> ProcessResult:
        """Process document: fetch → extract → summarize → write back.

        Args:
            record_id: Record ID in ArchitectureManagement.Zaprosinarazrabotky

        Returns:
            ProcessResult with success status, summary text, and any error
        """
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
            document_ref = record_data.get("Commerpredloshenie")
            user_prompt = record_data.get("prompt", "")

            # Extract document ID from reference
            document_id = None
            if isinstance(document_ref, dict):
                document_id = document_ref.get("id")
            elif isinstance(document_ref, str):
                document_id = document_ref

            if not document_id:
                return ProcessResult(
                    success=False,
                    error="No document attached to record",
                )

            # 2. Fetch document content
            doc_result = get_document_content(document_id, platform=self.platform)
            if not doc_result.get("success"):
                return ProcessResult(
                    success=False,
                    error=f"Failed to fetch document: {doc_result.get('error')}",
                )

            # 3. Extract text from document
            text_result = process_document(
                doc_result["content"],
                mime_type=doc_result.get("mime_type"),
                filename=doc_result.get("filename"),
            )

            if not text_result.get("success"):
                return ProcessResult(
                    success=False,
                    error=f"Failed to process document: {text_result.get('error')}",
                )

            document_text = text_result.get("text", "")

            if not document_text:
                return ProcessResult(
                    success=False,
                    error="Document contains no extractable text",
                )

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

    def _summarize(self, text: str, user_prompt: str) -> str:
        """Call LLM to summarize text.

        Args:
            text: Extracted document text
            user_prompt: User instructions for summarization

        Returns:
            Generated summary string
        """
        from rag_engine.llm.llm_manager import LLMManager

        llm = LLMManager(provider="openrouter", model="qwen/qwen3.5-27b")

        # Build prompt
        prompt = f"""{user_prompt}

Document to summarize:
{text[:50000]}

Provide a concise summary following the user's instructions."""

        result = llm.invoke(prompt)
        return result.content
