"""CMW Platform Document Summary Connector.

Orchestrates document fetch → extract → summarize → write back workflow.
Uses create_summary_agent for agentic LLM calls with web_search capability.
"""

import logging
import threading
from dataclasses import dataclass

from rag_engine.cmw_platform import config, records
from rag_engine.cmw_platform.attribute_types import to_api_alias
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


def _extract_document_id(ref: dict | str | None) -> str | None:
    """Extract document ID from a record attribute value."""
    if isinstance(ref, dict):
        return ref.get("id")
    if isinstance(ref, str):
        return ref
    return None


class DocumentSummaryConnector:
    """Process document through LLM agent for summarization.

    Workflow:
        1. Read record → get document_id from config-specified attribute
        2. Fetch document content → extract text
        3. Summarize with agent (web_search available for external data)
        4. Write summary back to record

    Args:
        platform: Platform name (e.g., "secondary"). Defaults to "secondary".
    """

    def __init__(self, platform: str = DEFAULT_PLATFORM):
        self.platform = platform or DEFAULT_PLATFORM

    def start(self, record_id: str) -> ProcessResult:
        """Verify record is readable, spawn background processing, return ACK.

        Fire-and-forget: returns immediately; process() runs in background thread.
        Mirrors PlatformConnector.start_request() pattern.

        Args:
            record_id: Record ID to process

        Returns:
            ProcessResult with success status (read succeeded, agent started)
        """
        try:
            pipeline = config.load_pipeline_config(self.platform)
            attr_map = pipeline.get("input", {}).get("attributes", {})
            document_attr = attr_map.get("document_file", "")
            prompt_attr = attr_map.get("user_prompt", "")

            if not document_attr:
                return ProcessResult(success=False, error="No document attribute configured")

            record = records.read_record(
                record_id, fields=[document_attr, prompt_attr], platform=self.platform,
            )

            if not record.get("success"):
                return ProcessResult(
                    success=False, error=f"Failed to read record: {record.get('error')}",
                )

            thread = threading.Thread(
                target=self.process,
                args=(record_id,),
                daemon=True,
            )
            thread.start()

            logger.info("Started background document processing for %s", record_id)
            return ProcessResult(success=True, message="Начата обработка данных")

        except Exception as e:
            logger.exception("Failed to start document processing for %s", record_id)
            return ProcessResult(success=False, error=str(e))

    def process(self, record_id: str) -> ProcessResult:
        """Process document: fetch → extract → summarize → write back."""
        try:
            pipeline = config.load_pipeline_config(self.platform)
            input_cfg = pipeline.get("input", {})
            output_cfg = pipeline.get("output", {})
            attr_map = input_cfg.get("attributes", {})

            document_attr = attr_map.get("document_file", "")
            prompt_attr = attr_map.get("user_prompt", "")

            if not document_attr:
                return ProcessResult(success=False, error="No document attribute configured")

            record = records.read_record(
                record_id, fields=[document_attr, prompt_attr], platform=self.platform,
            )

            if not record.get("success"):
                return ProcessResult(
                    success=False, error=f"Failed to read record: {record.get('error')}",
                )

            record_data = record.get("data", {}).get(record_id, {})

            document_id = _extract_document_id(record_data.get(to_api_alias(document_attr)))
            user_prompt = record_data.get(to_api_alias(prompt_attr), "") or ""

            if not document_id:
                return ProcessResult(success=False, error="No document attached to record")

            doc_result = get_document_content(document_id, platform=self.platform)
            if not doc_result.get("success"):
                return ProcessResult(
                    success=False, error=f"Failed to fetch document: {doc_result.get('error')}",
                )

            document_text = self._extract_text(doc_result)
            if not document_text:
                return ProcessResult(success=False, error="Failed to extract text from document")

            summary = self._summarize(document_text, user_prompt, pipeline)

            summary_attribute = output_cfg.get("summary_attribute", "summary")
            summary_value = (
                self._to_html(summary) if output_cfg.get("summary_as_html") else summary
            )

            write_result = records.update_record(
                record_id=record_id, values={summary_attribute: summary_value},
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

    @staticmethod
    def _extract_text(doc_result: dict) -> str:
        """Extract text from document content."""
        from rag_engine.cmw_platform.document_processor import process_document

        result = process_document(
            doc_result.get("content", ""), mime_type=doc_result.get("mime_type", ""),
        )
        if result.get("success"):
            return result.get("text", "")
        logger.error("Document processing failed: %s", result.get("error"))
        return ""

    @staticmethod
    def _to_html(markdown: str) -> str:
        """Convert markdown summary to HTML."""
        from rag_engine.cmw_platform.mapping import convert_markdown_to_html

        return convert_markdown_to_html(markdown)

    @staticmethod
    def _summarize(text: str, user_prompt: str, pipeline: dict) -> str:
        """Summarize document using agentic approach with create_summary_agent."""
        from langchain_core.messages import HumanMessage

        from rag_engine.llm.agent_factory import create_summary_agent

        system_prompt = pipeline.get("system_prompt", "")
        agent = create_summary_agent(system_prompt=system_prompt)

        user_content = f"Документ:\n{text[:50000]}\n\nЗапрос пользователя: {user_prompt}"

        result = agent.invoke({"messages": [HumanMessage(content=user_content)]})

        if isinstance(result, dict) and result.get("messages"):
            return result["messages"][-1].content
        return str(result)
