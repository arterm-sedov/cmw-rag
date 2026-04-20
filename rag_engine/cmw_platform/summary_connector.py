"""CMW Platform Document Summary Connector.

Orchestrates document fetch → process → summarize → write back workflow.
"""

import logging
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
        """Get LLM model from global settings."""
        from rag_engine.config import settings

        return settings.default_model

    def _get_system_prompt(self) -> str:
        """Get system prompt from platform config."""
        cfg = config.load_cmw_config(self.platform)
        return cfg.get("pipeline", {}).get("system_prompt", "")

    def _get_output_config(self) -> dict:
        """Get output config for summary attribute and formatting."""
        cfg = config.load_cmw_config(self.platform)
        return cfg.get("pipeline", {}).get("output", {})

    def _fetch_search_context(self, user_prompt: str, document_text: str) -> str:
        """Fetch web search results if prompt asks for external data."""
        import os

        if not any(kw in user_prompt.lower() for kw in ["конкурент", "сравни", "цена", "weather", "погода", "москва"]):
            return ""

        try:
            from langchain_tavily import TavilySearch
        except ImportError:
            return ""

        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            return ""

        try:
            search = TavilySearch(max_results=3)
            # Extract product name from document for competitor search
            lines = document_text.split("\n")[:20]
            product_hint = " ".join(lines[:3])[:100]

            queries = []
            if any(kw in user_prompt.lower() for kw in ["конкурент", "сравни", "цена"]):
                queries.append(f"competitor price {product_hint}")
            if any(kw in user_prompt.lower() for kw in ["погода", "weather", "москва"]):
                queries.append("Moscow Russia current temperature weather")

            context_parts = []
            for query in queries:
                try:
                    search_result = search.invoke(query)
                    if isinstance(search_result, dict):
                        results = search_result.get("results", [])
                    elif isinstance(search_result, list):
                        results = search_result
                    else:
                        results = []

                    for item in results[:3]:
                        if isinstance(item, dict):
                            content = item.get("content", "")[:500]
                            if content:
                                context_parts.append(f"[{item.get('title', 'Web')}]: {content}")
                except Exception:
                    pass

            if context_parts:
                return "\n".join(context_parts) + "\n\n"
            return ""
        except Exception:
            return ""

    def process(self, record_id: str) -> ProcessResult:
        """Process document: fetch → extract → summarize → write back."""
        try:
            # 1. Read record to get document_id and prompt
            output_config = self._get_output_config()
            input_config = config.load_cmw_config(self.platform).get("pipeline", {}).get("input", {})
            document_attr = input_config.get("attributes", {}).get("document_file", "Commerpredloshenie")
            prompt_attr = input_config.get("attributes", {}).get("user_prompt", "promt")

            record = records.read_record(
                record_id,
                fields=[document_attr, prompt_attr],
                platform=self.platform,
            )

            if not record.get("success"):
                return ProcessResult(
                    success=False,
                    error=f"Failed to read record: {record.get('error')}",
                )

            record_data = record.get("data", {}).get(record_id, {})

            # Get document ref (case-insensitive)
            document_ref = None
            for key in record_data:
                if key.lower() == document_attr.lower():
                    document_ref = record_data[key]
                    break

            user_prompt = record_data.get(prompt_attr.lower()) or record_data.get(prompt_attr) or ""

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

            # 5. Write summary back to record (as HTML if configured)
            output_config = self._get_output_config()
            summary_attribute = output_config.get("summary_attribute", "summary")
            summary_as_html = output_config.get("summary_as_html", False)

            if summary_as_html:
                from rag_engine.cmw_platform.mapping import convert_markdown_to_html

                summary_value = convert_markdown_to_html(summary)
            else:
                summary_value = summary

            write_result = records.update_record(
                record_id=record_id,
                values={summary_attribute: summary_value},
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
        """Process document using document_processor."""
        from rag_engine.cmw_platform.document_processor import process_document

        content = doc_result.get("content", "")
        mime_type = doc_result.get("mime_type", "")

        result = process_document(content, mime_type=mime_type)
        if result.get("success"):
            return result.get("text", "")
        logger.error(f"Document processing failed: {result.get('error')}")
        return ""

    def _summarize(self, text: str, user_prompt: str) -> str:
        """Call LLM to summarize text with platform-specific system prompt."""
        from rag_engine.llm.llm_manager import LLMManager

        model_name = self._get_model()
        system_prompt = self._get_system_prompt()
        llm = LLMManager(provider="openrouter", model=model_name)
        model = llm._chat_model()

        # Add web search context if prompt asks for it
        search_context = self._fetch_search_context(user_prompt, text)

        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("user", f"""{user_prompt}

{search_context}Document to summarize:
{text[:50000]}

Provide a concise summary following the user's instructions."""))

        resp = model.invoke(messages)
        return getattr(resp, "content", "")
