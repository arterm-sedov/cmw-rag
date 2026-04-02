"""CMW Platform connector orchestrator.

Provides a single entry point for processing platform requests through the RAG pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

from rag_engine.cmw_platform import config, records
from rag_engine.cmw_platform.mapping import map_agent_response
from rag_engine.cmw_platform.request_builder import build_request

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """Result of a platform request processing operation."""

    success: bool
    message: str | None = None
    error: str | None = None


def _build_success_message() -> str:
    """Build a concise success message with UTC timestamp.

    Returns:
        Human-readable status message for the platform.
    """
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    return f"Request fetched, agent started at {timestamp}"


class PlatformConnector:
    """Orchestrates the complete CMW Platform request → response pipeline.

    This class provides a fire-and-forget async model where:
    1. The platform sends a request ID via API
    2. We fetch the record and immediately return success
    3. The agent runs in the background and creates the linked response record
    """

    def start_request(self, request_id: str) -> ProcessResult:
        """Start processing a TPAIModel record through the RAG pipeline.

        This method is ASYNC - it fetches the record, starts the agent, and returns.
        The agent runs in the background and creates the linked response record.

        Args:
            request_id: The TPAIModel request ID to process

        Returns:
            ProcessResult with success status (fetch succeeded, agent started)
        """
        try:
            input_config = config.get_input_config()
            fields = [f["name"] for f in input_config.get("fields", [])]

            logger.info("Fetching request record %s from %s.%s",
                       request_id, input_config.get("application"), input_config.get("template"))

            record = records.read_record(request_id, fields=fields)

            if not record["success"]:
                logger.error("Failed to fetch record %s: %s", request_id, record.get("error"))
                return ProcessResult(
                    success=False,
                    error=f"Failed to fetch record: {record.get('error')}"
                )

            record_data = record["data"].get(request_id, {})
            logger.info("Successfully fetched record %s with %d fields", request_id, len(record_data))

            md_request = build_request(record_data)
            logger.debug("Built markdown request for record %s", request_id)

            thread = threading.Thread(target=_run_agent_background, args=(request_id, md_request), daemon=True)
            thread.start()

            logger.info("Started background agent for request %s", request_id)

            return ProcessResult(
                success=True,
                message=_build_success_message(),
            )

        except Exception as e:
            logger.exception("Failed to start request processing for %s: %s", request_id, e)
            return ProcessResult(success=False, error=str(e))


def _run_agent_background(request_id: str, md_request: str) -> None:
    """Run the RAG agent in a background thread.

    This function executes the full pipeline:
    1. Call the RAG agent with the markdown request
    2. Map the agent result to the output template
    3. Create the response record in CMW Platform

    Args:
        request_id: The original TPAIModel request ID
        md_request: The markdown request built from the input record
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        agent_result = loop.run_until_complete(_call_agent(md_request))

        output_config = config.get_output_config()
        template_config = config.get_template_config(
            output_config["application"],
            output_config["template"]
        )

        mapped_values = map_agent_response(
            agent_result=agent_result,
            input_record_id=request_id,
            attributes=template_config["attributes"],
            md_request=md_request,
        )

        response = records.create_record(
            application_alias=output_config["application"],
            template_alias=output_config["template"],
            values=mapped_values,
        )

        if response["success"]:
            logger.info("Created response record %s for request %s",
                       response.get("record_id"), request_id)
        else:
            logger.error("Failed to create response record for %s: %s",
                        request_id, response.get("error"))

    except Exception:
        logger.exception("Background agent failed for request %s", request_id)
    finally:
        loop.close()


async def _call_agent(md_request: str):
    """Call the RAG agent with the given request.

    Args:
        md_request: The markdown request string

    Returns:
        StructuredAgentResult from the agent
    """
    from rag_engine.api.app import ask_comindware_structured
    return await ask_comindware_structured(md_request)
