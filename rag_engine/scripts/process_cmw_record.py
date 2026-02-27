#!/usr/bin/env python
"""Process a CMW Platform record through the RAG agent pipeline.

This script:
1. Fetches a record from CMW Platform (TPAIModel template)
2. Builds a structured markdown request from fetched attributes
3. Calls the RAG agent and collects structured response
4. Creates a response record in the platform (agent_responses template)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load .env before importing anything that uses settings
from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from rag_engine.cmw_platform import config, records
from rag_engine.cmw_platform.mapping import map_agent_response
from rag_engine.cmw_platform.request_builder import build_request

logger = logging.getLogger(__name__)


def fetch_input_record(record_id: str) -> dict:
    """Fetch the input record from CMW Platform."""
    input_config = config.get_input_config()
    template = input_config.get("template")
    application = input_config.get("application")

    # Get fields to fetch
    fields = [f["name"] for f in input_config.get("fields", [])]

    logger.info(f"Fetching record {record_id} from {application}.{template}")
    logger.info(f"Fields: {fields}")

    result = records.read_record(record_id, fields=fields)

    if not result["success"]:
        raise RuntimeError(f"Failed to fetch record: {result.get('error')}")

    return result


def call_agent(md_request: str) -> any:
    """Call the RAG agent with the request."""
    from rag_engine.api import app as api_app

    logger.info("Calling RAG agent...")
    result = asyncio.run(api_app.ask_comindware_structured(md_request))
    logger.info("Agent call completed")

    return result


def create_response_record(input_record_id: str, agent_result: any, md_request: str) -> dict:
    """Create the response record in CMW Platform."""
    output_config = config.get_output_config()
    application = output_config.get("application")
    template = output_config.get("template")

    # Get attribute mapping from template config
    template_config = config.get_template_config(application, template)
    attributes = template_config.get("attributes", {})

    logger.info(f"Mapping agent response to {application}.{template}")

    # Map agent result to CMW fields - pass md_request directly
    mapped_values = map_agent_response(
        agent_result=agent_result,
        input_record_id=input_record_id,
        attributes=attributes,
        md_request=md_request,
    )

    logger.info(f"Mapped values: {list(mapped_values.keys())}")

    # Create the record
    logger.info(f"Creating response record in {application}.{template}")
    result = records.create_record(
        application_alias=application,
        template_alias=template,
        values=mapped_values,
    )

    return result


def run_pipeline(record_id: str, dry_run: bool = False) -> dict:
    """Run the complete pipeline.

    Args:
        record_id: The TPAIModel record ID to process
        dry_run: If True, don't create the response record

    Returns:
        Dictionary with pipeline results
    """
    results = {
        "input_record_id": record_id,
        "input_data": None,
        "md_request": None,
        "agent_result": None,
        "response_record_id": None,
        "error": None,
    }

    try:
        # Step 1: Fetch input record
        input_result = fetch_input_record(record_id)
        record_data = input_result["data"].get(record_id, {})
        results["input_data"] = record_data
        logger.info(f"Input record data: {record_data}")

        # Step 2: Build markdown request
        md_request = build_request(record_data)
        results["md_request"] = md_request
        logger.info(f"Markdown request:\n{md_request}")

        # Step 3: Call RAG agent
        agent_result = call_agent(md_request)
        results["agent_result"] = {
            "has_plan": hasattr(agent_result, "plan"),
            "has_answer": hasattr(agent_result, "answer_text"),
            "answer_length": len(getattr(agent_result, "answer_text", "") or ""),
        }

        if not dry_run:
            # Step 4: Create response record
            response_result = create_response_record(record_id, agent_result, md_request)

            if response_result["success"]:
                results["response_record_id"] = response_result.get("record_id")
                logger.info(f"Created response record: {results['response_record_id']}")
            else:
                results["error"] = f"Failed to create response: {response_result.get('error')}"
                logger.error(results["error"])
        else:
            logger.info("Dry run - skipping response record creation")

    except Exception as e:
        results["error"] = str(e)
        logger.exception(f"Pipeline failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Process CMW Platform record through RAG agent")
    parser.add_argument("--record-id", required=True, help="TPAIModel record ID to process")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and process but don't create response")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info(f"Starting pipeline for record {args.record_id}")
    logger.info(f"Dry run: {args.dry_run}")

    results = run_pipeline(args.record_id, dry_run=args.dry_run)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(f"Input Record ID: {results['input_record_id']}")
    print(f"Input Data: {results['input_data']}")
    print(f"Markdown Request:\n{results['md_request']}")
    print(f"Agent Result: {results['agent_result']}")
    print(f"Response Record ID: {results['response_record_id']}")
    print(f"Error: {results['error']}")
    print("=" * 60)

    if results["error"]:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
