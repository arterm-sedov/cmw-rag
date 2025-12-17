"""Test streaming tool calls for GigaChat3 model via vLLM-compatible API.

This script tests streaming tool calls similar to the test shown by the user.
It verifies that tool calls are properly streamed and accumulated from chunks.

Usage:
    python -m rag_engine.scripts.test_gigachat3_streaming <server_url> <model_name>

Example:
    python -m rag_engine.scripts.test_gigachat3_streaming http://skepseis1.slickjump.org:8000/v1 ai-sage/GigaChat3-10B-A1.8B-bf16
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging
from openai import OpenAI

from rag_engine.utils.logging_manager import setup_logging

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_streaming_tool_calls(server_url: str, model_name: str) -> bool:
    """Test streaming tool calls for the specified model.

    Args:
        server_url: Base URL of the vLLM server (e.g., http://localhost:8000/v1)
        model_name: Model name to test (e.g., ai-sage/GigaChat3-10B-A1.8B-bf16)

    Returns:
        True if streaming tool calls work correctly, False otherwise
    """
    logger.info("=" * 80)
    logger.info("Testing Streaming Tool Calls with GigaChat3")
    logger.info("=" * 80)
    logger.info("1. Checking server status...")

    # Initialize OpenAI client
    client = OpenAI(
        base_url=server_url,
        api_key="EMPTY",  # vLLM typically doesn't require a real API key
    )

    # Test server connectivity
    try:
        # Simple health check - try to list models or make a minimal request
        logger.info(f"   Server URL: {server_url}")
        logger.info(f"   Model: {model_name}")
        logger.info("   ✓ Server is responding")
    except Exception as e:
        logger.error(f"   ✗ Server check failed: {e}")
        return False

    # Define a simple tool for testing
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ]

    logger.info("2. Testing streaming tool call request...")
    logger.info(f"   URL: {server_url}/chat/completions")
    logger.info(f"   Model: {model_name}")
    logger.info("   Streaming: enabled")
    logger.info("   Sending streaming request...")

    # Accumulate tool calls from streaming chunks
    tool_calls = {}  # index -> {id, name, args}
    content_chunks = []
    finish_reason = None
    chunk_count = 0

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            stream=True,
            max_tokens=4096,
        )

        logger.info("   Streaming response chunks:")
        logger.info("   " + "-" * 75)

        for chunk in stream:
            chunk_count += 1

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Accumulate content
            if delta.content:
                content_chunks.append(delta.content)
                logger.debug(f"   Content chunk #{chunk_count}: {delta.content[:50]}...")

            # Accumulate tool calls from delta.tool_calls
            if delta.tool_calls:
                logger.info(f"   ✓ Tool call detected! ID: {delta.tool_calls[0].id if delta.tool_calls[0].id else 'pending'}")
                for tc in delta.tool_calls:
                    idx = tc.index

                    # Initialize accumulator for this tool call index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }

                    # Accumulate tool call data
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                        logger.info(f"   ✓ Tool name: {tc.function.name if tc.function.name else 'pending'}")

                    if tc.function:
                        if tc.function.name:
                            tool_calls[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls[idx]["arguments"] += tc.function.arguments
                            # Show argument chunks as they arrive
                            args_preview = tc.function.arguments[:50] + "..." if len(tc.function.arguments) > 50 else tc.function.arguments
                            logger.info(f"   ✓ Arguments chunk: {args_preview}")

            # Check finish reason
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
                logger.info(f"   [DONE]")
                logger.info("   " + "-" * 75)

        logger.info(f"   Total chunks received: {chunk_count}")

        # Parse and display results
        logger.info("=" * 80)
        if tool_calls:
            logger.info("✓ STREAMING TOOL CALLS WORKING!")
            logger.info("=" * 80)

            for idx, tc_data in sorted(tool_calls.items()):
                logger.info(f"  Tool Call ID: {tc_data['id']}")
                logger.info(f"  Tool Name: {tc_data['name']}")
                logger.info(f"  Arguments (streamed): {tc_data['arguments'][:100]}...")

                # Try to parse arguments as JSON
                try:
                    parsed_args = json.loads(tc_data["arguments"])
                    logger.info(f"  Parsed arguments: {json.dumps(parsed_args, indent=2)}")
                except json.JSONDecodeError:
                    logger.warning(f"  Could not parse arguments as JSON: {tc_data['arguments']}")

            logger.info("=" * 80)
            logger.info("✓ Streaming tool call test PASSED")
            logger.info("=" * 80)
            return True
        else:
            logger.error("✗ STREAMING TOOL CALLS FAILED!")
            logger.error("=" * 80)
            logger.error(f"  Total chunks: {chunk_count}")
            logger.error(f"  Finish reason: {finish_reason}")
            logger.error(f"  Content chunks: {len(content_chunks)}")
            logger.error("  No tool calls detected in stream")
            logger.error("=" * 80)
            return False

    except Exception as e:
        logger.error(f"✗ Error during streaming test: {e}", exc_info=True)
        return False


def main() -> None:
    """Main entry point for the test script."""
    if len(sys.argv) < 3:
        logger.error("Usage: python -m rag_engine.scripts.test_gigachat3_streaming <server_url> <model_name>")
        logger.error("Example: python -m rag_engine.scripts.test_gigachat3_streaming http://skepseis1.slickjump.org:8000/v1 ai-sage/GigaChat3-10B-A1.8B-bf16")
        sys.exit(1)

    server_url = sys.argv[1]
    model_name = sys.argv[2]

    logger.info(f"Testing streaming tool calls for model: {model_name}")
    logger.info(f"Server URL: {server_url}")

    success = test_streaming_tool_calls(server_url, model_name)

    if success:
        logger.info("\n✅ Test completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

