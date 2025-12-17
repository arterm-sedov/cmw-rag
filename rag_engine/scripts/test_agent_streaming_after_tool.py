"""Test agent streaming content after tool results using our agent patterns.

This test verifies that content streams correctly after tool results are provided,
matching the pattern used in our agent_chat_handler.

Usage:
    python -m rag_engine.scripts.test_agent_streaming_after_tool
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging

from rag_engine.api.app import _create_rag_agent
from rag_engine.utils.context_tracker import AgentContext, estimate_accumulated_tokens

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_agent_streaming_after_tool():
    """Test that agent streams content after tool results."""
    logger.info("=" * 80)
    logger.info("Testing Agent Streaming Content After Tool Results")
    logger.info("=" * 80)

    # Create agent (same as in app.py)
    agent = _create_rag_agent()

    # Simulate conversation: user question -> tool call -> tool result -> content response
    messages = [
        {"role": "user", "content": "What is Comindware Platform?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "retrieve_context", "arguments": '{"query": "Comindware Platform"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"articles": [{"title": "Test Article", "url": "http://test.com", "content": "Comindware Platform is a BPM system."}], "metadata": {"articles_count": 1}}',
            "tool_call_id": "call_123",
        },
    ]

    # Estimate tokens
    conversation_tokens, _ = estimate_accumulated_tokens(messages, [])
    agent_context = AgentContext(
        conversation_tokens=conversation_tokens,
        accumulated_tool_tokens=0,
    )

    logger.info("1. Starting agent stream with tool result in history...")
    logger.info(f"   Messages: {len(messages)}")
    logger.info("   Expected: Content should stream after tool result")

    answer = ""
    content_chunks = []
    chunk_count = 0
    tool_executing = False
    has_seen_tool_results = True  # We already have tool result in messages

    try:
        for stream_mode, chunk in agent.stream(
            {"messages": messages},
            context=agent_context,
            stream_mode=["updates", "messages"],
        ):
            chunk_count += 1

            if stream_mode == "messages":
                token, metadata = chunk
                token_type = getattr(token, "type", "unknown")
                token_class = type(token).__name__
                is_ai_message = token_type == "ai" or "AIMessage" in token_class

                logger.info(f"   Chunk #{chunk_count}: type={token_type}, class={token_class}")
                
                # Log token content for debugging
                token_content_attr = str(getattr(token, "content", ""))
                has_content_blocks = bool(getattr(token, "content_blocks", None))
                logger.info(f"      token.content length: {len(token_content_attr)}, has_content_blocks: {has_content_blocks}")

                # Skip tool messages
                if hasattr(token, "type") and token.type == "tool":
                    logger.info("   → Tool result (skipping)")
                    continue

                # Check for tool calls
                if hasattr(token, "tool_calls") and token.tool_calls:
                    logger.info("   → Tool call (skipping)")
                    tool_executing = True
                    continue

                # Process content blocks for text streaming
                text_chunk_found = False
                if hasattr(token, "content_blocks") and token.content_blocks:
                    logger.debug(f"   → Content blocks found: {len(token.content_blocks)}")
                    for block in token.content_blocks:
                        if block.get("type") == "tool_call_chunk":
                            tool_executing = True
                            continue
                        elif block.get("type") == "text" and block.get("text"):
                            if not tool_executing:
                                text_chunk = block["text"]
                                answer += text_chunk
                                content_chunks.append(text_chunk)
                                logger.info(f"   [Content] {text_chunk!r}")
                                text_chunk_found = True

                # Fallback: Check token.content directly
                if not text_chunk_found and is_ai_message and not tool_executing:
                    token_content = str(getattr(token, "content", ""))
                    if token_content:
                        # Determine if incremental or cumulative
                        new_chunk = None
                        if answer and token_content.startswith(answer):
                            new_chunk = token_content[len(answer):]
                        elif token_content != answer:
                            new_chunk = token_content

                        if new_chunk:
                            answer += new_chunk
                            content_chunks.append(new_chunk)
                            logger.info(f"   [Content] {new_chunk!r}")

            elif stream_mode == "updates":
                logger.debug(f"   Update: {list(chunk.keys()) if isinstance(chunk, dict) else chunk}")

        logger.info("=" * 80)
        logger.info(f"Stream completed: {chunk_count} chunks processed")
        logger.info(f"Content chunks: {len(content_chunks)}")
        logger.info(f"Final answer length: {len(answer)}")
        logger.info(f"Final answer: {answer[:200]}...")
        logger.info("=" * 80)

        if content_chunks:
            logger.info("✓ STREAMING CONTENT AFTER TOOL RESULT WORKS!")
            logger.info(f"   Received {len(content_chunks)} content chunks")
            return True
        else:
            logger.error("✗ NO CONTENT CHUNKS RECEIVED!")
            logger.error("   Streaming may not be working correctly")
            return False

    except Exception as e:
        logger.error(f"✗ Error during streaming test: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_agent_streaming_after_tool()
    sys.exit(0 if success else 1)

