"""vLLM streaming fallback utility.

Handles the workaround for vLLM's limitation where tool_choice parameter
is not respected in streaming mode. Falls back to invoke() mode to ensure
tool execution when streaming doesn't detect tool calls.
"""
from __future__ import annotations

import logging
from collections.abc import Generator

from rag_engine.config.settings import settings
from rag_engine.utils.context_tracker import AgentContext, estimate_accumulated_tokens

logger = logging.getLogger(__name__)


def is_vllm_provider() -> bool:
    """Check if current provider is vLLM."""
    return settings.default_llm_provider.lower() == "vllm"


def should_use_fallback(
    is_vllm: bool,
    has_seen_tool_results: bool,
    tool_calls_detected: bool,
    tool_results_count: int,
) -> bool:
    """Determine if fallback to invoke() mode is needed.

    Args:
        is_vllm: Whether provider is vLLM
        has_seen_tool_results: Whether tool results have been seen in this conversation
        tool_calls_detected: Whether tool calls were detected in the stream
        tool_results_count: Number of tool results received

    Returns:
        True if fallback should be used, False otherwise
    """
    if not is_vllm:
        return False

    fallback_enabled = getattr(settings, "vllm_streaming_fallback_enabled", True)
    if not fallback_enabled:
        return False

    # Fallback needed if we expected tool calls but didn't get them
    expected_tool_calls = not has_seen_tool_results
    return expected_tool_calls and not tool_calls_detected and tool_results_count == 0


def execute_fallback_invoke(
    agent,
    messages: list[dict],
    agent_context: AgentContext,
    has_seen_tool_results: bool,
    result_container: dict | None = None,
) -> Generator[str | dict, None, None]:
    """Execute agent.invoke() as fallback and yield results.

    This is used when vLLM streaming doesn't detect tool calls.
    The function executes tools via invoke() and yields the results
    in a streaming-like format for UX consistency.

    Args:
        agent: LangChain agent instance
        messages: Conversation messages
        agent_context: Agent context with token tracking
        has_seen_tool_results: Whether tool results have been seen before
        result_container: Optional dict to store results (tool_results, final_answer, disclaimer_prepended)

    Yields:
        Either string chunks (for text) or dict (for metadata like search_started/search_completed).
        Results are also stored in result_container if provided.
    """
    logger.info("Falling back to invoke() mode for tool execution")
    from rag_engine.api.stream_helpers import yield_search_started

    yield yield_search_started()

    # Invoke agent - this will execute tool calls and get final answer
    result = agent.invoke({"messages": messages}, context=agent_context)

    # Process result messages to extract tool results and final answer
    result_messages = result.get("messages", [])
    tool_results = []
    final_answer = ""
    disclaimer_prepended = False

    for msg in result_messages:
        # Check for tool results
        if hasattr(msg, "type") and msg.type == "tool":
            tool_results.append(msg.content)
            logger.debug("Tool result received from invoke(), %d total results", len(tool_results))

            # Update accumulated context
            _, accumulated_tool_tokens = estimate_accumulated_tokens([], tool_results)
            agent_context.accumulated_tool_tokens = accumulated_tool_tokens

            # Emit completion metadata
            from rag_engine.api.stream_helpers import (
                extract_article_count_from_tool_result,
                yield_search_completed,
            )

            articles_count = extract_article_count_from_tool_result(msg.content)
            yield yield_search_completed(articles_count)

        # Extract final answer from AI message (last AI message without tool_calls)
        elif hasattr(msg, "type") and msg.type == "ai":
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                # This is the final answer (no tool calls)
                content = str(getattr(msg, "content", ""))
                if content:
                    final_answer = content
                    logger.debug("Final answer extracted from invoke() result: %d chars", len(content))

    # Process answer with disclaimer if needed
    if final_answer:
        # Yield answer (simulate streaming by yielding in chunks for UX)
        chunk_size = 50  # Characters per chunk for simulated streaming
        for i in range(0, len(final_answer), chunk_size):
            chunk = final_answer[i : i + chunk_size]
            yield chunk

    # Store results in container for caller to access
    if result_container is not None:
        result_container["tool_results"] = tool_results
        result_container["final_answer"] = final_answer
        result_container["disclaimer_prepended"] = disclaimer_prepended


def check_stream_completion(
    is_vllm: bool,
    has_seen_tool_results: bool,
    tool_calls_detected: bool,
    tool_results_count: int,
    stream_chunk_count: int,
) -> tuple[bool, bool]:
    """Check if stream completed without expected tool calls.

    Args:
        is_vllm: Whether provider is vLLM
        has_seen_tool_results: Whether tool results have been seen
        tool_calls_detected: Whether tool calls were detected in stream
        tool_results_count: Number of tool results received
        stream_chunk_count: Total number of stream chunks processed

    Returns:
        Tuple of (should_fallback, fallback_enabled):
        - should_fallback: Whether fallback should be triggered
        - fallback_enabled: Whether fallback is enabled in settings
    """
    fallback_enabled = getattr(settings, "vllm_streaming_fallback_enabled", True)
    expected_tool_calls = is_vllm and not has_seen_tool_results

    if expected_tool_calls and tool_results_count == 0 and not tool_calls_detected:
        logger.warning(
            "vLLM streaming completed: chunks=%d, tool_calls_detected=%s, tool_results=%d",
            stream_chunk_count,
            tool_calls_detected,
            tool_results_count,
        )
        if fallback_enabled:
            logger.warning("Falling back to invoke() mode to ensure tool execution.")
        else:
            logger.warning(
                "Fallback is disabled. Tool calls may have been processed internally by agent framework. "
                "Check logs above for content_blocks, finish_reason, or token.tool_calls in AI tokens."
            )
        return (True, fallback_enabled)

    return (False, fallback_enabled)

