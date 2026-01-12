"""Gradio UI with Chatbot (reference agent pattern) and REST API endpoint."""
from __future__ import annotations

import sys
from collections.abc import AsyncGenerator
from pathlib import Path

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import logging
import os

import gradio as gr
from openai import APIError as OpenAIAPIError

from rag_engine.config.settings import get_allowed_fallback_models, settings  # noqa: F401
from rag_engine.llm.fallback import check_context_fallback
from rag_engine.llm.llm_manager import LLMManager
from rag_engine.retrieval.embedder import FRIDAEmbedder
from rag_engine.retrieval.retriever import RAGRetriever
from rag_engine.storage.vector_store import ChromaStore
from rag_engine.tools import retrieve_context
from rag_engine.utils.context_tracker import (
    AgentContext,
    compute_context_tokens,
    estimate_accumulated_context,
    estimate_accumulated_tokens,
)
from rag_engine.utils.conversation_store import salt_session_id
from rag_engine.utils.formatters import format_with_citations
from rag_engine.utils.logging_manager import setup_logging
from rag_engine.utils.vllm_fallback import (
    check_stream_completion,
    execute_fallback_invoke,
    is_vllm_provider,
    should_use_fallback,
)

setup_logging()

logger = logging.getLogger(__name__)


# Initialize singletons (order matters: llm_manager before retriever)
embedder = FRIDAEmbedder(
    model_name=settings.embedding_model,
    device=settings.embedding_device,
)
vector_store = ChromaStore(
    persist_dir=settings.chromadb_persist_dir,
    collection_name=settings.chromadb_collection,
)
llm_manager = LLMManager(
    provider=settings.default_llm_provider,
    model=settings.default_model,
    temperature=settings.llm_temperature,
)
retriever = RAGRetriever(
    embedder=embedder,
    vector_store=vector_store,
    llm_manager=llm_manager,  # NEW: Pass for dynamic context budgeting
    top_k_retrieve=settings.top_k_retrieve,
    top_k_rerank=settings.top_k_rerank,
    rerank_enabled=settings.rerank_enabled,
)


def _find_model_for_tokens(required_tokens: int) -> str | None:
    """Find a model that can handle the required token count.

    Args:
        required_tokens: Minimum token capacity needed

    Returns:
        Model name if found, None otherwise
    """
    from rag_engine.llm.fallback import find_fallback_model

    return find_fallback_model(required_tokens)


def _check_context_fallback(messages: list[dict]) -> str | None:
    """Check if context fallback is needed and return fallback model.

    Thin wrapper around ``rag_engine.llm.fallback.check_context_fallback`` so
    that all pre-agent fallback logic is centralized in the LLM fallback
    module. Kept for backward compatibility and tests.
    """
    # Early return if fallback is disabled
    if not getattr(settings, "llm_fallback_enabled", False):
        return None

    return check_context_fallback(messages)


def compress_tool_results(state: dict, runtime) -> dict | None:
    """Compress tool results before LLM call if approaching context limit.

    This middleware runs right before each LLM invocation, AFTER all tool calls complete.
    It extracts ALL articles from ALL tool messages, deduplicates, and compresses
    proportionally based on normalized_rank (0.0 = best, 1.0 = worst).

    Args:
        state: Agent state containing messages
        runtime: Runtime object with access to model config

    Returns:
        Updated state dict with compressed messages, or None if no changes needed
    """
    from rag_engine.llm.compression import compress_tool_messages

    messages = state.get("messages", [])
    if not messages:
        return None

    updated_messages = compress_tool_messages(
        messages=messages,
        runtime=runtime,
        llm_manager=llm_manager,
        threshold_pct=float(getattr(settings, "llm_compression_threshold_pct", 0.85)),
        target_pct=float(getattr(settings, "llm_compression_target_pct", 0.80)),
    )

    if updated_messages:
        return {"messages": updated_messages}

    return None


def _compute_context_tokens_from_state(messages: list[dict]) -> tuple[int, int]:
    """Compute (conversation_tokens, accumulated_tool_tokens) from agent state messages.

    This is a wrapper for the unified compute_context_tokens function.
    Uses configurable JSON overhead percentage from settings.

    - Conversation tokens: count non-tool message contents
    - Accumulated tool tokens: parse tool JSONs, dedupe by kb_id, sum content tokens,
      add configurable JSON overhead percentage (default 30%)
    """
    return compute_context_tokens(messages, tool_results=None, add_json_overhead=True)


def update_context_budget(state: dict, runtime) -> dict | None:
    """Middleware to populate runtime.context token figures before each model call.

    Ensures tools see accurate conversation and accumulated tool tokens via runtime.context.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    conv_toks, tool_toks = _compute_context_tokens_from_state(messages)

    # Mutate runtime.context (AgentContext) so tools can read accurate figures
    try:
        if hasattr(runtime, "context") and runtime.context:
            runtime.context.conversation_tokens = conv_toks
            runtime.context.accumulated_tool_tokens = tool_toks
            logger.debug(
                "Updated runtime.context: conversation_tokens=%d, accumulated_tool_tokens=%d",
                conv_toks,
                tool_toks,
            )
    except Exception:
        # Do not fail the run due to budgeting issues
        logger.debug("Unable to update runtime.context tokens")

    return None


from collections.abc import Callable  # noqa: E402

from langchain.agents.middleware import AgentMiddleware  # noqa: E402
from langchain.agents.middleware import wrap_tool_call as middleware_wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402


class ToolBudgetMiddleware(AgentMiddleware):
    """Populate runtime.context tokens right before each tool execution.

    Ensures tools see up-to-date conversation and accumulated tool tokens
    even when multiple tool calls happen within a single agent step.
    """

    @middleware_wrap_tool_call()
    def tool_budget_wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        try:
            state = getattr(request, "state", {}) or {}
            runtime = getattr(request, "runtime", None)
            if state and runtime is not None and hasattr(runtime, "context") and runtime.context:
                conv_toks, tool_toks = _compute_context_tokens_from_state(state.get("messages", []))
                runtime.context.conversation_tokens = conv_toks
                runtime.context.accumulated_tool_tokens = tool_toks
                logger.debug(
                    "[ToolBudget] runtime.context updated before tool: conv=%d, tools=%d",
                    conv_toks,
                    tool_toks,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ToolBudget] Failed to update context before tool: %s", exc)

        return handler(request)

def _create_rag_agent(override_model: str | None = None, force_tool_choice: bool = False):
    """Create LangChain agent with optional forced retrieval tool execution and memory compression.

    Uses centralized factory with app-specific middleware. The factory can enforce
    tool execution via tool_choice="retrieve_context" when needed, or allow model
    to choose tools freely by default.

    This wrapper preserves test patch points while delegating to the centralized
    agent_factory for consistent agent creation.

    Args:
        override_model: Optional model name to use instead of default
                       (for context window fallback)
        force_tool_choice: If True, forces retrieve_context tool execution.
                          If False, allows model to choose tools freely (default: False)

    Returns:
        Configured LangChain agent with retrieve_context tool and middleware
    """
    from rag_engine.llm.agent_factory import create_rag_agent

    return create_rag_agent(
        override_model=override_model,
        tool_budget_middleware=ToolBudgetMiddleware(),
        update_context_budget_middleware=update_context_budget,
        compress_tool_results_middleware=compress_tool_results,
        force_tool_choice=force_tool_choice,
    )


def _is_ui_only_message(msg: dict) -> bool:
    """Check if a message is UI-only (should not be sent to agent).

    UI-only messages include:
    - Disclaimer messages
    - Search started/completed metadata messages
    - Thinking blocks (tool execution metadata)
    - Model switch notices
    - Cancellation messages

    Regular assistant messages (actual answers) should NOT be filtered even if they
    have metadata, as they contain conversation content the agent needs to see.

    Args:
        msg: Message dict to check

    Returns:
        True if message is UI-only, False otherwise
    """
    if not isinstance(msg, dict):
        return False

    # Check for disclaimer content (disclaimer messages don't have metadata)
    from rag_engine.llm.prompts import AI_DISCLAIMER
    content = msg.get("content", "")
    if isinstance(content, str) and AI_DISCLAIMER.strip() in content:
        return True

    # Check for explicit UI-only metadata marker first (preferred, language-agnostic)
    metadata = msg.get("metadata", {})
    if isinstance(metadata, dict):
        ui_type = metadata.get("ui_type")
        if isinstance(ui_type, str) and ui_type in {
            "search_started",
            "search_completed",
            "thinking",
            "generating_answer",
            "model_switch",
            "cancelled",
        }:
            return True

    return False


def _messages_are_equivalent(msg1: dict, msg2: dict) -> bool:
    """Check if two messages are equivalent (same role and content).

    Used to detect duplicate messages in history.

    Args:
        msg1: First message dict
        msg2: Second message dict

    Returns:
        True if messages are equivalent, False otherwise
    """
    if not isinstance(msg1, dict) or not isinstance(msg2, dict):
        return False

    role1 = msg1.get("role")
    role2 = msg2.get("role")
    if role1 != role2:
        return False

    content1 = msg1.get("content", "")
    content2 = msg2.get("content", "")

    # Compare string content
    if isinstance(content1, str) and isinstance(content2, str):
        return content1.strip() == content2.strip()

    # For structured content, compare as strings
    return str(content1) == str(content2)


def _message_exists_in_history(message: dict, history: list[dict]) -> bool:
    """Check if a message already exists in history.

    Args:
        message: Message dict to check
        history: List of message dicts

    Returns:
        True if message exists in history, False otherwise
    """
    for existing_msg in history:
        if _messages_are_equivalent(message, existing_msg):
            return True
    return False


def _build_agent_messages_from_gradio_history(
    gradio_history: list[dict],
    current_message: str,
    wrapped_message: str,
) -> list[dict]:
    """Build messages for agent from Gradio history, filtering out UI-only messages.

    Filters gradio_history to exclude UI metadata messages (disclaimer, search_started, etc.)
    and builds a clean message list for the agent. This ensures the agent only sees
    actual conversation content, not UI elements.

    Args:
        gradio_history: Full Gradio history including UI messages
        current_message: Current user message (unwrapped, for filtering)
        wrapped_message: Current user message wrapped with template (for agent)

    Returns:
        List of message dicts in LangChain format for agent
    """
    messages = []

    # Filter gradio_history to exclude UI-only messages
    # We need to include previous conversation messages but exclude:
    # - Disclaimer messages
    # - Search started/completed metadata
    # - Model switch notices
    # - The current user message (we'll add it wrapped below)
    from rag_engine.utils.message_utils import normalize_gradio_history_message

    # Log all messages in gradio_history before filtering for debugging
    logger.info("gradio_history contents before filtering (%d messages):", len(gradio_history))
    for idx, msg in enumerate(gradio_history):
        msg_role = msg.get("role", "unknown")
        msg_content = msg.get("content", "")
        has_metadata = "metadata" in msg
        metadata_keys = list(msg.get("metadata", {}).keys()) if isinstance(msg.get("metadata"), dict) else []
        content_preview = str(msg_content)[:100] if isinstance(msg_content, str) else f"<{type(msg_content).__name__}>"
        logger.info(
            "  [%d] role=%s, has_metadata=%s, metadata_keys=%s, content_preview=%s",
            idx, msg_role, has_metadata, metadata_keys, content_preview
        )

    for idx, msg in enumerate(gradio_history):
        msg_role = msg.get("role")
        msg_content = msg.get("content", "")
        has_metadata = "metadata" in msg

        # Skip UI-only messages
        if _is_ui_only_message(msg):
            logger.debug(
                "Filtered out UI-only message [%d]: role=%s, has_metadata=%s, content_preview=%s",
                idx, msg_role, has_metadata, str(msg_content)[:100] if isinstance(msg_content, str) else "non-string"
            )
            continue

        # Skip the current user message (we'll add wrapped version below)
        if msg_role == "user" and isinstance(msg_content, str) and msg_content.strip() == current_message.strip():
            logger.debug("Skipping current user message [%d] (will add wrapped version)", idx)
            continue

        # Normalize message for LangChain (convert structured content to string)
        normalized_msg = normalize_gradio_history_message(msg)
        # Include actual conversation messages
        messages.append(normalized_msg)
        logger.info(
            "Included message [%d] in agent context: role=%s, content_preview=%s",
            idx, msg_role, str(msg_content)[:100] if isinstance(msg_content, str) else "non-string"
        )

    # Add wrapped current message for agent
    messages.append({"role": "user", "content": wrapped_message})

    return messages


def _process_text_chunk_for_streaming(
    text_chunk: str,
    answer: str,
    disclaimer_prepended: bool,
    has_seen_tool_results: bool,
) -> tuple[str, bool]:
    """Process a text chunk for streaming with disclaimer and formatting.

    Handles optional newline after tool results and accumulates the answer.

    Args:
        text_chunk: Raw text chunk from agent
        answer: Accumulated answer so far
        disclaimer_prepended: Whether disclaimer has already been added
        has_seen_tool_results: Whether tool results have been seen

    Returns:
        Tuple of (updated_answer, updated_disclaimer_prepended)
    """
    # Mark that we started streaming answer text
    if not disclaimer_prepended:
        disclaimer_prepended = True
    # Prepend newline before first text chunk after tool results
    elif has_seen_tool_results and not answer:
        text_chunk = "\n" + text_chunk

    answer = answer + text_chunk
    return answer, disclaimer_prepended


def _extract_content_string(content: str | list | None) -> str:
    """Extract plain text string from Gradio message content.

    Handles both string format and structured format (list of content blocks).
    Also handles double-encoded JSON strings that Gradio sometimes sends.

    Args:
        content: Message content (string, list of blocks, or None)

    Returns:
        Plain text string extracted from content
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Handle structured content format (Gradio 6): [{"type": "text", "text": "..."}]
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        # Recursively handle nested structures (double-encoded JSON)
                        # Handle case where text field contains a JSON string: "[{'text': '...', 'type': 'text'}]"
                        if isinstance(text, str) and text.strip().startswith("[") and text.strip().endswith("]"):
                            try:
                                parsed = json.loads(text)
                                if isinstance(parsed, list):
                                    # Recursively extract from nested structure
                                    extracted = _extract_content_string(parsed)
                                    if extracted:
                                        text_parts.append(extracted)
                                        continue
                            except (json.JSONDecodeError, ValueError):
                                # Not valid JSON, treat as plain text
                                pass
                        text_parts.append(str(text))
        return " ".join(text_parts) if text_parts else ""
    # Fallback: convert to string
    return str(content)


def _update_or_append_assistant_message(gradio_history: list[dict], content: str) -> None:
    """Update last assistant message or append new one if it has metadata.

    Only updates non-metadata assistant messages to preserve meta blocks.
    This ensures meta blocks (thinking, searching, etc.) are not overwritten.

    Args:
        gradio_history: List of Gradio message dictionaries
        content: Content to set for the assistant message
    """
    if gradio_history and gradio_history[-1].get("role") == "assistant" and not gradio_history[-1].get("metadata"):
        gradio_history[-1] = {"role": "assistant", "content": content}
    else:
        gradio_history.append({"role": "assistant", "content": content})


async def agent_chat_handler(
    message: str,
    history: list[dict],
    cancel_state: dict | None = None,
    request: gr.Request | None = None,
) -> AsyncGenerator[list[dict], None]:
    """Ask questions about Comindware Platform documentation and get intelligent answers with citations.

    The assistant automatically searches the knowledge base to find relevant articles
    and provides comprehensive answers based on official documentation. Use this for
    technical questions, configuration help, API usage, troubleshooting, and general
    platform guidance.

    Args:
        message: User's current message or question
        history: Chat history from Gradio
        cancel_state: Mutable dict with "cancelled" key for cooperative cancellation
        request: Gradio request object for session management

    Yields:
        Complete message history lists (for Chatbot) to preserve all messages
        including disclaimer, thinking blocks, and streaming answer.
        Uses reference agent pattern: always yields full working_history list.
    """
    # Helper to check if cancellation was requested
    def is_cancelled() -> bool:
        return cancel_state is not None and cancel_state.get("cancelled", False)

    if not message or not message.strip():
        yield history if history else []
        return

    # Build full message history starting from provided history
    # This ensures all messages (disclaimer, thinking blocks, answer) persist during streaming
    # Note: We keep original format for ChatInterface, normalize only when needed for agent

    # Initialize variables that might be needed in error handling
    current_model = settings.default_model
    messages = []
    tool_results = []

    # Build working history from provided history (like reference agent pattern)
    # Reference agent pattern: working_history = history + [new messages]
    # This ensures we start from ChatInterface's current state and build incrementally
    # Then we always yield the full working_history list
    gradio_history = list(history) if history else []

    # Log history state for debugging memory issues
    logger.info(
        "agent_chat_handler called: message='%s', history_length=%d, gradio_history_length=%d",
        message[:50] if message else "",
        len(history) if history else 0,
        len(gradio_history),
    )
    if gradio_history:
        logger.debug(
            "Previous conversation messages in history: %d user, %d assistant",
            sum(1 for msg in gradio_history if isinstance(msg, dict) and msg.get("role") == "user"),
            sum(1 for msg in gradio_history if isinstance(msg, dict) and msg.get("role") == "assistant"),
        )

    # Don't add user message here - it's already in history from ChatInterface pattern
    # The submit_event chain adds the user message to chatbot before calling agent_chat_handler
    # (pattern from test script: line 128-129)

    # Determine if this is the first message (for disclaimer and template)
    is_first_message = not history

    # Stream AI-generated content disclaimer as a persistent assistant message
    # so it stays above tool-call progress/thinking chunks in the Chatbot UI.
    # Only add it ONCE after the first question (pattern from test script)
    if is_first_message:
        from rag_engine.llm.prompts import AI_DISCLAIMER

        # Check if disclaimer already exists in history (safety check)
        disclaimer_exists = False
        for msg in gradio_history:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str) and AI_DISCLAIMER.strip() in content:
                    disclaimer_exists = True
                    break

        # Add disclaimer to history only if it doesn't exist yet (first question only)
        if not disclaimer_exists:
            disclaimer_msg = {
                "role": "assistant",
                "content": AI_DISCLAIMER,
            }
            # Double-check it doesn't exist (robust duplicate prevention)
            if not _message_exists_in_history(disclaimer_msg, gradio_history):
                gradio_history.append(disclaimer_msg)
                # Yield full history - ChatInterface will replace its internal history
                # According to Gradio docs, yielding a list should replace, not append
                # Create a new list to avoid mutating ChatInterface's internal state
                yield list(gradio_history)

    # Three UI blocks pattern (from test script):
    # 1. Thinking block
    # 2. Search started block
    # 3. Search completed block (added when tool results arrive)
    # Then the actual answer is streamed

    # 1. Add thinking block first (pattern from test script: line 136-142)
    from rag_engine.api.stream_helpers import yield_thinking_block
    thinking_msg = yield_thinking_block("retrieve_context")  # Use retrieve_context as tool name
    gradio_history.append(thinking_msg)  # Append, don't replace
    yield list(gradio_history)

    # 2. Add search started block (pattern from test script: line 161-167)
    from rag_engine.api.stream_helpers import yield_search_started

    # Use user's message as initial query (will be updated when tool call is detected)
    initial_query = message.strip() if message else ""
    search_started_msg = yield_search_started(initial_query)
    # Add search_started message (reference agent pattern: always append)
    gradio_history.append(search_started_msg)
    # Yield full history - always yield full working_history like reference agent
    yield list(gradio_history)

    # Session management (reuse existing pattern)
    base_session_id = getattr(request, "session_hash", None) if request is not None else None
    session_id = salt_session_id(base_session_id, history, message)

    # Wrap user message in template only for the first question in the conversation
    from rag_engine.llm.prompts import (
        USER_QUESTION_TEMPLATE_FIRST,
        USER_QUESTION_TEMPLATE_SUBSEQUENT,
    )

    # is_first_message already determined above (for disclaimer)
    wrapped_message = (
        USER_QUESTION_TEMPLATE_FIRST.format(question=message)
        if is_first_message
        else USER_QUESTION_TEMPLATE_SUBSEQUENT.format(question=message)
    )

    # Save user message to conversation store (BEFORE agent execution)
    # This ensures conversation history is tracked for memory compression
    if session_id:
        llm_manager._conversations.append(session_id, "user", message)

    # Build messages from gradio_history for agent (LangChain format)
    # This filters out UI-only messages (disclaimer, search_started, etc.)
    # and ensures the agent only sees actual conversation content
    messages = _build_agent_messages_from_gradio_history(
        gradio_history, message, wrapped_message
    )

    # Log messages being sent to agent for debugging memory issues
    logger.info(
        "Building agent messages: gradio_history_length=%d -> agent_messages_length=%d",
        len(gradio_history),
        len(messages),
    )
    if messages:
        logger.debug(
            "Agent messages breakdown: %d user, %d assistant",
            sum(1 for msg in messages if isinstance(msg, dict) and msg.get("role") == "user"),
            sum(1 for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant"),
        )

    # Note: pre-agent trimming removed by request; rely on existing middleware

    # Check if we need model fallback BEFORE creating agent
    # This matches old handler's upfront fallback check
    selected_model = None
    if settings.llm_fallback_enabled:
        selected_model = _check_context_fallback(messages)

    # Create agent (with fallback model if needed) and stream execution
    # Force tool choice only on first message; allow model to choose on subsequent turns
    # TEMPORARILY DISABLED FOR TESTING: force_tool_choice=False
    agent = _create_rag_agent(override_model=selected_model, force_tool_choice=False)  # was: is_first_message
    tool_results = []
    answer = ""
    current_model = selected_model or settings.default_model
    has_seen_tool_results = False
    disclaimer_prepended = False  # Track if disclaimer has been prepended to stream

    # Track incomplete response for memory saving if cancelled (pattern from test script)
    incomplete_response = None
    final_response = None

    # Track accumulated context for progressive budgeting
    # Agent is responsible for counting context, not the tool
    conversation_tokens, _ = estimate_accumulated_tokens(messages, [])

    # Initialize tool call accumulator for streaming chunks
    from rag_engine.api.stream_helpers import ToolCallAccumulator

    tool_call_accumulator = ToolCallAccumulator()

    try:
        # Track tool execution state
        # Only stream text content when NOT executing tools
        tool_executing = False

        # Pass accumulated context to agent via typed context parameter
        # Tools can access this via runtime.context (typed, clean!)
        agent_context = AgentContext(
            conversation_tokens=conversation_tokens,
            accumulated_tool_tokens=0,  # Updated as we go
        )

        # vLLM streaming limitation: tool_choice doesn't work in streaming mode
        # vLLM ignores tool_choice="retrieve_context" in streaming and returns finish_reason=stop
        # instead of finish_reason=tool_calls. This means forced tool execution requires invoke() mode.
        # The fallback ensures tool calls are executed even when streaming detection fails.
        # Can be disabled via vllm_streaming_fallback_enabled=False for testing (will fail to execute tools)
        is_vllm = is_vllm_provider()
        fallback_to_invoke = False

        # Use multiple stream modes for complete streaming experience
        # Per https://docs.langchain.com/oss/python/langchain/streaming#stream-multiple-modes
        # Always try streaming first - improved detection should catch tool calls via content_blocks/finish_reason
        logger.info("Starting agent.astream() with %d messages", len(messages))
        stream_chunk_count = 0
        tool_calls_detected_in_stream = False

        try:
            async for stream_mode, chunk in agent.astream(
                {"messages": messages},
                context=agent_context,
                stream_mode=["updates", "messages"]
            ):
                # Check for cancellation at each iteration
                if is_cancelled():
                    logger.info("Cancellation detected in stream loop - stopping")
                    break

                stream_chunk_count += 1
                logger.debug("Stream chunk #%d: mode=%s", stream_chunk_count, stream_mode)

                # Handle "messages" mode for token streaming
                if stream_mode == "messages":
                    token, metadata = chunk
                    token_type = getattr(token, "type", "unknown")
                    # Disabled verbose token logging for production
                    # logger.info("Messages token #%d: type=%s", stream_chunk_count, token_type)

                    # Debug logging for vLLM tool calling issues (only when expecting tool calls)
                    # Check for both "ai" type and AIMessageChunk class (LangChain uses different representations)
                    token_class = type(token).__name__
                    is_ai_message = token_type == "ai" or "AIMessage" in token_class or "AIMessage" in str(token_type)

                    # Process all AI tokens through accumulator to accumulate tool_call chunks
                    # This ensures we capture query even if chunks arrive before tool_call is detected
                    if is_ai_message:
                        tool_query_from_accumulator = tool_call_accumulator.process_token(token)
                        # If accumulator found a complete query, update search_started message
                        if tool_query_from_accumulator:
                            from rag_engine.api.stream_helpers import update_search_started_in_history

                            if update_search_started_in_history(gradio_history, tool_query_from_accumulator):
                                yield list(gradio_history)

                    if is_ai_message:
                        has_tool_calls = bool(getattr(token, "tool_calls", None))
                        content = str(getattr(token, "content", ""))
                        response_metadata = getattr(token, "response_metadata", {})
                        finish_reason = response_metadata.get("finish_reason", "N/A") if isinstance(response_metadata, dict) else "N/A"

                        # Check content_blocks for tool_call_chunk (critical for vLLM streaming)
                        content_blocks = getattr(token, "content_blocks", None)
                        has_content_blocks = bool(content_blocks)
                        tool_call_chunks_in_blocks = 0
                        if content_blocks:
                            tool_call_chunks_in_blocks = sum(1 for block in content_blocks if block.get("type") == "tool_call_chunk")
                            # Log first few content_blocks in detail for debugging (only when tool calls detected)
                            if tool_call_chunks_in_blocks > 0 or finish_reason == "tool_calls":
                                logger.debug("Content blocks detail: %s", content_blocks[:3] if len(content_blocks) > 3 else content_blocks)

                        # Enhanced logging only when we expect tool calls but haven't detected them yet
                        expected_tool_calls = is_vllm and not has_seen_tool_results
                        if expected_tool_calls and not tool_calls_detected_in_stream:
                            # Only log when we're actively debugging missing tool calls
                            logger.debug(
                                "AI token #%d: has_tool_calls=%s, content_length=%d, finish_reason=%s, "
                                "has_content_blocks=%s, tool_call_chunks=%d",
                                stream_chunk_count,
                                has_tool_calls,
                                len(content),
                                finish_reason,
                                has_content_blocks,
                                tool_call_chunks_in_blocks,
                            )
                        # Disabled verbose AI token logging for production
                        # else:
                        #     logger.debug(
                        #         "AI token: has_tool_calls=%s, content_length=%d, finish_reason=%s",
                        #         has_tool_calls,
                        #         len(content),
                        #         finish_reason,
                        #     )

                    # Filter out tool-related messages (DO NOT display in chat)
                    # 1. Tool results (type="tool") - processed internally for citations
                    if hasattr(token, "type") and token.type == "tool":
                        tool_results.append(token.content)
                        logger.debug("Tool result received, %d total results", len(tool_results))
                        tool_executing = False
                        has_seen_tool_results = True
                        
                        # Stop any pending thinking spinners (for all tools, not just search)
                        from rag_engine.api.stream_helpers import update_message_status_in_history
                        update_message_status_in_history(gradio_history, "thinking", "done")

                        # Update accumulated context for next tool call
                        # Agent tracks context, not the tool!
                        _, accumulated_tool_tokens = estimate_accumulated_tokens([], tool_results)
                        agent_context.accumulated_tool_tokens = accumulated_tool_tokens

                        logger.debug(
                            "Updated accumulated context: conversation=%d, tools=%d (total: %d)",
                            conversation_tokens,
                            accumulated_tool_tokens,
                            conversation_tokens + accumulated_tool_tokens,
                        )

                        # Parse result to get articles and emit completion metadata with sources
                        from rag_engine.api.stream_helpers import (
                            extract_article_count_from_tool_result,
                            yield_search_completed,
                        )

                        # Parse result to get articles and emit completion metadata with sources
                        from rag_engine.tools.utils import parse_tool_result_to_articles

                        articles_list = parse_tool_result_to_articles(token.content)
                        articles_count = len(articles_list) if articles_list else extract_article_count_from_tool_result(token.content)

                        # Format articles for display (title and URL)
                        articles_for_display = []
                        if articles_list:
                            for article in articles_list:
                                article_meta = article.metadata if hasattr(article, "metadata") else {}
                                title = article_meta.get("title", "Untitled")
                                url = article_meta.get("url", "")
                                articles_for_display.append({"title": title, "url": url})

                        search_completed_msg = yield_search_completed(
                            count=articles_count,
                            articles=articles_for_display if articles_for_display else None,
                        )
                        
                        # Update previous pending messages to done (stop spinners)
                        from rag_engine.api.stream_helpers import update_message_status_in_history
                        update_message_status_in_history(gradio_history, "thinking", "done")
                        update_message_status_in_history(gradio_history, "search_started", "done")
                        
                        # Add search completed to history and yield full history
                        gradio_history.append(search_completed_msg)
                        yield list(gradio_history)
                        
                        # Show "Generating answer" spinner while LLM processes the results
                        # This is especially helpful for slow LLM responses or non-streaming fallback
                        from rag_engine.api.stream_helpers import yield_generating_answer
                        generating_msg = yield_generating_answer()
                        gradio_history.append(generating_msg)
                        yield list(gradio_history)

                        # CRITICAL: Check if accumulated tool results exceed safe threshold
                        # This prevents overflow when agent makes multiple tool calls
                        if settings.llm_fallback_enabled and not selected_model:
                            from rag_engine.config.settings import get_allowed_fallback_models
                            from rag_engine.llm.fallback import select_mid_turn_fallback_model

                            fallback_model = select_mid_turn_fallback_model(
                                current_model,
                                messages,
                                tool_results,
                                get_allowed_fallback_models(),
                            )

                            if fallback_model:
                                from rag_engine.api.stream_helpers import yield_model_switch_notice

                                model_switch_msg = yield_model_switch_notice(fallback_model)
                                # Add model switch notice to history and yield full history
                                gradio_history.append(model_switch_msg)
                                yield list(gradio_history)

                                # Recreate agent with larger model
                                # Note: This is expensive but prevents catastrophic overflow
                                agent = _create_rag_agent(override_model=fallback_model)
                                current_model = fallback_model

                                # Note: Can't restart stream here - agent will continue with new model
                                # for subsequent calls

                        # Skip further processing of tool messages
                        continue

                    # 2. AI messages with tool_calls (when agent decides to call tools)
                    # Check multiple indicators: token.tool_calls, content_blocks, and finish_reason
                    # These should NEVER be displayed - only show metadata
                    has_tool_calls_attr = hasattr(token, "tool_calls") and bool(token.tool_calls)

                    # Check content_blocks for tool_call_chunk (critical for vLLM streaming)
                    token_content_blocks = getattr(token, "content_blocks", None)
                    has_tool_call_chunks = bool(token_content_blocks) and any(
                        block.get("type") == "tool_call_chunk" for block in token_content_blocks
                    )

                    # Check finish_reason (indicates tool calls completed)
                    token_response_metadata = getattr(token, "response_metadata", {})
                    token_finish_reason = token_response_metadata.get("finish_reason") if isinstance(token_response_metadata, dict) else None
                    finish_reason_is_tool_calls = token_finish_reason == "tool_calls"

                    # Tool call detected if any of these conditions are true
                    tool_call_detected = has_tool_calls_attr or has_tool_call_chunks or finish_reason_is_tool_calls

                    if tool_call_detected:
                        tool_calls_detected_in_stream = True
                        if not tool_executing:
                            tool_executing = True
                            # Log which method detected the tool call
                            if has_tool_calls_attr:
                                call_count = len(token.tool_calls) if isinstance(token.tool_calls, list) else "?"
                                logger.info("Agent calling tool(s) via token.tool_calls: %s call(s)", call_count)
                            elif has_tool_call_chunks:
                                logger.info("Agent calling tool(s) via content_blocks tool_call_chunk")
                            elif finish_reason_is_tool_calls:
                                logger.info("Agent calling tool(s) detected via finish_reason=tool_calls")
                                # Check if tool_calls are now available in the token
                                final_tool_calls = getattr(token, "tool_calls", None)
                                if final_tool_calls:
                                    logger.info("Final tool_calls after finish_reason: %s call(s)",
                                               len(final_tool_calls) if isinstance(final_tool_calls, list) else "?")

                            # Get tool name to determine which thinking block to show
                            tool_name = tool_call_accumulator.get_tool_name(token)

                            if tool_name == "retrieve_context":
                                tool_query = tool_call_accumulator.process_token(token)
                                # If we've already seen tool results, ADD a new search_started block
                                # Otherwise, UPDATE the existing one (which was added at handler start)
                                if has_seen_tool_results:
                                    # Add NEW search_started block for subsequent tool calls
                                    logger.info("Adding NEW search_started block (subsequent tool call): query=%s", tool_query[:50] if tool_query else "empty")
                                    from rag_engine.api.stream_helpers import yield_search_started
                                    search_started_msg = yield_search_started(tool_query or "")
                                    gradio_history.append(search_started_msg)
                                    yield list(gradio_history)
                                else:
                                    # Update existing search_started message with LLM-generated query
                                    logger.info("Updating existing search_started block: query=%s", tool_query[:50] if tool_query else "empty")
                                    from rag_engine.api.stream_helpers import update_search_started_in_history
                                    if tool_query:
                                        if update_search_started_in_history(gradio_history, tool_query):
                                            yield list(gradio_history)
                            elif tool_name:
                                # Show generic thinking block for non-search tools
                                # Reference agent pattern: always append, don't check duplicates
                                from rag_engine.api.stream_helpers import yield_thinking_block
                                thinking_msg = yield_thinking_block(tool_name)
                                gradio_history.append(thinking_msg)
                                yield list(gradio_history)
                        # Skip displaying the tool call itself and any content
                        continue

                    # 3. Only stream text content from messages WITHOUT tool_calls
                    # This ensures we only show the final answer, not tool reasoning
                    if hasattr(token, "tool_calls") and token.tool_calls:
                        # Skip any message that has tool_calls (redundant check for safety)
                        continue

                    # Process content blocks for final answer text streaming
                    text_chunk_found = False
                    if hasattr(token, "content_blocks") and token.content_blocks:
                        for block in token.content_blocks:
                            if block.get("type") == "tool_call_chunk":
                                # Tool call chunk detected - emit metadata if not already done
                                if not tool_executing:
                                    tool_executing = True
                                    logger.debug("Agent calling tool via chunk")

                                    # Get tool name to determine which thinking block to show
                                    tool_name = tool_call_accumulator.get_tool_name(token)

                                    if tool_name == "retrieve_context":
                                        tool_query = tool_call_accumulator.process_token(token)
                                        # If we've already seen tool results, ADD a new search_started block
                                        # Otherwise, UPDATE the existing one (which was added at handler start)
                                        if has_seen_tool_results:
                                            # Add NEW search_started block for subsequent tool calls
                                            logger.info("Adding NEW search_started block via chunk (subsequent): query=%s", tool_query[:50] if tool_query else "empty")
                                            from rag_engine.api.stream_helpers import yield_search_started
                                            search_started_msg = yield_search_started(tool_query or "")
                                            gradio_history.append(search_started_msg)
                                            yield list(gradio_history)
                                        else:
                                            # Update existing search_started message with LLM-generated query
                                            logger.info("Updating existing search_started block via chunk: query=%s", tool_query[:50] if tool_query else "empty")
                                            from rag_engine.api.stream_helpers import update_search_started_in_history
                                            if tool_query:
                                                if update_search_started_in_history(gradio_history, tool_query):
                                                    yield list(gradio_history)
                                    elif tool_name:
                                        # Show generic thinking block for non-search tools
                                        # Reference agent pattern: always append, don't check duplicates
                                        from rag_engine.api.stream_helpers import yield_thinking_block
                                        thinking_msg = yield_thinking_block(tool_name)
                                        gradio_history.append(thinking_msg)
                                        yield list(gradio_history)
                                # Never stream tool call chunks as text
                                continue

                            elif block.get("type") == "text" and block.get("text"):
                                # Only stream text if we're not currently executing tools
                                # This prevents streaming the agent's "reasoning" about tool calls
                                if not tool_executing:
                                    text_chunk = block["text"]
                                    
                                    # Stop "generating answer" spinner on first text chunk
                                    if not answer and has_seen_tool_results:
                                        update_message_status_in_history(gradio_history, "generating_answer", "done")
                                    
                                    answer, disclaimer_prepended = _process_text_chunk_for_streaming(
                                        text_chunk, answer, disclaimer_prepended, has_seen_tool_results
                                    )
                                    # Update last message (answer) in history and yield full history
                                    _update_or_append_assistant_message(gradio_history, answer)
                                    yield list(gradio_history)
                                    text_chunk_found = True

                    # Fallback: If no text found in content_blocks, check token.content directly
                    # This handles vLLM and other providers that provide text directly in token.content
                    # LangChain streaming provides incremental chunks (tested and confirmed)
                    if not text_chunk_found and is_ai_message and not tool_executing:
                        token_content = str(getattr(token, "content", ""))
                        if token_content:
                            # LangChain streaming provides incremental chunks, but handle both cases for robustness
                            new_chunk = None
                            if answer and token_content.startswith(answer):
                                # Cumulative content - extract only the new part
                                new_chunk = token_content[len(answer):]
                            elif token_content != answer:
                                # Incremental chunk - use as-is (typical case)
                                new_chunk = token_content

                            if new_chunk:
                                # Stop "generating answer" spinner on first text chunk
                                if not answer and has_seen_tool_results:
                                    update_message_status_in_history(gradio_history, "generating_answer", "done")
                                
                                answer, disclaimer_prepended = _process_text_chunk_for_streaming(
                                    new_chunk, answer, disclaimer_prepended, has_seen_tool_results
                                )
                                # Update last message (answer) in history and yield full history
                                _update_or_append_assistant_message(gradio_history, answer)
                                # Track incomplete response for memory saving if cancelled (pattern from test script)
                                incomplete_response = answer
                                final_response = answer  # Update final response as we stream
                                yield list(gradio_history)

                # Handle "updates" mode for agent state updates
                elif stream_mode == "updates":
                    # We can log updates but don't need to yield them
                    logger.debug("Agent update: %s", list(chunk.keys()) if isinstance(chunk, dict) else chunk)

            # After stream completes, check if we expected tool calls but didn't get results
            # Only check for vLLM on first message (when we expect tool calls)
            should_fallback, fallback_enabled = check_stream_completion(
                is_vllm=is_vllm,
                has_seen_tool_results=has_seen_tool_results,
                tool_calls_detected=tool_calls_detected_in_stream,
                tool_results_count=len(tool_results),
                stream_chunk_count=stream_chunk_count,
            )

        except OpenAIAPIError as api_error:
            # Handle OpenAI API errors (e.g., malformed streaming response)
            error_msg = str(api_error).lower()
            is_streaming_error = "list index out of range" in error_msg or "streaming" in error_msg

            if is_streaming_error:
                # If streaming fails, try falling back to invoke() mode
                # This can happen with any OpenAI-compatible provider, not just vLLM
                # Only fallback if we haven't seen tool results yet (first turn)
                if not has_seen_tool_results:
                    logger.warning(
                        "OpenAI API streaming error, falling back to invoke() mode: %s", api_error
                    )
                    fallback_to_invoke = True
                else:
                    # If we've already seen tool results, streaming error might be in the answer phase
                    # Re-raise to be handled by outer exception handler
                    logger.error("OpenAI API error during streaming (after tool execution): %s", api_error, exc_info=True)
                    raise
            else:
                # For other API errors, re-raise to be handled by outer exception handler
                logger.error("OpenAI API error: %s", api_error, exc_info=True)
                raise
        except Exception as stream_error:
            # If streaming fails and we expected tool calls, fall back to invoke()
            # Only if fallback is enabled
            if should_use_fallback(
                is_vllm=is_vllm,
                has_seen_tool_results=has_seen_tool_results,
                tool_calls_detected=False,
                tool_results_count=len(tool_results),
            ):
                logger.warning(
                    "Streaming failed for vLLM, falling back to invoke() mode: %s", stream_error
                )
                fallback_to_invoke = True
            else:
                raise

            # Fallback to invoke() only if we expected tool calls but didn't get them
            # This ensures tool execution happens even if streaming detection fails
            # Can be disabled via vllm_streaming_fallback_enabled setting for testing
            if fallback_to_invoke or should_fallback:
                # Show "Generating answer" spinner for invoke() fallback (non-streaming)
                # This is especially important since invoke() has no streaming feedback
                if has_seen_tool_results and not answer:
                    from rag_engine.api.stream_helpers import yield_generating_answer
                    generating_msg = yield_generating_answer()
                    gradio_history.append(generating_msg)
                    yield list(gradio_history)
                
                # Execute fallback and process results
                fallback_results = {}
                for chunk in execute_fallback_invoke(
                    agent=agent,
                    messages=messages,
                    agent_context=agent_context,
                    has_seen_tool_results=has_seen_tool_results,
                    result_container=fallback_results,
                ):
                    # Handle metadata (dict) or text chunks (str) and add to history
                    if isinstance(chunk, dict):
                        # Metadata message (search_started, search_completed, etc.)
                        gradio_history.append(chunk)
                        yield list(gradio_history)
                    elif isinstance(chunk, str):
                        # Text chunk - stop generating spinner on first chunk
                        if not answer and has_seen_tool_results:
                            update_message_status_in_history(gradio_history, "generating_answer", "done")
                        # Update last message or create new one
                        _update_or_append_assistant_message(gradio_history, chunk)
                        yield list(gradio_history)

            # Extract results from container
            if fallback_results:
                fallback_tool_results = fallback_results.get("tool_results", [])
                fallback_answer = fallback_results.get("final_answer", "")
                fallback_disclaimer = fallback_results.get("disclaimer_prepended", False)

                # Update tool results
                if fallback_tool_results:
                    tool_results.extend(fallback_tool_results)
                    has_seen_tool_results = True
                    # Update accumulated context
                    _, accumulated_tool_tokens = estimate_accumulated_tokens([], tool_results)
                    agent_context.accumulated_tool_tokens = accumulated_tool_tokens

                # Update answer if we got one from fallback
                if fallback_answer:
                    answer = fallback_answer
                    disclaimer_prepended = fallback_disclaimer or disclaimer_prepended

        # Check if we were cancelled during streaming
        if is_cancelled():
            logger.info("Stream cancelled by user - saving incomplete response and returning")
            if incomplete_response and session_id:
                llm_manager.save_assistant_turn(session_id, incomplete_response)
                logger.info(f"Saved INCOMPLETE response to memory ({len(incomplete_response)} chars)")
            yield list(gradio_history)
            return

        # Accumulate articles from tool results and add citations
        logger.info("Stream completed: %d chunks processed, %d tool results", stream_chunk_count, len(tool_results))
        from rag_engine.tools import accumulate_articles_from_tool_results
        articles = accumulate_articles_from_tool_results(tool_results)

        # Handle no results case
        if not articles:
            final_text = answer
            logger.info("Agent completed with no retrieved articles")
        else:
            final_text = format_with_citations(answer, articles)
            logger.info("Agent completed with %d articles", len(articles))

        # Only update/append if we have actual content (don't overwrite with empty)
        # If answer is empty, the agent didn't produce any text - don't create empty message
        if final_text and final_text.strip():
            # Update last message (answer) with final formatted text and yield full history
            _update_or_append_assistant_message(gradio_history, final_text)
        elif not final_text or not final_text.strip():
            # Empty answer - log warning but don't create empty message
            logger.warning("Agent completed with empty answer - no text content was produced")

        # Update final response tracking
        final_response = final_text

        # Save conversation turn (reuse existing pattern)
        # Streaming completed successfully - save complete response to memory
        if final_response and session_id:
            llm_manager.save_assistant_turn(session_id, final_response)
            logger.info(f"Saved complete response to memory ({len(final_response)} chars)")

        # Log final history state for debugging
        logger.info(
            "Final history yield: total_messages=%d (user=%d, assistant=%d, ui_metadata=%d)",
            len(gradio_history),
            sum(1 for msg in gradio_history if isinstance(msg, dict) and msg.get("role") == "user"),
            sum(1 for msg in gradio_history if isinstance(msg, dict) and msg.get("role") == "assistant" and not msg.get("metadata")),
            sum(1 for msg in gradio_history if isinstance(msg, dict) and msg.get("metadata")),
        )

        yield list(gradio_history)

    except GeneratorExit:
        # Stream was cancelled - save incomplete response to memory if available (pattern from test script)
        logger.info("Stream cancelled (GeneratorExit) - saving incomplete response")
        if incomplete_response and session_id:
            llm_manager.save_assistant_turn(session_id, incomplete_response)
            logger.info(
                f"Saved INCOMPLETE response to memory: {incomplete_response[:50]}... ({len(incomplete_response)} chars)"
            )
        raise  # Re-raise to allow Gradio to handle cancellation properly

    except Exception as e:
        logger.error("Error in agent_chat_handler: %s", e, exc_info=True)
        # Variables are initialized at function start, so they should always exist
        # Handle context-length overflow gracefully by switching to a larger model (once)
        err_text = str(e).lower()
        is_context_overflow = (
            "maximum context length" in err_text
            or "context length" in err_text
            or "token limit" in err_text
            or "too many tokens" in err_text
        )

        if settings.llm_fallback_enabled and is_context_overflow and messages:
            try:  # Single-shot fallback retry
                # Estimate required tokens and pick a capable fallback model
                # Get current model context window for adaptive overhead calculation
                from rag_engine.llm.model_configs import MODEL_CONFIGS

                current_model_config = MODEL_CONFIGS.get(current_model)
                if not current_model_config:
                    for key in MODEL_CONFIGS:
                        if key != "default" and key in current_model:
                            current_model_config = MODEL_CONFIGS[key]
                            break
                if not current_model_config:
                    current_model_config = MODEL_CONFIGS["default"]
                current_window = int(current_model_config.get("token_limit", 0))

                required_tokens = estimate_accumulated_context(
                    messages,
                    tool_results,
                    context_window=current_window,
                )
                fallback_model = _find_model_for_tokens(required_tokens) or None

                if fallback_model and fallback_model != current_model:
                    logger.warning(
                        "Retrying with fallback model due to context overflow: %s -> %s (required≈%d)",
                        current_model,
                        fallback_model,
                        required_tokens,
                    )

                    # Inform UI about the switch
                    from rag_engine.api.stream_helpers import yield_model_switch_notice

                    model_switch_msg = yield_model_switch_notice(fallback_model)
                    # Add model switch notice to history and yield full history
                    gradio_history.append(model_switch_msg)
                    yield list(gradio_history)

                    # Recreate agent and re-run the stream once
                    agent = _create_rag_agent(override_model=fallback_model)
                    current_model = fallback_model

                    conversation_tokens, _ = estimate_accumulated_tokens(messages, [])
                    agent_context = AgentContext(
                        conversation_tokens=conversation_tokens,
                        accumulated_tool_tokens=0,
                    )

                    answer = ""
                    tool_results = []
                    has_seen_tool_results = False
                    disclaimer_prepended = False  # Track if disclaimer has been prepended to stream

                    async for stream_mode, chunk in agent.astream(
                        {"messages": messages},
                        context=agent_context,
                        stream_mode=["updates", "messages"],
                    ):
                        if stream_mode == "messages":
                            token, metadata = chunk
                            # Collect tool results silently; only stream final text
                            if hasattr(token, "type") and token.type == "tool":
                                tool_results.append(token.content)
                                has_seen_tool_results = True
                                # keep agent_context.accumulated_tool_tokens updated
                                _, acc_tool_toks = estimate_accumulated_tokens([], tool_results)
                                agent_context.accumulated_tool_tokens = acc_tool_toks
                                
                                # Show generating answer spinner after tool results in fallback retry
                                if not answer:
                                    from rag_engine.api.stream_helpers import yield_generating_answer
                                    generating_msg = yield_generating_answer()
                                    gradio_history.append(generating_msg)
                                    yield list(gradio_history)
                                continue

                            if hasattr(token, "tool_calls") and token.tool_calls:
                                # Do not stream tool call reasoning
                                continue

                            if hasattr(token, "content_blocks") and token.content_blocks:
                                for block in token.content_blocks:
                                    if block.get("type") == "text" and block.get("text"):
                                        text_chunk = block["text"]
                                        
                                        # Stop generating spinner on first text chunk
                                        if not answer and has_seen_tool_results:
                                            update_message_status_in_history(gradio_history, "generating_answer", "done")
                                        
                                        answer, disclaimer_prepended = _process_text_chunk_for_streaming(
                                            text_chunk, answer, disclaimer_prepended, has_seen_tool_results
                                        )
                                        # Update last message (answer) in history and yield full history
                                        _update_or_append_assistant_message(gradio_history, answer)
                                        # Track incomplete response for memory saving if cancelled (pattern from test script)
                                        incomplete_response = answer
                                        final_response = answer  # Update final response as we stream
                                        yield list(gradio_history)
                        elif stream_mode == "updates":
                            # No-op for UI
                            pass

                    from rag_engine.tools import accumulate_articles_from_tool_results

                    articles = accumulate_articles_from_tool_results(tool_results)
                    if not articles:
                        final_text = answer
                    else:
                        final_text = format_with_citations(answer, articles)

                    # Only update/append if we have actual content (don't overwrite with empty)
                    if final_text and final_text.strip():
                        # Update last message with final formatted text
                        _update_or_append_assistant_message(gradio_history, final_text)
                    elif not final_text or not final_text.strip():
                        # Empty answer - log warning but don't create empty message
                        logger.warning("Fallback agent completed with empty answer - no text content was produced")

                    # Update final response tracking
                    final_response = final_text

                    # Save conversation turn
                    if final_response and session_id:
                        llm_manager.save_assistant_turn(session_id, final_response)
                        logger.info(f"Saved complete response to memory ({len(final_response)} chars)")
                    yield list(gradio_history)
                    return
            except Exception as retry_exc:  # If fallback retry fails, emit original-style error
                logger.error("Fallback retry failed: %s", retry_exc, exc_info=True)

        # Default error path - add error message to history
        error_msg = f"Извините, произошла ошибка / Sorry, an error occurred: {str(e)}"
        # gradio_history should always exist (initialized early), but ensure it's not empty
        if not gradio_history:
            # Add user message if we have it
            if message:
                gradio_history.append({"role": "user", "content": message})

        _update_or_append_assistant_message(gradio_history, error_msg)
        yield list(gradio_history)




def query_rag(question: str, provider: str = "gemini", top_k: int = 5) -> str:
    if not question or not question.strip():
        return "Error: Empty question"
    docs = retriever.retrieve(question, top_k=top_k)
    # If no documents found, inject a message into the context
    has_no_results_doc = False
    if not docs:
        from rag_engine.retrieval.retriever import Article
        no_results_msg = "К сожалению, не найдено релевантных материалов / No relevant results found."
        no_results_doc = Article(
            kb_id="",
            content=no_results_msg,
            metadata={"title": "No Results", "kbId": "", "_is_no_results": True}
        )
        docs = [no_results_doc]
        has_no_results_doc = True

    answer = llm_manager.generate(question, docs, provider=provider)
    # If we injected the "no results" message, don't add citations
    if has_no_results_doc:
        return answer  # Don't add citations for "no results" message
    return format_with_citations(answer, docs)


# Configure chatbot height and UI elements based on embedded widget setting
if settings.gradio_embedded_widget:
    # For embedded widget
    chatbot_height = "400px"
    chatbot_max_height = "65vh"
    chat_title = None
else:
    # For standalone app
    chatbot_height = "70vh"  # 70% of viewport height for standalone
    chatbot_max_height = "70vh"  # Same as height for consistency
    chat_title = "Ассистент базы знаний Comindware Platform"

# Force agent-based handler; legacy direct handler removed
handler_fn = agent_chat_handler
logger.info("Using agent-based (LangChain) handler for chat interface")

# Load CSS theme from reference agent
css_file_path = Path(__file__).parent.parent / "resources" / "css" / "cmw_copilot_theme.css"

# Setup Gradio static resource paths (for logo and other assets)
# Must use absolute path for GRADIO_ALLOWED_PATHS
RESOURCES_DIR = (Path(__file__).parent.parent / "resources").resolve()
try:
    existing_allowed = os.environ.get("GRADIO_ALLOWED_PATHS", "")
    parts = [p for p in existing_allowed.split(os.pathsep) if p]
    resources_dir_str = str(RESOURCES_DIR)
    if resources_dir_str not in parts:
        parts.append(resources_dir_str)
    os.environ["GRADIO_ALLOWED_PATHS"] = os.pathsep.join(parts)
    logger.info(f"Gradio static allowed paths: {os.environ['GRADIO_ALLOWED_PATHS']}")
except Exception as e:
    logger.warning(f"Could not set GRADIO_ALLOWED_PATHS: {e}")

# Wrapper function to expose retrieve_context tool as API endpoint
# The tool is a StructuredTool, so we need to extract the underlying function
def get_knowledge_base_articles(query: str, top_k: int | str | None = None) -> str:
    """Search and retrieve documentation articles from the Comindware Platform knowledge base.

    Use this tool when you need raw search results with article metadata. For intelligent
    answers with automatic context retrieval, use agent_chat_handler instead.

    Args:
        query: Search query or question to find relevant documentation articles.
               Examples: "authentication", "API integration", "user management"
        top_k: Optional limit on number of articles to return. If not specified,
               returns the default number of most relevant articles (typically 10-20).
               Can be provided as int or string (will be converted).

    Returns:
        JSON string containing an array of articles, each with:
        - kb_id: Article identifier
        - title: Article title
        - url: Link to the article
        - content: Full article content (markdown format)
        - metadata: Additional metadata including rerank scores and source information
    """
    # Convert top_k from string to int if needed (Gradio/MCP may pass strings)
    converted_top_k: int | None = None
    if top_k is not None:
        if isinstance(top_k, str):
            try:
                converted_top_k = int(top_k)
            except (ValueError, TypeError) as e:
                raise ValueError(f"top_k must be a valid integer, got: {top_k!r}") from e
        elif isinstance(top_k, int):
            converted_top_k = top_k
        else:
            raise ValueError(f"top_k must be an integer or None, got: {type(top_k).__name__}")

        # Validate positive integer
        if converted_top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {converted_top_k}")

    # Access the underlying function from the StructuredTool
    return retrieve_context.func(query=query, top_k=converted_top_k)


# MCP-compatible wrapper for agent_chat_handler
# Collects streaming response into a single string for MCP tools
async def ask_comindware(message: str) -> str:
    """Ask questions about Comindware Platform documentation and get intelligent answers with citations.

    The assistant automatically searches the knowledge base to find relevant articles
    and provides comprehensive answers based on official documentation. Use this for
    technical questions, configuration help, API usage, troubleshooting, and general
    platform guidance.

    Args:
        message: User's current message or question

    Returns:
        Complete response text with citations
    """
    # Collect all chunks from the generator into a single response
    response_parts = []
    last_text_response = None
    generator = None
    try:
        # Call the handler with empty history and None request (MCP context)
        # Note: agent_chat_handler is an async generator function used by ChatInterface
        generator = agent_chat_handler(message=message, history=[], request=None)

        # Consume the entire async generator to collect all responses
        # The generator now yields: full message history lists (for ChatInterface)
        # Extract the final answer from the last assistant message in the history
        async for chunk in generator:
            if chunk is None:
                continue

            # Handle full history lists (new format - preserves all messages)
            if isinstance(chunk, list):
                # Extract final answer from last assistant message
                for msg in reversed(chunk):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        # Skip metadata-only messages (thinking blocks)
                        if content and not msg.get("metadata"):
                            if content.strip():
                                last_text_response = content
                                break
            # Handle string responses (backward compatibility)
            elif isinstance(chunk, str):
                if chunk.strip():  # Only add non-empty strings
                    response_parts.append(chunk)
                    last_text_response = chunk
            # Handle dict responses (backward compatibility)
            elif isinstance(chunk, dict):
                # Only extract text content from dicts, skip pure metadata
                content = chunk.get("content", "")
                if content and isinstance(content, str) and content.strip():
                    response_parts.append(content)
                    last_text_response = content

        # Ensure generator is fully consumed
        if generator:
            try:
                # Try to close the async generator if it's still open
                await generator.aclose()
            except Exception:
                pass

        # Return the final accumulated response (last chunk contains the full formatted text)
        if last_text_response:
            return last_text_response
        elif response_parts:
            # Fallback: join all parts if no single final response
            return "\n".join(response_parts)
        else:
            return "No response generated. Please try rephrasing your question."

    except StopIteration:
        # Generator exhausted normally
        if last_text_response:
            return last_text_response
        elif response_parts:
            return "\n".join(response_parts)
        return "No response generated."
    except IndexError as e:
        # Handle specific "pop index out of range" error
        logger.error("IndexError in ask_comindware (pop index): %s", e, exc_info=True)
        import traceback
        logger.error("Traceback: %s", traceback.format_exc())
        # Try to return whatever we collected
        if last_text_response:
            return last_text_response
        elif response_parts:
            return "\n".join(response_parts)
        return f"Error processing response: {str(e)}. Please try again."
    except Exception as e:
        logger.error("Error in ask_comindware: %s", e, exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        logger.error("Full traceback: %s", error_details)
        # Try to return whatever we collected before the error
        if last_text_response:
            return last_text_response + f"\n\n[Note: An error occurred during processing: {type(e).__name__}]"
        elif response_parts:
            return "\n".join(response_parts) + f"\n\n[Note: An error occurred: {str(e)}]"
        return f"Error: {str(e)}. Please try rephrasing your question or contact support."

with gr.Blocks(
    title=chat_title or "Comindware Platform Documentation Assistant",
) as demo:
    # Header (like reference agent)
    if chat_title:
        gr.Markdown(f"# {chat_title}", elem_classes=["hero-title"])

    # Chatbot component (like reference agent - NOT ChatInterface)
    # In Gradio 6, Chatbot uses messages format by default (no type parameter needed)
    # Conditional sizing: embedded widget uses fixed height and follows container, standalone is resizable
    chatbot = gr.Chatbot(
        label="Диалог с агентом",
        height=chatbot_height,  # Always set height (400px for embedded, 70vh for standalone)
        max_height=chatbot_max_height,  # Always set max_height (65vh for embedded, 70vh for standalone)
        show_label=True,
        container=True,
        buttons=["copy", "copy_all"],
        elem_id="chatbot-main",
        elem_classes=["chatbot-card"],
        resizable=not settings.gradio_embedded_widget,  # Resizable only in standalone mode, not embedded
    )

    # Message input (regular single-line Textbox, NOT MultimodalTextbox)
    # Small built-in submit and stop buttons (icons) in the Textbox
    # Pattern from test script: dynamic stop button visibility
    msg = gr.Textbox(
        label="Сообщение",
        placeholder="Введите ваш вопрос...",
        # Single-line input so Enter submits the message instead of inserting newlines
        # See: https://www.gradio.app/docs/textbox and
        # https://www.gradio.app/guides/blocks-and-event-listeners#running-events-consecutively
        lines=1,
        max_lines=1,
        show_label=False,  # Hide label for cleaner UI
        elem_id="message-input",
        elem_classes=["message-card"],
        submit_btn=True,  # Small built-in submit icon button
        stop_btn=False,  # Start hidden, will be shown when streaming starts
    )

    # State to store saved message (pattern from ChatInterface/test script)
    saved_input = gr.State()
    # State to store current session_id for memory clearing
    current_session_id = gr.State(None)
    # Cancellation state - mutable dict so changes propagate to running generator
    # Note: Using direct dict value (not lambda) for Gradio 6 compatibility
    cancellation_state = gr.State(value={"cancelled": False})

    def handle_stop_click(history: list[dict], cancel_state: dict | None) -> tuple[list[dict], dict]:
        """Handle built-in stop button click - set cancellation flag and add UI block.

        Sets cancellation flag in shared state so the running generator can check it,
        and adds a cancellation message as a separate UI block.
        """
        logger.info("Stop button clicked - setting cancellation flag")
        # Ensure cancel_state is a valid dict
        if cancel_state is None or not isinstance(cancel_state, dict):
            cancel_state = {"cancelled": True}
        else:
            cancel_state["cancelled"] = True
        # Add cancellation UI block as separate message (like other UI blocks)
        from rag_engine.api.stream_helpers import yield_cancelled

        cancellation_msg = yield_cancelled()
        history.append(cancellation_msg)
        return history, cancel_state

    def clear_and_save_textbox(message: str) -> tuple[gr.Textbox, str]:
        """Clear textbox and save message to state (pattern from ChatInterface/test script)."""
        return (
            gr.Textbox(value="", interactive=False, placeholder=""),
            message,
        )

    def handle_chatbot_clear(session_id: str | None) -> None:
        """Handle chatbot clear event - also clear memory when chat is cleared."""
        logger.info("Clear button clicked - clearing memory")
        if session_id:
            llm_manager._conversations.clear(session_id)
            logger.info(f"Memory cleared for session {session_id[:8]}...")
        return None  # Clear session_id state

    # Store original stop_btn value (True, but we start with False)
    original_stop_btn = True

    # Submit event - main handler
    # Pattern from ChatInterface/test script: show stop button after submit succeeds, hide when streaming completes
    user_submit = msg.submit(
        fn=clear_and_save_textbox,
        inputs=[msg],
        outputs=[msg, saved_input],  # Clear textbox and save message to state
        queue=False,
        api_visibility="private",
    )

    # Show stop button when submit succeeds (before streaming starts)
    # Pattern from ChatInterface: after_success.success() shows stop button
    user_submit.success(
        lambda: gr.Textbox(submit_btn=False, stop_btn=original_stop_btn),
        outputs=[msg],
        queue=False,
        api_visibility="private",
    )

    def save_session_id(message: str, history: list[dict], request: gr.Request | None) -> str | None:
        """Save session_id to state for memory clearing."""
        base_session_id = getattr(request, "session_hash", None) if request is not None else None
        session_id = salt_session_id(base_session_id, history, message)
        return session_id

    def reset_cancellation_state(cancel_state: dict | None) -> dict:
        """Reset cancellation state at start of new submission."""
        if cancel_state is None or not isinstance(cancel_state, dict):
            cancel_state = {"cancelled": False}
        else:
            cancel_state["cancelled"] = False
        return cancel_state

    # Main streaming handler - chained from user_submit
    # Pattern from ChatInterface: append user message to chatbot first, then call handler
    submit_event = user_submit.then(
        fn=reset_cancellation_state,
        inputs=[cancellation_state],
        outputs=[cancellation_state],
        queue=False,
        api_visibility="private",
    ).then(
        lambda message, history: history + [{"role": "user", "content": message}],
        inputs=[saved_input, chatbot],
        outputs=[chatbot],
        queue=False,
        api_visibility="private",
    ).then(
        fn=save_session_id,
        inputs=[saved_input, chatbot],  # Request is automatically passed to functions that accept it
        outputs=[current_session_id],
        queue=False,
        api_visibility="private",
    ).then(
        fn=handler_fn,
        inputs=[saved_input, chatbot, cancellation_state],  # Pass cancellation state to handler
        outputs=[chatbot],
        concurrency_limit=settings.gradio_default_concurrency_limit,
        api_visibility="private",  # Hide agent_chat_handler from MCP tools
    ).then(
        lambda: gr.Textbox(value="", interactive=True),  # Clear and re-enable input after completion
        outputs=[msg],
        api_visibility="private",
    )

    # Hide stop button when streaming completes
    # Pattern from ChatInterface: events_to_cancel.then() hides stop button
    submit_event.then(
        lambda: gr.Textbox(submit_btn=True, stop_btn=False),
        outputs=[msg],
        queue=False,
        api_visibility="private",
    )

    # Built-in stop button automatically cancels submit_event when stop_btn=True
    # Wire up the stop event to handle cancellation and update history
    # Also hide stop button when stop is clicked
    msg.stop(
        fn=handle_stop_click,
        inputs=[chatbot, cancellation_state],
        outputs=[chatbot, cancellation_state],
        cancels=[submit_event],  # Explicitly cancel submit event (though it's automatic)
        api_visibility="private",
    ).then(
        lambda: gr.Textbox(submit_btn=True, stop_btn=False),  # Hide stop button after cancellation
        outputs=[msg],
        queue=False,
        api_visibility="private",
    )

    # Bind to the built-in clear button's clear event
    # Also clear memory when chat is cleared
    chatbot.clear(
        fn=handle_chatbot_clear,
        inputs=[current_session_id],  # Use stored session_id
        outputs=[current_session_id],  # Clear session_id after clearing memory
        api_visibility="private",
    )

    # Explicitly expose MCP tools (public by default when using gr.api())
    # All other functions (agent_chat_handler, lambda, etc.) are set to private above
    # Functions exposed via gr.api() are automatically public and visible in MCP
    gr.api(
        fn=get_knowledge_base_articles,
        api_name="get_knowledge_base_articles",
        api_description="Search the Comindware Platform documentation knowledge base and retrieve relevant articles with full content. Returns structured JSON with article titles, URLs, content, and metadata. Use this for programmatic access to documentation content. For conversational answers, use ask_comindware instead.",
    )
    # Register the working wrapper function with a business-oriented name for MCP consumers
    # This provides a clean API name that's meaningful to external tools like Cursor
    gr.api(
        fn=ask_comindware,
        api_name="ask_comindware",
        api_description="Ask questions about Comindware Platform documentation and get intelligent answers with citations. The assistant automatically searches the knowledge base to find relevant articles and provides comprehensive answers based on official documentation. Use this for technical questions, configuration help, API usage, troubleshooting, and general platform guidance.",
    )

    # Explicitly set a plain attribute for tests and downstream code to read
    demo.title = "Comindware Platform Documentation Assistant"

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    logger.info(
        "Starting Gradio server at %s:%s (share=%s)",
        settings.gradio_server_name,
        settings.gradio_server_port,
        settings.gradio_share,
    )

    if settings.gradio_share:
        logger.info(
            "Share link enabled. If share link creation fails, the app will still run locally."
        )

    # Configure queue with default concurrency limit per Gradio queuing best practices
    # https://www.gradio.app/guides/queuing
    # This sets the default for all event listeners unless overridden
    demo.queue(
        default_concurrency_limit=settings.gradio_default_concurrency_limit,
        status_update_rate="auto",
    ).launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        share=settings.gradio_share,
        mcp_server=True,
        footer_links=["api"],
        theme=gr.themes.Soft(),
        css_paths=[css_file_path] if css_file_path.exists() else [],
    )


