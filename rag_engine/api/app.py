"""Gradio UI with ChatInterface and REST API endpoint."""
from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging

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

def _create_rag_agent(override_model: str | None = None):
    """Create LangChain agent with forced retrieval tool execution and memory compression.

    Uses centralized factory with app-specific middleware. The factory enforces
    tool execution via tool_choice="retrieve_context" to ensure the agent always
    searches the knowledge base before answering.

    This wrapper preserves test patch points while delegating to the centralized
    agent_factory for consistent agent creation.

    Args:
        override_model: Optional model name to use instead of default
                       (for context window fallback)

    Returns:
        Configured LangChain agent with retrieve_context tool and middleware
    """
    from rag_engine.llm.agent_factory import create_rag_agent

    return create_rag_agent(
        override_model=override_model,
        tool_budget_middleware=ToolBudgetMiddleware(),
        update_context_budget_middleware=update_context_budget,
        compress_tool_results_middleware=compress_tool_results,
    )


def _is_ui_only_message(msg: dict) -> bool:
    """Check if a message is UI-only (should not be sent to agent).

    UI-only messages include:
    - Disclaimer messages
    - Search started/completed metadata messages
    - Model switch notices
    - Any message with metadata field (UI metadata)

    Args:
        msg: Message dict to check

    Returns:
        True if message is UI-only, False otherwise
    """
    if not isinstance(msg, dict):
        return False

    # Check for metadata field (UI-only messages have this)
    if "metadata" in msg:
        return True

    # Check for disclaimer content
    from rag_engine.llm.prompts import AI_DISCLAIMER
    content = msg.get("content", "")
    if isinstance(content, str) and AI_DISCLAIMER.strip() in content:
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
    for msg in gradio_history:
        # Skip UI-only messages
        if _is_ui_only_message(msg):
            continue

        # Skip the current user message (we'll add wrapped version below)
        msg_role = msg.get("role")
        msg_content = msg.get("content", "")
        if msg_role == "user" and isinstance(msg_content, str) and msg_content.strip() == current_message.strip():
            continue

        # Include actual conversation messages
        messages.append(msg)

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


def agent_chat_handler(
    message: str,
    history: list[dict],
    request: gr.Request | None = None,
) -> Generator[str | dict | list[dict], None, None]:
    """Ask questions about Comindware Platform documentation and get intelligent answers with citations.

    The assistant automatically searches the knowledge base to find relevant articles
    and provides comprehensive answers based on official documentation. Use this for
    technical questions, configuration help, API usage, troubleshooting, and general
    platform guidance.

    Args:
        message: User's current message or question
        history: Chat history from Gradio
        request: Gradio request object for session management

    Yields:
        Complete message history lists (for ChatInterface) to preserve all messages
        including disclaimer, thinking blocks, and streaming answer.
    """
    if not message or not message.strip():
        yield "Пожалуйста, введите вопрос / Please enter a question."
        return

    # Build full message history starting from provided history
    # This ensures all messages (disclaimer, thinking blocks, answer) persist during streaming
    from rag_engine.utils.message_utils import normalize_gradio_history_message

    # Initialize variables that might be needed in error handling
    current_model = settings.default_model
    messages = []
    tool_results = []

    # Start with normalized history
    gradio_history = []
    for msg in history:
        normalized_msg = normalize_gradio_history_message(msg)
        gradio_history.append(normalized_msg)

    # Add user message to history
    gradio_history.append({"role": "user", "content": message})

    # Stream AI-generated content disclaimer as a persistent assistant message
    # so it stays above tool-call progress/thinking chunks in the Chatbot UI.
    from rag_engine.llm.prompts import AI_DISCLAIMER

    # Add disclaimer to history and yield full history
    gradio_history.append({
        "role": "assistant",
        "content": AI_DISCLAIMER,
    })
    yield gradio_history.copy()

    # Show "search started" immediately with user's message (before LLM tool call)
    # This provides instant feedback. We'll update it with LLM-generated query when available.
    from rag_engine.api.stream_helpers import yield_search_started

    # Use user's message as initial query (will be updated when tool call is detected)
    initial_query = message.strip() if message else ""
    search_started_msg = yield_search_started(initial_query)
    gradio_history.append(search_started_msg)
    yield gradio_history.copy()

    # Session management (reuse existing pattern)
    base_session_id = getattr(request, "session_hash", None) if request is not None else None
    session_id = salt_session_id(base_session_id, history, message)

    # Wrap user message in template only for the first question in the conversation
    from rag_engine.llm.prompts import (
        USER_QUESTION_TEMPLATE_FIRST,
        USER_QUESTION_TEMPLATE_SUBSEQUENT,
    )

    # Apply template only if this is the first message (empty history)
    is_first_message = not history
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

    # Note: pre-agent trimming removed by request; rely on existing middleware

    # Check if we need model fallback BEFORE creating agent
    # This matches old handler's upfront fallback check
    selected_model = None
    if settings.llm_fallback_enabled:
        selected_model = _check_context_fallback(messages)

    # Create agent (with fallback model if needed) and stream execution
    agent = _create_rag_agent(override_model=selected_model)
    tool_results = []
    answer = ""
    current_model = selected_model or settings.default_model
    has_seen_tool_results = False
    disclaimer_prepended = False  # Track if disclaimer has been prepended to stream

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
        logger.info("Starting agent.stream() with %d messages", len(messages))
        stream_chunk_count = 0
        tool_calls_detected_in_stream = False

        try:
            for stream_mode, chunk in agent.stream(
                {"messages": messages},
                context=agent_context,
                stream_mode=["updates", "messages"]
            ):
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
                                yield gradio_history.copy()

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
                        # Add search completed to history and yield full history
                        gradio_history.append(search_completed_msg)
                        yield gradio_history.copy()

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
                                yield gradio_history.copy()

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

                            # Update existing "search started" message with LLM-generated query
                            # Use accumulator to handle streaming chunks properly
                            from rag_engine.api.stream_helpers import update_search_started_in_history

                            tool_query = tool_call_accumulator.process_token(token)
                            # If we got a query from tool call, update the existing search_started message
                            if tool_query:
                                if update_search_started_in_history(gradio_history, tool_query):
                                    yield gradio_history.copy()
                            # If no query extracted yet, keep the existing message with user's query
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

                                    # Update existing "search started" message with LLM-generated query
                                    # Use accumulator to handle streaming chunks properly
                                    from rag_engine.api.stream_helpers import update_search_started_in_history

                                    tool_query = tool_call_accumulator.process_token(token)
                                    if tool_query:
                                        if update_search_started_in_history(gradio_history, tool_query):
                                            yield gradio_history.copy()
                                    # If no query extracted yet, keep the existing message with user's query
                                # Never stream tool call chunks as text
                                continue

                            elif block.get("type") == "text" and block.get("text"):
                                # Only stream text if we're not currently executing tools
                                # This prevents streaming the agent's "reasoning" about tool calls
                                if not tool_executing:
                                    text_chunk = block["text"]
                                    answer, disclaimer_prepended = _process_text_chunk_for_streaming(
                                        text_chunk, answer, disclaimer_prepended, has_seen_tool_results
                                    )
                                    # Update last message (answer) in history and yield full history
                                    if gradio_history and gradio_history[-1].get("role") == "assistant" and not gradio_history[-1].get("metadata"):
                                        # Update existing answer message
                                        gradio_history[-1] = {"role": "assistant", "content": answer}
                                    else:
                                        # Create new answer message
                                        gradio_history.append({"role": "assistant", "content": answer})
                                    yield gradio_history.copy()
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
                                answer, disclaimer_prepended = _process_text_chunk_for_streaming(
                                    new_chunk, answer, disclaimer_prepended, has_seen_tool_results
                                )
                                # Update last message (answer) in history and yield full history
                                if gradio_history and gradio_history[-1].get("role") == "assistant" and not gradio_history[-1].get("metadata"):
                                    # Update existing answer message
                                    gradio_history[-1] = {"role": "assistant", "content": answer}
                                else:
                                    # Create new answer message
                                    gradio_history.append({"role": "assistant", "content": answer})
                                yield gradio_history.copy()

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
                    yield gradio_history.copy()
                elif isinstance(chunk, str):
                    # Text chunk - update last message or create new one
                    if gradio_history and gradio_history[-1].get("role") == "assistant" and not gradio_history[-1].get("metadata"):
                        # Update existing answer message
                        gradio_history[-1] = {"role": "assistant", "content": chunk}
                    else:
                        # Create new answer message
                        gradio_history.append({"role": "assistant", "content": chunk})
                    yield gradio_history.copy()

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

        # Update last message (answer) with final formatted text and yield full history
        if gradio_history and gradio_history[-1].get("role") == "assistant" and not gradio_history[-1].get("metadata"):
            # Update existing answer message
            gradio_history[-1] = {"role": "assistant", "content": final_text}
        else:
            # Create new answer message
            gradio_history.append({"role": "assistant", "content": final_text})

        # Save conversation turn (reuse existing pattern)
        if session_id:
            llm_manager.save_assistant_turn(session_id, final_text)

        yield gradio_history.copy()

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
                    yield gradio_history.copy()

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

                    for stream_mode, chunk in agent.stream(
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
                                continue

                            if hasattr(token, "tool_calls") and token.tool_calls:
                                # Do not stream tool call reasoning
                                continue

                            if hasattr(token, "content_blocks") and token.content_blocks:
                                for block in token.content_blocks:
                                    if block.get("type") == "text" and block.get("text"):
                                        text_chunk = block["text"]
                                        answer, disclaimer_prepended = _process_text_chunk_for_streaming(
                                            text_chunk, answer, disclaimer_prepended, has_seen_tool_results
                                        )
                                        # Update last message (answer) in history and yield full history
                                        if gradio_history and gradio_history[-1].get("role") == "assistant" and not gradio_history[-1].get("metadata"):
                                            gradio_history[-1] = {"role": "assistant", "content": answer}
                                        else:
                                            gradio_history.append({"role": "assistant", "content": answer})
                                        yield gradio_history.copy()
                        elif stream_mode == "updates":
                            # No-op for UI
                            pass

                    from rag_engine.tools import accumulate_articles_from_tool_results

                    articles = accumulate_articles_from_tool_results(tool_results)
                    if not articles:
                        final_text = answer
                    else:
                        final_text = format_with_citations(answer, articles)

                    # Update last message with final formatted text
                    if gradio_history and gradio_history[-1].get("role") == "assistant" and not gradio_history[-1].get("metadata"):
                        gradio_history[-1] = {"role": "assistant", "content": final_text}
                    else:
                        gradio_history.append({"role": "assistant", "content": final_text})

                    if session_id:
                        llm_manager.save_assistant_turn(session_id, final_text)
                    yield gradio_history.copy()
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

        if gradio_history and gradio_history[-1].get("role") == "assistant" and not gradio_history[-1].get("metadata"):
            gradio_history[-1] = {"role": "assistant", "content": error_msg}
        else:
            gradio_history.append({"role": "assistant", "content": error_msg})
        yield gradio_history.copy()




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
    chat_description = None
else:
    # For standalone app
    chatbot_height = "85vh"
    chatbot_max_height = "80vh"
    chat_title = "Ассистент базы знаний Comindware Platform"
    chat_description = None  # "RAG-агент базы знаний Comindware Platform"

chatbot_config = gr.Chatbot(
    min_height="30vh",
    height=chatbot_height,
    max_height=chatbot_max_height,
    resizable=True,
    elem_classes=["gradio-chatbot"],
    label="Диалог с агентом",
    buttons=["copy", "copy_all"],
)

# Force agent-based handler; legacy direct handler removed
handler_fn = agent_chat_handler
logger.info("Using agent-based (LangChain) handler for chat interface")

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
def ask_comindware(message: str) -> str:
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
        # Note: agent_chat_handler is the generator function used by ChatInterface
        generator = agent_chat_handler(message=message, history=[], request=None)

        # Consume the entire generator to collect all responses
        # The generator now yields: full message history lists (for ChatInterface)
        # Extract the final answer from the last assistant message in the history
        for chunk in generator:
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
                # Try to close the generator if it's still open
                generator.close()
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

with gr.Blocks() as demo:
    # ChatInterface for UI only
    # Note: ChatInterface automatically exposes its function (agent_chat_handler generator),
    # but we register ask_comindware separately for MCP access via filtered endpoint
    gr.ChatInterface(
        fn=handler_fn,
        title=chat_title,
        description=chat_description,
        save_history=True,
        #fill_width=True,
        chatbot=chatbot_config,
        # Attempt to hide auto-generated API endpoint from API docs and MCP
        # Note: According to docs, this may not be effective for MCP, but worth trying
        api_visibility="private",  # Completely disable the API endpoint
    )
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

    demo.queue().launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        share=settings.gradio_share,
        mcp_server=True,
        footer_links=["api"],
    )


