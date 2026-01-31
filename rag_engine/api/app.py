# ruff: noqa: E402
"""Gradio UI with Chatbot (reference agent pattern) and REST API endpoint."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import gradio as gr
from openai import APIError as OpenAIAPIError

from rag_engine.api.i18n import i18n_resolve
from rag_engine.config.settings import get_allowed_fallback_models, settings  # noqa: F401
from rag_engine.llm.fallback import check_context_fallback
from rag_engine.llm.llm_manager import LLMManager
from rag_engine.llm.schemas import StructuredAgentResult
from rag_engine.retrieval.embedder import create_embedder
from rag_engine.retrieval.retriever import RAGRetriever
from rag_engine.storage.vector_store import ChromaStore
from rag_engine.tools import retrieve_context
from rag_engine.tools.retrieve_context import set_app_retriever
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


def _article_to_dict(article) -> dict:
    meta = getattr(article, "metadata", None) or {}
    title = meta.get("title", getattr(article, "kb_id", ""))
    url = (
        meta.get("article_url")
        or meta.get("url")
        or f"https://kb.comindware.ru/article.php?id={getattr(article, 'kb_id', '')}"
    )
    return {
        "kb_id": getattr(article, "kb_id", ""),
        "title": title,
        "url": url,
        "metadata": dict(meta),
    }


def _badge_html(*, label: str, value: str, color: str) -> str:
    return (
        f'<span style="background:{color};padding:2px 8px;border-radius:4px;">'
        f"{label}: {value}"
        "</span>"
    )


def format_spam_badge(score: float) -> str:
    """Format spam score as colored HTML badge (localized)."""
    label = i18n_resolve("spam_badge_label")
    if score < 0.3:
        color, level = "green", i18n_resolve("spam_level_low")
    elif score < 0.6:
        color, level = "orange", i18n_resolve("spam_level_medium")
    else:
        color, level = "red", i18n_resolve("spam_level_high")
    return _badge_html(label=label, value=f"{score:.2f} {level}", color=color)


def format_confidence_badge(query_traces: list[dict]) -> str:
    """Format overall retrieval confidence as colored HTML badge (localized)."""
    from rag_engine.retrieval.confidence import compute_normalized_confidence_from_traces

    label = i18n_resolve("confidence_badge_label")
    avg = compute_normalized_confidence_from_traces(query_traces)

    if avg is None:
        return _badge_html(label=label, value=i18n_resolve("confidence_level_na"), color="gray")

    if avg > 0.7:
        color, level = "green", i18n_resolve("confidence_level_high")
    elif avg > 0.4:
        color, level = "orange", i18n_resolve("confidence_level_medium")
    else:
        color, level = "red", i18n_resolve("confidence_level_low")

    return _badge_html(label=label, value=level, color=color)


def format_queries_badge(query_traces: list[dict]) -> str:
    """Format queries count as colored HTML badge (localized)."""
    label = i18n_resolve("queries_badge_label")
    count = len(query_traces) if query_traces else 0
    # Use blue color for queries badge to distinguish from confidence/spam
    color = "#87CEEB"  # skyblue - valid CSS color
    return _badge_html(label=label, value=str(count), color=color)


def format_articles_dataframe(articles: list[dict]) -> list[list]:
    """Format final articles list for gr.Dataframe."""
    rows: list[list] = []
    for idx, article in enumerate(articles or [], start=1):
        meta = article.get("metadata", {}) if isinstance(article, dict) else {}
        rows.append(
            [
                idx,
                meta.get(
                    "title",
                    article.get("title", "Untitled") if isinstance(article, dict) else "Untitled",
                ),
                f"{meta.get('rerank_score', 0):.2f}"
                if isinstance(meta.get("rerank_score"), (int, float))
                else "",
                meta.get("article_url") or meta.get("url") or article.get("url", "")
                if isinstance(article, dict)
                else "",
            ]
        )
    return rows


# Initialize singletons (order matters: llm_manager before retriever)
embedder = create_embedder(settings)
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
# Use same retriever in retrieve_context tool so FRIDA/direct embedder is never
# loaded in a worker thread (avoids crash when agent first calls the tool).
set_app_retriever(retriever)


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

            # Initialize runtime.context if it doesn't exist (LangChain should set this, but ensure it's there)
            if state and runtime is not None:
                if not hasattr(runtime, "context") or runtime.context is None:
                    # Try to get context from request.config if available
                    config = getattr(request, "config", None) or {}
                    context_from_config = config.get("context") if config else None
                    if context_from_config is None:
                        # Fallback: create a new AgentContext if none exists
                        from rag_engine.utils.context_tracker import AgentContext

                        context_from_config = AgentContext()

                    # Set runtime.context so tools can access it
                    runtime.context = context_from_config
                    logger.debug("[ToolBudget] Initialized runtime.context (was missing)")

                if hasattr(runtime, "context") and runtime.context:
                    conv_toks, tool_toks = _compute_context_tokens_from_state(
                        state.get("messages", [])
                    )
                    runtime.context.conversation_tokens = conv_toks
                    runtime.context.accumulated_tool_tokens = tool_toks
                    logger.debug(
                        "[ToolBudget] runtime.context updated before tool: conv=%d, tools=%d",
                        conv_toks,
                        tool_toks,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ToolBudget] Failed to update context before tool: %s", exc, exc_info=True
            )

        return handler(request)


def _create_rag_agent(
    override_model: str | None = None,
    *,
    force_tool_choice: bool = False,
    enable_sgr_planning: bool = True,
    sgr_spam_threshold: float = 0.8,
):
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
        enable_sgr_planning=enable_sgr_planning,
        sgr_spam_threshold=sgr_spam_threshold,
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
            "search_bubble",
            "search_started",
            "search_completed",
            "thinking",
            "generating_answer",
            "model_switch",
            "cancelled",
            "user_intent_display",  # User intent message (UI only, not for agent context)
            "disclaimer_display",  # AI disclaimer (UI only, not for agent context)
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
        metadata_keys = (
            list(msg.get("metadata", {}).keys()) if isinstance(msg.get("metadata"), dict) else []
        )
        content_preview = (
            str(msg_content)[:100]
            if isinstance(msg_content, str)
            else f"<{type(msg_content).__name__}>"
        )
        logger.info(
            "  [%d] role=%s, has_metadata=%s, metadata_keys=%s, content_preview=%s",
            idx,
            msg_role,
            has_metadata,
            metadata_keys,
            content_preview,
        )

    for idx, msg in enumerate(gradio_history):
        msg_role = msg.get("role")
        msg_content = msg.get("content", "")
        has_metadata = "metadata" in msg

        # Skip UI-only messages
        if _is_ui_only_message(msg):
            logger.debug(
                "Filtered out UI-only message [%d]: role=%s, has_metadata=%s, content_preview=%s",
                idx,
                msg_role,
                has_metadata,
                str(msg_content)[:100] if isinstance(msg_content, str) else "non-string",
            )
            continue

        # Skip the current user message (we'll add wrapped version below)
        if (
            msg_role == "user"
            and isinstance(msg_content, str)
            and msg_content.strip() == current_message.strip()
        ):
            logger.debug("Skipping current user message [%d] (will add wrapped version)", idx)
            continue

        # Normalize message for LangChain (convert structured content to string)
        normalized_msg = normalize_gradio_history_message(msg)
        # Include actual conversation messages
        messages.append(normalized_msg)
        logger.info(
            "Included message [%d] in agent context: role=%s, content_preview=%s",
            idx,
            msg_role,
            str(msg_content)[:100] if isinstance(msg_content, str) else "non-string",
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
    """Process a text chunk for streaming and accumulate the answer.

    Disclaimer is injected as a separate message before the first chunk (UI only).
    Handles optional newline after tool results.

    Args:
        text_chunk: Raw text chunk from agent
        answer: Accumulated answer so far
        disclaimer_prepended: Whether disclaimer message has already been injected
        has_seen_tool_results: Whether tool results have been seen

    Returns:
        Tuple of (updated_answer, updated_disclaimer_prepended)
    """
    # Newline before first text chunk after tool results (disclaimer is a separate message)
    if has_seen_tool_results and not answer:
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
                        if (
                            isinstance(text, str)
                            and text.strip().startswith("[")
                            and text.strip().endswith("]")
                        ):
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
    if (
        gradio_history
        and gradio_history[-1].get("role") == "assistant"
        and not gradio_history[-1].get("metadata")
    ):
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
            sum(
                1
                for msg in gradio_history
                if isinstance(msg, dict) and msg.get("role") == "assistant"
            ),
        )

    # Don't add user message here - it's already in history from ChatInterface pattern
    # The submit_event chain adds the user message to chatbot before calling agent_chat_handler
    # (pattern from test script: line 128-129)

    # Determine if this is the first message (for template only; disclaimer injected before first answer chunk)
    is_first_message = not history

    # Three UI blocks pattern (from test script):
    # Thinking/search blocks are added dynamically when tools are actually called
    # (not here, to maintain correct order: SGR planning -> tool calls -> answer generation)
    # This prevents showing spinners for tools that aren't actually called

    # Session management (reuse existing pattern)
    base_session_id = getattr(request, "session_hash", None) if request is not None else None
    session_id = salt_session_id(base_session_id, history, message)

    # Wrap user message in template only for the first question in the conversation
    from rag_engine.llm.prompts import (
        USER_QUESTION_TEMPLATE_FIRST,
        USER_QUESTION_TEMPLATE_SUBSEQUENT,
    )

    # is_first_message already determined above (for template)
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
    messages = _build_agent_messages_from_gradio_history(gradio_history, message, wrapped_message)

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

    # --- Forced user request analysis as tool call (per-turn) ---
    # We force an "analyse_user_request" tool call once per user turn, capture the plan, and
    # then inject the resulting tool-call transcript into the agent messages.
    # SGR planning is enabled by default (can be disabled via _create_rag_agent parameter)
    sgr_plan_dict: dict | None = None
    enable_sgr_planning_flag = True  # Always enabled for now (matches _create_rag_agent default)
    if enable_sgr_planning_flag:
        logger.info("SGR planning enabled, executing forced analyse_user_request tool call")
        try:
            from rag_engine.api.stream_helpers import yield_sgr_planning_started

            # Show dedicated SGR bubble (like other tool bubbles)
            gradio_history.append(yield_sgr_planning_started())
            yield list(gradio_history)
            logger.info("SGR planning UI bubble added to history")

            from rag_engine.llm.llm_manager import LLMManager
            from rag_engine.llm.schemas import SGRPlanResult
            from rag_engine.tools import analyse_user_request as analyse_user_request_tool

            sgr_llm = LLMManager(
                provider=settings.default_llm_provider,
                model=selected_model or settings.default_model,
                temperature=settings.llm_temperature,
            )._chat_model()

            sgr_model = sgr_llm.bind_tools(
                [analyse_user_request_tool],
                tool_choice={"type": "function", "function": {"name": "analyse_user_request"}},
            )

            logger.info("Calling SGR planning LLM with %d messages", len(messages))
            sgr_msg = await sgr_model.ainvoke(messages)
            tool_calls = getattr(sgr_msg, "tool_calls", None) or []
            tool_call = tool_calls[0] if isinstance(tool_calls, list) and tool_calls else None
            logger.info("SGR planning LLM returned %d tool calls", len(tool_calls))

            if isinstance(tool_call, dict):
                call_id = tool_call.get("id") or "sgr_plan_call"
                args = tool_call.get("args")
                if isinstance(args, dict):
                    plan = SGRPlanResult.model_validate(args)
                    sgr_plan_dict = plan.model_dump()
                    logger.info(
                        "SGR plan extracted from tool_call.args: spam_score=%.2f, user_intent_len=%d, subqueries_count=%d",
                        sgr_plan_dict.get("spam_score", 0.0),
                        len(sgr_plan_dict.get("user_intent", "")),
                        len(sgr_plan_dict.get("subqueries", [])),
                    )
                else:
                    fn = (
                        tool_call.get("function")
                        if isinstance(tool_call.get("function"), dict)
                        else {}
                    )
                    arg_str = fn.get("arguments") if isinstance(fn, dict) else None
                    if isinstance(arg_str, str) and arg_str.strip():
                        plan = SGRPlanResult.model_validate_json(arg_str)
                        sgr_plan_dict = plan.model_dump()
                        logger.info(
                            "SGR plan extracted from tool_call.function.arguments: spam_score=%.2f, user_intent_len=%d, subqueries_count=%d",
                            sgr_plan_dict.get("spam_score", 0.0),
                            len(sgr_plan_dict.get("user_intent", "")),
                            len(sgr_plan_dict.get("subqueries", [])),
                        )
                    else:
                        logger.warning(
                            "SGR planning tool_call missing arguments: tool_call=%s", tool_call
                        )
            else:
                logger.warning(
                    "SGR planning did not return valid tool_call: tool_calls=%s", tool_calls
                )

            if sgr_plan_dict:
                plan_json = json.dumps(sgr_plan_dict, ensure_ascii=False, separators=(",", ":"))

                messages = list(messages) + [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "analyse_user_request",
                                    "arguments": plan_json,
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": plan_json,
                        "tool_call_id": call_id,
                    },
                ]
                logger.info(
                    "SGR plan injected into agent messages (total messages: %d)", len(messages)
                )
            else:
                logger.warning("SGR plan dict is None or empty, not injecting into messages")

            from rag_engine.api.stream_helpers import (
                get_text,
                update_message_status_in_history,
            )

            update_message_status_in_history(gradio_history, "sgr_planning", "done")

            # Replace SGR planning bubble with user intent message (UI only, not context)
            if sgr_plan_dict and sgr_plan_dict.get("user_intent"):
                user_intent = sgr_plan_dict.get("user_intent", "").strip()
                if user_intent:
                    # Find and replace the sgr_planning bubble with normal assistant message
                    for i, msg in enumerate(gradio_history):
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            metadata = msg.get("metadata")
                            if isinstance(metadata, dict) and metadata.get("ui_type") == "sgr_planning":
                                # Replace with normal assistant message with type metadata for context management
                                prefix = get_text("user_intent_prefix")
                                gradio_history[i] = {
                                    "role": "assistant",
                                    "content": f"**{prefix}**\n\n{user_intent}\n",
                                    "metadata": {
                                        "ui_type": "user_intent_display",  # Type for future context management
                                    },
                                }
                                logger.info(
                                    "Replaced SGR planning bubble with user intent: '%s'...",
                                    user_intent[:100]
                                )
                                break

            yield list(gradio_history)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SGR forced tool call failed; continuing without plan: %s", exc, exc_info=True
            )
            from rag_engine.api.stream_helpers import update_message_status_in_history

            update_message_status_in_history(gradio_history, "sgr_planning", "done")
            yield list(gradio_history)
    else:
        logger.info("SGR planning disabled (enable_sgr_planning=False), skipping")

    # Add thinking spinner to show progress while LLM generates tool calls (same id used for removal in stream)
    from rag_engine.api.stream_helpers import short_uid, yield_thinking_block
    thinking_id = short_uid()  # single id for initial block; stream loop reuses it for remove_message_by_id
    thinking_msg = yield_thinking_block("agent", block_id=thinking_id)
    gradio_history.append(thinking_msg)
    yield list(gradio_history)

    # Create agent (with fallback model if needed) and stream execution
    # Force tool choice only on first message; allow model to choose on subsequent turns
    # TEMPORARILY DISABLED FOR TESTING: force_tool_choice=False
    agent = _create_rag_agent(
        override_model=selected_model, force_tool_choice=False
    )  # was: is_first_message
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
    from rag_engine.api.stream_helpers import (
        ToolCallAccumulator,
        remove_message_by_id,
        short_uid,
        update_search_bubble_by_id,
        yield_search_bubble,
        yield_thinking_block,
    )

    tool_call_accumulator = ToolCallAccumulator()
    # Unified bubble approach: track query -> search_id mapping
    # Key: normalized query string (stripped), Value: search_id generated when bubble was created
    search_id_by_query: dict[str, str] = {}
    # Track which search bubbles have been marked as completed to prevent duplicates
    completed_search_ids: set[str] = set()
    # Reuse thinking_id from initial block above for removal when first search bubble or text arrives
    # Track generating answer block ID for removal when done
    generating_answer_id = short_uid()

    agent_context: AgentContext | None = None
    try:
        # Track tool execution state
        # Only stream text content when NOT executing tools
        tool_executing = False

        # Pass accumulated context to agent via typed context parameter
        # Tools can access this via runtime.context (typed, clean!)
        agent_context = AgentContext(
            conversation_tokens=conversation_tokens,
            accumulated_tool_tokens=0,  # Updated as we go
            fetched_kb_ids=set(),  # Reset for new turn - tracks articles fetched in this turn
        )
        if sgr_plan_dict:
            agent_context.sgr_plan = sgr_plan_dict

        # Workaround for LangChain streaming bug: store context in thread-local storage
        # so tools can access it even when runtime.context is None during astream()
        from rag_engine.utils.context_tracker import set_current_context

        set_current_context(agent_context)

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
                {"messages": messages}, context=agent_context, stream_mode=["updates", "messages"]
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
                    is_ai_message = (
                        token_type == "ai"
                        or "AIMessage" in token_class
                        or "AIMessage" in str(token_type)
                    )

                    # Process all AI tokens through accumulator to accumulate tool_call chunks
                    # This ensures we capture query even if chunks arrive before tool_call is detected
                    if is_ai_message:
                        # Process token through accumulator to extract query (works for both streaming chunks and complete tool_calls)
                        tool_query_from_accumulator = tool_call_accumulator.process_token(token)
                        tool_name_from_accumulator = tool_call_accumulator.get_tool_name(token)
                        # Continuously try to update search_started block if query becomes available
                        # This handles cases where query arrives after the block was created with empty query
                        if (
                            tool_name_from_accumulator == "retrieve_context"
                            and not has_seen_tool_results
                        ):
                            # Always try to update if we have a query, even if block was created with empty query
                            if tool_query_from_accumulator:
                                # Query is available - check if we already have a bubble for this query (avoid duplicates)
                                has_search_bubble = any(
                                    isinstance(msg, dict)
                                    and msg.get("role") == "assistant"
                                    and (msg.get("metadata") or {}).get("ui_type")
                                    == "search_bubble"
                                    and (msg.get("metadata") or {}).get("query") == tool_query_from_accumulator
                                    for msg in gradio_history
                                )
                                if not has_search_bubble:
                                    # Check if any search bubble exists at all
                                    any_search_bubble = any(
                                        isinstance(msg, dict)
                                        and msg.get("role") == "assistant"
                                        and (msg.get("metadata") or {}).get("ui_type")
                                        == "search_bubble"
                                        for msg in gradio_history
                                    )
                                    if not any_search_bubble:
                                        logger.info(
                                            "No existing bubble found, creating new search bubble with query='%s'",
                                            tool_query_from_accumulator[:50],
                                        )
                                        # Generate unique search_id for this query
                                        search_id = short_uid()
                                        search_id_by_query[tool_query_from_accumulator.strip()] = search_id
                                        search_started_msg = yield_search_bubble(
                                            tool_query_from_accumulator, search_id=search_id
                                        )
                                        gradio_history.append(search_started_msg)
                                        logger.info(
                                            "YIELDING search bubble: query='%s', search_id=%s, history_len=%d",
                                            tool_query_from_accumulator[:50],
                                            search_id,
                                            len(gradio_history)
                                        )
                                        yield list(gradio_history)
                                    else:
                                        # Existing bubble for different query - add new bubble
                                        logger.info(
                                            "Adding new search bubble for subsequent tool call: query='%s'",
                                            tool_query_from_accumulator[:50],
                                        )
                                        # Generate unique search_id for this query
                                        search_id = short_uid()
                                        search_id_by_query[tool_query_from_accumulator.strip()] = search_id
                                        search_started_msg = yield_search_bubble(
                                            tool_query_from_accumulator, search_id=search_id
                                        )
                                        gradio_history.append(search_started_msg)
                                        logger.info(
                                            "YIELDING subsequent search bubble: query='%s', search_id=%s, history_len=%d",
                                            tool_query_from_accumulator[:50],
                                            search_id,
                                            len(gradio_history)
                                        )
                                        yield list(gradio_history)
                            elif not tool_query_from_accumulator:
                                # Tool detected but query not ready yet - do NOT create empty block
                                # The block will be created when we receive the tool result with query
                                logger.debug(
                                    "retrieve_context detected but query not ready yet, deferring block creation"
                                )

                    if is_ai_message:
                        has_tool_calls = bool(getattr(token, "tool_calls", None))
                        content = str(getattr(token, "content", ""))
                        response_metadata = getattr(token, "response_metadata", {})
                        finish_reason = (
                            response_metadata.get("finish_reason", "N/A")
                            if isinstance(response_metadata, dict)
                            else "N/A"
                        )

                        # Check content_blocks for tool_call_chunk (critical for vLLM streaming)
                        content_blocks = getattr(token, "content_blocks", None)
                        has_content_blocks = bool(content_blocks)
                        tool_call_chunks_in_blocks = 0
                        if content_blocks:
                            tool_call_chunks_in_blocks = sum(
                                1
                                for block in content_blocks
                                if block.get("type") == "tool_call_chunk"
                            )
                            # Log first few content_blocks in detail for debugging (only when tool calls detected)
                            if tool_call_chunks_in_blocks > 0 or finish_reason == "tool_calls":
                                logger.debug(
                                    "Content blocks detail: %s",
                                    content_blocks[:3]
                                    if len(content_blocks) > 3
                                    else content_blocks,
                                )

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

                        # Remove thinking block when tool result arrives (if still present)
                        remove_message_by_id(gradio_history, thinking_id)

                        # For unified bubble, update it with results
                        bubble_updated = False
                        try:
                            result_data = json.loads(token.content) if isinstance(token.content, str) else {}
                            query_from_result = result_data.get("metadata", {}).get("query", "")
                            articles_list = result_data.get("articles", [])
                            count = result_data.get("metadata", {}).get("articles_count", 0)

                            if query_from_result:
                                search_id = search_id_by_query.get(query_from_result.strip())
                                if search_id:
                                    articles_for_display = [
                                        {"title": a.get("title", "Untitled"), "url": a.get("url", "")}
                                        for a in articles_list
                                    ] if articles_list else None

                                    # Only update bubble if not already completed to prevent duplicates
                                    if search_id not in completed_search_ids:
                                        # Use query from bubble metadata (not from tool result) for consistent matching
                                        # The bubble stores the exact query used for search, which is most reliable
                                        query_from_bubble = None
                                        for msg in gradio_history:
                                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                                metadata = msg.get("metadata") or {}
                                                if metadata.get("ui_type") == "search_bubble":
                                                    query_from_bubble = metadata.get("query")
                                                    if query_from_bubble and query_from_bubble == tool_query_from_accumulator:
                                                        # Found matching bubble by stored query
                                                        search_id = metadata.get("search_id")
                                                        break
                                        if search_id:
                                            update_search_bubble_by_id(
                                                gradio_history,
                                                search_id,
                                                count=count,
                                                articles=articles_for_display
                                            )
                                            completed_search_ids.add(search_id)
                                            bubble_updated = True
                                            yield list(gradio_history)
                                    else:
                                        logger.debug(
                                            "Skipping bubble update for search_id=%s (already completed)",
                                            search_id
                                        )
                                    del search_id_by_query[query_from_result.strip()]
                                    yield list(gradio_history)
                                else:
                                    logger.warning("No search_id found for query: %s", query_from_result[:50])
                        except (json.JSONDecodeError, AttributeError) as e:
                            logger.debug("Failed to parse tool result for bubble update: %s", e)

                        # Update accumulated context for next tool call
                        # Agent tracks context, not the tool!
                        _, accumulated_tool_tokens = estimate_accumulated_tokens([], tool_results)
                        agent_context.accumulated_tool_tokens = accumulated_tool_tokens

                        # Update fetched_kb_ids in context (if context exists)
                        if agent_context:
                            try:
                                from rag_engine.tools.utils import parse_tool_result_to_articles

                                articles_list = parse_tool_result_to_articles(token.content)
                                if articles_list:
                                    for article in articles_list:
                                        if article.kb_id:
                                            agent_context.fetched_kb_ids.add(article.kb_id)

                                    logger.debug(
                                        "Updated fetched_kb_ids: %d total articles fetched in this turn",
                                        len(agent_context.fetched_kb_ids),
                                    )
                            except Exception as exc:
                                logger.warning("Failed to update fetched_kb_ids: %s", exc)

                        logger.debug(
                            "Updated accumulated context: conversation=%d, tools=%d (total: %d)",
                            conversation_tokens,
                            accumulated_tool_tokens,
                            conversation_tokens + accumulated_tool_tokens,
                        )

                        # Parse result to check if it's from retrieve_context tool (has articles)
                        from rag_engine.api.stream_helpers import (
                            extract_article_count_from_tool_result,
                            update_message_status_in_history,
                            yield_search_completed,
                        )
                        from rag_engine.tools.utils import parse_tool_result_to_articles

                        # Check if this is a retrieve_context result (has "articles" key)
                        is_search_result = False
                        try:
                            result_json = (
                                json.loads(token.content) if isinstance(token.content, str) else {}
                            )
                            is_search_result = "articles" in result_json
                        except (json.JSONDecodeError, TypeError):
                            pass

                        if is_search_result and not bubble_updated:
                            # This is a retrieve_context result - show search completed with sources
                            # Only emit if bubble wasn't already updated (to avoid duplicates)
                            articles_list = parse_tool_result_to_articles(token.content)
                            articles_count = (
                                len(articles_list)
                                if articles_list
                                else extract_article_count_from_tool_result(token.content)
                            )

                            # Format articles for display (title and URL)
                            articles_for_display = []
                            if articles_list:
                                for article in articles_list:
                                    article_meta = (
                                        article.metadata if hasattr(article, "metadata") else {}
                                    )
                                    title = article_meta.get("title", "Untitled")
                                    url = article_meta.get("url", "")
                                    articles_for_display.append({"title": title, "url": url})

                            search_completed_msg = yield_search_completed(
                                count=articles_count,
                                articles=articles_for_display if articles_for_display else None,
                            )

                            # Update previous pending messages to done (stop spinners)
                            update_message_status_in_history(
                                gradio_history, "search_started", "done"
                            )

                            # Add search completed to history and yield full history
                            gradio_history.append(search_completed_msg)
                            yield list(gradio_history)
                        else:
                            # Non-search tool result (e.g., get_current_datetime): remove initial block, collapse this tool's thinking block
                            logger.debug(
                                "Non-search tool result received, removing initial thinking block and collapsing tool thinking block"
                            )
                            remove_message_by_id(gradio_history, thinking_id)
                            update_message_status_in_history(
                                gradio_history, "thinking", "done"
                            )

                        # Stop SGR planning spinner after any tool result
                        update_message_status_in_history(gradio_history, "sgr_planning", "done")

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
                    token_finish_reason = (
                        token_response_metadata.get("finish_reason")
                        if isinstance(token_response_metadata, dict)
                        else None
                    )
                    finish_reason_is_tool_calls = token_finish_reason == "tool_calls"

                    # Tool call detected if any of these conditions are true
                    tool_call_detected = (
                        has_tool_calls_attr or has_tool_call_chunks or finish_reason_is_tool_calls
                    )

                    if tool_call_detected:
                        tool_calls_detected_in_stream = True
                        if not tool_executing:
                            tool_executing = True
                            # Remove generating_answer block when a new tool call is detected
                            # So the next generating_answer block can appear after this tool completes
                            if has_seen_tool_results:
                                remove_message_by_id(gradio_history, generating_answer_id)
                            # Log which method detected the tool call
                            if has_tool_calls_attr:
                                call_count = (
                                    len(token.tool_calls)
                                    if isinstance(token.tool_calls, list)
                                    else "?"
                                )
                                logger.info(
                                    "Agent calling tool(s) via token.tool_calls: %s call(s)",
                                    call_count,
                                )
                            elif has_tool_call_chunks:
                                logger.info(
                                    "Agent calling tool(s) via content_blocks tool_call_chunk"
                                )
                            elif finish_reason_is_tool_calls:
                                logger.info(
                                    "Agent calling tool(s) detected via finish_reason=tool_calls"
                                )
                                # Check if tool_calls are now available in the token
                                final_tool_calls = getattr(token, "tool_calls", None)
                                if final_tool_calls:
                                    logger.info(
                                        "Final tool_calls after finish_reason: %s call(s)",
                                        len(final_tool_calls)
                                        if isinstance(final_tool_calls, list)
                                        else "?",
                                    )

                            # Get tool name to determine which thinking block to show
                            tool_name = tool_call_accumulator.get_tool_name(token)

                            logger.debug("Tool name detected: %s", tool_name)

                            if tool_name == "retrieve_context":
                                tool_query = tool_call_accumulator.process_token(token)

                                # Fallback: directly extract query from token.tool_calls if accumulator failed
                                if not tool_query:
                                    tool_calls = getattr(token, "tool_calls", None)
                                    if tool_calls:
                                        for tc in tool_calls:
                                            if isinstance(tc, dict):
                                                tc_name = tc.get("name", "")
                                            else:
                                                tc_name = getattr(tc, "name", "")
                                            if tc_name == "retrieve_context":
                                                tc_args = tc.get("args", {}) or tc.get("arguments", "")
                                                if isinstance(tc_args, dict):
                                                    tool_query = tc_args.get("query", "")
                                                elif isinstance(tc_args, str):
                                                    try:
                                                        parsed = json.loads(tc_args)
                                                        tool_query = parsed.get("query", "")
                                                    except (json.JSONDecodeError, ValueError):
                                                        tool_query = ""
                                                else:
                                                    tool_query = ""

                                                if tool_query:
                                                    logger.debug(
                                                        "Fallback: extracted query: %s",
                                                        tool_query[:50]
                                                    )
                                                    break

                                # Create or update search bubble based on query availability
                                if tool_query:
                                    # Query is complete - check if this is a subsequent search or first one
                                    if has_seen_tool_results:
                                        # Add NEW search bubble for subsequent tool calls
                                        logger.info(
                                            "Adding NEW search bubble via tool_calls (subsequent): query=%s",
                                            tool_query[:50],
                                        )
                                        # Generate unique search_id for this query
                                        search_id = short_uid()
                                        search_id_by_query[tool_query.strip()] = search_id
                                        search_started_msg = yield_search_bubble(tool_query, search_id=search_id)
                                        gradio_history.append(search_started_msg)
                                        # Remove thinking spinner completely now that search bubble is shown
                                        remove_message_by_id(gradio_history, thinking_id)
                                        yield list(gradio_history)
                                    else:
                                        # First search - check if bubble already exists for this query
                                        has_search_bubble = any(
                                            isinstance(msg, dict)
                                            and msg.get("role") == "assistant"
                                            and (msg.get("metadata") or {}).get("ui_type")
                                            == "search_bubble"
                                            and (msg.get("metadata") or {}).get("query") == tool_query
                                            for msg in gradio_history
                                        )
                                        if not has_search_bubble:
                                            logger.info(
                                                "Adding search bubble via tool_calls (new): query=%s",
                                                tool_query[:50],
                                            )
                                            # Generate unique search_id for this query
                                            search_id = short_uid()
                                            search_id_by_query[tool_query.strip()] = search_id
                                            search_started_msg = yield_search_bubble(
                                                tool_query, search_id=search_id
                                            )
                                            gradio_history.append(search_started_msg)
                                            # Remove thinking spinner completely now that search bubble is shown
                                            remove_message_by_id(gradio_history, thinking_id)
                                            yield list(gradio_history)
                                else:
                                    # Query not ready yet - do NOT create empty block
                                    # The block will be created when we receive the tool result with query
                                    logger.debug(
                                        "retrieve_context detected via tool_calls but query not ready, deferring"
                                    )
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
                                    # Remove generating_answer block when a new tool call is detected
                                    if has_seen_tool_results:
                                        remove_message_by_id(
                                            gradio_history, generating_answer_id
                                        )
                                    logger.debug("Agent calling tool via chunk")

                                    # Get tool name to determine which thinking block to show
                                    tool_name = tool_call_accumulator.get_tool_name(token)

                                    if tool_name == "retrieve_context":
                                        tool_query = tool_call_accumulator.process_token(token)
                                        # Create or update search bubble based on query availability
                                        if tool_query:
                                            # Query is complete - check if this is a subsequent search or first one
                                            if has_seen_tool_results:
                                                # Add NEW search bubble for subsequent tool calls
                                                logger.info(
                                                    "Adding NEW search bubble via chunk (subsequent): query=%s",
                                                    tool_query[:50],
                                                )
                                                # Generate unique search_id for this query
                                                search_id = short_uid()
                                                search_id_by_query[tool_query.strip()] = search_id
                                                search_started_msg = yield_search_bubble(
                                                    tool_query, search_id=search_id
                                                )
                                                gradio_history.append(search_started_msg)
                                                yield list(gradio_history)
                                            else:
                                                # First search - check if bubble already exists for this query
                                                has_search_bubble = any(
                                                    isinstance(msg, dict)
                                                    and msg.get("role") == "assistant"
                                                    and (msg.get("metadata") or {}).get("ui_type")
                                                    == "search_bubble"
                                                    and (msg.get("metadata") or {}).get("query") == tool_query
                                                    for msg in gradio_history
                                                )
                                                if not has_search_bubble:
                                                    logger.info(
                                                        "Adding search bubble via chunk (new): query=%s",
                                                        tool_query[:50],
                                                    )
                                                    # Generate unique search_id for this query
                                                    search_id = short_uid()
                                                    search_id_by_query[tool_query.strip()] = search_id
                                                    search_started_msg = yield_search_bubble(
                                                        tool_query, search_id=search_id
                                                    )
                                                    gradio_history.append(search_started_msg)
                                                    yield list(gradio_history)
                                # Skip displaying the tool call itself and any content
                                continue

                            elif block.get("type") == "text" and block.get("text"):
                                # Only stream text if we're not currently executing tools
                                # This prevents streaming the agent's "reasoning" about tool calls
                                if not tool_executing:
                                    text_chunk = block["text"]

                                    # On first text chunk: inject disclaimer as separate message (UI only), then stream answer
                                    if not answer:
                                        if not disclaimer_prepended:
                                            from rag_engine.api.stream_helpers import (
                                                yield_disclaimer_display,
                                            )

                                            gradio_history.append(
                                                yield_disclaimer_display()
                                            )
                                            disclaimer_prepended = True
                                            yield list(gradio_history)
                                        remove_message_by_id(gradio_history, thinking_id)
                                        update_message_status_in_history(
                                            gradio_history, "search_started", "done"
                                        )
                                        update_message_status_in_history(
                                            gradio_history, "sgr_planning", "done"
                                        )
                                        # Show "Generating Answer" spinner while answer is being generated
                                        if has_seen_tool_results:
                                            from rag_engine.api.stream_helpers import (
                                                yield_generating_answer,
                                            )

                                            generating_msg = yield_generating_answer(block_id=generating_answer_id)
                                            gradio_history.append(generating_msg)
                                            yield list(gradio_history)

                                    answer, disclaimer_prepended = (
                                        _process_text_chunk_for_streaming(
                                            text_chunk,
                                            answer,
                                            disclaimer_prepended,
                                            has_seen_tool_results,
                                        )
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
                                new_chunk = token_content[len(answer) :]
                            elif token_content != answer:
                                # Incremental chunk - use as-is (typical case)
                                new_chunk = token_content

                            if new_chunk:
                                # On first text chunk: inject disclaimer as separate message (UI only)
                                if not answer and not disclaimer_prepended:
                                    from rag_engine.api.stream_helpers import (
                                        yield_disclaimer_display,
                                    )

                                    gradio_history.append(
                                        yield_disclaimer_display()
                                    )
                                    disclaimer_prepended = True
                                    yield list(gradio_history)
                                # Show "Generating Answer" spinner on first chunk after tool results
                                if not answer and has_seen_tool_results:
                                    from rag_engine.api.stream_helpers import (
                                        yield_generating_answer,
                                    )

                                    generating_msg = yield_generating_answer(block_id=generating_answer_id)
                                    gradio_history.append(generating_msg)
                                    yield list(gradio_history)

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
                    logger.debug(
                        "Agent update: %s", list(chunk.keys()) if isinstance(chunk, dict) else chunk
                    )

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
                    logger.error(
                        "OpenAI API error during streaming (after tool execution): %s",
                        api_error,
                        exc_info=True,
                    )
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
                    from rag_engine.api.stream_helpers import (
                        update_message_status_in_history,
                        yield_generating_answer,
                    )

                    # Stop SGR planning spinner when starting to generate answer
                    update_message_status_in_history(gradio_history, "sgr_planning", "done")
                    generating_msg = yield_generating_answer(block_id=generating_answer_id)
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
                        # Text chunk - remove generating_answer block on first chunk
                        if not answer and has_seen_tool_results:
                            remove_message_by_id(gradio_history, generating_answer_id)
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
                logger.info(
                    f"Saved INCOMPLETE response to memory ({len(incomplete_response)} chars)"
                )
            yield list(gradio_history)
            try:
                agent_context.final_answer = incomplete_response or ""
                agent_context.diagnostics = {
                    "model": current_model,
                    "cancelled": True,
                    "tool_results_count": len(tool_results),
                }
            except Exception:
                pass
            yield agent_context
            return

        # Accumulate articles from tool results and add citations
        logger.info(
            "Stream completed: %d chunks processed, %d tool results",
            stream_chunk_count,
            len(tool_results),
        )
        from rag_engine.tools import accumulate_articles_from_tool_results

        articles = accumulate_articles_from_tool_results(tool_results)

        # Inject disclaimer as separate message if it wasn't added during streaming
        if not disclaimer_prepended and answer:
            from rag_engine.api.stream_helpers import yield_disclaimer_display

            gradio_history.append(yield_disclaimer_display())
            disclaimer_prepended = True
            logger.info("Injected disclaimer message (was missing from stream)")

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

        # Remove transient blocks and mark pending spinners done before final yield
        from rag_engine.api.stream_helpers import update_message_status_in_history

        remove_message_by_id(gradio_history, thinking_id)
        remove_message_by_id(gradio_history, generating_answer_id)
        update_message_status_in_history(gradio_history, "search_started", "done")
        update_message_status_in_history(gradio_history, "sgr_planning", "done")
        logger.info("Marked all pending UI spinners as done before final yield")

        # Log final history state for debugging
        logger.info(
            "Final history yield: total_messages=%d (user=%d, assistant=%d, ui_metadata=%d)",
            len(gradio_history),
            sum(1 for msg in gradio_history if isinstance(msg, dict) and msg.get("role") == "user"),
            sum(
                1
                for msg in gradio_history
                if isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and not msg.get("metadata")
            ),
            sum(1 for msg in gradio_history if isinstance(msg, dict) and msg.get("metadata")),
        )

        yield list(gradio_history)

        # Populate AgentContext for structured output / UI metadata, then yield it.
        # Ensure sgr_plan is set if it was created during SGR planning
        if sgr_plan_dict and not getattr(agent_context, "sgr_plan", None):
            agent_context.sgr_plan = sgr_plan_dict
            logger.info(
                "agent_chat_handler: restored sgr_plan_dict to agent_context before yielding"
            )
        try:
            agent_context.final_answer = final_text or ""
            agent_context.final_articles = (
                [_article_to_dict(a) for a in articles] if articles else []
            )
            # Log rerank_score presence for debugging
            if articles:
                scores = [(a.kb_id, (a.metadata or {}).get("rerank_score")) for a in articles[:5]]
                logger.debug(
                    f"agent_chat_handler: sample rerank_scores from final_articles: {scores}"
                )
            agent_context.diagnostics = {
                "model": current_model,
                "stream_chunks": stream_chunk_count,
                "tool_results_count": len(tool_results),
                "conversation_tokens": agent_context.conversation_tokens,
                "accumulated_tool_tokens": agent_context.accumulated_tool_tokens,
            }
            logger.info(
                f"agent_chat_handler: yielding AgentContext - "
                f"sgr_plan_present={agent_context.sgr_plan is not None}, "
                f"sgr_plan_dict_present={sgr_plan_dict is not None}, "
                f"query_traces_count={len(agent_context.query_traces)}, "
                f"final_articles_count={len(agent_context.final_articles)}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to populate AgentContext final fields: %s", exc)

        yield agent_context
        return

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
                                    from rag_engine.api.stream_helpers import (
                                        update_message_status_in_history,
                                        yield_generating_answer,
                                    )

                                    # Stop SGR planning spinner when starting to generate answer
                                    update_message_status_in_history(
                                        gradio_history, "sgr_planning", "done"
                                    )
                                    generating_msg = yield_generating_answer(block_id=generating_answer_id)
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

                                        # Remove generating_answer block on first text chunk
                                        if not answer and has_seen_tool_results:
                                            remove_message_by_id(
                                                gradio_history, generating_answer_id
                                            )

                                        answer, disclaimer_prepended = (
                                            _process_text_chunk_for_streaming(
                                                text_chunk,
                                                answer,
                                                disclaimer_prepended,
                                                has_seen_tool_results,
                                            )
                                        )
                                        # Update last message (answer) in history and yield full history
                                        _update_or_append_assistant_message(gradio_history, answer)
                                        # Track incomplete response for memory saving if cancelled (pattern from test script)
                                        incomplete_response = answer
                                        final_response = (
                                            answer  # Update final response as we stream
                                        )
                                        yield list(gradio_history)
                        elif stream_mode == "updates":
                            # No-op for UI
                            pass

                    # Inject disclaimer as separate message if it wasn't added during streaming
                    if not disclaimer_prepended and answer:
                        from rag_engine.api.stream_helpers import yield_disclaimer_display

                        gradio_history.append(yield_disclaimer_display())
                        disclaimer_prepended = True
                        logger.info(
                            "Injected disclaimer message (fallback retry, was missing from stream)"
                        )

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
                        logger.warning(
                            "Fallback agent completed with empty answer - no text content was produced"
                        )

                    # Update final response tracking
                    final_response = final_text

                    # Save conversation turn
                    if final_response and session_id:
                        llm_manager.save_assistant_turn(session_id, final_response)
                        logger.info(
                            f"Saved complete response to memory ({len(final_response)} chars)"
                        )
                    yield list(gradio_history)
                    # Yield AgentContext so metadata UI can update after fallback retry.
                    try:
                        agent_context.final_answer = final_text or ""
                        agent_context.final_articles = (
                            [_article_to_dict(a) for a in articles] if articles else []
                        )
                        agent_context.diagnostics = {
                            "model": current_model,
                            "fallback_retry": True,
                            "tool_results_count": len(tool_results),
                        }
                        if sgr_plan_dict and not getattr(agent_context, "sgr_plan", None):
                            agent_context.sgr_plan = sgr_plan_dict
                    except Exception:
                        pass
                    yield agent_context
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

        # Remove transient blocks and mark pending spinners done before yielding error
        from rag_engine.api.stream_helpers import update_message_status_in_history

        remove_message_by_id(gradio_history, thinking_id)
        remove_message_by_id(gradio_history, generating_answer_id)
        update_message_status_in_history(gradio_history, "search_started", "done")
        update_message_status_in_history(gradio_history, "sgr_planning", "done")

        yield list(gradio_history)

        # Always yield a final AgentContext so `chat_with_metadata` can update UI panels.
        if agent_context is None:
            agent_context = AgentContext(conversation_tokens=0, accumulated_tool_tokens=0)
        if sgr_plan_dict and not getattr(agent_context, "sgr_plan", None):
            agent_context.sgr_plan = sgr_plan_dict
        try:
            agent_context.final_answer = error_msg
            agent_context.final_articles = []
            agent_context.diagnostics = {"model": current_model, "error": str(e)}
        except Exception:
            pass
        yield agent_context


def query_rag(question: str, provider: str = "gemini", top_k: int = 5) -> str:
    if not question or not question.strip():
        return "Error: Empty question"
    docs = retriever.retrieve(question, top_k=top_k)
    # If no documents found, inject a message into the context
    has_no_results_doc = False
    if not docs:
        from rag_engine.retrieval.retriever import Article

        no_results_msg = (
            "К сожалению, не найдено релевантных материалов / No relevant results found."
        )
        no_results_doc = Article(
            kb_id="",
            content=no_results_msg,
            metadata={"title": "No Results", "kbId": "", "_is_no_results": True},
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

# Force agent-based handler; legacy direct handler removed.
# handler_fn is assigned after chat_with_metadata is defined (below).

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
def get_knowledge_base_articles(
    query: str, top_k: int | str | None = None, exclude_kb_ids: list[str] | None = None
) -> str:
    """Search and retrieve documentation articles from the Comindware Platform knowledge base.

    Use this tool when you need raw search results with article metadata. For intelligent
    answers with automatic context retrieval, use agent_chat_handler instead.

    Args:
        query: Search query or question to find relevant documentation articles.
               Examples: "authentication", "API integration", "user management"
        top_k: Optional limit on number of articles to return. If not specified,
               returns the default number of most relevant articles (typically 10-20).
               Can be provided as int or string (will be converted).
        exclude_kb_ids: Optional list of article kb_ids to exclude from results (for deduplication).
                       Use this to prevent retrieving articles you've already fetched in previous calls.
                       Example: exclude_kb_ids=['12345', '67890'].

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
    return retrieve_context.func(query=query, top_k=converted_top_k, exclude_kb_ids=exclude_kb_ids)


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
    response_parts: list[str] = []
    last_text_response: str | None = None
    final_context: AgentContext | None = None
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

            # New: final AgentContext yield (preferred for exact final answer)
            if isinstance(chunk, AgentContext):
                final_context = chunk
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

        # Prefer context-captured final answer if available
        if (
            final_context
            and isinstance(final_context.final_answer, str)
            and final_context.final_answer.strip()
        ):
            return final_context.final_answer

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
            return (
                last_text_response
                + f"\n\n[Note: An error occurred during processing: {type(e).__name__}]"
            )
        elif response_parts:
            return "\n".join(response_parts) + f"\n\n[Note: An error occurred: {str(e)}]"
        return f"Error: {str(e)}. Please try rephrasing your question or contact support."


async def ask_comindware_structured(
    message: str,
    *,
    include_per_query_trace: bool = True,
) -> StructuredAgentResult:
    """Non-streaming structured callable for batch processing / datasets.

    Runs the same streaming handler, but captures the final AgentContext object
    yielded at the end of the generator.
    """
    context: AgentContext | None = None
    generator = agent_chat_handler(message=message, history=[], request=None)
    async for chunk in generator:
        if isinstance(chunk, AgentContext):
            context = chunk

    if context is None:
        # Defensive fallback: keep shape stable
        from rag_engine.llm.schemas import SGRPlanResult

        empty_plan = SGRPlanResult(
            spam_score=0.0,
            spam_reason="",
            user_intent="",
            subqueries=[""],
        )
        return StructuredAgentResult(plan=empty_plan, answer_text="")

    from pydantic import ValidationError

    from rag_engine.llm.schemas import SGRPlanResult

    plan_dict = context.sgr_plan or {}
    try:
        plan = SGRPlanResult.model_validate(plan_dict)
    except ValidationError:
        plan = SGRPlanResult(
            spam_score=0.0,
            spam_reason="",
            user_intent="",
            subqueries=[""],
        )

    return StructuredAgentResult(
        plan=plan,
        per_query_results=context.query_traces if include_per_query_trace else [],
        final_articles=context.final_articles,
        answer_text=context.final_answer,
        diagnostics=context.diagnostics,
    )


async def chat_with_metadata(
    message: str,
    history: list[dict],
    cancel_state: dict | None = None,
    request: gr.Request | None = None,
) -> AsyncGenerator[
    tuple[
        list[dict],
        str | gr.HTML,
        str | gr.HTML,
        str | gr.HTML,
        str | gr.Textbox,
        list | gr.JSON,
        list | gr.JSON,
        list | gr.Dataframe,
        dict | None,  # metadata_state
    ],
    None,
]:
    """Streaming UI handler with metadata enabled and 1s delays around culprits for debugging.

    Adds 1 second delays around formatting operations to identify which causes frontend hang.
    """
    last_history: list[dict] = history if history else []
    ctx: AgentContext | None = None
    metadata_start_time = None
    user_message = message  # Store original message for fallback metadata

    async for chunk in agent_chat_handler(
        message=message,
        history=history,
        cancel_state=cancel_state,
        request=request,
    ):
        if isinstance(chunk, list):
            last_history = chunk
            # Yield history with hidden metadata during streaming
            yield (
                chunk,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=[]),
                gr.update(visible=False, value=[]),
                gr.update(visible=False, value=[]),
                None,  # metadata_state - not updated during streaming
            )
        elif isinstance(chunk, AgentContext):
            ctx = chunk
            metadata_start_time = time.perf_counter()
            logger.info("chat_with_metadata: received AgentContext, starting metadata processing")

    # After streaming completes, populate metadata components with delays
    if ctx is None:
        logger.warning("chat_with_metadata: no AgentContext received, yielding hidden metadata")
        yield (
            last_history,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=[]),
            gr.update(visible=False, value=[]),
            gr.update(visible=False, value=[]),
            None,  # metadata_state - no metadata to store
        )
        return

    try:
        # Extract plan data with timing
        plan_start = time.perf_counter()
        plan = ctx.sgr_plan or {}
        logger.info(
            f"chat_with_metadata: sgr_plan present={ctx.sgr_plan is not None}, "
            f"plan_keys={list(plan.keys()) if plan else []}, "
            f"user_intent_present={'user_intent' in plan}, "
            f"subqueries_present={'subqueries' in plan}, "
            f"action_plan_present={'action_plan' in plan}, "
            f"user_message='{user_message[:50] if user_message else 'None'}...'"
        )
        spam_score = float(plan.get("spam_score", 0.0) or 0.0)
        user_intent = (
            plan.get("user_intent", "") if isinstance(plan.get("user_intent"), str) else ""
        )
        subqueries = plan.get("subqueries", [])
        action_plan = plan.get("action_plan", [])

        # Log if sgr_plan is missing (shouldn't happen if SGR planning executed)
        if not ctx.sgr_plan:
            logger.warning(
                "chat_with_metadata: sgr_plan is None - SGR planning may not have executed. "
                "Will use fallback user_intent from message if available."
            )

        # Fallback: if user_intent is empty but we have a plan, try to derive it from the message
        # This handles cases where SGR planning returned empty values for math/time questions
        if not user_intent and plan and user_message:
            # Use the original user message as a fallback intent
            user_intent = user_message[:200]  # Truncate to reasonable length
            logger.info(
                f"chat_with_metadata: using fallback user_intent from message (len={len(user_intent)})"
            )

        # Ensure subqueries is a list (even if empty)
        if not isinstance(subqueries, list):
            subqueries = []

        # Ensure action_plan is a list (even if empty)
        if not isinstance(action_plan, list):
            action_plan = []
        plan_elapsed = (time.perf_counter() - plan_start) * 1000
        logger.info(
            f"chat_with_metadata: plan extraction took {plan_elapsed:.2f}ms - "
            f"spam_score={spam_score}, user_intent_len={len(user_intent)}, "
            f"subqueries_count={len(subqueries) if isinstance(subqueries, list) else 0}, "
            f"action_plan_count={len(action_plan) if isinstance(action_plan, list) else 0}"
        )

        # Format badges with timing
        spam_start = time.perf_counter()
        try:
            spam_badge_html = format_spam_badge(spam_score)
        except Exception as exc:
            logger.error("Failed to format spam badge: %s", exc, exc_info=True)
            spam_badge_html = ""
        spam_elapsed = (time.perf_counter() - spam_start) * 1000
        logger.info(f"chat_with_metadata: spam badge formatting took {spam_elapsed:.2f}ms")

        conf_start = time.perf_counter()
        query_traces = ctx.query_traces or []
        logger.info(
            f"chat_with_metadata: formatting confidence badge - query_traces_count={len(query_traces)}, "
            f"has_traces={bool(query_traces)}"
        )
        if query_traces:
            for idx, trace in enumerate(query_traces):
                has_conf = trace.get("confidence") is not None if isinstance(trace, dict) else False
                conf_type = (
                    type(trace.get("confidence")).__name__ if isinstance(trace, dict) else "N/A"
                )
                logger.debug(
                    f"chat_with_metadata: trace[{idx}] - has_confidence={has_conf}, "
                    f"confidence_type={conf_type}, articles_count={len(trace.get('articles', [])) if isinstance(trace, dict) else 0}"
                )
        try:
            confidence_badge_html = format_confidence_badge(query_traces)
        except Exception as exc:
            logger.error("Failed to format confidence badge: %s", exc, exc_info=True)
            confidence_badge_html = ""
        conf_elapsed = (time.perf_counter() - conf_start) * 1000
        logger.info(f"chat_with_metadata: confidence badge formatting took {conf_elapsed:.2f}ms")

        queries_start = time.perf_counter()
        try:
            queries_badge_html = format_queries_badge(query_traces)
        except Exception as exc:
            logger.error("Failed to format queries badge: %s", exc, exc_info=True)
            queries_badge_html = ""
        queries_elapsed = (time.perf_counter() - queries_start) * 1000
        logger.info(
            f"chat_with_metadata: queries badge formatting took {queries_elapsed:.2f}ms - "
            f"queries_count={len(query_traces)}"
        )

        final_start = time.perf_counter()
        try:
            # format_articles_dataframe is disabled, returns empty list
            articles_df_data = format_articles_dataframe(ctx.final_articles or [])
        except Exception as exc:
            logger.error("Failed to format articles dataframe: %s", exc, exc_info=True)
            articles_df_data = []
        final_elapsed = (time.perf_counter() - final_start) * 1000
        logger.info(f"chat_with_metadata: final formatting took {final_elapsed:.2f}ms")

        # Yield badges immediately, store metadata in state for later UI update
        logger.info("chat_with_metadata: yielding badges and storing metadata in state")
        yield_start = time.perf_counter()

        # Prepare metadata for state storage
        # Ensure we always show SGR metadata when plan exists
        has_user_intent = bool(user_intent and isinstance(user_intent, str) and user_intent.strip())
        has_subqueries = bool(isinstance(subqueries, list) and len(subqueries) > 0)
        has_action_plan = bool(isinstance(action_plan, list) and len(action_plan) > 0)
        has_articles = bool(isinstance(articles_df_data, list) and len(articles_df_data) > 0)

        # Always show user_intent if we have a user message (even if SGR plan is missing/empty)
        # This ensures metadata appears for math/date questions that don't call retrieve_context
        if not has_user_intent and user_message:
            has_user_intent = True
            user_intent = user_intent or user_message[:200]  # Use fallback if empty
            logger.info(
                f"chat_with_metadata: using fallback user_intent from message - "
                f"sgr_plan_present={ctx.sgr_plan is not None}, user_intent_len={len(user_intent)}"
            )

        # Store metadata in state for later UI update (after input is unlocked)
        metadata_dict = {
            "user_intent": user_intent if has_user_intent else "",
            "has_user_intent": has_user_intent,
            "subqueries": subqueries if isinstance(subqueries, list) else [],
            "has_subqueries": has_subqueries,
            "action_plan": action_plan if isinstance(action_plan, list) else [],
            "has_action_plan": has_action_plan,
            "articles_df_data": articles_df_data,
            "has_articles": has_articles,
        }

        logger.info(
            "chat_with_metadata: storing metadata in state - "
            f"user_intent={user_intent[:50] if user_intent else 'empty'}, "
            f"subqueries_count={len(subqueries) if isinstance(subqueries, list) else 0}, "
            f"action_plan_count={len(action_plan) if isinstance(action_plan, list) else 0}, "
            f"articles_df_rows={len(articles_df_data) if isinstance(articles_df_data, list) else 0}"
        )

        try:
            yield (
                last_history,
                gr.update(visible=True, value=spam_badge_html),
                gr.update(visible=True, value=confidence_badge_html),
                gr.update(visible=True, value=queries_badge_html),
                gr.update(visible=False, value=""),  # intent_text - hide for now, will update later
                gr.update(
                    visible=False, value=[]
                ),  # subqueries_json - hide for now, will update later
                gr.update(
                    visible=False, value=[]
                ),  # action_plan_json - hide for now, will update later
                gr.update(visible=False, value=[]),  # articles_df - hide for now, will update later
                metadata_dict,  # Store metadata in state
            )
            yield_elapsed = (time.perf_counter() - yield_start) * 1000
            logger.info(
                f"chat_with_metadata: badges yield and metadata storage completed, took {yield_elapsed:.2f}ms"
            )
        except Exception as yield_exc:
            logger.error(f"chat_with_metadata: badges yield failed: {yield_exc}", exc_info=True)
            raise

        if metadata_start_time:
            total_elapsed = (time.perf_counter() - metadata_start_time) * 1000
            logger.info(f"chat_with_metadata: total metadata processing took {total_elapsed:.2f}ms")

        logger.info("chat_with_metadata: generator completing normally")

    except Exception as exc:
        logger.error("Error in chat_with_metadata metadata processing: %s", exc, exc_info=True)
        # Yield safe fallback
        yield (
            last_history,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=[]),
            gr.update(visible=False, value=[]),
            gr.update(visible=False, value=[]),
            None,  # metadata_state - no metadata to store on error
        )


# Use metadata-enabled wrapper to populate debug UI panels after streaming.
handler_fn = chat_with_metadata
logger.info("Using agent-based (LangChain) handler for chat interface")

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
    # State to store metadata for UI update after streaming completes
    metadata_state = gr.State(value=None)

    # --- Metadata badges (populated after streaming completes, shown below chat) ---
    with gr.Row():
        spam_badge = gr.HTML(visible=False)
        confidence_badge = gr.HTML(visible=False)
        queries_badge = gr.HTML(visible=False)

    # Metadata panels (populated after streaming completes) - displayed directly, no accordions
    # Markdown headers are always visible (static text)
    gr.Markdown(f"### {i18n_resolve('analysis_summary_title')}")
    intent_text = gr.Textbox(
        label=i18n_resolve("user_intent_label"), interactive=False, visible=False
    )
    subqueries_json = gr.JSON(label=i18n_resolve("subqueries_label"), visible=False)
    action_plan_json = gr.JSON(label=i18n_resolve("action_plan_label"), visible=False)

    gr.Markdown(f"### {i18n_resolve('retrieved_articles_title')}")
    articles_df = gr.Dataframe(
        headers=[
            i18n_resolve("articles_rank_header"),
            i18n_resolve("articles_title_header"),
            i18n_resolve("articles_confidence_header"),
            i18n_resolve("articles_url_header"),
        ],
        interactive=False,
        visible=False,
    )

    def handle_stop_click(
        history: list[dict], cancel_state: dict | None
    ) -> tuple[list[dict], dict]:
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

    def save_session_id(
        message: str, history: list[dict], request: gr.Request | None
    ) -> str | None:
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
    submit_event = (
        user_submit.then(
            fn=reset_cancellation_state,
            inputs=[cancellation_state],
            outputs=[cancellation_state],
            queue=False,
            api_visibility="private",
        )
        .then(
            lambda message, history: history + [{"role": "user", "content": message}],
            inputs=[saved_input, chatbot],
            outputs=[chatbot],
            queue=False,
            api_visibility="private",
        )
        .then(
            fn=save_session_id,
            inputs=[
                saved_input,
                chatbot,
            ],  # Request is automatically passed to functions that accept it
            outputs=[current_session_id],
            queue=False,
            api_visibility="private",
        )
    )

    def re_enable_textbox_and_hide_stop():
        """Re-enable textbox and hide stop button after handler completion."""
        logger.info("Re-enabling textbox and hiding stop button after handler completion")
        return gr.Textbox(value="", interactive=True, submit_btn=True, stop_btn=False)

    def update_metadata_ui(metadata: dict | None) -> tuple:
        """Update metadata UI components from stored metadata state.

        Called after input is unlocked to populate SGR metadata.
        """
        if not metadata:
            logger.info("update_metadata_ui: no metadata to display")
            return (
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=[]),
                gr.update(visible=False, value=[]),
                gr.update(visible=False, value=[]),
            )

        logger.info(
            f"update_metadata_ui: updating UI with metadata - "
            f"has_user_intent={metadata.get('has_user_intent', False)}, "
            f"has_subqueries={metadata.get('has_subqueries', False)}, "
            f"has_action_plan={metadata.get('has_action_plan', False)}, "
            f"has_articles={metadata.get('has_articles', False)}"
        )

        return (
            gr.update(
                visible=metadata.get("has_user_intent", False),
                value=metadata.get("user_intent", ""),
            ),
            gr.update(
                visible=metadata.get("has_subqueries", False), value=metadata.get("subqueries", [])
            ),
            gr.update(
                visible=metadata.get("has_action_plan", False),
                value=metadata.get("action_plan", []),
            ),
            gr.update(
                visible=metadata.get("has_articles", False),
                value=metadata.get("articles_df_data", []),
            ),
        )

    submit_event = (
        submit_event.then(
            fn=handler_fn,
            inputs=[saved_input, chatbot, cancellation_state],  # Pass cancellation state to handler
            outputs=[
                chatbot,
                spam_badge,
                confidence_badge,
                queries_badge,
                intent_text,
                subqueries_json,
                action_plan_json,
                articles_df,
                metadata_state,  # Store metadata for later UI update
            ],
            concurrency_limit=settings.gradio_default_concurrency_limit,
            api_visibility="private",  # Hide agent_chat_handler from MCP tools
        )
        .then(
            # Chain re-enable directly from handler completion
            # .then() fires after generator completes and all yields are processed
            fn=re_enable_textbox_and_hide_stop,
            outputs=[msg],
            api_visibility="private",
        )
        .then(
            # Update metadata UI after input is unlocked
            fn=update_metadata_ui,
            inputs=[metadata_state],
            outputs=[intent_text, subqueries_json, action_plan_json, articles_df],
            api_visibility="private",
        )
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
