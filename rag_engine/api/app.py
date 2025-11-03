"""Gradio UI with ChatInterface and REST API endpoint."""
from __future__ import annotations

import hashlib
import sys
from collections.abc import Generator
from pathlib import Path

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging

import gradio as gr

from rag_engine.config.settings import get_allowed_fallback_models, settings
from rag_engine.llm.llm_manager import LLMManager
from rag_engine.retrieval.embedder import FRIDAEmbedder
from rag_engine.retrieval.retriever import RAGRetriever
from rag_engine.storage.vector_store import ChromaStore
from rag_engine.utils.formatters import format_with_citations
from rag_engine.utils.logging_manager import setup_logging

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


def _estimate_accumulated_context(messages: list[dict], tool_results: list) -> int:
    """Estimate total tokens for messages + tool results (JSON format).

    Args:
        messages: Conversation messages
        tool_results: List of tool result JSON strings

    Returns:
        Estimated token count
    """
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    fast_path_threshold = 50_000
    total_tokens = 0

    # Count message tokens
    for msg in messages:
        if hasattr(msg, "content"):
            content = msg.content
        else:
            content = msg.get("content", "")
        if isinstance(content, str) and content:
            if len(content) > fast_path_threshold:
                total_tokens += len(content) // 4
            else:
                total_tokens += len(encoding.encode(content))

    # Count tool result tokens (JSON format is verbose!)
    for result in tool_results:
        if isinstance(result, str):
            if len(result) > fast_path_threshold:
                total_tokens += len(result) // 4
            else:
                total_tokens += len(encoding.encode(result))

    # Add buffer for system prompt, tool schemas, overhead
    total_tokens += 40000

    return total_tokens


def _find_model_for_tokens(required_tokens: int) -> str | None:
    """Find a model that can handle the required token count.

    Args:
        required_tokens: Minimum token capacity needed

    Returns:
        Model name if found, None otherwise
    """
    from rag_engine.llm.llm_manager import MODEL_CONFIGS

    allowed = get_allowed_fallback_models()
    if not allowed:
        return None

    # Add 10% buffer
    required_tokens = int(required_tokens * 1.1)

    for candidate in allowed:
        if candidate == settings.default_model:
            continue

        candidate_config = MODEL_CONFIGS.get(candidate)
        if not candidate_config:
            # Try partial match
            for key in MODEL_CONFIGS:
                if key != "default" and key in candidate:
                    candidate_config = MODEL_CONFIGS[key]
                    break

        if candidate_config:
            candidate_window = candidate_config.get("token_limit", 0)
            if candidate_window >= required_tokens:
                logger.info(
                    "Found model %s with capacity %d tokens (required: %d)",
                    candidate,
                    candidate_window,
                    required_tokens,
                )
                return candidate

    logger.error("No model found with capacity for %d tokens", required_tokens)
    return None


def _check_context_fallback(messages: list[dict]) -> str | None:
    """Check if context fallback is needed and return fallback model.

    Estimates token usage for the conversation and checks against the current
    model's context window. If approaching limit (90%), selects a larger model
    from allowed fallbacks. Matches old chat_handler's fallback logic.

    Args:
        messages: List of message dicts with 'content' field

    Returns:
        Fallback model name if needed, None otherwise
    """
    import tiktoken

    from rag_engine.llm.llm_manager import MODEL_CONFIGS

    # Get current model config
    model_config = MODEL_CONFIGS.get(settings.default_model)
    if not model_config:
        # Try partial match
        for key in MODEL_CONFIGS:
            if key != "default" and key in settings.default_model:
                model_config = MODEL_CONFIGS[key]
                break
    if not model_config:
        model_config = MODEL_CONFIGS["default"]

    current_window = model_config["token_limit"]

    # Estimate tokens using tiktoken with fast path for large strings
    # Matches token_utils.py pattern: exact for <50K chars, approximation for larger
    encoding = tiktoken.get_encoding("cl100k_base")
    fast_path_threshold = 50_000  # chars
    total_tokens = 0
    for msg in messages:
        # Handle both dict (Gradio) and LangChain message objects
        if hasattr(msg, "content"):
            content = msg.content  # LangChain message object
        else:
            content = msg.get("content", "")  # Dict from Gradio
        if isinstance(content, str) and content:
            # Use fast approximation for very large content
            if len(content) > fast_path_threshold:
                total_tokens += len(content) // 4
            else:
                total_tokens += len(encoding.encode(content))

    # Add buffer for system prompt and output (~35K tokens)
    total_tokens += 35000

    # Check if approaching limit (90% threshold)
    threshold = int(current_window * 0.9)

    if total_tokens > threshold:
        logger.warning(
            "Context size %d tokens exceeds %.1f%% threshold (%d) of %d window for %s",
            total_tokens,
            90.0,
            threshold,
            current_window,
            settings.default_model,
        )

        # Find fallback model with sufficient capacity
        allowed = get_allowed_fallback_models()
        if not allowed:
            logger.warning("No fallback models configured")
            return None

        # Add 10% buffer
        required_tokens = int(total_tokens * 1.1)

        for candidate in allowed:
            if candidate == settings.default_model:
                continue

            candidate_config = MODEL_CONFIGS.get(candidate)
            if not candidate_config:
                # Try partial match
                for key in MODEL_CONFIGS:
                    if key != "default" and key in candidate:
                        candidate_config = MODEL_CONFIGS[key]
                        break
            if not candidate_config:
                continue

            candidate_window = candidate_config.get("token_limit", 0)
            if candidate_window >= required_tokens:
                logger.warning(
                    "Falling back from %s to %s (window: %d → %d tokens)",
                    settings.default_model,
                    candidate,
                    current_window,
                    candidate_window,
                )
                return candidate

        logger.error(
            "No fallback model found with capacity for %d tokens", required_tokens
        )

    return None


def compress_tool_results_if_needed(state: dict, runtime) -> dict | None:
    """Compress tool results before LLM call if approaching context limit.

    This middleware runs right before each LLM invocation. It checks if the
    accumulated tool results + conversation would exceed 85% of the context window.
    If so, it compresses the least relevant articles using the summarization utility.

    This is the lean way to handle dynamic compression without modifying the
    streaming loop or tool logic.

    Args:
        state: Agent state containing messages
        runtime: Runtime object with access to model config

    Returns:
        Updated state dict with compressed messages, or None if no changes needed
    """
    import json

    from rag_engine.llm.llm_manager import MODEL_CONFIGS
    from rag_engine.llm.summarization import summarize_to_tokens
    from rag_engine.llm.token_utils import count_tokens

    messages = state.get("messages", [])
    if not messages:
        return None

    # Get current model's context window
    current_model = getattr(runtime, "model", None) or settings.default_model
    model_config = MODEL_CONFIGS.get(current_model)
    if not model_config:
        for key in MODEL_CONFIGS:
            if key != "default" and key in str(current_model):
                model_config = MODEL_CONFIGS[key]
                break
    if not model_config:
        model_config = MODEL_CONFIGS["default"]

    context_window = model_config.get("token_limit", 262144)
    threshold = int(context_window * 0.85)  # 85% threshold

    # Count total tokens in messages
    total_tokens = 0
    tool_message_indices = []

    for idx, msg in enumerate(messages):
        content = getattr(msg, "content", "") if hasattr(msg, "content") else msg.get("content", "")
        if content and isinstance(content, str):
            total_tokens += count_tokens(content)

            # Track tool messages (type="tool" or ToolMessage class)
            msg_type = getattr(msg, "type", None) if hasattr(msg, "type") else msg.get("type")
            if msg_type == "tool":
                tool_message_indices.append(idx)

    # Check if we need compression
    if total_tokens <= threshold:
        return None  # All good, no changes needed

    if not tool_message_indices:
        return None  # No tool messages to compress

    logger.warning(
        "Context at %d tokens (%.1f%% of %d window), compressing tool results",
        total_tokens,
        100 * total_tokens / context_window,
        context_window
    )

    # Calculate how much to compress (target: get below 80% to leave room for output)
    target_tokens = int(context_window * 0.80)
    tokens_to_save = total_tokens - target_tokens

    # Find the user's question for summarization guidance
    user_question = ""
    for msg in messages:
        msg_type = getattr(msg, "type", None) if hasattr(msg, "type") else msg.get("type")
        if msg_type == "human":
            content = getattr(msg, "content", "") if hasattr(msg, "content") else msg.get("content", "")
            if content:
                user_question = content
                break  # Use the first (most recent) user message

    # Compress tool messages, starting from the last one (least relevant)
    tokens_saved = 0
    updated_messages = list(messages)  # Copy to avoid mutating original

    for idx in reversed(tool_message_indices):
        if tokens_saved >= tokens_to_save:
            break

        msg = updated_messages[idx]
        content = getattr(msg, "content", "") if hasattr(msg, "content") else msg.get("content", "")

        try:
            # Parse tool result JSON
            result = json.loads(content)
            articles = result.get("articles", [])

            if not articles:
                continue

            # Compress articles from the end (least relevant first)
            for i in range(len(articles) - 1, -1, -1):
                if tokens_saved >= tokens_to_save:
                    break

                article = articles[i]
                original_content = article.get("content", "")
                original_tokens = count_tokens(original_content)

                # Target: compress to 30% of original size
                article_target = max(300, int(original_tokens * 0.30))

                # Compress using existing summarization utility
                compressed = summarize_to_tokens(
                    title=article.get("title", "Article"),
                    url=article.get("url", ""),
                    matched_chunks=[original_content],  # Use content as chunk
                    full_body=None,
                    target_tokens=article_target,
                    guidance=user_question,
                    llm=llm_manager,  # Use global llm_manager
                    max_retries=1,  # Quick compression
                )

                # Update article
                compressed_tokens = count_tokens(compressed)
                articles[i]["content"] = compressed
                articles[i]["metadata"]["compressed"] = True

                saved = original_tokens - compressed_tokens
                tokens_saved += saved

                logger.info(
                    "Compressed article '%s': %d → %d tokens (saved %d)",
                    article.get("title", "")[:50],
                    original_tokens,
                    compressed_tokens,
                    saved
                )

            # Update the message content with compressed articles
            result["metadata"]["compressed_articles_count"] = sum(
                1 for a in articles if a.get("metadata", {}).get("compressed")
            )
            result["metadata"]["tokens_saved_by_compression"] = tokens_saved

            # Create new compact JSON
            new_content = json.dumps(result, ensure_ascii=False, separators=(',', ':'))

            # Update message content in place
            if hasattr(msg, "content"):
                msg.content = new_content  # LangChain message object
            else:
                updated_messages[idx] = {**msg, "content": new_content}  # Dict

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Failed to compress tool message at index %d: %s", idx, exc)
            continue

    if tokens_saved > 0:
        logger.info(
            "Compression complete: saved %d tokens, new total ~%d (%.1f%% of window)",
            tokens_saved,
            total_tokens - tokens_saved,
            100 * (total_tokens - tokens_saved) / context_window
        )
        # Return updated messages
        return {"messages": updated_messages}

    return None  # No compression happened


def _compute_context_tokens_from_state(messages: list[dict]) -> tuple[int, int]:
    """Compute (conversation_tokens, accumulated_tool_tokens) from agent state messages.

    - Conversation tokens: count non-tool message contents
    - Accumulated tool tokens: parse tool JSONs, dedupe by kb_id, sum content tokens, add ~30% JSON overhead
    """
    from rag_engine.llm.token_utils import count_tokens
    from rag_engine.tools.utils import parse_tool_result_to_articles

    conversation_tokens = 0
    accumulated_tool_tokens = 0
    seen_kb_ids: set[str] = set()

    for msg in messages:
        content = getattr(msg, "content", "") if hasattr(msg, "content") else msg.get("content", "")
        if not isinstance(content, str) or not content:
            continue
        msg_type = getattr(msg, "type", None) if hasattr(msg, "type") else msg.get("type")
        if msg_type == "tool":
            try:
                articles = parse_tool_result_to_articles(content)
                for art in articles:
                    if art.kb_id and art.kb_id not in seen_kb_ids:
                        accumulated_tool_tokens += count_tokens(art.content)
                        seen_kb_ids.add(art.kb_id)
            except Exception as exc:
                # If parsing fails, log and skip rather than guessing; keep accounting strict
                logger.warning("Failed to parse tool result for token accounting: %s", exc)
                continue
        else:
            conversation_tokens += count_tokens(content)

    # JSON overhead ~30%
    accumulated_tool_tokens = int(accumulated_tool_tokens * 1.3)
    return conversation_tokens, accumulated_tool_tokens


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


from typing import Callable  # noqa: E402
from langchain.agents.middleware import AgentMiddleware, wrap_tool_call as middleware_wrap_tool_call  # noqa: E402
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

    Uses tool_choice parameter to enforce tool calling, the standard
    Comindware Platform system prompt, and SummarizationMiddleware for
    automatic conversation history compression at 85% context threshold.

    Args:
        override_model: Optional model name to use instead of default
                       (for context window fallback)

    Returns:
        Configured LangChain agent with retrieve_context tool and middleware
    """
    import tiktoken
    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware, before_model

    from rag_engine.llm.llm_manager import MODEL_CONFIGS
    from rag_engine.llm.prompts import SUMMARIZATION_PROMPT, SYSTEM_PROMPT
    from rag_engine.tools import retrieve_context
    from rag_engine.utils.context_tracker import AgentContext

    # Use override model if provided (for fallback), otherwise use default
    selected_model = override_model or settings.default_model

    # Select model based on provider
    if settings.default_llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        base_model = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
        )
    else:  # openrouter or other
        from langchain_openai import ChatOpenAI
        base_model = ChatOpenAI(
            model=selected_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
        )

    # Get model configuration for context window
    model_config = MODEL_CONFIGS.get(selected_model)
    if not model_config:
        # Try partial match
        for key in MODEL_CONFIGS:
            if key != "default" and key in selected_model:
                model_config = MODEL_CONFIGS[key]
                break
    if not model_config:
        model_config = MODEL_CONFIGS["default"]

    context_window = model_config["token_limit"]

    # Calculate threshold (70% default, more aggressive for agent with tool calls)
    threshold_tokens = int(context_window * (settings.memory_compression_threshold_pct / 100))

    # Custom token counter using tiktoken with fast path for large strings
    # Matches token_utils.py pattern: exact for <50K chars, approximation for larger
    encoding = tiktoken.get_encoding("cl100k_base")
    fast_path_threshold = 50_000  # chars

    def tiktoken_counter(messages: list) -> int:
        """Count tokens using tiktoken for accuracy with fast path for large content.

        Handles both dict messages and LangChain message objects.
        Uses exact tiktoken encoding for normal content, but switches to
        fast approximation (chars // 4) for very large strings (>50K chars)
        to avoid performance issues, matching the pattern in token_utils.py.
        """
        total = 0
        for msg in messages:
            # Handle both dict (Gradio) and LangChain message objects
            if hasattr(msg, "content"):
                content = msg.content  # LangChain message object
            else:
                content = msg.get("content", "")  # Dict from Gradio
            if isinstance(content, str) and content:
                # Use fast approximation for very large content
                if len(content) > fast_path_threshold:
                    total += len(content) // 4
                else:
                    total += len(encoding.encode(content))
        return total

    # CRITICAL: Use tool_choice to force retrieval tool execution
    # This ensures the agent always searches the knowledge base
    model_with_tools = base_model.bind_tools(
        [retrieve_context],
        tool_choice="retrieve_context"
    )

    # Get messages_to_keep from settings (default 2, matching old handler)
    messages_to_keep = getattr(settings, "memory_compression_messages_to_keep", 2)

    agent = create_agent(
        model=model_with_tools,
        tools=[retrieve_context],
        system_prompt=SYSTEM_PROMPT,
        context_schema=AgentContext,  # Typed context for tools
        middleware=[
            ToolBudgetMiddleware(),  # Ensure tokens are fresh before tool execution
            before_model(update_context_budget),  # Keep runtime.context tokens fresh
            before_model(compress_tool_results_if_needed),  # Dynamic tool result compression
            SummarizationMiddleware(
                model=base_model,  # Use same model for summarization
                token_counter=tiktoken_counter,  # Use our tiktoken-based counter
                max_tokens_before_summary=threshold_tokens,  # Configurable threshold (default 70%)
                messages_to_keep=messages_to_keep,  # Configurable, default 2 (matches old handler)
                summary_prompt=SUMMARIZATION_PROMPT,  # Use existing prompt from prompts.py
                summary_prefix="## Предыдущее обсуждение / Previous conversation:",
            ),
        ],
    )

    if override_model:
        logger.info(
            "RAG agent created with FALLBACK MODEL %s: forced tool execution, "
            "memory compression (threshold: %d tokens at %d%%, keep: %d msgs, window: %d)",
            selected_model,
            threshold_tokens,
            settings.memory_compression_threshold_pct,
            messages_to_keep,
            context_window,
        )
    else:
        logger.info(
            "RAG agent created with forced tool execution and memory compression "
            "(threshold: %d tokens at %d%%, keep: %d msgs, window: %d)",
            threshold_tokens,
            settings.memory_compression_threshold_pct,
            messages_to_keep,
            context_window,
        )
    return agent


def agent_chat_handler(
    message: str,
    history: list[dict],
    request: gr.Request | None = None,
) -> Generator[str, None, None]:
    """Agent-based chat handler using LangChain agent with tool calling.

    This handler uses a LangChain agent that decides when to call the
    retrieve_context tool. The agent is prompted to always search the
    knowledge base before answering.

    Args:
        message: User's current message
        history: Chat history from Gradio
        request: Gradio request object for session management

    Yields:
        Streaming response with citations
    """
    if not message or not message.strip():
        yield "Пожалуйста, введите вопрос / Please enter a question."
        return

    # Session management (reuse existing pattern)
    base_session_id = getattr(request, "session_hash", None) if request is not None else None
    session_id = _salt_session_id(base_session_id, history, message)

    # Save user message to conversation store (BEFORE agent execution)
    # This ensures conversation history is tracked for memory compression
    if session_id:
        llm_manager._conversations.append(session_id, "user", message)

    # Build messages from history for agent
    messages = []
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": message})

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

    # Track accumulated context for progressive budgeting
    # Agent is responsible for counting context, not the tool
    from rag_engine.utils.context_tracker import AgentContext, estimate_accumulated_tokens

    conversation_tokens, _ = estimate_accumulated_tokens(messages, [])

    try:
        # Track tool execution state
        # Only stream text content when NOT executing tools
        tool_executing = False
        import json

        # Pass accumulated context to agent via typed context parameter
        # Tools can access this via runtime.context (typed, clean!)
        agent_context = AgentContext(
            conversation_tokens=conversation_tokens,
            accumulated_tool_tokens=0,  # Updated as we go
        )

        # Use multiple stream modes for complete streaming experience
        # Per https://docs.langchain.com/oss/python/langchain/streaming#stream-multiple-modes
        for stream_mode, chunk in agent.stream(
            {"messages": messages},
            context=agent_context,
            stream_mode=["updates", "messages"]
        ):
            # Handle "messages" mode for token streaming
            if stream_mode == "messages":
                token, metadata = chunk

                # Filter out tool-related messages (DO NOT display in chat)
                # 1. Tool results (type="tool") - processed internally for citations
                if hasattr(token, "type") and token.type == "tool":
                    tool_results.append(token.content)
                    logger.debug("Tool result received, %d total results", len(tool_results))
                    tool_executing = False

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

                    # Parse result to get article count and emit completion metadata
                    try:
                        result = json.loads(token.content)
                        articles_count = result.get("metadata", {}).get("articles_count", 0)

                        # Yield completion metadata to Gradio
                        article_word = "article" if articles_count == 1 else "articles"
                        yield {
                            "role": "assistant",
                            "content": "",
                            "metadata": {"title": f"✅ Found {articles_count} {article_word}"}
                        }
                    except (json.JSONDecodeError, KeyError):
                        # If parsing fails, emit generic completion
                        yield {
                            "role": "assistant",
                            "content": "",
                            "metadata": {"title": "✅ Search completed"}
                        }

                    # CRITICAL: Check if accumulated tool results exceed safe threshold
                    # This prevents overflow when agent makes multiple tool calls
                    if settings.llm_fallback_enabled and not selected_model:
                        # Estimate total tokens: conversation + all tool results (JSON)
                        accumulated_tokens = _estimate_accumulated_context(messages, tool_results)

                        from rag_engine.llm.llm_manager import MODEL_CONFIGS
                        model_config = MODEL_CONFIGS.get(current_model, MODEL_CONFIGS["default"])
                        context_window = model_config.get("token_limit", 262144)

                        # Use 80% threshold for post-tool check (more conservative)
                        post_tool_threshold = int(context_window * 0.80)

                        if accumulated_tokens > post_tool_threshold:
                            logger.warning(
                                "Accumulated context (%d tokens) exceeds 80%% threshold (%d) after tool calls. "
                                "Checking for model fallback before final answer generation.",
                                accumulated_tokens,
                                post_tool_threshold,
                            )

                            # Find fallback model
                            fallback_model = _find_model_for_tokens(accumulated_tokens)

                            if fallback_model and fallback_model != current_model:
                                logger.warning(
                                    "⚠️ Switching from %s to %s mid-turn due to accumulated tool results (%d tokens)",
                                    current_model,
                                    fallback_model,
                                    accumulated_tokens,
                                )

                                # Notify user of fallback
                                yield {
                                    "role": "assistant",
                                    "content": "",
                                    "metadata": {"title": f"⚡ Switching to {fallback_model} (larger context needed)"}
                                }

                                # Recreate agent with larger model
                                # Note: This is expensive but prevents catastrophic overflow
                                agent = _create_rag_agent(override_model=fallback_model)
                                current_model = fallback_model

                                # Note: Can't restart stream here - agent will continue with new model
                                # for subsequent calls

                    # Skip further processing of tool messages
                    continue

                # 2. AI messages with tool_calls (when agent decides to call tools)
                # These should NEVER be displayed - only show metadata
                if hasattr(token, "tool_calls") and token.tool_calls:
                    if not tool_executing:
                        tool_executing = True
                        # Log tool call count safely (tool_calls might be True or a list)
                        call_count = len(token.tool_calls) if isinstance(token.tool_calls, list) else "?"
                        logger.debug("Agent calling tool(s): %s call(s)", call_count)
                        # Yield metadata message to Gradio
                        yield {
                            "role": "assistant",
                            "content": "",
                            "metadata": {
                                "title": "🔍 Searching information in the knowledge base"
                            },
                        }
                    # Skip displaying the tool call itself and any content
                    continue

                # 3. Only stream text content from messages WITHOUT tool_calls
                # This ensures we only show the final answer, not tool reasoning
                if hasattr(token, "tool_calls") and token.tool_calls:
                    # Skip any message that has tool_calls (redundant check for safety)
                    continue

                # Process content blocks for final answer text streaming
                if hasattr(token, "content_blocks") and token.content_blocks:
                    for block in token.content_blocks:
                        if block.get("type") == "tool_call_chunk":
                            # Tool call chunk detected - emit metadata if not already done
                            if not tool_executing:
                                tool_executing = True
                                logger.debug("Agent calling tool via chunk")
                                yield {
                                    "role": "assistant",
                                    "content": "",
                                    "metadata": {
                                        "title": "🔍 Searching information in the knowledge base"
                                    },
                                }
                            # Never stream tool call chunks as text
                            continue

                        elif block.get("type") == "text" and block.get("text"):
                            # Only stream text if we're not currently executing tools
                            # This prevents streaming the agent's "reasoning" about tool calls
                            if not tool_executing:
                                text_chunk = block["text"]
                                answer += text_chunk
                                yield answer

            # Handle "updates" mode for agent state updates
            elif stream_mode == "updates":
                # We can log updates but don't need to yield them
                logger.debug("Agent update: %s", list(chunk.keys()) if isinstance(chunk, dict) else chunk)

        # Accumulate articles from tool results and add citations
        from rag_engine.tools import accumulate_articles_from_tool_results
        articles = accumulate_articles_from_tool_results(tool_results)

        # Handle no results case
        if not articles:
            final_text = answer
            logger.info("Agent completed with no retrieved articles")
        else:
            final_text = format_with_citations(answer, articles)
            logger.info("Agent completed with %d articles", len(articles))

        # Save conversation turn (reuse existing pattern)
        if session_id:
            llm_manager.save_assistant_turn(session_id, final_text)

        yield final_text

    except Exception as e:
        logger.error("Error in agent_chat_handler: %s", e, exc_info=True)
        error_msg = f"Извините, произошла ошибка / Sorry, an error occurred: {str(e)}"
        yield error_msg


def _salt_session_id(base_session_id: str | None, history: list[dict], current_message: str = "") -> str | None:
    """Salt session_id with chat history to isolate memory per chat.

    When Gradio's save_history=True is enabled, different chats share the same
    session_hash. Salting ensures each chat (new or loaded from history) gets
    its own isolated session memory.

    Strategy: Generate deterministically from the first user message.
    - For new chats: uses current_message (first message being sent)
    - For loaded chats: uses first user message from history
    - For continuing same chat: same first message → same salt → same session_id → memory preserved

    This approach is robust because:
    1. The first message is stable across Gradio's save/load cycle
    2. Each distinct chat has a different first message → different session_id
    3. Same chat always generates the same session_id → memory continuity

    Args:
        base_session_id: Base session hash from Gradio request
        history: Current chat history from Gradio (includes loaded chats)
        current_message: Current message being sent (used for new chats)

    Returns:
        Salted session_id or None if base_session_id is None
    """
    if not base_session_id:
        return None

    # Extract first user message as salt
    # For loaded chats: from history; for new chats: from current_message
    salt = ""
    if history:
        # Loaded or continuing chat: use first user message from history
        for msg in history:
            role = msg.get("role", "")
            if role != "user":
                continue
            content = msg.get("content", "")
            # Handle both string and dict content (multimodal)
            if isinstance(content, dict):
                # Extract text from dict if available, otherwise use path
                text = content.get("text", "") or str(content.get("path", ""))
            else:
                text = str(content)
            if text:
                salt = text[:100]  # First 100 chars as salt
                break
    elif current_message:
        # New chat: use current message as salt
        salt = str(current_message)[:100]

    # Create salted session_id deterministically
    salted = f"{base_session_id}:{salt}"
    return hashlib.sha256(salted.encode()).hexdigest()[:32]


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
    type="messages",
    show_copy_button=True,
    min_height="30vh",
    height=chatbot_height,
    max_height=chatbot_max_height,
    resizable=True,
    elem_classes=["gradio-chatbot"],
)

# Force agent-based handler; legacy direct handler removed
handler_fn = agent_chat_handler
logger.info("Using agent-based (LangChain) handler for chat interface")

demo = gr.ChatInterface(
    fn=handler_fn,
    title=chat_title,
    description=chat_description,
    type="messages",
    save_history=True,
    #fill_width=True,
    chatbot=chatbot_config,
)
# Explicitly set a plain attribute for tests and downstream code to read
demo.title = "Comindware Platform Documentation Assistant"

try:
    gr.api(fn=query_rag, api_name="query_rag")
except Exception:  # noqa: BLE001
    # Older/newer Gradio builds without gr.api support
    pass


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
    )


