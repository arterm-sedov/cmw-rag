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

from rag_engine.config.settings import get_allowed_fallback_models, settings
from rag_engine.llm.llm_manager import LLMManager
from rag_engine.retrieval.embedder import FRIDAEmbedder
from rag_engine.retrieval.retriever import RAGRetriever
from rag_engine.storage.vector_store import ChromaStore
from rag_engine.utils.context_tracker import (
    AgentContext,
    estimate_accumulated_context,
    estimate_accumulated_tokens,
)
from rag_engine.utils.conversation_store import salt_session_id
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

    Uses app-level settings and allowed fallbacks (patch-friendly for tests).
    """
    # Early return if fallback is disabled
    if not getattr(settings, "llm_fallback_enabled", False):
        return None

    from rag_engine.llm.model_configs import MODEL_CONFIGS
    from rag_engine.llm.token_utils import count_messages_tokens

    model_config = MODEL_CONFIGS.get(settings.default_model)
    if not model_config:
        for key in MODEL_CONFIGS:
            if key != "default" and key in settings.default_model:
                model_config = MODEL_CONFIGS[key]
                break
    if not model_config:
        model_config = MODEL_CONFIGS["default"]

    current_window = int(model_config.get("token_limit", 0))
    total_tokens = count_messages_tokens(messages)
    # Get overhead from settings (with fallback for tests that don't set it)
    overhead = int(getattr(settings, "llm_tool_results_overhead_tokens", None) or 40000)
    total_tokens += overhead

    # Get threshold percentage from settings (with fallback for tests)
    pre_pct = float(getattr(settings, "llm_pre_context_threshold_pct", None) or 0.90)
    threshold = int(current_window * pre_pct)

    if total_tokens <= threshold:
        return None

    allowed = get_allowed_fallback_models()
    if not allowed:
        return None

    required = int(total_tokens * 1.1)
    for candidate in allowed:
        if candidate == settings.default_model:
            continue
        cfg = MODEL_CONFIGS.get(candidate)
        if not cfg:
            for key in MODEL_CONFIGS:
                if key != "default" and key in candidate:
                    cfg = MODEL_CONFIGS[key]
                    break
        if not cfg:
            continue
        if int(cfg.get("token_limit", 0)) >= required:
            return candidate
    return None


def compress_tool_results(state: dict, runtime) -> dict | None:
    """Compress tool results before LLM call if approaching context limit.

    This middleware runs right before each LLM invocation. It checks if the
    accumulated tool results + conversation would exceed 85% of the context window.
    If so, it compresses the least relevant articles using the summarization utility.

    This is a thin wrapper around the compression utility module.

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

    Local implementation to preserve test patch points for model construction
    and create_agent invocation.

    Args:
        override_model: Optional model name to use instead of default
                       (for context window fallback)

    Returns:
        Configured LangChain agent with retrieve_context tool and middleware
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware, before_model

    from rag_engine.llm.llm_manager import get_context_window
    from rag_engine.llm.prompts import SUMMARIZATION_PROMPT, SYSTEM_PROMPT
    from rag_engine.tools import retrieve_context

    # Use override model if provided (for fallback), otherwise use default
    selected_model = override_model or settings.default_model

    # Create LLMManager instance to handle all LLM operations
    # This centralizes model construction, config lookup, and provider logic
    temp_llm_manager = LLMManager(
        provider=settings.default_llm_provider,
        model=selected_model,
        temperature=settings.llm_temperature,
    )
    base_model = temp_llm_manager._chat_model()

    # Force tool execution
    model_with_tools = base_model.bind_tools([retrieve_context], tool_choice="retrieve_context")

    # Memory compression threshold (messages to keep from settings)
    messages_to_keep = getattr(settings, "memory_compression_messages_to_keep", 2)
    threshold_tokens = int(
        get_context_window(selected_model)
        * (getattr(settings, "memory_compression_threshold_pct", 80) / 100)
    )

    middleware_list = [
        ToolBudgetMiddleware(),
        before_model(update_context_budget),
        before_model(compress_tool_results),
        SummarizationMiddleware(
            model=base_model,
            token_counter=lambda _msgs: 0,
            max_tokens_before_summary=threshold_tokens,
            messages_to_keep=messages_to_keep,
            summary_prompt=SUMMARIZATION_PROMPT,
            summary_prefix="## Предыдущее обсуждение / Previous conversation:",
        ),
    ]

    agent = create_agent(
        model=model_with_tools,
        tools=[retrieve_context],
        system_prompt=SYSTEM_PROMPT,
        context_schema=AgentContext,
        middleware=middleware_list,
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
    session_id = salt_session_id(base_session_id, history, message)

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
    has_seen_tool_results = False

    # Track accumulated context for progressive budgeting
    # Agent is responsible for counting context, not the tool
    conversation_tokens, _ = estimate_accumulated_tokens(messages, [])

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

                    # Parse result to get article count and emit completion metadata
                    from rag_engine.api.stream_helpers import (
                        extract_article_count_from_tool_result,
                        yield_search_completed,
                    )

                    articles_count = extract_article_count_from_tool_result(token.content)
                    yield yield_search_completed(articles_count)

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

                            yield yield_model_switch_notice(fallback_model)

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
                        from rag_engine.api.stream_helpers import yield_search_started

                        yield yield_search_started()
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
                                from rag_engine.api.stream_helpers import yield_search_started

                                yield yield_search_started()
                            # Never stream tool call chunks as text
                            continue

                        elif block.get("type") == "text" and block.get("text"):
                            # Only stream text if we're not currently executing tools
                            # This prevents streaming the agent's "reasoning" about tool calls
                            if not tool_executing:
                                text_chunk = block["text"]
                                # Prepend newline before first text chunk after tool results
                                if has_seen_tool_results and not answer:
                                    text_chunk = "\n" + text_chunk
                                answer = answer + text_chunk
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
        # Handle context-length overflow gracefully by switching to a larger model (once)
        err_text = str(e).lower()
        is_context_overflow = (
            "maximum context length" in err_text
            or "context length" in err_text
            or "token limit" in err_text
            or "too many tokens" in err_text
        )

        if settings.llm_fallback_enabled and is_context_overflow:
            try:  # Single-shot fallback retry
                # Estimate required tokens and pick a capable fallback model
                required_tokens = estimate_accumulated_context(
                    messages,
                    tool_results,
                    overhead=int(getattr(settings, "llm_tool_results_overhead_tokens", 40000)),
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

                    yield yield_model_switch_notice(fallback_model)

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
                                        # Prepend newline before first text chunk after tool results
                                        if has_seen_tool_results and not answer:
                                            text_chunk = "\n" + text_chunk
                                        answer = answer + text_chunk
                                        yield answer
                        elif stream_mode == "updates":
                            # No-op for UI
                            pass

                    from rag_engine.tools import accumulate_articles_from_tool_results
                    articles = accumulate_articles_from_tool_results(tool_results)
                    final_text = answer if not articles else format_with_citations(answer, articles)

                    if session_id:
                        llm_manager.save_assistant_turn(session_id, final_text)
                    yield final_text
                    return
            except Exception as retry_exc:  # If fallback retry fails, emit original-style error
                logger.error("Fallback retry failed: %s", retry_exc, exc_info=True)

        # Default error path
        error_msg = f"Извините, произошла ошибка / Sorry, an error occurred: {str(e)}"
        yield error_msg




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


