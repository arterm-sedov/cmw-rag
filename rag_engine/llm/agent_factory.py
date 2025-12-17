"""Agent factory for creating RAG agents with consistent configuration."""
from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, before_model

from rag_engine.config.settings import settings
from rag_engine.llm.llm_manager import LLMManager, get_context_window
from rag_engine.llm.prompts import SUMMARIZATION_PROMPT, SYSTEM_PROMPT
from rag_engine.llm.token_utils import count_messages_tokens
from rag_engine.utils.context_tracker import AgentContext

logger = logging.getLogger(__name__)


def create_rag_agent(
    override_model: str | None = None,
    retrieve_context_tool=None,
    update_context_budget_middleware=None,
    compress_tool_results_middleware=None,
    tool_budget_middleware=None,
) -> any:
    """Create LangChain agent with forced retrieval tool execution and memory compression.

    Uses tool_choice parameter to enforce tool calling, the standard
    Comindware Platform system prompt, and SummarizationMiddleware for
    automatic conversation history compression at configurable context threshold.

    Args:
        override_model: Optional model name to use instead of default
                       (for context window fallback)
        retrieve_context_tool: Optional retrieve_context tool (defaults to import)
        update_context_budget_middleware: Optional middleware function
        compress_tool_results_middleware: Optional compression middleware function
        tool_budget_middleware: Optional ToolBudgetMiddleware instance

    Returns:
        Configured LangChain agent with retrieve_context tool and middleware

    Example:
        >>> from rag_engine.llm.agent_factory import create_rag_agent
        >>> agent = create_rag_agent()
        >>> # Use agent.stream() or agent.invoke()
    """
    if retrieve_context_tool is None:
        # Import here to avoid circular dependencies
        from rag_engine.tools import retrieve_context

        retrieve_context_tool = retrieve_context

    # Use override model if provided (for fallback), otherwise use default
    selected_model = override_model or settings.default_model

    # Use centralized LLMManager for consistent model construction
    # This ensures max_tokens, OpenRouter headers, streaming, and config lookup
    temp_llm_manager = LLMManager(
        provider=settings.default_llm_provider,
        model=selected_model,
        temperature=settings.llm_temperature,
    )
    base_model = temp_llm_manager._chat_model()

    # Get model configuration for context window
    context_window = get_context_window(selected_model)

    # Calculate threshold (configurable, default 70%)
    threshold_tokens = int(context_window * (settings.memory_compression_threshold_pct / 100))

    # Use centralized token counter from token_utils
    def tiktoken_counter(messages: list) -> int:
        """Count tokens using centralized utility.

        Uses count_messages_tokens which handles both dict and LangChain
        message objects with exact tiktoken encoding.
        """
        return count_messages_tokens(messages)

    # CRITICAL: Use tool_choice to force retrieval tool execution
    # This ensures the agent always searches the knowledge base
    model_with_tools = base_model.bind_tools(
        [retrieve_context_tool],
        tool_choice={
            "type": "function",
            "function": {"name": "retrieve_context"},
        },
    )

    # Get messages_to_keep from settings (default 2, matching old handler)
    messages_to_keep = getattr(settings, "memory_compression_messages_to_keep", 2)

    # Build middleware list
    middleware_list = []

    # Add tool budget middleware if provided
    if tool_budget_middleware:
        middleware_list.append(tool_budget_middleware)

    # Add context budget update middleware if provided
    if update_context_budget_middleware:
        middleware_list.append(before_model(update_context_budget_middleware))

    # Add compression middleware if provided
    if compress_tool_results_middleware:
        middleware_list.append(before_model(compress_tool_results_middleware))

    # Add summarization middleware
    middleware_list.append(
        SummarizationMiddleware(
            model=base_model,  # Use same model for summarization
            token_counter=tiktoken_counter,  # Use our centralized counter
            max_tokens_before_summary=threshold_tokens,  # Configurable threshold
            messages_to_keep=messages_to_keep,  # Configurable, default 2
            summary_prompt=SUMMARIZATION_PROMPT,  # Use existing prompt
            summary_prefix="## Предыдущее обсуждение / Previous conversation:",
        ),
    )

    agent = create_agent(
        model=model_with_tools,
        tools=[retrieve_context_tool],
        system_prompt=SYSTEM_PROMPT,
        context_schema=AgentContext,  # Typed context for tools
        middleware=middleware_list,
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

