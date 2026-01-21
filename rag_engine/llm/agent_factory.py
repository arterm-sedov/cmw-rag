"""Agent factory for creating RAG agents with consistent configuration."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, before_model

from rag_engine.config.settings import settings
from rag_engine.llm.llm_manager import LLMManager, get_context_window
from rag_engine.llm.prompts import SUMMARIZATION_PROMPT, get_system_prompt
from rag_engine.llm.token_utils import count_messages_tokens
from rag_engine.utils.context_tracker import AgentContext

logger = logging.getLogger(__name__)


def create_rag_agent(
    override_model: str | None = None,
    retrieve_context_tool=None,
    update_context_budget_middleware=None,
    compress_tool_results_middleware=None,
    tool_budget_middleware=None,
    force_tool_choice: bool = False,
    enable_sgr_planning: bool = True,
    sgr_spam_threshold: float = 0.8,
) -> any:
    """Create LangChain agent with optional forced retrieval tool execution and memory compression.

    Uses tool_choice parameter to optionally enforce tool calling, the standard
    Comindware Platform system prompt, and SummarizationMiddleware for
    automatic conversation history compression at configurable context threshold.

    Args:
        override_model: Optional model name to use instead of default
                       (for context window fallback)
        retrieve_context_tool: Optional retrieve_context tool (defaults to import)
        update_context_budget_middleware: Optional middleware function
        compress_tool_results_middleware: Optional compression middleware function
        tool_budget_middleware: Optional ToolBudgetMiddleware instance
        force_tool_choice: If True, forces retrieve_context tool execution.
                          If False, allows model to choose tools freely (default: False)

    Returns:
        Configured LangChain agent with retrieve_context tool and middleware

    Example:
        >>> from rag_engine.llm.agent_factory import create_rag_agent
        >>> agent = create_rag_agent(force_tool_choice=True)  # Force on first call
        >>> agent2 = create_rag_agent()  # Allow model choice (default)
        >>> # Use agent.stream() or agent.invoke()
    """
    if retrieve_context_tool is None:
        # Import here to avoid circular dependencies
        from rag_engine.tools import (
            add,
            divide,
            get_current_datetime,
            modulus,
            multiply,
            power,
            retrieve_context,
            analyse_user_request,
            square_root,
            subtract,
        )

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

    # Get mild_limit for system prompt guidance (soft limit, separate from hard max_tokens cutoff)
    mild_limit = settings.llm_mild_limit

    # Calculate threshold (configurable, default 70%)
    threshold_tokens = int(context_window * (settings.memory_compression_threshold_pct / 100))

    # Use centralized token counter from token_utils
    def tiktoken_counter(messages: Iterable[Any]) -> int:
        """Count tokens using centralized utility.

        Uses count_messages_tokens which handles both dict and LangChain
        message objects with exact tiktoken encoding.
        """
        return count_messages_tokens(list(messages))

    # Conditionally use tool_choice to force retrieval tool execution
    # On first call: force retrieve_context to ensure knowledge base search
    # On subsequent calls: allow model to choose tools freely
    all_tools = [
        # User request analysis tool (forced externally per turn)
        # If enable_sgr_planning is False, we won't register it at all.
        *( [analyse_user_request] if enable_sgr_planning else [] ),
        retrieve_context_tool,
        get_current_datetime,
        add,
        subtract,
        multiply,
        divide,
        power,
        square_root,
        modulus,
    ]

    # Set tool_choice based on force_tool_choice parameter
    # If True, force retrieve_context; if False, allow model to choose (None = auto)
    tool_choice_value = (
        {
            "type": "function",
            "function": {"name": "retrieve_context"},
        }
        if force_tool_choice
        else None
    )

    model_with_tools = base_model.bind_tools(
        all_tools,
        tool_choice=tool_choice_value,
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
            token_counter=tiktoken_counter,  # type: ignore[arg-type]  # Use our centralized counter
            max_tokens_before_summary=threshold_tokens,  # Configurable threshold
            messages_to_keep=messages_to_keep,  # Configurable, default 2
            summary_prompt=SUMMARIZATION_PROMPT,  # Use existing prompt
            summary_prefix="## Предыдущее обсуждение / Previous conversation:",
        ),
    )

    agent = create_agent(
        model=model_with_tools,
        tools=all_tools,
        system_prompt=get_system_prompt(mild_limit=mild_limit),
        context_schema=AgentContext,  # Typed context for tools
        middleware=middleware_list,
    )

    tool_choice_status = "forced" if force_tool_choice else "optional"
    if override_model:
        logger.info(
            "RAG agent created with FALLBACK MODEL %s: tool choice=%s, "
            "memory compression (threshold: %d tokens at %d%%, keep: %d msgs, window: %d)",
            selected_model,
            tool_choice_status,
            threshold_tokens,
            settings.memory_compression_threshold_pct,
            messages_to_keep,
            context_window,
        )
    else:
        logger.info(
            "RAG agent created with tool choice=%s and memory compression "
            "(threshold: %d tokens at %d%%, keep: %d msgs, window: %d)",
            tool_choice_status,
            threshold_tokens,
            settings.memory_compression_threshold_pct,
            messages_to_keep,
            context_window,
        )
    return agent

