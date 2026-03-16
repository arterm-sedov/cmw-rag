"""Ad-hoc test: full OpenRouter model with tools, reasoning, usage in real agent harness.

Uses OpenRouterNativeFullChatModel from rag_engine.llm.openrouter_native.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

from rag_engine.config.settings import settings
from rag_engine.llm.openrouter_native import create_openrouter_native_model


def _pp(title: str, payload: Any) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(payload)
    print()


async def run_agent_test() -> None:
    """Run agent with our custom model and verify usage, reasoning, costs."""
    load_dotenv()
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware

    from rag_engine.llm.model_configs import MODEL_CONFIGS
    from rag_engine.llm.prompts import SUMMARIZATION_PROMPT, get_system_prompt
    from rag_engine.llm.token_utils import count_messages_tokens
    from rag_engine.tools import (
        add,
        divide,
        get_current_datetime,
        modulus,
        multiply,
        power,
        retrieve_context,
        square_root,
        subtract,
    )
    from rag_engine.utils.context_tracker import AgentContext, set_current_context

    base_model = create_openrouter_native_model()
    from rag_engine.llm.llm_manager import get_context_window

    context_window = get_context_window(settings.default_model)
    threshold = int(context_window * (settings.memory_compression_threshold_pct / 100))

    def tiktoken_counter(messages: list) -> int:
        return count_messages_tokens(messages)

    all_tools = [
        retrieve_context,
        get_current_datetime,
        add,
        subtract,
        multiply,
        divide,
        power,
        square_root,
        modulus,
    ]

    cfg = MODEL_CONFIGS.get(settings.default_model, {})
    supports_forced = cfg.get("supports_forced_tool_choice", True)
    tool_choice = (
        {"type": "function", "function": {"name": "retrieve_context"}}
        if supports_forced
        else "auto"
    )

    model_with_tools = base_model.bind_tools(all_tools, tool_choice=tool_choice)

    agent = create_agent(
        model=model_with_tools,
        tools=all_tools,
        system_prompt=get_system_prompt(mild_limit=settings.llm_mild_limit),
        context_schema=AgentContext,
        middleware=[
            SummarizationMiddleware(
                model=base_model,
                token_counter=tiktoken_counter,
                max_tokens_before_summary=threshold,
                messages_to_keep=getattr(settings, "memory_compression_messages_to_keep", 2),
                summary_prompt=SUMMARIZATION_PROMPT,
                summary_prefix="## Previous conversation:",
            )
        ],
    )

    _pp("Agent created", {"model": settings.default_model, "tools": len(all_tools)})

    query = "How do I install Comindware Platform? Give a brief answer."
    messages = [HumanMessage(content=query)]
    ctx = AgentContext()
    set_current_context(ctx)

    _pp("Query", query)

    usage_samples: list[dict[str, Any]] = []

    class UsageCapture(BaseCallbackHandler):
        def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # type: ignore[override]
            lo = response.llm_output or {}
            if tu := lo.get("token_usage"):
                usage_samples.append({"token_usage": tu})

    try:
        result = await agent.ainvoke(
            {"messages": messages},
            context=ctx,
            config={"callbacks": [UsageCapture()]},
        )
    finally:
        set_current_context(None)

    msgs = result.get("messages", [])
    last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)

    _pp("Usage samples (from callbacks)", usage_samples)
    _pp("Last AIMessage.usage_metadata", getattr(last_ai, "usage_metadata", None))
    _pp("Last AIMessage.response_metadata", getattr(last_ai, "response_metadata", None))
    _pp("Last AIMessage.additional_kwargs (reasoning)", getattr(last_ai, "additional_kwargs", None))
    _pp("Answer preview", (last_ai.content or "")[:500] if last_ai else "N/A")
    _pp(
        "AgentContext.usage_turn_summary",
        getattr(ctx, "usage_turn_summary", "N/A (no UsageAccountingCallback)"),
    )

    total_cost = 0.0
    for s in usage_samples:
        tu = s.get("token_usage") or {}
        total_cost += float(tu.get("cost", 0) or 0)
    _pp("Total cost (sum of samples)", total_cost)


if __name__ == "__main__":
    asyncio.run(run_agent_test())
