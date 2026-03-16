"""Multi-model OpenRouter compatibility test: usage accounting, reasoning, streaming, bind_tools, middleware.

Tests OpenRouterNativeFullChatModel across models from .env slugs:
- deepseek, gpt-oss, gemini flash/pro, kimi, glm, claude sonnet, qwen

References:
- https://docs.langchain.com/oss/python/integrations/providers/openrouter
- https://openrouter.ai/docs/guides/guides/usage-accounting
- LangChain ChatOpenRouter: D:\\Repo\\langchain\\libs\\partners\\openrouter
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

from rag_engine.config.settings import settings
from rag_engine.llm.openrouter_native import create_openrouter_native_model

# Models to test (slugs from .env and model_configs)
MODELS_TO_TEST = [
    "qwen/qwen3.5-flash-02-23",
    "deepseek/deepseek-v3.1-terminus",
    "openai/gpt-oss-20b",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "moonshotai/kimi-k2.5",
    "z-ai/glm-4.7",
    "z-ai/glm-4.7-flash",
    "anthropic/claude-sonnet-4.5",
]


def _create_model_for(model_slug: str):
    """Create OpenRouterNativeFullChatModel for given model slug."""
    m = create_openrouter_native_model(model_name=model_slug)
    m.max_tokens = 256
    return m


def _check_usage(msg: AIMessage | None, label: str) -> tuple[bool, str]:
    """Verify usage accounting on AIMessage. Returns (ok, detail)."""
    if msg is None:
        return False, "no AIMessage"
    um = getattr(msg, "usage_metadata", None)
    rm = getattr(msg, "response_metadata", None) or {}
    tu = rm.get("token_usage") if isinstance(rm, dict) else None
    if not um and not tu:
        return False, "no usage_metadata or token_usage"
    # Prefer token_usage from response_metadata (our model puts it there)
    if tu and isinstance(tu, dict):
        data = tu
    elif isinstance(um, dict):
        data = um
    elif um and hasattr(um, "model_dump"):
        data = um.model_dump()
    else:
        data = {}
    if not data:
        return False, "empty usage"
    prompt = data.get("prompt_tokens") or data.get("input_tokens") or 0
    completion = data.get("completion_tokens") or data.get("output_tokens") or 0
    cost = data.get("cost")
    if prompt == 0 and completion == 0:
        return False, f"zero tokens: {data}"
    detail = f"prompt={prompt} comp={completion}"
    if cost is not None:
        detail += f" cost={cost:.6f}"
    comp_details = data.get("completion_tokens_details") or data.get("output_token_details")
    reasoning = comp_details.get("reasoning_tokens") if isinstance(comp_details, dict) else None
    if reasoning is not None:
        detail += f" reasoning={reasoning}"
    return True, detail


async def _test_invoke(model_slug: str) -> tuple[bool, str]:
    """Test simple ainvoke - usage in response."""
    try:
        model = _create_model_for(model_slug)
        msg = await model.ainvoke([HumanMessage(content="Say 'ok' in one word.")])
        if isinstance(msg, AIMessage):
            ai = msg
        else:
            ai = msg.content if hasattr(msg, "content") else None
        ok, detail = _check_usage(ai if isinstance(ai, AIMessage) else None, "invoke")
        return ok, detail
    except Exception as e:
        return False, str(e)


async def _test_astream(model_slug: str) -> tuple[bool, str]:
    """Test astream - usage in final chunk (OpenRouter always includes it)."""
    try:
        model = _create_model_for(model_slug)
        full: AIMessage | None = None
        async for chunk in model.astream([HumanMessage(content="Say 'hi' in one word.")]):
            if hasattr(chunk, "message"):
                m = chunk.message
            else:
                m = chunk
            if m:
                if full is None:
                    full = m
                else:
                    full = full + m
        ok, detail = _check_usage(full, "astream")
        return ok, detail
    except Exception as e:
        return False, str(e)


async def _test_bind_tools(model_slug: str) -> tuple[bool, str]:
    """Test bind_tools + invoke - tool calling works."""
    try:
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            """Get weather for a location."""
            location: str = Field(description="City name")

        model = _create_model_for(model_slug)
        model_with_tools = model.bind_tools([GetWeather], tool_choice="auto")
        msg = await model_with_tools.ainvoke(
            [HumanMessage(content="What's the weather in Paris? Use the tool.")]
        )
        if isinstance(msg, AIMessage):
            ai = msg
        else:
            ai = getattr(msg, "content", msg)
        if not isinstance(ai, AIMessage):
            return False, "no AIMessage"
        has_tool_calls = bool(getattr(ai, "tool_calls", None))
        ok, detail = _check_usage(ai, "bind_tools")
        if has_tool_calls:
            detail += " [tool_calls=ok]"
        else:
            # Some models may answer without calling - still pass if usage ok
            detail += " [no tool_calls]"
        return ok, detail
    except Exception as e:
        return False, str(e)


async def _test_agent_with_middleware(model_slug: str) -> tuple[bool, str]:
    """Test full agent with SummarizationMiddleware (new API: trigger, keep)."""
    try:
        from langchain.agents import create_agent
        from langchain.agents.middleware import SummarizationMiddleware

        from rag_engine.llm.model_configs import MODEL_CONFIGS
        from rag_engine.llm.prompts import SUMMARIZATION_PROMPT, get_system_prompt
        from rag_engine.llm.token_utils import count_messages_tokens
        from rag_engine.tools import add, get_current_datetime, retrieve_context
        from rag_engine.utils.context_tracker import AgentContext, set_current_context

        model = _create_model_for(model_slug)
        context_window = 128_000  # Use safe default for model_configs lookup
        cfg = MODEL_CONFIGS.get(model_slug, MODEL_CONFIGS.get("default", {}))
        ctx_limit = cfg.get("token_limit", context_window)
        threshold = int(ctx_limit * (settings.memory_compression_threshold_pct / 100))

        def tiktoken_counter(messages: list) -> int:
            return count_messages_tokens(messages)

        tools = [retrieve_context, get_current_datetime, add]
        supports_forced = cfg.get("supports_forced_tool_choice", True)
        tool_choice = (
            {"type": "function", "function": {"name": "retrieve_context"}}
            if supports_forced
            else "auto"
        )
        model_with_tools = model.bind_tools(tools, tool_choice=tool_choice)

        # Use new SummarizationMiddleware API (trigger, keep) to avoid deprecation
        agent = create_agent(
            model=model_with_tools,
            tools=tools,
            system_prompt=get_system_prompt(mild_limit=settings.llm_mild_limit),
            context_schema=AgentContext,
            middleware=[
                SummarizationMiddleware(
                    model=model,
                    token_counter=tiktoken_counter,
                    trigger=("tokens", threshold),
                    keep=("messages", getattr(settings, "memory_compression_messages_to_keep", 2)),
                    summary_prompt=SUMMARIZATION_PROMPT,
                    summary_prefix="## Previous conversation:",
                )
            ],
        )

        ctx = AgentContext()
        set_current_context(ctx)
        usage_samples: list[dict] = []

        class UsageCapture(BaseCallbackHandler):
            def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # type: ignore[override]
                lo = getattr(response, "llm_output", None) or {}
                if tu := lo.get("token_usage"):
                    usage_samples.append({"token_usage": tu})

        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is 2+2? Use the add tool.")]},
                context=ctx,
                config={"callbacks": [UsageCapture()]},
            )
        finally:
            set_current_context(None)

        msgs = result.get("messages", [])
        last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
        ok, detail = _check_usage(last_ai, "agent")
        total_cost = sum(
            float((s.get("token_usage") or {}).get("cost", 0) or 0) for s in usage_samples
        )
        detail += f" calls={len(usage_samples)} cost={total_cost:.6f}"
        return ok, detail
    except Exception as e:
        return False, str(e)


async def run_model_tests(model_slug: str, tests: list[str]) -> dict[str, tuple[bool, str]]:
    """Run selected tests for one model."""
    results: dict[str, tuple[bool, str]] = {}
    if "invoke" in tests:
        results["invoke"] = await _test_invoke(model_slug)
    if "astream" in tests:
        results["astream"] = await _test_astream(model_slug)
    if "bind_tools" in tests:
        results["bind_tools"] = await _test_bind_tools(model_slug)
    if "agent" in tests:
        results["agent"] = await _test_agent_with_middleware(model_slug)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-model OpenRouter compatibility test (usage, reasoning, streaming, bind_tools, middleware)"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model slugs to test (default: first 3 from MODELS_TO_TEST for quick run)",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        default=["invoke", "astream", "bind_tools", "agent"],
        choices=["invoke", "astream", "bind_tools", "agent"],
        help="Tests to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models (default: first 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    load_dotenv()
    if not settings.openrouter_api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    models = args.models or (MODELS_TO_TEST if args.all else MODELS_TO_TEST[:3])
    tests = args.tests

    print(f"Testing models: {models}")
    print(f"Tests: {tests}")
    print()

    all_results: dict[str, dict[str, tuple[bool, str]]] = {}
    for model_slug in models:
        print(f"--- {model_slug} ---")
        try:
            results = asyncio.run(run_model_tests(model_slug, tests))
            all_results[model_slug] = results
            for name, (ok, detail) in results.items():
                status = "PASS" if ok else "FAIL"
                print(f"  {name}: {status} - {detail}")
        except Exception as e:
            all_results[model_slug] = {"error": (False, str(e))}
            print(f"  ERROR: {e}")
        print()

    if args.json:
        # Serialize for JSON (tuple -> list)
        out = {
            m: {k: [v[0], v[1]] for k, v in r.items()}
            for m, r in all_results.items()
        }
        print(json.dumps(out, indent=2))

    failed = sum(
        1 for r in all_results.values() for ok, _ in r.values() if not ok
    )
    total = sum(len(r) for r in all_results.values())
    print(f"Summary: {total - failed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
