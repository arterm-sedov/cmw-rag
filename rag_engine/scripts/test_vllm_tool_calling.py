"""Standalone script to rehearse tool-calling against vLLM/OpenRouter.

Uses LangChain's ChatOpenAI and the actual RAG agent mechanics to verify whether the
backend emits tool calls (function calling) for a simple query, with comprehensive
logging to pinpoint issues.

Usage (from repo root, with .venv activated):

    python -m rag_engine.scripts.test_vllm_tool_calling --provider vllm --mode rag_stream_like_app
    python -m rag_engine.scripts.test_vllm_tool_calling --provider openrouter --mode rag_stream_like_app

Environment:
- For vLLM:
    VLLM_BASE_URL (e.g. http://localhost:8000/v1 or your gpt-oss vLLM URL)
    VLLM_API_KEY   (optional, default "EMPTY")
- For OpenRouter:
    OPENROUTER_BASE_URL (default: https://openrouter.ai/api/v1)
    OPENROUTER_API_KEY  (required)
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware, before_model
from langchain.agents.middleware import wrap_tool_call as middleware_wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from rag_engine.config.settings import settings
from rag_engine.llm.llm_manager import LLMManager, get_context_window
from rag_engine.llm.prompts import SUMMARIZATION_PROMPT, get_system_prompt
from rag_engine.llm.token_utils import count_messages_tokens
from rag_engine.utils.context_tracker import (
    AgentContext,
    compute_context_tokens,
    estimate_accumulated_tokens,
)
from rag_engine.utils.logging_manager import setup_logging

# Setup logging with detailed format
setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@tool
def get_weather(city: str) -> str:
    """Mock weather tool used to test function calling.

    Returns a deterministic string so we can easily see if the tool was
    actually invoked by the model.
    """
    logger.debug(f"[get_weather] Called with city={city}")
    return f"The weather in {city} is sunny and 20°C."


class ToolBudgetMiddleware(AgentMiddleware):
    """Populate runtime.context tokens right before each tool execution.

    Ensures tools see up-to-date conversation and accumulated tool tokens
    even when multiple tool calls happen within a single agent step.
    """

    @middleware_wrap_tool_call()
    def tool_budget_wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> ToolMessage | Command:
        try:
            state = getattr(request, "state", {}) or {}
            runtime = getattr(request, "runtime", None)
            if state and runtime is not None and hasattr(runtime, "context") and runtime.context:
                conv_toks, tool_toks = compute_context_tokens(
                    state.get("messages", []), tool_results=None, add_json_overhead=True
                )
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


def _compute_context_tokens_from_state(messages: list[dict]) -> tuple[int, int]:
    """Compute (conversation_tokens, accumulated_tool_tokens) from agent state messages."""
    return compute_context_tokens(messages, tool_results=None, add_json_overhead=True)


def update_context_budget(state: dict, runtime) -> dict | None:  # noqa: ANN001
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
                "[update_context_budget] Updated runtime.context: conversation_tokens=%d, accumulated_tool_tokens=%d",
                conv_toks,
                tool_toks,
            )
    except Exception:
        # Do not fail the run due to budgeting issues
        logger.debug("[update_context_budget] Unable to update runtime.context tokens")

    return None


def compress_tool_results(state: dict, runtime) -> dict | None:  # noqa: ANN001
    """Compress tool results before LLM call if approaching context limit.

    This middleware runs right before each LLM invocation, AFTER all tool calls complete.
    For testing purposes, we use a no-op version to match app structure.
    """
    # In the real app, this compresses tool messages. For testing, we just log.
    messages = state.get("messages", [])
    if messages:
        tool_msg_count = sum(1 for msg in messages if getattr(msg, "type", None) == "tool")
        if tool_msg_count > 0:
            logger.debug("[compress_tool_results] Found %d tool messages (no-op in test)", tool_msg_count)
    return None


def _build_llm(provider: str, model: str) -> ChatOpenAI:
    """Build ChatOpenAI client for given provider and model."""
    provider = provider.lower()
    if provider == "vllm":
        # Reuse centralized .env-driven configuration
        base_url = settings.vllm_base_url
        api_key = getattr(settings, "vllm_api_key", "EMPTY") or "EMPTY"
    elif provider == "openrouter":
        base_url = settings.openrouter_base_url
        api_key = settings.openrouter_api_key
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    logger.info(f"[_build_llm] Building LLM: provider={provider}, model={model}, base_url={base_url}")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        streaming=False,  # CRITICAL: streaming=False for vLLM compatibility
    )


def _create_rag_agent(
    provider: str,
    model: str,
    retrieve_context_tool=None,
) -> Any:
    """Create LangChain agent with forced retrieval tool execution and memory compression.

    Uses the exact same pattern as app.py's _create_rag_agent to reproduce real behavior.
    """
    if retrieve_context_tool is None:
        from rag_engine.tools import retrieve_context

        retrieve_context_tool = retrieve_context

    logger.info(f"[_create_rag_agent] Creating RAG agent: provider={provider}, model={model}")

    # Use LLMManager exactly like create_rag_agent does
    temp_llm_manager = LLMManager(
        provider=provider,
        model=model,
        temperature=settings.llm_temperature,
    )
    base_model = temp_llm_manager._chat_model()

    # Get model configuration for context window
    context_window = get_context_window(model)
    logger.debug(f"[_create_rag_agent] Context window: {context_window}")

    # Calculate threshold (configurable, default 70%)
    threshold_tokens = int(context_window * (settings.memory_compression_threshold_pct / 100))
    logger.debug(f"[_create_rag_agent] Compression threshold: {threshold_tokens} tokens")

    # Use centralized token counter from token_utils
    def tiktoken_counter(messages: list) -> int:
        """Count tokens using centralized utility."""
        return count_messages_tokens(messages)

    # CRITICAL: Use tool_choice to force retrieval tool execution
    # This ensures the agent always searches the knowledge base
    logger.info("[_create_rag_agent] Binding tools with tool_choice='retrieve_context'")
    model_with_tools = base_model.bind_tools(
        [retrieve_context_tool],
        tool_choice="retrieve_context",
    )

    # Get messages_to_keep from settings (default 2, matching old handler)
    messages_to_keep = getattr(settings, "memory_compression_messages_to_keep", 2)

    # Build middleware list (exact same as app.py)
    middleware_list = []

    # Add tool budget middleware
    middleware_list.append(ToolBudgetMiddleware())

    # Add context budget update middleware
    middleware_list.append(before_model(update_context_budget))

    # Add compression middleware
    middleware_list.append(before_model(compress_tool_results))

    # Add summarization middleware
    middleware_list.append(
        SummarizationMiddleware(
            model=base_model,
            token_counter=tiktoken_counter,
            max_tokens_before_summary=threshold_tokens,
            messages_to_keep=messages_to_keep,
            summary_prompt=SUMMARIZATION_PROMPT,
            summary_prefix="## Предыдущее обсуждение / Previous conversation:",
        ),
    )

    logger.info(f"[_create_rag_agent] Created agent with {len(middleware_list)} middleware(s)")
    agent = create_agent(
        model=model_with_tools,
        tools=[retrieve_context_tool],
        system_prompt=get_system_prompt(),  # Use function for consistency
        context_schema=AgentContext,
        middleware=middleware_list,
    )

    return agent


def _describe_response(msg: Any) -> None:
    """Print a compact summary of the AIMessage/tool calls."""
    logger.info("=== Raw message ===")
    logger.info(repr(msg))

    tool_calls = getattr(msg, "tool_calls", None) or getattr(
        msg, "additional_kwargs", {}
    ).get("tool_calls")
    logger.info(f"Has tool_calls: {bool(tool_calls)}")
    if tool_calls:
        logger.info(f"tool_calls count: {len(tool_calls)}")
        for i, tc in enumerate(tool_calls, start=1):
            logger.info(f"  [{i}] name={tc.get('function', {}).get('name')}, type={tc.get('type')}")
            logger.info(f"      arguments={tc.get('function', {}).get('arguments')}")

    content = getattr(msg, "content", "")
    logger.info(f"=== Assistant content (length={len(str(content))}) ===")
    logger.info(content)


def _test_direct_openai_client_streaming(provider: str, model: str):
    """Test vLLM streaming tool calls using raw OpenAI client to see raw responses.

    This bypasses LangChain to see exactly what vLLM sends in streaming mode.
    Based on GitHub issue: https://github.com/vllm-project/vllm/issues/27641
    """
    from openai import OpenAI

    provider = provider.lower()
    if provider == "vllm":
        base_url = settings.vllm_base_url
        api_key = getattr(settings, "vllm_api_key", "EMPTY") or "EMPTY"
    elif provider == "openrouter":
        base_url = settings.openrouter_base_url
        api_key = settings.openrouter_api_key
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in a given city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What's the weather in Moscow?"}
    ]

    logger.info("=" * 80)
    logger.info("Testing RAW OpenAI Client Streaming (bypassing LangChain)")
    logger.info("=" * 80)

    # Test streaming mode
    logger.info("[RAW STREAM] Testing streaming mode...")
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=True,
        max_tokens=4096,
    )

    tool_calls = []
    content = ""
    finish_reason = None
    chunk_count = 0

    # Accumulate tool calls from chunks (as per vLLM docs)
    for chunk in stream:
        chunk_count += 1
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if delta.content:
            content += delta.content
            logger.debug(f"[RAW STREAM] Chunk #{chunk_count}: Content chunk: {delta.content[:50]}...")

        # Accumulate tool calls from delta.tool_calls (critical for vLLM)
        if delta.tool_calls:
            logger.info(f"[RAW STREAM] Chunk #{chunk_count}: Tool call delta received: {len(delta.tool_calls)} chunk(s)")
            for tc in delta.tool_calls:
                idx = tc.index
                # Ensure we have enough slots
                while len(tool_calls) <= idx:
                    tool_calls.append({
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })

                if tc.id:
                    tool_calls[idx]["id"] = tc.id
                    logger.debug(f"[RAW STREAM] Tool call {idx} ID: {tc.id}")

                if tc.function:
                    if tc.function.name:
                        tool_calls[idx]["function"]["name"] = tc.function.name
                        logger.info(f"[RAW STREAM] Tool call {idx} name: {tc.function.name}")
                    if tc.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc.function.arguments
                        logger.debug(f"[RAW STREAM] Tool call {idx} arguments chunk: {len(tc.function.arguments)} chars")

        # Check finish reason
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
            logger.info(f"[RAW STREAM] Chunk #{chunk_count}: Finish reason: {finish_reason}")

    logger.info("=" * 80)
    logger.info("RAW Streaming Results:")
    logger.info(f"  Total chunks: {chunk_count}")
    logger.info(f"  Content length: {len(content)}")
    logger.info(f"  Tool calls detected: {len(tool_calls)}")
    logger.info(f"  Finish reason: {finish_reason}")
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            logger.info(f"  Tool call {i}: id={tc.get('id', 'N/A')}, name={tc.get('function', {}).get('name', 'N/A')}")
            args = tc.get('function', {}).get('arguments', '')
            logger.info(f"    Arguments length: {len(args)} chars")
            if args:
                try:
                    import json
                    args_dict = json.loads(args)
                    logger.info(f"    Arguments parsed: {args_dict}")
                except Exception as e:
                    logger.debug(f"    Arguments (raw): {args[:100]}... (parse error: {e})")
    logger.info("=" * 80)

    return {
        "tool_calls": tool_calls,
        "content": content,
        "finish_reason": finish_reason,
        "chunk_count": chunk_count,
    }


def _test_direct_openai_client_non_streaming(provider: str, model: str):
    """Test vLLM non-streaming tool calls using raw OpenAI client."""
    from openai import OpenAI

    provider = provider.lower()
    if provider == "vllm":
        base_url = settings.vllm_base_url
        api_key = getattr(settings, "vllm_api_key", "EMPTY") or "EMPTY"
    elif provider == "openrouter":
        base_url = settings.openrouter_base_url
        api_key = settings.openrouter_api_key
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in a given city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What's the weather in Moscow?"}
    ]

    logger.info("=" * 80)
    logger.info("Testing RAW OpenAI Client Non-Streaming")
    logger.info("=" * 80)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=False,
        max_tokens=4096,
    )

    msg = response.choices[0].message
    tool_calls = msg.tool_calls
    content = msg.content
    finish_reason = response.choices[0].finish_reason

    logger.info("RAW Non-Streaming Results:")
    logger.info(f"  Content: {content}")
    logger.info(f"  Tool calls: {len(tool_calls) if tool_calls else 0}")
    logger.info(f"  Finish reason: {finish_reason}")
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            logger.info(f"  Tool call {i}: id={tc.id}, name={tc.function.name}")
            logger.info(f"    Arguments: {tc.function.arguments}")
    logger.info("=" * 80)

    return {
        "tool_calls": [{"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in (tool_calls or [])],
        "content": content,
        "finish_reason": finish_reason,
    }


def _test_langchain_chatopenai_streaming(provider: str, model: str):
    """Test LangChain ChatOpenAI streaming to diagnose why tool calls aren't detected.

    This replicates the exact behavior used in the app to see what LangChain receives
    and how it processes streaming tool calls from vLLM.
    """
    logger.info("=" * 80)
    logger.info("Testing LangChain ChatOpenAI Streaming (replicating app behavior)")
    logger.info("=" * 80)

    # Build ChatOpenAI exactly like the app does
    llm = _build_llm(provider, model)
    
    # Enable streaming (like agent.stream() does)
    # Note: ChatOpenAI streaming is controlled at call site, not construction
    logger.info("[LangChain Stream] Creating ChatOpenAI with tools...")
    
    # Bind tools exactly like the app does
    from langchain_core.tools import tool
    
    @tool
    def get_weather(city: str) -> str:
        """Get current weather in a given city."""
        return f"The weather in {city} is sunny and 20°C."
    
    llm_with_tools = llm.bind_tools([get_weather])
    
    # Test streaming invoke (like agent.stream() does internally)
    logger.info("[LangChain Stream] Invoking with streaming=True...")
    
    messages = [{"role": "user", "content": "What's the weather in Moscow?"}]
    
    chunk_count = 0
    tool_calls_detected = []
    content_chunks = []
    finish_reason = None
    
    # Accumulate tool calls from content_blocks (like vLLM docs show)
    accumulated_tool_calls = {}  # index -> {id, name, args}
    
    try:
        # Use .stream() method (this is what LangChain uses internally)
        for chunk in llm_with_tools.stream(messages):
            chunk_count += 1
            logger.info(f"[LangChain Stream] Chunk #{chunk_count}: type={type(chunk).__name__}")
            
            # Check what attributes the chunk has
            attrs = dir(chunk)
            logger.debug(f"[LangChain Stream] Chunk attributes: {[a for a in attrs if not a.startswith('_')]}")
            
            # Check for tool_calls (may be incomplete during streaming)
            has_tool_calls = hasattr(chunk, "tool_calls") and bool(chunk.tool_calls)
            tool_calls = getattr(chunk, "tool_calls", None)
            
            # Check for content_blocks (tool calls come here incrementally)
            has_content_blocks = hasattr(chunk, "content_blocks") and bool(chunk.content_blocks)
            content_blocks = getattr(chunk, "content_blocks", None)
            
            # Check response_metadata
            response_metadata = getattr(chunk, "response_metadata", {})
            finish_reason_chunk = response_metadata.get("finish_reason") if isinstance(response_metadata, dict) else None
            
            logger.info(
                f"[LangChain Stream] Chunk #{chunk_count}: "
                f"has_tool_calls={has_tool_calls}, "
                f"has_content_blocks={has_content_blocks}, "
                f"finish_reason={finish_reason_chunk}"
            )
            
            # Process content_blocks to accumulate tool calls (CRITICAL for vLLM)
            if has_content_blocks:
                logger.info(f"[LangChain Stream] Content blocks found: {len(content_blocks)}")
                for i, block in enumerate(content_blocks):
                    block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", "unknown")
                    logger.info(f"[LangChain Stream]   Block {i}: type={block_type}")
                    
                    if block_type == "tool_call_chunk":
                        logger.info(f"[LangChain Stream]     TOOL CALL CHUNK detected!")
                        # Extract tool call chunk data
                        chunk_index = block.get("index", 0) if isinstance(block, dict) else getattr(block, "index", 0)
                        chunk_id = block.get("id") if isinstance(block, dict) else getattr(block, "id", None)
                        chunk_name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
                        chunk_args = block.get("args", "") if isinstance(block, dict) else getattr(block, "args", "")
                        
                        logger.debug(
                            f"[LangChain Stream]     Chunk data: index={chunk_index}, "
                            f"id={chunk_id}, name={chunk_name}, args_length={len(str(chunk_args))}"
                        )
                        
                        # Initialize accumulator for this tool call index
                        if chunk_index not in accumulated_tool_calls:
                            accumulated_tool_calls[chunk_index] = {
                                "id": "",
                                "name": "",
                                "args": ""
                            }
                        
                        # Accumulate tool call data
                        if chunk_id:
                            accumulated_tool_calls[chunk_index]["id"] = chunk_id
                        if chunk_name:
                            accumulated_tool_calls[chunk_index]["name"] = chunk_name
                        if chunk_args:
                            accumulated_tool_calls[chunk_index]["args"] += str(chunk_args)
            
            # Check tool_calls attribute (may be incomplete during streaming)
            if has_tool_calls:
                logger.info(f"[LangChain Stream] tool_calls attribute found in chunk #{chunk_count}!")
                if isinstance(tool_calls, list):
                    logger.info(f"[LangChain Stream] Tool calls count: {len(tool_calls)}")
                    for i, tc in enumerate(tool_calls):
                        tc_dict = tc if isinstance(tc, dict) else tc.__dict__ if hasattr(tc, "__dict__") else str(tc)
                        logger.info(f"[LangChain Stream]   Tool call {i}: {tc_dict}")
                        # Note: This may be incomplete - check accumulated_tool_calls for complete data
                else:
                    logger.info(f"[LangChain Stream] Tool calls (non-list): {tool_calls}")
            
            # Check content
            content = getattr(chunk, "content", "")
            if content:
                content_chunks.append(str(content))
                logger.debug(f"[LangChain Stream] Content chunk: {content[:50]}...")
            
            if finish_reason_chunk:
                finish_reason = finish_reason_chunk
                logger.info(f"[LangChain Stream] Stream finished with reason: {finish_reason}")
        
        # After stream completes, check accumulated tool calls
        logger.info("=" * 80)
        logger.info("Accumulated Tool Calls from content_blocks:")
        for idx, tc_data in sorted(accumulated_tool_calls.items()):
            logger.info(f"  Tool call index {idx}:")
            logger.info(f"    ID: {tc_data['id']}")
            logger.info(f"    Name: {tc_data['name']}")
            logger.info(f"    Args length: {len(tc_data['args'])} chars")
            if tc_data['args']:
                try:
                    import json
                    args_dict = json.loads(tc_data['args'])
                    logger.info(f"    Args parsed: {args_dict}")
                    tool_calls_detected.append({
                        "id": tc_data['id'],
                        "name": tc_data['name'],
                        "args": args_dict
                    })
                except Exception as e:
                    logger.warning(f"    Args parse error: {e}, raw: {tc_data['args'][:100]}...")
                    tool_calls_detected.append({
                        "id": tc_data['id'],
                        "name": tc_data['name'],
                        "args": tc_data['args']
                    })
        
        logger.info("=" * 80)
        logger.info("LangChain Streaming Results:")
        logger.info(f"  Total chunks: {chunk_count}")
        logger.info(f"  Tool calls detected: {len(tool_calls_detected)}")
        logger.info(f"  Content chunks: {len(content_chunks)}")
        logger.info(f"  Finish reason: {finish_reason}")
        if tool_calls_detected:
            logger.info("  ✅ Tool calls were detected by LangChain!")
            for i, tc in enumerate(tool_calls_detected):
                logger.info(f"    Tool call {i}: {tc}")
        else:
            logger.warning("  ❌ No tool calls detected by LangChain!")
        logger.info("=" * 80)
        
        return {
            "chunk_count": chunk_count,
            "tool_calls": tool_calls_detected,
            "content_chunks": content_chunks,
            "finish_reason": finish_reason,
        }
        
    except Exception as e:
        logger.error(f"[LangChain Stream] Error during streaming: {e}", exc_info=True)
        raise


def _test_streaming_with_retries(provider: str, model: str, num_retries: int = 5):
    """Test streaming mode multiple times to catch random failures.

    Based on GitHub issue showing streaming tool calls can randomly fail.
    """
    logger.info("=" * 80)
    logger.info(f"Testing Streaming Mode with {num_retries} Retries")
    logger.info("(Testing for random failures as reported in vLLM issue #27641)")
    logger.info("=" * 80)

    results = []
    for attempt in range(1, num_retries + 1):
        logger.info(f"\n--- Attempt {attempt}/{num_retries} ---")
        try:
            result = _test_direct_openai_client_streaming(provider, model)
            success = bool(result["tool_calls"])
            results.append({
                "attempt": attempt,
                "success": success,
                "tool_calls_count": len(result["tool_calls"]),
                "finish_reason": result["finish_reason"],
                "chunk_count": result.get("chunk_count", 0),
            })
            logger.info(f"Attempt {attempt}: {'✅ SUCCESS' if success else '❌ FAILED'} - {len(result['tool_calls'])} tool call(s)")
        except Exception as e:
            logger.error(f"Attempt {attempt} raised exception: {e}", exc_info=True)
            results.append({
                "attempt": attempt,
                "success": False,
                "error": str(e),
            })

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Streaming Retry Test Summary")
    logger.info("=" * 80)
    successes = sum(1 for r in results if r.get("success", False))
    logger.info(f"Successes: {successes}/{num_retries}")
    logger.info(f"Failure rate: {(num_retries - successes) / num_retries * 100:.1f}%")

    for r in results:
        status = "✅" if r.get("success") else "❌"
        error_info = f" - Error: {r.get('error', 'N/A')}" if not r.get("success") and r.get("error") else ""
        logger.info(f"  {status} Attempt {r['attempt']}: {r.get('tool_calls_count', 0)} tool call(s), finish_reason={r.get('finish_reason', 'N/A')}, chunks={r.get('chunk_count', 0)}{error_info}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test vLLM/OpenRouter tool-calling via LangChain ChatOpenAI.",
    )
    parser.add_argument(
        "--provider",
        choices=["vllm", "openrouter"],
        required=True,
        help="Backend provider to test.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="Model name to use (default: openai/gpt-oss-20b).",
    )
    parser.add_argument(
        "--question",
        default="Опиши кратко возможности платформы Comindware.",
        help=(
            "User question to send to the model. "
            "Default is a generic Comindware-related question."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[
            "direct",
            "agent",
            "rag_base",
            "rag_with_summarization",
            "rag_with_update_budget",
            "rag_with_llm_manager",
            "rag_with_middlewares",
            "rag_stream_like_app",
            "raw_stream",
            "raw_non_stream",
            "stream_retry_test",
            "langchain_stream",
        ],
        default="direct",
        help=(
            "Test mode: direct ChatOpenAI.bind_tools, minimal LangChain agent, "
            "or layered RAG-like agents (base, with summarization, with "
            "update_context_budget middleware, with all middlewares, and streaming "
            "similar to app.agent_chat_handler). "
            "NEW: raw_stream, raw_non_stream, stream_retry_test, langchain_stream for vLLM diagnosis."
        ),
    )
    args = parser.parse_args()

    question = args.question
    logger.info("=" * 80)
    logger.info("Test Configuration")
    logger.info("=" * 80)
    logger.info(f"Provider: {args.provider}")
    logger.info(f"Model:    {args.model}")
    logger.info(f"Mode:     {args.mode}")
    logger.info(f"Question: {question}")
    logger.info("=" * 80)

    if args.mode in ("direct", "agent"):
        llm = _build_llm(args.provider, args.model)

        if args.mode == "direct":
            # Bind mock tool using the same pattern as our main agent code.
            logger.info("[direct] Binding tools and invoking LLM")
            llm_with_tools = llm.bind_tools([get_weather])
            msg = llm_with_tools.invoke(question)
            _describe_response(msg)
        else:
            # Minimal LangChain agent similar to our production agent, but isolated.
            logger.info("[agent] Creating minimal agent")
            llm_with_tools = llm.bind_tools([get_weather])
            agent = create_agent(
                model=llm_with_tools,
                tools=[get_weather],
                system_prompt="You are a helpful assistant that uses tools when appropriate.",
            )
            # Agent expects a list of messages as input state in simple setups.
            logger.info("[agent] Invoking agent")
            result = agent.invoke({"messages": [HumanMessage(content=question)]})
            logger.info("=== Agent result ===")
            logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            logger.info(f"Messages count: {len(result.get('messages', [])) if isinstance(result, dict) else 'N/A'}")
    else:
        # Use actual RAG agent creation pattern from app.py
        logger.info("=" * 80)
        logger.info("Building RAG agent with actual mechanics from app.py")
        logger.info("=" * 80)

        # Temporarily override provider to match test argument
        original_provider = settings.default_llm_provider
        try:
            settings.default_llm_provider = args.provider

            # Create agent using the exact same pattern as app.py
            agent = _create_rag_agent(
                provider=args.provider,
                model=args.model,
            )
        finally:
            # Restore original provider
            settings.default_llm_provider = original_provider

        # Prepare messages (matching app.py pattern)
        from rag_engine.llm.prompts import (
            USER_QUESTION_TEMPLATE_FIRST,
            USER_QUESTION_TEMPLATE_SUBSEQUENT,
        )

        # For testing, we use first message template
        wrapped_message = USER_QUESTION_TEMPLATE_FIRST.format(question=question)
        messages = [{"role": "user", "content": wrapped_message}]

        logger.info(f"[main] Prepared {len(messages)} message(s) for agent")
        logger.info(f"[main] Wrapped message length: {len(wrapped_message)} chars")

        # Track accumulated context (matching app.py)
        conversation_tokens, _ = estimate_accumulated_tokens(messages, [])
        logger.info(f"[main] Initial conversation_tokens: {conversation_tokens}")

        if args.mode in (
            "rag_base",
            "rag_with_summarization",
            "rag_with_update_budget",
            "rag_with_llm_manager",
            "rag_with_middlewares",
        ):
            # Non-streaming invoke mode
            logger.info(f"[main] Running in invoke mode: {args.mode}")
            agent_context = AgentContext(
                conversation_tokens=conversation_tokens,
                accumulated_tool_tokens=0,
            )
            result = agent.invoke({"messages": messages}, context=agent_context)
            logger.info(f"=== {args.mode} result (messages) ===")
            for i, msg in enumerate(result.get("messages", []), start=1):
                msg_type = type(msg).__name__
                content = getattr(msg, "content", "")
                tool_calls = getattr(msg, "tool_calls", None)
                logger.info(
                    f"[{i}] type={msg_type}, "
                    f"has_tool_calls={bool(tool_calls)}, "
                    f"content_length={len(str(content))}"
                )
                if tool_calls:
                    logger.info(f"      tool_calls: {len(tool_calls) if isinstance(tool_calls, list) else '?'} call(s)")
        elif args.mode == "rag_stream_like_app":
            # Streaming mode with comprehensive logging, matching app.agent_chat_handler exactly
            logger.info("=" * 80)
            logger.info("Starting streaming mode (rag_stream_like_app)")
            logger.info("=" * 80)

            tool_results = []
            answer = ""
            has_seen_tool_results = False
            tool_executing = False

            agent_context = AgentContext(
                conversation_tokens=conversation_tokens,
                accumulated_tool_tokens=0,
            )

            logger.info(f"[main] Starting agent.stream() with {len(messages)} messages")
            logger.info(f"[main] Initial agent_context: conv={agent_context.conversation_tokens}, tools={agent_context.accumulated_tool_tokens}")

            stream_chunk_count = 0
            messages_token_count = 0
            tool_calls_detected = 0
            tool_results_count = 0

            try:
                for stream_mode, chunk in agent.stream(
                    {"messages": messages},
                    context=agent_context,
                    stream_mode=["updates", "messages"],
                ):
                    stream_chunk_count += 1
                    logger.debug(f"[stream] Chunk #{stream_chunk_count}: mode={stream_mode}")

                    # Handle "messages" mode for token streaming
                    if stream_mode == "messages":
                        token, metadata = chunk
                        messages_token_count += 1
                        token_type = getattr(token, "type", "unknown")
                        token_class = type(token).__name__
                        logger.info(f"[stream] Messages token #{messages_token_count}: type={token_type}, class={token_class}")

                        # Enhanced debug logging for vLLM tool calling issues
                        # Check both AIMessage and AIMessageChunk
                        is_ai_message = token_type == "ai" or token_class in ("AIMessage", "AIMessageChunk")
                        if is_ai_message:
                            has_tool_calls = bool(getattr(token, "tool_calls", None))
                            content = str(getattr(token, "content", ""))
                            response_metadata = getattr(token, "response_metadata", {})
                            finish_reason = response_metadata.get("finish_reason", "N/A") if isinstance(response_metadata, dict) else "N/A"
                            
                            # Check content_blocks for tool_call_chunk
                            content_blocks = getattr(token, "content_blocks", None)
                            has_content_blocks = bool(content_blocks)
                            tool_call_chunks_in_blocks = 0
                            if content_blocks:
                                tool_call_chunks_in_blocks = sum(1 for block in content_blocks if block.get("type") == "tool_call_chunk")
                            
                            logger.info(
                                f"[stream] AI token: has_tool_calls={has_tool_calls}, "
                                f"content_length={len(content)}, finish_reason={finish_reason}, "
                                f"has_content_blocks={has_content_blocks}, tool_call_chunks={tool_call_chunks_in_blocks}"
                            )
                            
                            # Log full token structure for debugging
                            if messages_token_count <= 3 or has_tool_calls or tool_call_chunks_in_blocks > 0:
                                logger.debug(f"[stream] Token repr: {repr(token)}")
                                if content_blocks:
                                    logger.debug(f"[stream] Content blocks: {content_blocks}")

                            # Log tool calls in detail
                            if has_tool_calls:
                                tool_calls = getattr(token, "tool_calls", None)
                                if tool_calls:
                                    tool_calls_detected += len(tool_calls) if isinstance(tool_calls, list) else 1
                                    logger.info(f"[stream] TOOL CALLS DETECTED: {tool_calls_detected} total")
                                    if isinstance(tool_calls, list):
                                        for i, tc in enumerate(tool_calls, start=1):
                                            tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                                            logger.info(f"[stream]   Tool call #{i}: name={tc_name}")
                            
                            # Also check content_blocks for tool calls
                            if tool_call_chunks_in_blocks > 0:
                                logger.info(f"[stream] TOOL CALL CHUNKS DETECTED in content_blocks: {tool_call_chunks_in_blocks}")
                                tool_calls_detected += tool_call_chunks_in_blocks

                        # Filter out tool-related messages (matching app.py logic)
                        # 1. Tool results (type="tool") - processed internally
                        if hasattr(token, "type") and token.type == "tool":
                            tool_results.append(token.content)
                            tool_results_count += 1
                            logger.info(f"[stream] Tool result received: #{tool_results_count} total")
                            logger.debug(f"[stream] Tool result content length: {len(str(token.content))}")
                            tool_executing = False
                            has_seen_tool_results = True

                            # Update accumulated context for next tool call (matching app.py)
                            _, accumulated_tool_tokens = estimate_accumulated_tokens([], tool_results)
                            agent_context.accumulated_tool_tokens = accumulated_tool_tokens

                            logger.info(
                                f"[stream] Updated context: conv={agent_context.conversation_tokens}, "
                                f"tools={accumulated_tool_tokens} (total: {agent_context.conversation_tokens + accumulated_tool_tokens})"
                            )

                            # Parse result to get article count
                            try:
                                import json
                                result_dict = json.loads(token.content) if isinstance(token.content, str) else {}
                                articles = result_dict.get("articles", [])
                                logger.info(f"[stream] Tool result contains {len(articles)} article(s)")
                            except Exception as e:
                                logger.debug(f"[stream] Could not parse tool result: {e}")

                            continue

                        # 2. AI messages with tool_calls (when agent decides to call tools)
                        # Check both token.tool_calls AND content_blocks for tool_call_chunk
                        has_tool_calls_attr = hasattr(token, "tool_calls") and bool(token.tool_calls)
                        
                        # Get content_blocks and finish_reason for this token
                        token_content_blocks = getattr(token, "content_blocks", None)
                        has_tool_call_chunks = bool(token_content_blocks) and any(
                            block.get("type") == "tool_call_chunk" for block in token_content_blocks
                        )
                        
                        token_response_metadata = getattr(token, "response_metadata", {})
                        token_finish_reason = token_response_metadata.get("finish_reason") if isinstance(token_response_metadata, dict) else None
                        finish_reason_is_tool_calls = token_finish_reason == "tool_calls"
                        
                        # Tool call detected if any of these conditions are true
                        tool_call_detected = has_tool_calls_attr or has_tool_call_chunks or finish_reason_is_tool_calls
                        
                        if tool_call_detected:
                            if not tool_executing:
                                tool_executing = True
                                if has_tool_calls_attr:
                                    call_count = len(token.tool_calls) if isinstance(token.tool_calls, list) else "?"
                                    logger.info(f"[stream] Agent calling tool(s) via token.tool_calls: {call_count} call(s)")
                                elif has_tool_call_chunks:
                                    logger.info(f"[stream] Agent calling tool(s) via content_blocks tool_call_chunk")
                                elif finish_reason_is_tool_calls:
                                    logger.info(f"[stream] Agent calling tool(s) detected via finish_reason=tool_calls")
                                    # Check if tool_calls are now available in the token
                                    final_tool_calls = getattr(token, "tool_calls", None)
                                    if final_tool_calls:
                                        logger.info(f"[stream] Final tool_calls after finish_reason: {len(final_tool_calls) if isinstance(final_tool_calls, list) else '?'} call(s)")
                            continue

                        # 3. Only stream text content from messages WITHOUT tool_calls
                        if hasattr(token, "tool_calls") and token.tool_calls:
                            continue

                        # Process content blocks for final answer text streaming
                        if hasattr(token, "content_blocks") and token.content_blocks:
                            for block in token.content_blocks:
                                if block.get("type") == "tool_call_chunk":
                                    if not tool_executing:
                                        tool_executing = True
                                        logger.debug("[stream] Agent calling tool via chunk")
                                    continue
                                elif block.get("type") == "text" and block.get("text"):
                                    if not tool_executing:
                                        text_chunk = block["text"]
                                        answer = answer + text_chunk
                                        logger.debug(f"[stream] Text chunk: length={len(text_chunk)}, total_answer_length={len(answer)}")

                    # Handle "updates" mode for agent state updates
                    elif stream_mode == "updates":
                        logger.debug(f"[stream] Agent update: {list(chunk.keys()) if isinstance(chunk, dict) else chunk}")

                # Final summary
                logger.info("=" * 80)
                logger.info("Stream completed")
                logger.info(f"  Total chunks: {stream_chunk_count}")
                logger.info(f"  Messages tokens: {messages_token_count}")
                logger.info(f"  Tool calls detected: {tool_calls_detected}")
                logger.info(f"  Tool results: {tool_results_count}")
                logger.info(f"  Final answer length: {len(answer)}")
                logger.info("=" * 80)

                # Accumulate articles from tool results
                from rag_engine.tools import accumulate_articles_from_tool_results

                articles = accumulate_articles_from_tool_results(tool_results)
                logger.info(f"[main] Accumulated {len(articles)} article(s) from tool results")

                if not articles:
                    logger.warning("[main] Agent completed with NO retrieved articles")
                else:
                    logger.info(f"[main] Agent completed with {len(articles)} article(s)")

            except Exception as e:
                logger.error(f"[main] Error during streaming: {e}", exc_info=True)
                raise
        elif args.mode == "raw_stream":
            # Test raw OpenAI client streaming (bypasses LangChain)
            _test_direct_openai_client_streaming(args.provider, args.model)
        elif args.mode == "raw_non_stream":
            # Test raw OpenAI client non-streaming (baseline)
            _test_direct_openai_client_non_streaming(args.provider, args.model)
        elif args.mode == "stream_retry_test":
            # Test streaming with multiple retries to catch random failures
            _test_streaming_with_retries(args.provider, args.model, num_retries=5)
        elif args.mode == "langchain_stream":
            # Test LangChain ChatOpenAI streaming to diagnose tool call detection
            _test_langchain_chatopenai_streaming(args.provider, args.model)
        else:
            # Default streaming mode (simplified)
            logger.info(f"[main] Running in default streaming mode: {args.mode}")
            agent_context = AgentContext(
                conversation_tokens=conversation_tokens,
                accumulated_tool_tokens=0,
            )
            for stream_mode, chunk in agent.stream(
                {"messages": messages},
                context=agent_context,
                stream_mode=["updates", "messages"],
            ):
                logger.info(f"[stream] stream_mode={stream_mode!r}")
                if stream_mode == "messages":
                    token, metadata = chunk
                    t_type = getattr(token, "type", None)
                    t_tool_calls = getattr(token, "tool_calls", None)
                    t_content = getattr(token, "content", None)
                    logger.info(
                        f"[stream]   token.type={t_type!r}, "
                        f"has_tool_calls={bool(t_tool_calls)}, "
                        f"content_length={len(str(t_content))}"
                    )
                else:
                    logger.debug(f"[stream]   update={chunk!r}")


if __name__ == "__main__":
    main()


