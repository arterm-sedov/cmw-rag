# vLLM Streaming Tool Calls Issue

**Date**: December 16, 2025  
**Status**: ✅ Resolved with workaround

## Problem

When using vLLM provider with streaming mode (`agent.stream()`), tool calls are not detected. The agent completes without retrieving any articles from the knowledge base, even when `tool_choice="retrieve_context"` is configured.

## Root Cause

**vLLM does NOT respect `tool_choice` parameter in streaming mode.**

### Key Findings:
- ✅ vLLM supports streaming tool calls when the model **decides** to call tools
- ✅ Raw OpenAI client streaming: Tool calls work reliably (100% success rate in tests)
- ✅ LangChain ChatOpenAI streaming: Tool calls detected via `content_blocks` and `finish_reason`
- ❌ **vLLM ignores `tool_choice="retrieve_context"` in streaming mode**
- ❌ Agent framework returns `finish_reason=stop` instead of `finish_reason=tool_calls` when `tool_choice` is set

### Evidence from Logs:
```
All 49 chunks: has_tool_calls=False, content_length=0, has_content_blocks=False
Final chunk: finish_reason=stop (NOT tool_calls)
Agent configured with: tool_choice="retrieve_context"
Result: No tool calls executed, agent completes without retrieval
```

## Test Results

### Raw OpenAI Client Tests (Bypassing LangChain)

#### Non-Streaming Mode
- **Result**: ✅ Success
- Tool calls detected correctly
- `finish_reason: 'tool_calls'`
- Tool call arguments parsed correctly

#### Streaming Mode
- **Result**: ✅ Success (100% success rate, 5/5 attempts)
- Tool calls arrive incrementally via `delta.tool_calls` chunks
- Successfully accumulated from multiple chunks
- Arguments parsed correctly: `{'city': 'Moscow'}`
- Finish reason: `tool_calls`

### LangChain Agent Streaming Test
- **Result**: ❌ Failure
- No tool calls detected in `agent.stream()` mode
- All chunks have `has_tool_calls=False`, `content_length=0`
- Issue: vLLM ignores `tool_choice` in streaming mode

## Solution

### Current Implementation (Required Workaround)

Modified `agent_chat_handler` in `rag_engine/api/app.py` to:
1. Detect vLLM provider via `settings.default_llm_provider`
2. For vLLM with no existing tool results: Use `agent.invoke()` to get tool calls (non-streaming)
3. Extract tool results and final answer from invoke() result
4. Simulate streaming by yielding answer in chunks for UX
5. Fallback to full streaming for other providers (OpenRouter, Gemini)

**Note:** This workaround is **required** because vLLM doesn't respect `tool_choice` in streaming mode. The fallback can be disabled for testing, but tools won't execute without it.

### Configuration

Control the fallback behavior via environment variable:

```bash
# Enable fallback (default - ensures tools execute with vLLM)
VLLM_STREAMING_FALLBACK_ENABLED=true

# Disable fallback (for testing - tools won't execute with vLLM)
VLLM_STREAMING_FALLBACK_ENABLED=false
```

### Impact on Other Providers

✅ **No impact** - The workaround is isolated to vLLM only:
- OpenRouter: Uses normal streaming, no fallback
- Gemini: Uses normal streaming, no fallback
- Other providers: Unchanged behavior

## Code Changes

- Added vLLM detection: `is_vllm_provider = settings.default_llm_provider.lower() == "vllm"`
- Added fallback logic: Falls back to `invoke()` if tool calls aren't detected in stream
- Enhanced tool call detection: Checks `content_blocks`, `finish_reason`, and `token.tool_calls`
- Added configuration: `VLLM_STREAMING_FALLBACK_ENABLED` setting

## Next Steps

1. ✅ **COMPLETED:** Confirmed vLLM streaming tool calls work when model decides to call tools
2. ✅ **COMPLETED:** Confirmed vLLM ignores `tool_choice` parameter in streaming mode
3. ✅ **COMPLETED:** Implemented fallback to `invoke()` for forced tool execution
4. ⏳ **FUTURE:** Monitor vLLM updates for `tool_choice` support in streaming mode
5. ⏳ **FUTURE:** Consider alternative approaches if vLLM never supports `tool_choice` in streaming

## Related Files

- `rag_engine/api/app.py` - Main agent handler with fallback logic
- `rag_engine/llm/agent_factory.py` - Agent creation with `tool_choice`
- `rag_engine/config/settings.py` - Configuration settings
- `rag_engine/scripts/test_vllm_tool_calling.py` - Test script

## Test Commands

```bash
# Test raw streaming (bypasses LangChain)
python -m rag_engine.scripts.test_vllm_tool_calling --provider vllm --mode raw_stream

# Test raw non-streaming (baseline)
python -m rag_engine.scripts.test_vllm_tool_calling --provider vllm --mode raw_non_stream

# Test LangChain streaming (shows the issue)
python -m rag_engine.scripts.test_vllm_tool_calling --provider vllm --mode rag_stream_like_app

# Test LangChain ChatOpenAI streaming (works)
python -m rag_engine.scripts.test_vllm_tool_calling --provider vllm --mode langchain_stream
```

## References

- vLLM GitHub: https://github.com/vllm-project/vllm
- LangChain Streaming: https://docs.langchain.com/oss/python/langchain/streaming
- vLLM Streaming Tool Calls Issue: https://github.com/vllm-project/vllm/issues/27641
- vLLM OpenAI Cookbook: https://cookbook.openai.com/articles/gpt-oss/run-vllm
- vLLM Streaming Example: https://docs.vllm.ai/en/latest/examples/online_serving/openai_chat_completion_client_with_tools_xlam_streaming/
