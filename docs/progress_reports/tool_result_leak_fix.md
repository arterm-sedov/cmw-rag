# Tool Result Leak Fix - Progress Report

**Date**: 2025-11-03  
**Status**: ✅ Complete

## Issue

Tool results (JSON output from `retrieve_context`) were being displayed in the Gradio chat interface, showing the entire article content to users instead of only the final answer with citations.

## Root Cause

The streaming loop was processing tool result messages but not explicitly preventing them from being displayed. When `token.type == "tool"`, the content was being tracked for internal use but the message was still being processed through the rest of the streaming logic.

## Solution

Modified `rag_engine/api/app.py` to:

1. **Move tool result handling to the top** of the `if stream_mode == "messages"` block
2. **Add explicit `continue` statement** after processing tool results to skip further processing
3. **Add clear documentation** explaining that tool results are for internal use only

### Code Changes

```python
# Track tool results but DO NOT yield them to chat
# Tool results are processed internally for citation generation
if hasattr(token, "type") and token.type == "tool":
    tool_results.append(token.content)
    logger.debug("Tool result received, %d total results", len(tool_results))
    tool_executing = False

    # Parse result to get article count and emit completion metadata
    try:
        result = json.loads(token.content)
        articles_count = result.get("metadata", {}).get("articles_count", 0)

        # Yield completion metadata to Gradio
        yield {
            "role": "assistant",
            "content": "",
            "metadata": {"title": f"✅ Found {articles_count} article{'s' if articles_count != 1 else ''}"}
        }
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, emit generic completion
        yield {
            "role": "assistant",
            "content": "",
            "metadata": {"title": "✅ Search completed"}
        }
    
    # Skip further processing of tool messages
    continue
```

## Result

✅ Tool results are now processed internally only  
✅ Users see only metadata messages ("🔍 Searching..." and "✅ Found X articles")  
✅ Final answer with citations is displayed correctly  
✅ Token streaming remains functional  
✅ All tests pass (10/10)  

## Files Modified

- `rag_engine/api/app.py` - Updated `agent_chat_handler` streaming loop

## Testing

All agent handler tests pass:
```
rag_engine/tests/test_agent_handler.py::TestCreateRagAgent::test_create_agent_gemini PASSED
rag_engine/tests/test_agent_handler.py::TestCreateRagAgent::test_create_agent_openrouter PASSED
rag_engine/tests/test_agent_handler.py::TestCreateRagAgent::test_system_prompt_uses_standard_prompt PASSED
rag_engine/tests/test_agent_handler.py::TestAgentChatHandler::test_agent_handler_empty_message PASSED
rag_engine/tests/test_agent_handler.py::TestAgentChatHandler::test_agent_handler_success_with_articles PASSED
rag_engine/tests/test_agent_handler.py::TestAgentChatHandler::test_agent_handler_no_articles PASSED
rag_engine/tests/test_agent_handler.py::TestAgentChatHandler::test_agent_handler_error_handling PASSED
rag_engine/tests/test_agent_handler.py::TestAgentChatHandler::test_agent_handler_with_history PASSED
rag_engine/tests/test_agent_handler.py::TestAgentIntegration::test_handler_selection_agent_mode PASSED
rag_engine/tests/test_agent_handler.py::TestAgentIntegration::test_handler_selection_direct_mode PASSED
```

## Impact

- **User Experience**: ✅ Clean chat interface showing only relevant information
- **Functionality**: ✅ Preserved (all internal processing remains unchanged)
- **Performance**: ✅ No impact
- **Backward Compatibility**: ✅ Maintained

## Next Steps

None required. The fix is complete and validated.

