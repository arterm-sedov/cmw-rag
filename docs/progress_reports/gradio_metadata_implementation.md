# Gradio Metadata Messages for Tool Execution Status

## Date
November 2, 2025

## Overview
Implemented real-time tool execution status messages using Gradio's metadata feature, following the [official Gradio agents pattern](https://www.gradio.app/4.44.1/guides/agents-and-tool-usage#the-metadata-key).

## User Experience

### Before Implementation
```
User: How to configure authentication?
[waiting...]
Agent: Based on the articles I found...
```

No visual feedback during retrieval, user doesn't know what's happening.

### After Implementation
```
User: How to configure authentication?

🔍 Searching information in the knowledge base [collapsible]

✅ Found 3 articles [collapsible]

Agent: Based on the articles I found...
```

Clear visual feedback at each stage of agent execution.

## Implementation Details

### Code Changes

```python
# Track tool calls - emit metadata message
if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
    tool_name = latest_msg.tool_calls[0].get("name", "retrieve_context")
    logger.debug("Agent calling tool: %s", tool_name)
    
    # Emit status message to Gradio with metadata
    status_msg = {
        "role": "assistant",
        "content": "",
        "metadata": {"title": "🔍 Searching information in the knowledge base"}
    }
    # In Gradio, this will appear as a collapsible message
    messages.append(status_msg)
    continue

# Track tool results - emit completion metadata
if hasattr(latest_msg, "type") and latest_msg.type == "tool":
    tool_results.append(latest_msg.content)
    logger.debug("Tool result received, %d total results", len(tool_results))
    
    # Parse result to get article count
    try:
        import json
        result = json.loads(latest_msg.content)
        articles_count = result.get("metadata", {}).get("articles_count", 0)
        
        # Emit completion message with metadata
        completion_msg = {
            "role": "assistant",
            "content": "",
            "metadata": {"title": f"✅ Found {articles_count} article{'s' if articles_count != 1 else ''}"}
        }
        messages.append(completion_msg)
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, emit generic completion message
        completion_msg = {
            "role": "assistant",
            "content": "",
            "metadata": {"title": "✅ Search completed"}
        }
        messages.append(completion_msg)
    
    continue
```

### Message Format

Per [Gradio documentation](https://www.gradio.app/4.44.1/guides/agents-and-tool-usage#the-metadata-key), the metadata key accepts a dictionary with a `title` key:

```python
{
    "role": "assistant",
    "content": "",  # Empty for status-only messages
    "metadata": {
        "title": "Message shown in collapsible box"
    }
}
```

## Features

### 1. Start Message
- **Trigger**: When agent decides to call a tool
- **Message**: "🔍 Searching information in the knowledge base"
- **Icon**: 🔍 (magnifying glass) indicates search in progress
- **Display**: Collapsible message box in Gradio UI

### 2. Completion Message
- **Trigger**: When tool returns results
- **Message**: "✅ Found X article(s)" or "✅ Search completed"
- **Icon**: ✅ (checkmark) indicates successful completion
- **Display**: Collapsible message box in Gradio UI
- **Dynamic**: Shows actual article count from tool result

### 3. Error Handling
- If JSON parsing fails → shows generic "✅ Search completed"
- Gracefully handles missing metadata fields
- Doesn't break agent execution if metadata fails

## Benefits

### 1. Improved User Experience
- **Transparency**: Users see what the agent is doing
- **Perceived Performance**: Visual feedback reduces perceived wait time
- **Trust**: Showing tool execution builds user confidence
- **Debugging**: Users can see if retrieval succeeded

### 2. Professional UI
- Follows [Gradio best practices](https://www.gradio.app/4.44.1/guides/agents-and-tool-usage) for agent UIs
- Matches patterns used by other AI assistants
- Clean, collapsible messages don't clutter the chat
- Emoji indicators provide quick visual scanning

### 3. Extensibility
- Easy to add more tool status messages
- Can show different messages for different tools
- Can include additional metadata (e.g., query used, time taken)
- Framework supports rich metadata beyond just titles

## Comparison with Other Implementations

### Gradio's LangChain Example
From their [documentation](https://www.gradio.app/4.44.1/guides/agents-and-tool-usage#a-real-example-using-langchain-agents):

```python
messages.append(ChatMessage(
    role="assistant",
    content=step.action.log,
    metadata={"title": f"🛠️ Used tool {step.action.tool}"}
))
```

**Our approach is similar but optimized for RAG:**
- Clearer messages specific to knowledge base search
- Dynamic article count in completion message
- Better error handling for JSON parsing

### Transformers Agents Example
Gradio also shows patterns for transformers agents with similar metadata usage.

**Our implementation is consistent** with these established patterns.

## Testing

All existing tests pass with the new metadata feature:

```bash
pytest rag_engine/tests/test_agent_handler.py -v
```

**Result**: ✅ 10/10 tests passing

The metadata messages:
- Don't break existing functionality
- Don't interfere with streaming responses
- Don't affect citation generation
- Are purely additive UX improvements

## References

1. [Gradio Agents & Tool Usage - The metadata key](https://www.gradio.app/4.44.1/guides/agents-and-tool-usage#the-metadata-key)
2. [Gradio's LangChain Agent Example](https://www.gradio.app/4.44.1/guides/agents-and-tool-usage#a-real-example-using-langchain-agents)
3. [LangChain Middleware Documentation](https://docs.langchain.com/oss/python/langchain/middleware)

## Code Location

- **File**: `rag_engine/api/app.py`
- **Function**: `agent_chat_handler()`
- **Lines**: 147-189 (metadata message emission logic)

## Future Enhancements

Possible improvements:
1. **Time tracking**: Show elapsed time for retrieval
2. **Query display**: Show the actual query sent to retriever
3. **Multiple tools**: Different icons/messages for different tools
4. **Error metadata**: Show error details in metadata on failure
5. **Progress updates**: For long-running retrievals, show incremental progress

## Conclusion

The implementation follows official Gradio patterns and provides professional-grade visual feedback for agent tool execution. Users can now see:
- When retrieval starts
- When retrieval completes
- How many articles were found

This improves transparency, trust, and user experience without affecting the core functionality.

