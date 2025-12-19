# Native Gradio Spinners Implementation

**Date**: December 19, 2025  
**Status**: ✅ Implemented  
**Component**: UI/UX Enhancement - Loading Indicators

## Overview

Implemented native Gradio spinner support for chat metadata messages, providing visual feedback while the agent is thinking or waiting for tool results. This leverages Gradio's built-in `status` field in message metadata to display loading indicators.

## Problem

Previously, when the agent was processing (thinking, searching, executing tools), users had no visual indication that work was in progress beyond text messages. This could make the UI feel unresponsive, especially during longer operations.

## Solution

Gradio's Chatbot component has native support for spinners through the `status` field in message metadata:
- `status: "pending"` → Shows a spinner next to the message title
- `status: "done"` → Hides the spinner (operation complete)

This is documented in Gradio's codebase:
```python
# From gradio/components/chatbot.py
status: NotRequired[Literal["pending", "done"]]
# if set to "pending", a spinner appears next to the thought title
# If "done", the thought accordion is initialized closed
```

## Implementation

### Updated Functions in `rag_engine/api/stream_helpers.py`

#### 1. `yield_thinking_block(tool_name: str)`
- **Status**: `"pending"`
- **Purpose**: Shows spinner while generic tools are executing
- **Example**: When calling `add`, `get_current_datetime`, or other non-search tools

```python
return {
    "role": "assistant",
    "content": content,
    "metadata": {
        "title": title,
        "ui_type": "thinking",
        "status": "pending",  # ← Shows spinner
    },
}
```

#### 2. `yield_search_started(query: str | None)`
- **Status**: `"pending"`
- **Purpose**: Shows spinner while searching the knowledge base
- **Example**: "🔍 Searching for: user authentication"

```python
return {
    "role": "assistant",
    "content": content,
    "metadata": {
        "title": title,
        "ui_type": "search_started",
        "status": "pending",  # ← Shows spinner
    },
}
```

#### 3. `yield_search_completed(count: int, articles: list)`
- **Status**: `"done"`
- **Purpose**: Removes spinner when search completes, shows results
- **Example**: "✅ Found 5 articles"

```python
return {
    "role": "assistant",
    "content": content,
    "metadata": {
        "title": title,
        "ui_type": "search_completed",
        "status": "done",  # ← Removes spinner
    },
}
```

#### 4. `yield_model_switch_notice(model: str)`
- **Status**: `"done"`
- **Purpose**: No spinner needed for notices (instant event)
- **Example**: "🔄 Switched to gemini-2.0-flash-exp"

#### 5. `yield_cancelled()`
- **Status**: `"done"`
- **Purpose**: Stops any pending spinners when user cancels
- **Example**: "🛑 Response cancelled by user"

#### 6. New Helper: `update_message_status_in_history()`
- **Purpose**: Update status of existing messages (pending → done)
- **Use case**: Mark thinking/search as done without replacing the message

```python
def update_message_status_in_history(
    gradio_history: list[dict],
    ui_type: str,
    new_status: str,
) -> bool:
    """Update the status of the last message with given ui_type.
    
    Args:
        gradio_history: List of Gradio message dictionaries
        ui_type: The ui_type to search for ("thinking", "search_started", etc.)
        new_status: New status value ("pending" or "done")
    
    Returns:
        True if message was updated, False otherwise
    """
```

## User Experience Flow

### Example 1: Simple Search Query

1. User asks: *"How do I configure authentication?"*
2. UI shows: **🧠 Thinking... [spinner]** (status: pending)
3. UI shows: **🔍 Searching for: authentication configuration [spinner]** (status: pending)
4. UI shows: **✅ Found 5 articles** (status: done, spinner removed)
5. UI shows: Assistant's answer with citations

### Example 2: Multi-Tool Usage

1. User asks: *"What's the date and how many articles mention workflows?"*
2. UI shows: **🧠 Thinking... [spinner]** (tool: get_current_datetime, status: pending)
3. Tool completes → spinner removed
4. UI shows: **🔍 Searching for: workflows [spinner]** (status: pending)
5. UI shows: **✅ Found 12 articles** (status: done, spinner removed)
6. UI shows: Assistant's answer

### Example 3: Cancellation

1. User asks a question
2. UI shows: **🔍 Searching... [spinner]** (status: pending)
3. User clicks Stop button
4. UI shows: **🛑 Response cancelled by user** (status: done, all spinners removed)

## Technical Details

### Status Field Behavior

The `status` field in message metadata is processed by Gradio's frontend:

```typescript
// Gradio frontend behavior
if (metadata.status === "pending") {
  showSpinner(message);
  initializeAccordionOpen(message);
} else if (metadata.status === "done") {
  hideSpinner(message);
  initializeAccordionClosed(message);
}
```

### Compatibility

- **Gradio Version**: 6.0+ (tested with native Chatbot component)
- **Backward Compatible**: Old messages without `status` field work normally
- **Browser Support**: All modern browsers (native CSS spinners)

### Integration with Existing Code

The implementation integrates seamlessly with existing patterns:

1. **No changes to `app.py` flow**: Spinners work automatically
2. **i18n Support**: Status field is language-agnostic
3. **Metadata Filtering**: UI-only messages still filtered from agent context
4. **Streaming Compatible**: Status updates work during streaming

## Benefits

1. **Native Implementation**: Uses Gradio's built-in feature (no custom CSS/JS)
2. **Automatic Behavior**: Frontend handles spinner display automatically
3. **Consistent UX**: Same spinner style as other Gradio components
4. **Zero Overhead**: No additional requests or processing needed
5. **Accessibility**: Native spinners support screen readers

## Testing Recommendations

### Manual Testing

1. **Simple Query**: Ask a question, observe spinner during search
2. **Multi-Tool**: Ask question requiring multiple tools, observe sequential spinners
3. **Cancellation**: Cancel mid-search, verify spinner disappears
4. **Long Search**: Verify spinner stays visible during long operations
5. **Error Case**: Trigger error, verify spinner disappears

### Browser Testing

Test in:
- Chrome/Edge (Chromium)
- Firefox
- Safari (if accessible)

### Accessibility Testing

- Screen reader: Verify status announcements
- Keyboard navigation: Verify accordion keyboard control
- High contrast mode: Verify spinner visibility

## Future Enhancements

### Potential Improvements

1. **Progress Percentage**: If tools support progress reporting
   ```python
   metadata["progress"] = 0.75  # 75% complete
   ```

2. **Estimated Time**: Show time remaining for long operations
   ```python
   metadata["duration"] = 5.2  # seconds elapsed
   ```

3. **Nested Thoughts**: Support hierarchical tool execution
   ```python
   metadata["parent_id"] = parent_message_id
   ```

4. **Custom Spinner Text**: Dynamic status updates
   ```python
   metadata["log"] = "Processing 3 of 10 articles..."
   ```

## References

- **Gradio Documentation**: [Chatbot Component](https://www.gradio.app/docs/chatbot)
- **Source Code**: `.reference-repos/.gradio/gradio/components/chatbot.py`
- **Implementation**: `rag_engine/api/stream_helpers.py`
- **Usage**: `rag_engine/api/app.py` (agent_chat_handler)

## Conclusion

This implementation provides clear visual feedback during agent operations using Gradio's native capabilities. The spinners automatically appear when operations start (status: "pending") and disappear when they complete (status: "done"), creating a more responsive and professional user experience.

No changes to application logic were required—only metadata updates in helper functions. The feature works seamlessly with existing streaming, i18n, and metadata filtering patterns.

