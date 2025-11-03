# Clean Architecture Refactor: Agent-Driven Context Tracking

**Date**: 2025-11-03  
**Status**: ✅ Complete  
**Key Principle**: Agent counts, Tool receives

## Problem

The tool was doing the agent's job:
- ❌ Tool was counting tokens
- ❌ Tool was parsing messages  
- ❌ Tool was deduplicating articles
- ❌ Tool was doing heavy computation

**Why this was wrong**:
1. Tool is not "self-contained" - doing complex state management
2. Duplicated logic - agent also counts context for fallback
3. Tool has too many responsibilities
4. Hard to test and maintain

## Solution: Separation of Concerns

**Clear responsibility split**:

```
┌─────────────────────────────────────────────┐
│ AGENT (app.py)                              │
├─────────────────────────────────────────────┤
│ ✅ Counts conversation tokens               │
│ ✅ Tracks tool results as they arrive       │
│ ✅ Counts accumulated tool tokens           │
│ ✅ Deduplicates by kb_id                    │
│ ✅ Passes counts via runtime.config         │
└─────────────────────────────────────────────┘
                    ↓
        runtime.config.configurable = {
            conversation_tokens: 10000,
            accumulated_tool_tokens: 71000
        }
                    ↓
┌─────────────────────────────────────────────┐
│ TOOL (retrieve_context.py)                  │
├─────────────────────────────────────────────┤
│ ✅ Reads tokens from runtime.config         │
│ ✅ Passes to retriever                      │
│ ✅ Returns articles                         │
│ ✅ Simple, lightweight, fast                │
└─────────────────────────────────────────────┘
```

## Implementation

### 1. New Utility: `context_tracker.py` ✅

Centralized agent utilities for context tracking:

```python
def estimate_accumulated_tokens(
    conversation_messages: list[dict],
    tool_results: list[str],
) -> tuple[int, int]:
    """Estimate tokens from conversation and accumulated tool results.

    This is the agent's responsibility - counting context as tool calls accumulate.
    The tool should just receive this information and pass it to the retriever.
    
    Returns:
        Tuple of (conversation_tokens, accumulated_tool_tokens)
    """
    from rag_engine.llm.token_utils import count_tokens
    from rag_engine.tools.utils import parse_tool_result_to_articles

    conversation_tokens = 0
    accumulated_tool_tokens = 0

    # Count conversation history
    for msg in conversation_messages:
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            conversation_tokens += count_tokens(content)

    # Count accumulated tools (deduplicated by kb_id)
    if tool_results:
        seen_kb_ids = set()
        for tool_result in tool_results:
            articles = parse_tool_result_to_articles(tool_result)
            for article in articles:
                if article.kb_id and article.kb_id not in seen_kb_ids:
                    seen_kb_ids.add(article.kb_id)
                    accumulated_tool_tokens += count_tokens(article.content)

    return conversation_tokens, accumulated_tool_tokens
```

**Key point**: Agent utility, not tool logic!

### 2. Agent Tracks Context (`app.py`) ✅

**Before**:
```python
# Agent did nothing, tool did everything ❌
agent.stream({"messages": messages})
```

**After**:
```python
# Agent counts and passes context ✅
from rag_engine.utils.context_tracker import estimate_accumulated_tokens

# Count initial conversation
conversation_tokens, _ = estimate_accumulated_tokens(messages, [])

# Pass to agent via config
config = {
    "configurable": {
        "conversation_tokens": conversation_tokens,
        "accumulated_tool_tokens": 0,  # Updated as tool calls complete
    }
}

# Agent streams with config
for stream_mode, chunk in agent.stream(
    {"messages": messages},
    config=config,  # Tool can access via runtime.config
    stream_mode=["updates", "messages"]
):
    if is_tool_result:
        # Update accumulated context for NEXT tool call
        _, accumulated_tool_tokens = estimate_accumulated_tokens([], tool_results)
        config["configurable"]["accumulated_tool_tokens"] = accumulated_tool_tokens
```

**Benefits**:
- ✅ Agent controls context tracking
- ✅ Agent updates config as tool calls complete
- ✅ Tool just receives the numbers

### 3. Tool Receives Context (`retrieve_context.py`) ✅

**Before** (~80 lines):
```python
# Tool did heavy lifting ❌
conversation_tokens = 0
accumulated_tool_tokens = 0

messages = runtime.state.get("messages", [])
seen_kb_ids = set()

for msg in messages:
    content = get_content(msg)
    msg_type = get_type(msg)
    
    if msg_type == "tool":
        articles = parse_tool_result_to_articles(content)
        for article in articles:
            if article.kb_id not in seen_kb_ids:
                seen_kb_ids.add(article.kb_id)
                accumulated_tool_tokens += count_tokens(article.content)
    elif msg_type not in ("tool_call",):
        conversation_tokens += count_tokens(content)

total_reserved_tokens = conversation_tokens + accumulated_tool_tokens
```

**After** (~10 lines):
```python
# Tool just receives ✅
conversation_tokens = 0
accumulated_tool_tokens = 0

if runtime and hasattr(runtime, "config"):
    # Agent provides pre-calculated token counts
    configurable = runtime.config.get("configurable", {})
    conversation_tokens = configurable.get("conversation_tokens", 0)
    accumulated_tool_tokens = configurable.get("accumulated_tool_tokens", 0)

total_reserved_tokens = conversation_tokens + accumulated_tool_tokens
```

**Benefits**:
- ✅ 80+ lines → ~10 lines
- ✅ No parsing, no looping, no deduplication
- ✅ Just reads and uses
- ✅ True "self-contained" tool

## Flow Diagram

### Before (Tool Does Everything)

```
Agent:
  → stream(messages)
  → (does nothing with context)

Tool Call 1:
  → Receives runtime.state
  → Parses all messages ❌
  → Counts conversation ❌
  → No tool results yet
  → reserved_tokens = 10K
  
Tool Call 2:
  → Receives runtime.state
  → Parses all messages AGAIN ❌
  → Counts conversation AGAIN ❌
  → Finds tool_result_1, parses it ❌
  → Deduplicates articles ❌
  → Counts tokens ❌
  → reserved_tokens = 81K
  
Tool Call 3:
  → (Repeats all the work AGAIN) ❌
```

### After (Agent Does Heavy Lifting)

```
Agent:
  → Counts conversation: 10K ✅
  → config = {conversation_tokens: 10K, accumulated_tool_tokens: 0}
  → stream(messages, config)
  
Tool Call 1:
  → Reads runtime.config ✅
  → conversation_tokens = 10K (from config)
  → accumulated_tool_tokens = 0 (from config)
  → reserved_tokens = 10K
  → Returns articles
  
Agent:
  → Receives tool_result_1
  → Counts accumulated: 71K ✅
  → Updates config: {conversation_tokens: 10K, accumulated_tool_tokens: 71K}
  
Tool Call 2:
  → Reads runtime.config ✅
  → conversation_tokens = 10K (from config)
  → accumulated_tool_tokens = 71K (from config)
  → reserved_tokens = 81K
  → Returns articles
  
Agent:
  → Receives tool_result_2
  → Counts accumulated: 116K ✅
  → Updates config: {conversation_tokens: 10K, accumulated_tool_tokens: 116K}
  
Tool Call 3:
  → Reads runtime.config ✅
  → conversation_tokens = 10K (from config)
  → accumulated_tool_tokens = 116K (from config)
  → reserved_tokens = 126K
  → Returns articles
```

## Benefits

### 1. Clean Separation of Concerns ✅

| Component | Responsibility |
|-----------|----------------|
| **Agent** | Context tracking, token counting, deduplication |
| **Tool** | Receive context, retrieve articles, return JSON |
| **Utils** | Shared utilities for counting (reusable) |

### 2. Performance ✅

**Before**:
- Tool Call 1: Parse 1 message
- Tool Call 2: Parse 3 messages (message + tool_result_1 + new message)
- Tool Call 3: Parse 5 messages (accumulated)
- **Total**: 1 + 3 + 5 = 9 message parses

**After**:
- Agent: Parse messages once ✅
- Tool Call 1: Read 2 numbers from config
- Tool Call 2: Read 2 numbers from config
- Tool Call 3: Read 2 numbers from config
- **Total**: 1 parse + 6 reads (much faster!)

### 3. Simplicity ✅

**Tool complexity**:
- Before: 80+ lines of parsing, counting, deduplication
- After: ~10 lines to read config
- **Reduction**: 87% less code in tool!

### 4. Testability ✅

**Before**:
- Had to mock complex runtime.state with messages
- Had to simulate tool result structures
- Hard to test edge cases

**After**:
- Just set config values in test
- Simple, predictable, easy to test

### 5. Maintainability ✅

**Before**:
- Logic scattered between agent and tool
- Duplicated token counting (agent for fallback, tool for budgeting)
- Hard to modify

**After**:
- Centralized logic in `context_tracker.py`
- Agent uses it, tool uses results
- Single place to modify

## Files Modified

1. **`rag_engine/utils/context_tracker.py`** - NEW ✅
   - `estimate_accumulated_tokens()` - Agent utility
   - `extract_articles_from_runtime_state()` - Helper for tools

2. **`rag_engine/api/app.py`** ✅
   - Imports `estimate_accumulated_tokens`
   - Counts conversation upfront
   - Updates config after each tool result
   - Passes config to agent.stream()

3. **`rag_engine/tools/retrieve_context.py`** ✅
   - Simplified from ~80 lines to ~10 lines
   - Just reads from runtime.config
   - No parsing, no counting, no deduplication

## Comparison: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **Tool complexity** | 80+ lines | ~10 lines |
| **Logic location** | Tool | Agent |
| **Token counting** | Tool parses messages | Agent counts, passes number |
| **Deduplication** | Tool logic | Agent utility |
| **Config passing** | None | runtime.config |
| **Performance** | Repeats work each call | Work done once |
| **Testability** | Complex mocks needed | Simple config values |
| **Maintainability** | Logic scattered | Centralized in utils |

## Technical Details

### How runtime.config Works

LangGraph allows passing config to agent.stream():

```python
config = {
    "configurable": {
        "key": "value",  # Any custom data
    }
}

agent.stream(input, config=config)
```

Tools can access it via `runtime.config`:

```python
def my_tool(param: str, runtime: ToolRuntime) -> str:
    my_value = runtime.config.get("configurable", {}).get("key")
    return f"Got {my_value}"
```

**Key point**: Agent controls config, tool reads it. Perfect separation!

### Why This Architecture is Better

**Single Responsibility Principle**:
- Agent: Orchestration, state management
- Tool: Domain logic (retrieval)
- Utils: Shared calculations

**DRY (Don't Repeat Yourself)**:
- Token counting logic in one place (`context_tracker.py`)
- Both agent (for fallback) and tool (for budgeting) use same logic

**Testable**:
- Agent logic tested with real tool results
- Tool logic tested with mock config values
- Utils tested independently

## Conclusion

✅ **Clean Architecture Achieved**

**Key improvements**:
1. ✅ Agent owns context tracking
2. ✅ Tool is truly self-contained (just reads, retrieves, returns)
3. ✅ Centralized utilities for reusability
4. ✅ 87% reduction in tool complexity
5. ✅ Better performance (no repeated parsing)
6. ✅ Easier to test and maintain

The architecture now follows proper separation of concerns, with each component doing one thing well.

