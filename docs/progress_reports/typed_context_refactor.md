# Typed Context Refactor: From `runtime.config` to `runtime.context`

## Overview

This document describes the refactoring from untyped `runtime.config` to typed `runtime.context` for passing context information from the agent to tools. This follows the official LangChain 1.0 pattern for clean, type-safe context sharing.

## Evolution

### Phase 1: Tool Sniffing State (❌ Anti-pattern)
The tool accessed `runtime.state.messages` directly to count context.

### Phase 2: Untyped Config (⚠️ Functional but verbose)
The tool read from `runtime.config["configurable"]["conversation_tokens"]` (untyped dict access).

### Phase 3: Typed Context (✅ LangChain 1.0 Official Pattern)
The tool accesses `runtime.context.conversation_tokens` (typed Pydantic model).

## Problem with Phase 2

While `runtime.config` worked, it had several issues:
- **Untyped**: No IDE autocomplete, easy to make typos
- **Verbose**: Nested `config.get("configurable", {}).get("field", 0)` calls
- **Error-prone**: No validation, silent failures on typos
- **Not idiomatic**: Not the recommended LangChain 1.0 pattern

## Solution: Typed Context

### Architecture

According to [LangChain docs](https://docs.langchain.com/oss/python/langchain/short-term-memory#write-short-term-memory-from-tools), the clean pattern is:

1. Define a Pydantic `context_schema`
2. Pass context via `agent.invoke(..., context=TypedContext(...))`
3. Tools access via `runtime.context.field` (typed!)

### Implementation Details

#### New Schema: `rag_engine/utils/context_tracker.py`

```python
from pydantic import BaseModel, Field

class AgentContext(BaseModel):
    """Typed context passed from agent to tools for progressive budgeting.
    
    This context is passed via the `context` parameter in agent.invoke()
    and accessed by tools via `runtime.context` for clean, typed access.
    Follows LangChain 1.0 official pattern for context sharing.
    """
    
    conversation_tokens: int = Field(
        default=0,
        description="Tokens used by conversation history (user/assistant messages)",
    )
    
    accumulated_tool_tokens: int = Field(
        default=0,
        description="Tokens accumulated from previous tool calls in this turn (deduplicated)",
    )
```

#### Updated: `rag_engine/api/app.py`

**Agent creation:**
```python
from rag_engine.utils.context_tracker import AgentContext

agent = create_agent(
    model=model_with_tools,
    tools=[retrieve_context],
    system_prompt=SYSTEM_PROMPT,
    context_schema=AgentContext,  # Define typed context schema
    middleware=[...],
)
```

**Agent invocation:**
```python
# Create typed context
agent_context = AgentContext(
    conversation_tokens=conversation_tokens,
    accumulated_tool_tokens=0,  # Updated as we go
)

# Pass via context parameter (not config!)
for stream_mode, chunk in agent.stream(
    {"messages": messages},
    context=agent_context,  # Typed, validated!
    stream_mode=["updates", "messages"]
):
    # Update context after each tool result
    agent_context.accumulated_tool_tokens = accumulated_tool_tokens
```

#### Updated: `rag_engine/tools/retrieve_context.py`

**Tool signature:**
```python
from rag_engine.utils.context_tracker import AgentContext

@tool("retrieve_context", args_schema=RetrieveContextSchema)
def retrieve_context(
    query: str,
    top_k: int | None = None,
    runtime: ToolRuntime[AgentContext, None] | None = None,  # Typed!
) -> str:
```

**Tool implementation:**
```python
# Clean, typed access!
if runtime and hasattr(runtime, "context") and runtime.context:
    conversation_tokens = runtime.context.conversation_tokens
    accumulated_tool_tokens = runtime.context.accumulated_tool_tokens

total_reserved_tokens = conversation_tokens + accumulated_tool_tokens
docs = retriever.retrieve(query, top_k=top_k, reserved_tokens=total_reserved_tokens)
```

## Code Comparison

**Before (untyped config):**
```python
if runtime and hasattr(runtime, "config"):
    configurable = runtime.config.get("configurable", {})
    conversation_tokens = configurable.get("conversation_tokens", 0)
    accumulated_tool_tokens = configurable.get("accumulated_tool_tokens", 0)
```

**After (typed context):**
```python
if runtime and hasattr(runtime, "context") and runtime.context:
    conversation_tokens = runtime.context.conversation_tokens
    accumulated_tool_tokens = runtime.context.accumulated_tool_tokens
```

**Improvement**: 3 lines → 2 lines, typed access, no nested dicts!

## Key Benefits

1. **Type Safety**: Pydantic validation ensures correct data types
2. **IDE Autocomplete**: `runtime.context.` shows available fields
3. **Cleaner Syntax**: Direct attribute access vs nested dict lookups
4. **Catch Typos**: `runtime.context.conversaton_tokens` (typo) raises `AttributeError` immediately
5. **Self-Documenting**: Field descriptions explain what each value represents
6. **Official Pattern**: Follows [LangChain 1.0 docs](https://docs.langchain.com/oss/python/langchain/short-term-memory#write-short-term-memory-from-tools)
7. **Future-Proof**: Standard LangChain pattern, less likely to break

## Distinction: Three Ways to Pass Data to Tools

According to [LangChain docs](https://docs.langchain.com/oss/python/langchain/short-term-memory#read-short-term-memory-in-a-tool):

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| `runtime.state` | Read agent's state | Direct access | ❌ Tool sniffs state (anti-pattern) |
| `runtime.config` | Pass config dicts | Works | ⚠️ Untyped, verbose, error-prone |
| `runtime.context` | Pass typed data | ✅ Typed, clean, official | Requires schema definition |

**Our choice**: `runtime.context` for clean, typed, idiomatic LangChain 1.0 code.

## Files Changed

### Modified
- `rag_engine/utils/context_tracker.py` - Added `AgentContext` Pydantic schema
- `rag_engine/api/app.py` - Use `context_schema` and `context` parameter
- `rag_engine/tools/retrieve_context.py` - Type hint `runtime` and access `runtime.context`

### Tests
- All existing tests pass (behavior identical, only API changed)

## Migration Guide

If you have other tools that need context, follow this pattern:

1. Import `AgentContext` from `rag_engine.utils.context_tracker`
2. Type your runtime parameter: `runtime: ToolRuntime[AgentContext, None]`
3. Access fields: `runtime.context.field_name`
4. The agent will automatically pass the context to all tools

## Example: Adding a New Context Field

```python
# 1. Update AgentContext schema
class AgentContext(BaseModel):
    conversation_tokens: int = Field(default=0)
    accumulated_tool_tokens: int = Field(default=0)
    user_id: str = Field(default="", description="Current user ID")  # NEW

# 2. Agent passes it
agent_context = AgentContext(
    conversation_tokens=conv_tokens,
    accumulated_tool_tokens=tool_tokens,
    user_id=session_id,  # NEW
)

# 3. Tools access it
@tool
def my_tool(runtime: ToolRuntime[AgentContext, None]) -> str:
    user_id = runtime.context.user_id  # Typed, autocompleted!
    ...
```

## References

- [LangChain: Write short-term memory from tools](https://docs.langchain.com/oss/python/langchain/short-term-memory#write-short-term-memory-from-tools)
- [LangChain: Read short-term memory in a tool](https://docs.langchain.com/oss/python/langchain/short-term-memory#read-short-term-memory-in-a-tool)
- [LangChain: Customizing agent memory](https://docs.langchain.com/oss/python/langchain/short-term-memory#customizing-agent-memory)

## Related Documents

- [Clean Architecture Refactor](./clean_architecture_refactor.md) - Previous refactor from state sniffing to config
- [Progressive Budgeting Fix](./progressive_budgeting_fix.md) - Implementation of progressive context tracking
- [Article Deduplication Fix](./article_deduplication_fix.md) - Deduplication logic centralized in agent

---

**Date**: 2025-11-03  
**Status**: ✅ Completed  
**Pattern**: LangChain 1.0 Official (`runtime.context` with Pydantic)

