# Tool Architecture Fix: Self-Contained `retrieve_context`

**Date**: 2025-11-03  
**Status**: ✅ Complete  
**Issue**: Tool was trying to do too much - tracking and deduplicating tool results

## Problem

The `retrieve_context` tool was trying to be "smart" by tracking tool results from other calls:

```python
# WRONG: Tool trying to track OTHER tool calls
if tool_results_for_dedup:
    for tool_result in tool_results_for_dedup:
        articles = parse_tool_result_to_articles(tool_result)
        for article in articles:
            if article.kb_id not in seen_kb_ids:
                ...
```

**Why this was wrong**:
1. ❌ Tool results are **ephemeral** - only exist during agent execution
2. ❌ Tool results are **NOT stored** in conversation history
3. ❌ Tool can't predict OTHER tool calls in the SAME turn
4. ❌ This is the AGENT's responsibility, not the tool's

## Correct Architecture

### Tool's Responsibility (`retrieve_context`):
✅ Count conversation history (user + assistant messages)  
✅ Pass `reserved_tokens` to retriever  
✅ Return articles as JSON  
✅ **That's it!** Keep it simple and self-contained

### Agent's Responsibility (`agent_chat_handler`):
✅ Call tool multiple times  
✅ Accumulate tool results  
✅ **Deduplicate articles** using `accumulate_articles_from_tool_results()`  
✅ Pass deduplicated articles to LLM

## Implementation

### Fixed `retrieve_context.py`

**Before** (80+ lines, complex):
```python
# WRONG: Trying to track and deduplicate tool results
conversation_tokens = 0
tool_result_tokens = 0

for msg in messages:
    if is_tool_result(msg):
        tool_results_for_dedup.append(content)
    else:
        conversation_tokens += count_tokens(content)

# Complex deduplication logic...
if tool_results_for_dedup:
    seen_kb_ids = set()
    for tool_result in tool_results_for_dedup:
        articles = parse_tool_result_to_articles(tool_result)
        for article in articles:
            if article.kb_id not in seen_kb_ids:
                unique_article_content.append(article.content)
                seen_kb_ids.add(article.kb_id)
    
    tool_result_tokens = sum(count_tokens(c) for c in unique_article_content)
    tool_result_tokens = int(tool_result_tokens * 1.3)

total_reserved_tokens = conversation_tokens + tool_result_tokens
```

**After** (~30 lines, simple):
```python
# CORRECT: Only count conversation history
conversation_tokens = 0

if runtime and hasattr(runtime, "state"):
    from rag_engine.llm.token_utils import count_tokens

    messages = runtime.state.get("messages", [])

    for msg in messages:
        if hasattr(msg, "content"):
            content = msg.content
        else:
            content = msg.get("content", "") if isinstance(msg, dict) else ""

        if isinstance(content, str) and content:
            # Skip tool-related messages - only count user/assistant conversation
            msg_type = getattr(msg, "type", None)
            if msg_type not in ("tool", "tool_call"):
                conversation_tokens += count_tokens(content)

# Pass conversation history to retriever for budgeting
docs = retriever.retrieve(query, top_k=top_k, reserved_tokens=conversation_tokens)
```

### Agent Handles Deduplication

**In `agent_chat_handler` (app.py)**:
```python
# Accumulate articles from tool results and add citations
from rag_engine.tools import accumulate_articles_from_tool_results

# This function deduplicates by kb_id automatically
articles = accumulate_articles_from_tool_results(tool_results)

# Format with citations
final_text = format_with_citations(answer, articles)
```

## Flow Diagram

### Before (Wrong)

```
Turn 1:
┌─────────────────────────────────────────────┐
│ User: "Tell me about features"              │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Tool Call 1: retrieve_context("features")   │
│   - Counts conversation: 10K tokens         │
│   - Tries to count tool results: 0K (none)  │  ❌ Unnecessary
│   - Returns: Article A, B, C                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Tool Call 2: retrieve_context("integrations")│
│   - Counts conversation: 10K tokens         │
│   - Tries to dedupe tool results from Call 1│  ❌ Can't access!
│   - Returns: Article A (dup), D, E          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Agent: Accumulate → deduplicate manually?   │  ❌ Extra work
└─────────────────────────────────────────────┘
```

### After (Correct)

```
Turn 1:
┌─────────────────────────────────────────────┐
│ User: "Tell me about features"              │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Tool Call 1: retrieve_context("features")   │
│   - Counts conversation: 10K tokens         │  ✅ Simple!
│   - Returns: Article A, B, C                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Tool Call 2: retrieve_context("integrations")│
│   - Counts conversation: 10K tokens         │  ✅ Simple!
│   - Returns: Article A (dup), D, E          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Agent: accumulate_articles_from_tool_results│  ✅ Automatic dedup!
│   - Input: [A,B,C], [A,D,E]                 │
│   - Output: [A,B,C,D,E] (deduplicated)      │
└─────────────────────────────────────────────┘
```

## Benefits

### 1. Correct Separation of Concerns ✅

| Component | Responsibility |
|-----------|----------------|
| **Tool** | Count conversation history, retrieve articles |
| **Agent** | Accumulate results, deduplicate, generate answer |

### 2. Simplified Tool Logic ✅

- **Before**: 80+ lines with complex deduplication
- **After**: ~30 lines of simple counting
- **Result**: Easier to test, maintain, understand

### 3. Better Testability ✅

Tool tests don't need to mock complex message states:
```python
# Simple test: conversation history → reserved tokens
def test_retrieve_with_history():
    runtime = MockRuntime(messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ])
    
    result = retrieve_context.invoke({
        "query": "test",
        "runtime": runtime
    })
    
    # Tool just counts conversation, that's it!
    assert "articles" in result
```

### 4. Correct Data Flow ✅

```
Conversation History (persisted)
    ↓
retrieve_context counts it
    ↓
Passes reserved_tokens to retriever
    ↓
Returns articles (may include duplicates)
    ↓
Agent accumulates from multiple calls
    ↓
Agent deduplicates by kb_id
    ↓
LLM sees clean, unique articles ✅
```

## Key Insight

**Tool results are ephemeral** - they only exist during agent execution and are **NOT** stored in conversation history:

```python
# Conversation history stored by llm_manager:
llm_manager._conversations.append(session_id, "user", message)      # ✅ Stored
llm_manager.save_assistant_turn(session_id, final_answer)          # ✅ Stored

# Tool results are NOT stored:
tool_result = retrieve_context.invoke({"query": "..."})            # ❌ NOT stored
# Tool results are internal to agent execution, not persisted!
```

Therefore:
- Tool can't deduplicate across calls (no access to other tool results yet)
- Agent can deduplicate (has all tool results in memory during turn)
- Next turn: only conversation history matters, tool results are gone

## Comparison

| Aspect | Before | After |
|--------|--------|-------|
| **Tool lines of code** | ~80 | ~30 |
| **Tool complexity** | High (dedup logic) | Low (simple counting) |
| **Architecture** | Tool does agent's job | Clean separation |
| **Testability** | Hard (complex mocks) | Easy (simple mocks) |
| **Maintainability** | Poor (scattered logic) | Good (single responsibility) |
| **Correctness** | Wrong (can't access other tool results) | Correct (agent deduplicates) |

## Conclusion

✅ **Problem Solved**: Tool is now self-contained and simple

**Key Changes**:
1. ✅ Tool only counts conversation history
2. ✅ Tool doesn't try to deduplicate
3. ✅ Agent handles deduplication (where it belongs)
4. ✅ Cleaner, simpler, more maintainable code

The `retrieve_context` tool now follows the **Single Responsibility Principle** - it retrieves articles based on conversation context, nothing more.

