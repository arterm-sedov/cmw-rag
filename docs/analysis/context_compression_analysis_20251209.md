# Context Compression Analysis: Proactive and Robust Compression for Small Context Windows

**Date**: 2025-01-28  
**Status**: Analysis Complete

## Executive Summary

The RAG agent has **reactive compression** that triggers at 85% of context window, but it is **not proactively robust** for small context windows. Compression happens AFTER tool calls complete, which means:

1. ✅ **Works well** for large context windows (100K+ tokens) with headroom
2. ⚠️ **Marginal** for medium windows (32K-100K tokens) 
3. ❌ **Not robust** for small context windows (<16K tokens)

## Current Compression Architecture

### 1. Two-Layer Compression System

#### A. Memory Compression (Conversation History)
- **Location**: `SummarizationMiddleware` in `agent_factory.py`
- **Trigger**: 80% of context window (`memory_compression_threshold_pct`)
- **Target**: Compress to ~1000 tokens (`memory_compression_target_tokens`)
- **Keeps**: Last 2 messages (`memory_compression_messages_to_keep`)
- **Timing**: Runs before each LLM call

#### B. Tool Result Compression
- **Location**: `compress_tool_results()` middleware → `compress_tool_messages()` in `compression.py`
- **Trigger**: 85% of context window (`llm_compression_threshold_pct`)
- **Target**: 80% after compression (`llm_compression_target_pct`)
- **Method**: Proportional compression by rank (higher-ranked articles get more tokens)
- **Timing**: Runs AFTER all tool calls complete, BEFORE LLM answer generation

### 2. Compression Flow

```
User Question
    ↓
Agent calls retrieve_context tool → Returns uncompressed articles
    ↓
Agent calls retrieve_context tool → Returns uncompressed articles (accumulates)
    ↓
[All tool calls complete]
    ↓
@before_model middleware: compress_tool_results()
    ├─ Count total tokens (messages + tool results + JSON overhead)
    ├─ Check if > 85% threshold
    ├─ If yes: Extract ALL articles, deduplicate, compress proportionally
    └─ Update tool messages with compressed articles
    ↓
SummarizationMiddleware: Compress conversation history if > 80%
    ↓
LLM generates answer
```

## Analysis: Robustness for Small Context Windows

### Scenario: 8K Token Context Window

**Example**: Model with `token_limit: 8192` (e.g., some Mistral variants)

#### Current Behavior:

1. **Threshold Calculation**:
   - Compression threshold: `8192 * 0.85 = 6,963 tokens`
   - Target after compression: `8192 * 0.80 = 6,554 tokens`

2. **Overhead Breakdown**:
   - System prompt: ~2,000 tokens
   - Tool schema: ~500 tokens
   - Safety margin: ~2,000 tokens
   - **Total overhead**: ~4,500 tokens
   - **Available for articles**: `6,554 - 4,500 = 2,054 tokens` (at target)

3. **Problem**: If conversation history = 1,500 tokens:
   - **Available for articles**: `6,554 - 4,500 - 1,500 = 554 tokens`
   - **Minimum per article**: 300 tokens (`llm_compression_min_tokens`)
   - **Can fit**: Only 1 article (300 tokens) ❌

4. **Fallback Logic** (line 425 in `compression.py`):
   ```python
   if available_for_articles <= 0:
       available_for_articles = max(300 * len(all_articles), int(context_window * 0.10))
   ```
   - For 8K window: `max(300 * 5, 819) = max(1500, 819) = 1500 tokens`
   - But overhead + conversation = 6,000 tokens
   - **Total needed**: 6,000 + 1,500 = 7,500 tokens > 8,192 ✅ (barely fits)
   - **But**: This assumes compression succeeds, which may not happen

### Issues Identified

#### 1. ❌ **No Proactive Prevention**

**Problem**: Compression only triggers reactively at 85% threshold. For small windows:
- Tool calls retrieve full articles without checking if they'll fit
- Multiple tool calls accumulate uncompressed articles
- Compression may fail if articles are too large relative to available budget

**Example**:
- Context window: 8K tokens
- Overhead + conversation: 6K tokens
- Available: 2K tokens
- Tool call 1: Returns 5 articles × 1K tokens = 5K tokens ❌
- Compression tries to fit 5K → 2K (60% reduction needed)
- If compression fails or is insufficient → **Context overflow**

#### 2. ⚠️ **Compression Can Fail Silently**

**Location**: `compression.py` lines 464-480

```python
if tokens_saved == 0:
    # If compression didn't help but we're still over limit, force more aggressive compression
    if current_article_tokens > available_for_articles:
        # Try again with even more aggressive target (50% of available)
        compressed_articles, tokens_saved = compress_all_articles_proportionally_by_rank(
            articles=all_articles,
            target_tokens=int(available_for_articles * 0.5),
            ...
        )
        if tokens_saved == 0:
            return None  # Still failed
```

**Problem**: If compression fails twice, returns `None` (no compression applied). The agent proceeds with potentially overflowing context.

#### 3. ⚠️ **No Early Warning System**

**Problem**: No mechanism to:
- Warn before tool calls that context is tight
- Prevent tool calls if context is already near limit
- Suggest fallback model proactively

**Current**: Fallback only checked at agent creation time (`_check_context_fallback()`), not during execution.

#### 4. ⚠️ **Minimum Token Constraint**

**Problem**: `llm_compression_min_tokens = 300` per article may be too high for small windows.

**Example**:
- Available budget: 1,000 tokens
- Articles: 5 articles
- Minimum needed: `5 × 300 = 1,500 tokens` > 1,000 ❌
- Result: Cannot compress enough, may overflow

#### 5. ⚠️ **JSON Overhead Not Accounted Proactively**

**Problem**: JSON overhead (30% by default) is added during compression check, but not considered when retrieving articles.

**Current Flow**:
1. Tool retrieves articles based on raw content size
2. Articles wrapped in JSON (adds 30% overhead)
3. Compression checks total with JSON overhead
4. May be too late if articles already retrieved

## Recommendations

### 1. ✅ **Add Proactive Context Checking**

**Location**: `retrieve_context` tool or retriever

**Implementation**:
```python
def retrieve_context(query: str, top_k: int | None = None, runtime: ToolRuntime | None = None):
    # Check available context BEFORE retrieving
    if runtime and hasattr(runtime, 'context'):
        context_window = get_context_window(runtime.model)
        conv_tokens = runtime.context.conversation_tokens
        tool_tokens = runtime.context.accumulated_tool_tokens
        overhead = compute_overhead_tokens()
        
        available = context_window - conv_tokens - tool_tokens - overhead
        
        # If available < threshold, reduce top_k proactively
        if available < context_window * 0.20:  # Less than 20% available
            logger.warning("Low context budget (%d tokens), reducing top_k", available)
            top_k = min(top_k or 5, max(1, int(available / 2000)))  # Estimate 2K per article
```

### 2. ✅ **Make Minimum Tokens Adaptive**

**Location**: `compression.py`

**Implementation**:
```python
def compress_all_articles_proportionally_by_rank(...):
    # Adaptive minimum based on available budget
    if target_tokens < 2000:  # Small context window
        min_tokens_per_article = max(100, int(target_tokens / (len(articles) * 2)))
    else:
        min_tokens_per_article = getattr(settings, "llm_compression_min_tokens", 300)
```

### 3. ✅ **Add Compression Failure Handling**

**Location**: `compression.py` and `app.py`

**Implementation**:
```python
def compress_tool_messages(...):
    # ... existing compression logic ...
    
    if tokens_saved == 0 and current_article_tokens > available_for_articles:
        # Compression failed - need to truncate or fail gracefully
        logger.error(
            "Compression failed: %d tokens needed, %d available. "
            "Truncating articles or requesting fallback model.",
            current_article_tokens,
            available_for_articles,
        )
        
        # Option 1: Truncate articles to fit
        truncated = _truncate_articles_to_fit(all_articles, available_for_articles)
        
        # Option 2: Return error signal for fallback
        # (Requires agent-level handling)
        
        return updated_messages
```

### 4. ✅ **Add Early Fallback Detection**

**Location**: `compress_tool_results()` middleware

**Implementation**:
```python
def compress_tool_results(state: dict, runtime) -> dict | None:
    messages = state.get("messages", [])
    context_window = get_context_window(runtime.model)
    total_tokens = count_messages_tokens(messages)
    
    # Early warning: if already at 90%+, suggest fallback
    if total_tokens > context_window * 0.90:
        logger.warning(
            "Context at %.1f%% of window (%d/%d tokens). "
            "Consider fallback model or aggressive compression.",
            total_tokens / context_window * 100,
            total_tokens,
            context_window,
        )
    
    # ... existing compression logic ...
```

### 5. ✅ **Add Context Window Size Detection**

**Location**: `compression.py`

**Implementation**:
```python
def compress_tool_messages(...):
    context_window = get_context_window(current_model)
    
    # Detect small context window
    is_small_window = context_window < 16_384  # 16K threshold
    
    if is_small_window:
        # More aggressive thresholds for small windows
        threshold_pct = 0.75  # Trigger earlier (75% instead of 85%)
        target_pct = 0.70    # Target lower (70% instead of 80%)
        min_tokens = max(100, int(context_window * 0.01))  # 1% of window, min 100
    else:
        threshold_pct = 0.85
        target_pct = 0.80
        min_tokens = 300
```

## Testing Recommendations

### Test Cases for Small Context Windows

1. **8K Token Window**:
   - Conversation: 1,500 tokens
   - Overhead: 4,500 tokens
   - Available: 2,000 tokens
   - Tool returns: 5 articles × 1K = 5K tokens
   - **Expected**: Compression reduces to ~2K tokens

2. **4K Token Window**:
   - Conversation: 1,000 tokens
   - Overhead: 2,500 tokens
   - Available: 500 tokens
   - Tool returns: 3 articles × 500 = 1,500 tokens
   - **Expected**: Compression reduces to ~500 tokens OR fallback triggered

3. **Multiple Tool Calls**:
   - Context: 8K tokens
   - Tool call 1: 3K tokens
   - Tool call 2: 3K tokens
   - Tool call 3: 3K tokens
   - **Expected**: Compression handles all accumulated articles

## Conclusion

### Current State: ⚠️ **Partially Robust**

**Strengths**:
- ✅ Reactive compression works for large/medium windows
- ✅ Proportional compression by rank preserves important content
- ✅ Fallback mechanism exists (though not proactive)
- ✅ JSON overhead accounted for

**Weaknesses**:
- ❌ No proactive prevention for small windows
- ❌ Compression can fail silently
- ❌ Fixed minimum tokens may be too high for small windows
- ❌ No early warning system

### Recommendation Priority

1. **HIGH**: Add proactive context checking in tool (prevent overflow before it happens)
2. **HIGH**: Make minimum tokens adaptive based on context window size
3. **MEDIUM**: Add compression failure handling (truncate or error)
4. **MEDIUM**: Add early fallback detection
5. **LOW**: Add context window size detection for adaptive thresholds

### Final Verdict

**The RAG agent is NOT proactively robust for small context windows (<16K tokens).** It relies on reactive compression that may fail if articles are too large relative to available budget. For production use with small context windows, implement the proactive checks and adaptive minimums recommended above.
