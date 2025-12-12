# Memory Management Limits: Application, Timing, and Precedence

**Date**: 2025-01-28  
**Status**: Complete Documentation

## Overview

The RAG system uses a multi-layered memory management system with different thresholds and limits that apply at different stages of the conversation and tool execution pipeline. This document explains how each limit is applied, when it triggers, their relative precedence, and compares configuration values against defaults.

---

## Configuration Parameters

### Memory Compression (Conversation History)

#### `MEMORY_COMPRESSION_THRESHOLD_PCT=80`
- **Type**: Integer percentage (0-100)
- **Default**: 80
- **Purpose**: Triggers compression of conversation history when total tokens exceed this percentage of the context window
- **Location**: `rag_engine/config/settings.py:84`
- **Applied in**: `SummarizationMiddleware` in `rag_engine/llm/agent_factory.py:69-110`
- **Code**: `threshold_tokens = int(context_window * (settings.memory_compression_threshold_pct / 100))`
- **When**: **BEFORE each LLM call** (runs as middleware before model invocation)
- **Calculation**: `threshold_tokens = context_window * (memory_compression_threshold_pct / 100)`
- **Example**: For 262K window: `262,144 * 0.80 = 209,715 tokens`

#### `MEMORY_COMPRESSION_TARGET_TOKENS=1000`
- **Type**: Integer (absolute token count)
- **Default**: 1000
- **Purpose**: Target size for the compressed conversation history summary
- **Location**: `rag_engine/config/settings.py:86`
- **Applied in**: `SummarizationMiddleware` (LangChain built-in)
- **When**: When memory compression is triggered
- **Behavior**: Compresses old conversation history to approximately 1000 tokens, keeping recent messages uncompressed

#### `MEMORY_COMPRESSION_MESSAGES_TO_KEEP=2`
- **Type**: Integer (absolute count)
- **Default**: 2
- **Purpose**: Number of recent messages to keep uncompressed during compression
- **Location**: `rag_engine/config/settings.py:88`
- **Applied in**: `rag_engine/llm/agent_factory.py:88`
- **Code**: `messages_to_keep = getattr(settings, "memory_compression_messages_to_keep", 2)`
- **When**: During memory compression
- **Behavior**: Preserves the last N messages (default: 2) in their original form

---

### Context Thresholds (Pre/Post Checks)

#### `LLM_PRE_CONTEXT_THRESHOLD_PCT=0.90`
- **Type**: Float (0.0-1.0)
- **Default**: 0.90 (90%)
- **Purpose**: Safety threshold for checking context size **BEFORE** agent execution (pre-agent check)
- **Location**: `rag_engine/config/settings.py:92`
- **Applied in**:
  - `rag_engine/api/app.py:_check_context_fallback()` (line 107)
  - `rag_engine/llm/fallback.py:check_context_fallback()` (line 97)
- **Code**: `pre_pct = float(getattr(settings, "llm_pre_context_threshold_pct", None) or 0.90)`
- **When**: **BEFORE agent starts processing** (pre-flight check)
- **Calculation**: `pre_threshold = context_window * llm_pre_context_threshold_pct`
- **Action**: If exceeded, triggers context fallback to a larger model (if enabled)
- **Example**: For 262K window: `262,144 * 0.90 = 235,930 tokens`

#### `LLM_POST_CONTEXT_THRESHOLD_PCT=0.80`
- **Type**: Float (0.0-1.0)
- **Default**: 0.80 (80%)
- **Purpose**: Safety threshold for checking context size **AFTER** tool calls complete (post-tool check)
- **Location**: `rag_engine/config/settings.py:93`
- **Applied in**: `rag_engine/utils/context_tracker.py:compute_thresholds()` (line 226)
- **Code**: `return int(window * pre_pct), int(window * post_pct)`
- **When**: **AFTER tool calls complete**, before final LLM answer generation
- **Calculation**: `post_threshold = context_window * llm_post_context_threshold_pct`
- **Action**: Used for validation and monitoring (not directly triggering compression, but used in context tracking)
- **Example**: For 262K window: `262,144 * 0.80 = 209,715 tokens`

---

### Tool Results Compression

#### `LLM_COMPRESSION_THRESHOLD_PCT=0.85`
- **Type**: Float (0.0-1.0)
- **Default**: 0.85 (85%)
- **Purpose**: Triggers compression of tool results (retrieved articles) when total context exceeds this percentage
- **Location**: `rag_engine/config/settings.py:109`
- **Applied in**:
  - `rag_engine/api/app.py:compress_tool_results()` (line 158)
  - `rag_engine/llm/compression.py:compress_tool_messages()` (line 288)
- **Code**: `threshold = int(context_window * threshold_pct)`
- **When**: **AFTER all tool calls complete**, **BEFORE** LLM answer generation (runs as `@before_model` middleware)
- **Calculation**: `threshold = context_window * llm_compression_threshold_pct`
- **Note**: Uses **adjusted token count** including JSON overhead (see below)
- **Example**: For 262K window: `262,144 * 0.85 = 222,822 tokens`

#### `LLM_COMPRESSION_TARGET_PCT=0.80`
- **Type**: Float (0.0-1.0)
- **Default**: 0.80 (80%)
- **Purpose**: Target percentage of context window to achieve **AFTER** tool results compression
- **Location**: `rag_engine/config/settings.py:111`
- **Applied in**: `rag_engine/llm/compression.py:compress_tool_messages()` (line 396)
- **Code**: `target_tokens = int(context_window * target_pct)`
- **When**: When tool results compression is triggered
- **Calculation**: `target_tokens = context_window * llm_compression_target_pct`
- **Behavior**: Compresses articles proportionally by rank to fit within this target
- **Example**: For 262K window: `262,144 * 0.80 = 209,715 tokens`

#### `LLM_COMPRESSION_ARTICLE_RATIO=0.30`
- **Type**: Float (0.0-1.0)
- **Default**: 0.30 (30%)
- **Purpose**: Target compression ratio for individual articles (used in legacy compression path)
- **Location**: `rag_engine/config/settings.py:113`
- **Applied in**: `rag_engine/llm/compression.py:compress_articles_to_target_tokens()` (line 190)
- **Code**: `target_ratio = settings.llm_compression_article_ratio`
- **When**: When compressing articles using the legacy method (not the proportional-by-rank method)
- **Calculation**: `article_target = max(min_tokens, original_tokens * llm_compression_article_ratio)`
- **Note**: This is used in `compress_articles_to_target_tokens()` but **NOT** in the main `compress_tool_messages()` flow which uses proportional allocation

#### `LLM_COMPRESSION_MIN_TOKENS=300`
- **Type**: Integer (absolute token count)
- **Default**: 300
- **Purpose**: Minimum tokens to preserve per article during compression
- **Location**: `rag_engine/config/settings.py:114`
- **Applied in**:
  - `rag_engine/llm/compression.py:compress_all_articles_proportionally_by_rank()` (line 74)
  - `rag_engine/llm/compression.py:compress_articles_to_target_tokens()` (line 192)
- **Code**: `min_tokens_per_article = getattr(settings, "llm_compression_min_tokens", 300)`
- **When**: During article compression to ensure minimum quality
- **Behavior**: Ensures each compressed article has at least 300 tokens (unless original is smaller)

---

### Overhead and Safety Margins

#### `LLM_CONTEXT_OVERHEAD_SAFETY_MARGIN=2000`
- **Type**: Integer (absolute token count)
- **Default**: 2000
- **Purpose**: Additional safety buffer for message formatting, JSON structure, output buffer
- **Location**: `rag_engine/config/settings.py:100`
- **Applied in**: `rag_engine/utils/context_tracker.py:compute_overhead_tokens()` (line 307)
- **Code**: `safety_margin = getattr(settings, "llm_context_overhead_safety_margin", 2000)`
- **When**: During overhead calculation for system prompt + tool schemas + safety margin
- **Calculation**: `total_overhead += safety_margin` (line 312)
- **Note**: Accounts for message formatting, JSON structure overhead, and output buffer beyond actual system prompt and tool schema counts

#### `LLM_TOOL_RESULTS_JSON_OVERHEAD_PCT=0.30`
- **Type**: Float (0.0-1.0)
- **Default**: 0.30 (30%)
- **Purpose**: Accounts for JSON serialization overhead in tool messages
- **Location**: `rag_engine/config/settings.py:105`
- **Applied in**:
  - `rag_engine/llm/compression.py:compress_tool_messages()` (line 304)
  - `rag_engine/utils/context_tracker.py:compute_context_tokens()` (line 140)
- **Code**: `json_overhead_pct = getattr(settings, "llm_tool_results_json_overhead_pct", 0.30)`
- **When**: During tool results compression check and context token counting
- **Calculation**: `json_overhead = int(tool_tokens_raw * json_overhead_pct)`
- **Note**: This is added to the token count **before** comparing against `llm_compression_threshold_pct`

#### `LLM_TOOL_RESULTS_OVERHEAD_TOKENS=3000`
- **Note**: This parameter is **NOT found in the codebase**. The system uses:
  - `llm_tool_results_json_overhead_pct` (percentage-based JSON overhead)
  - `llm_context_overhead_safety_margin` (absolute safety margin)
  
  If this parameter exists in `.env`, it would need to be added to `settings.py` and used in the compression logic.

---

## Parameter Usage Verification

All parameters listed are **actively used** in the codebase:

1. ✅ `MEMORY_COMPRESSION_THRESHOLD_PCT` - Used in `agent_factory.py`
2. ✅ `MEMORY_COMPRESSION_TARGET_TOKENS` - Used in `SummarizationMiddleware`
3. ✅ `MEMORY_COMPRESSION_MESSAGES_TO_KEEP` - Used in `agent_factory.py`
4. ✅ `LLM_PRE_CONTEXT_THRESHOLD_PCT` - Used in `app.py` and `fallback.py`
5. ✅ `LLM_POST_CONTEXT_THRESHOLD_PCT` - Used in `context_tracker.py`
6. ✅ `LLM_CONTEXT_OVERHEAD_SAFETY_MARGIN` - Used in `context_tracker.py`
7. ✅ `LLM_COMPRESSION_THRESHOLD_PCT` - Used in `app.py` and `compression.py`
8. ✅ `LLM_COMPRESSION_TARGET_PCT` - Used in `compression.py`
9. ✅ `LLM_COMPRESSION_ARTICLE_RATIO` - Used in `compression.py`
10. ✅ `LLM_COMPRESSION_MIN_TOKENS` - Used in `compression.py`
11. ✅ `LLM_TOOL_RESULTS_JSON_OVERHEAD_PCT` - Used in `compression.py` and `context_tracker.py`

---

## Comparison: Values vs Defaults

| Parameter | Default Value | Typical Production Value | Difference | Impact |
|-----------|---------------|-------------------------|------------|--------|
| `MEMORY_COMPRESSION_THRESHOLD_PCT` | 80 | 80 | ✅ Same | No change |
| `MEMORY_COMPRESSION_TARGET_TOKENS` | 1000 | 1000 | ✅ Same | No change |
| `MEMORY_COMPRESSION_MESSAGES_TO_KEEP` | 2 | 2 | ✅ Same | No change |
| `LLM_PRE_CONTEXT_THRESHOLD_PCT` | 0.90 | 0.80 | ⚠️ **-10%** | **More aggressive** - triggers fallback earlier |
| `LLM_POST_CONTEXT_THRESHOLD_PCT` | 0.80 | 0.70 | ⚠️ **-10%** | **More aggressive** - stricter monitoring |
| `LLM_CONTEXT_OVERHEAD_SAFETY_MARGIN` | 2000 | 4000 | ⚠️ **+2000** | **More conservative** - reserves more tokens |
| `LLM_COMPRESSION_THRESHOLD_PCT` | 0.85 | 0.80 | ⚠️ **-5%** | **More aggressive** - triggers compression earlier |
| `LLM_COMPRESSION_TARGET_PCT` | 0.80 | 0.80 | ✅ Same | No change |
| `LLM_COMPRESSION_ARTICLE_RATIO` | 0.30 | 0.30 | ✅ Same | No change |
| `LLM_COMPRESSION_MIN_TOKENS` | 300 | 300 | ✅ Same | No change |
| `LLM_TOOL_RESULTS_JSON_OVERHEAD_PCT` | 0.30 | 0.30 | ✅ Same | No change |

### Impact Analysis: More Aggressive Configuration

A more aggressive configuration (lower thresholds, higher safety margins) makes the system:

1. **Trigger compression earlier**:
   - Pre-context: 80% vs 90% (10% earlier) → ~26K tokens earlier fallback trigger (262K window)
   - Tool compression: 80% vs 85% (5% earlier) → ~13K tokens earlier compression trigger (262K window)

2. **Reserve more safety margin**:
   - Overhead safety: 4000 vs 2000 (+2000 tokens) → Reduces available context by 2000 tokens

3. **Stricter monitoring**:
   - Post-context: 70% vs 80% (10% stricter) → ~26K tokens stricter monitoring (262K window)

### Example: 262K Context Window Comparison

| Check | Default Threshold | Aggressive Threshold | Difference |
|-------|-------------------|---------------------|------------|
| Pre-context fallback | 235,930 tokens (90%) | 209,715 tokens (80%) | **-26,215 tokens** |
| Memory compression | 209,715 tokens (80%) | 209,715 tokens (80%) | Same |
| Tool compression | 222,822 tokens (85%) | 209,715 tokens (80%) | **-13,107 tokens** |
| Post-context monitoring | 209,715 tokens (80%) | 183,501 tokens (70%) | **-26,214 tokens** |
| Overhead safety margin | +2000 tokens | +4000 tokens | **+2000 tokens** |

### Benefits of More Aggressive Configuration

✅ **More proactive**: Compression triggers earlier, preventing overflow  
✅ **More headroom**: Larger safety margin (4000 vs 2000) provides buffer  
✅ **Better for small windows**: More aggressive thresholds help with <32K windows  
✅ **Prevents edge cases**: Earlier triggers catch issues before they become problems

### Potential Trade-offs

⚠️ **More frequent compression**: May compress more often, potentially losing some context  
⚠️ **Less available context**: 2000 more tokens reserved for safety (but this is intentional)  
⚠️ **Earlier fallback**: May switch to larger models more often (if fallback enabled)

---

## Execution Flow and Precedence

### 1. Pre-Agent Check (Highest Priority - Prevents Overflow)

```text
User Question
    ↓
[PRE-AGENT CHECK]
    ↓
Check: total_tokens > (context_window * LLM_PRE_CONTEXT_THRESHOLD_PCT)
    ↓
If YES → Trigger context fallback to larger model (if enabled)
If NO → Continue
```

**Applied**: `LLM_PRE_CONTEXT_THRESHOLD_PCT=0.90` (default) or `0.80` (aggressive)

**Purpose**: Early detection to prevent context overflow before agent starts

---

### 2. Memory Compression (Before Each LLM Call)

```text
[Before LLM Call]
    ↓
[MEMORY COMPRESSION CHECK]
    ↓
Check: total_tokens > (context_window * MEMORY_COMPRESSION_THRESHOLD_PCT / 100)
    ↓
If YES → Compress old history to MEMORY_COMPRESSION_TARGET_TOKENS (1000)
         Keep last N messages uncompressed (MEMORY_COMPRESSION_MESSAGES_TO_KEEP)
If NO → Continue
```

**Applied**: 
- `MEMORY_COMPRESSION_THRESHOLD_PCT=80` (80%)
- `MEMORY_COMPRESSION_TARGET_TOKENS=1000`
- `MEMORY_COMPRESSION_MESSAGES_TO_KEEP=2`

**Purpose**: Prevents conversation history from growing unbounded

**Timing**: Runs **BEFORE each LLM call** (including before tool calls and before final answer)

---

### 3. Tool Execution

```text
Agent calls retrieve_context tool
    ↓
Tool returns articles (uncompressed)
    ↓
Agent may call tool multiple times (accumulates articles)
    ↓
[All tool calls complete]
```

**Note**: No compression happens during tool execution - articles accumulate

---

### 4. Tool Results Compression (After Tool Calls, Before Answer)

```text
[All tool calls complete]
    ↓
[TOOL RESULTS COMPRESSION CHECK]
    ↓
Calculate: total_tokens + (tool_tokens_raw * LLM_TOOL_RESULTS_JSON_OVERHEAD_PCT)
    ↓
Check: adjusted_tokens > (context_window * LLM_COMPRESSION_THRESHOLD_PCT)
    ↓
If YES → Compress articles proportionally by rank
         Target: context_window * LLM_COMPRESSION_TARGET_PCT
         Min per article: LLM_COMPRESSION_MIN_TOKENS
If NO → Continue
```

**Applied**:
- `LLM_COMPRESSION_THRESHOLD_PCT=0.85` (default) or `0.80` (aggressive) - **trigger threshold**
- `LLM_COMPRESSION_TARGET_PCT=0.80` (80%) - **target after compression**
- `LLM_COMPRESSION_MIN_TOKENS=300` - **minimum per article**
- `LLM_TOOL_RESULTS_JSON_OVERHEAD_PCT=0.30` (30%) - **JSON overhead adjustment**

**Purpose**: Compresses retrieved articles to fit within context window

**Timing**: Runs **AFTER all tool calls complete**, **BEFORE** LLM answer generation

---

### 5. Post-Tool Check (Monitoring/Validation)

```text
[After compression]
    ↓
[POST-TOOL CHECK]
    ↓
Check: total_tokens > (context_window * LLM_POST_CONTEXT_THRESHOLD_PCT)
    ↓
Used for: Monitoring, validation, logging
```

**Applied**: `LLM_POST_CONTEXT_THRESHOLD_PCT=0.80` (default) or `0.70` (aggressive)

**Purpose**: Validation and monitoring (does not trigger actions, but used in context tracking)

---

## Precedence Summary

### Order of Application (Chronological)

1. **`LLM_PRE_CONTEXT_THRESHOLD_PCT`** (90% default / 80% aggressive) - Pre-agent check, triggers fallback
2. **`MEMORY_COMPRESSION_THRESHOLD_PCT`** (80%) - Before each LLM call, compresses history
3. **`LLM_COMPRESSION_THRESHOLD_PCT`** (85% default / 80% aggressive) - After tool calls, compresses articles
4. **`LLM_POST_CONTEXT_THRESHOLD_PCT`** (80% default / 70% aggressive) - Post-tool validation/monitoring

### Threshold Hierarchy (By Strictness)

From **most strict** (triggers earliest) to **least strict**:

1. **`LLM_PRE_CONTEXT_THRESHOLD_PCT=0.90`** (90% default) - **Most strict** - Prevents overflow before agent starts
2. **`LLM_COMPRESSION_THRESHOLD_PCT=0.85`** (85% default) - Triggers tool results compression
3. **`MEMORY_COMPRESSION_THRESHOLD_PCT=80`** (80%) - Triggers history compression
4. **`LLM_POST_CONTEXT_THRESHOLD_PCT=0.80`** (80% default) - **Least strict** - Only for monitoring

### Key Relationships

- **Memory compression** (80%) runs **more frequently** (before every LLM call) but is **less strict** than tool compression (85%)
- **Tool compression** (85% default) runs **less frequently** (only after tool calls) but is **more strict** to handle large article accumulation
- **Pre-context check** (90% default) is the **most strict** and runs **earliest** to prevent overflow
- **Post-context check** (80% default) is for **monitoring only**, doesn't trigger actions

---

## Example Scenario

### Context Window: 262,144 tokens (262K)

#### Default Configuration

1. **Pre-Agent Check** (90% = 235,930 tokens)
   - If conversation + overhead > 235,930 → Fallback to larger model

2. **Memory Compression** (80% = 209,715 tokens)
   - Before each LLM call, if total > 209,715 → Compress history to 1000 tokens

3. **Tool Results Compression** (85% = 222,822 tokens)
   - After tool calls, if adjusted total > 222,822 → Compress articles to 80% (209,715 tokens)
   - Each article compressed with minimum 300 tokens

4. **Post-Tool Check** (80% = 209,715 tokens)
   - Validation/monitoring only

#### Aggressive Configuration

1. **Pre-Agent Check** (80% = 209,715 tokens)
   - If conversation + overhead > 209,715 → Fallback to larger model

2. **Memory Compression** (80% = 209,715 tokens)
   - Before each LLM call, if total > 209,715 → Compress history to 1000 tokens

3. **Tool Results Compression** (80% = 209,715 tokens)
   - After tool calls, if adjusted total > 209,715 → Compress articles to 80% (209,715 tokens)
   - Each article compressed with minimum 300 tokens

4. **Post-Tool Check** (70% = 183,501 tokens)
   - Validation/monitoring only

### Typical Flow (Default Configuration)

```text
User: "Question about X"
    ↓
Pre-check: 50K tokens < 235,930 ✓ (90%) → Continue
    ↓
Memory check: 50K tokens < 209,715 ✓ (80%) → No compression
    ↓
Agent calls retrieve_context → Returns 7 articles (150K tokens)
    ↓
Total: 200K tokens
    ↓
Tool compression check: 200K < 222,822 ✓ (85%) → No compression
    ↓
LLM generates answer
    ↓
Next turn: Memory check: 220K tokens > 209,715 ✗ (80%) → Compress history to 1000 tokens
```

---

## Configuration Recommendations

### For Large Context Windows (100K+ tokens)
- `MEMORY_COMPRESSION_THRESHOLD_PCT=80` (default) - Good balance
- `LLM_COMPRESSION_THRESHOLD_PCT=0.85` (default) - Allows more articles
- `LLM_PRE_CONTEXT_THRESHOLD_PCT=0.90` (default) - Standard safety margin
- `LLM_CONTEXT_OVERHEAD_SAFETY_MARGIN=2000` (default) - Standard overhead

### For Medium Context Windows (32K-100K tokens)
- `MEMORY_COMPRESSION_THRESHOLD_PCT=75` - More aggressive
- `LLM_COMPRESSION_THRESHOLD_PCT=0.80` - Trigger earlier
- `LLM_PRE_CONTEXT_THRESHOLD_PCT=0.85` - More aggressive fallback
- `LLM_CONTEXT_OVERHEAD_SAFETY_MARGIN=3000` - Increased safety margin

### For Small Context Windows (<16K tokens)
- `MEMORY_COMPRESSION_THRESHOLD_PCT=70` - Very aggressive
- `LLM_COMPRESSION_THRESHOLD_PCT=0.75` - Trigger much earlier
- `LLM_COMPRESSION_TARGET_PCT=0.70` - More aggressive compression
- `LLM_PRE_CONTEXT_THRESHOLD_PCT=0.80` - Early fallback
- `LLM_CONTEXT_OVERHEAD_SAFETY_MARGIN=4000` - Maximum safety margin

### For Production with Agent Mode and Multiple Tool Calls
- `MEMORY_COMPRESSION_THRESHOLD_PCT=80` - Standard
- `LLM_COMPRESSION_THRESHOLD_PCT=0.80` - More aggressive to handle accumulation
- `LLM_PRE_CONTEXT_THRESHOLD_PCT=0.80` - Early fallback detection
- `LLM_CONTEXT_OVERHEAD_SAFETY_MARGIN=4000` - Extra headroom for JSON overhead
- `LLM_POST_CONTEXT_THRESHOLD_PCT=0.70` - Stricter monitoring

---

## Notes

1. **`LLM_TOOL_RESULTS_OVERHEAD_TOKENS=3000`** is **not found** in the codebase. The system uses:
   - `llm_tool_results_json_overhead_pct` (percentage-based JSON overhead)
   - `llm_context_overhead_safety_margin` (absolute safety margin)
   
   If you need this parameter, it should be added to `settings.py` and used in the compression logic.

2. **`LLM_COMPRESSION_ARTICLE_RATIO=0.30`** is used in the legacy `compress_articles_to_target_tokens()` function but **NOT** in the main `compress_tool_messages()` flow which uses proportional allocation by rank.

3. **JSON Overhead**: Tool messages are JSON strings, so `llm_tool_results_json_overhead_pct` (30%) is added to the token count **before** comparing against `llm_compression_threshold_pct`.

4. **Memory compression** and **tool compression** are **independent** - both can trigger in the same turn if thresholds are exceeded.

5. **Overhead Safety Margin**: The `llm_context_overhead_safety_margin` is added to the actual counted tokens for system prompt and tool schemas. It accounts for:
   - Message formatting overhead
   - JSON structure overhead
   - Output buffer
   - Edge cases and rounding errors

6. **Two Overhead Mechanisms**:
   - **`LLM_CONTEXT_OVERHEAD_SAFETY_MARGIN`**: Absolute tokens added to system prompt + tool schema overhead
   - **`LLM_TOOL_RESULTS_JSON_OVERHEAD_PCT`**: Percentage multiplier applied to tool result tokens for JSON serialization overhead
