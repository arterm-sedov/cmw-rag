# System Prompt Redundancy Analysis

## Date
2025-01-28

## Overview
Analysis of `SYSTEM_PROMPT` in `rag_engine/llm/prompts.py` for redundancies, duplicates, and unnecessary parts relevant to the RAG engine internals.

## Key Findings

### 1. **Major Redundancy: Tool Usage Instructions**

**Location**: Lines 4, 8, 10, 14-20, 21-50, 52-55

**Issue**: The prompt contains extensive instructions about using `retrieve_context` tool, but:

- **Tool docstring already covers this**: The `retrieve_context` tool (lines 166-223 in `retrieve_context.py`) has comprehensive documentation explaining:
  - When to use the tool
  - How to use it
  - Query best practices
  - Iterative search strategy
  - Examples of query variations

- **Tool forcing is handled programmatically**: The agent uses `tool_choice` parameter (see `agent_factory.py:111-118`) to force tool execution, not prompt instructions. According to `docs/progress_reports/agent_tool_choice_update.md`, this was intentionally moved away from prompt-based forcing.

- **User question templates already instruct search**: `USER_QUESTION_TEMPLATE_FIRST` (line 204-208) already wraps user messages with "Найди информацию в базе знаний" instruction.

**Redundant sections**:
- Lines 4: "You are focused on searching the knowledge base using retrieve_context tool"
- Lines 8-10: "**FIRST, before answering ANY question, ALWAYS call retrieve_context tool**" + "Never answer without searching first"
- Lines 14-20: `<content_to_search>` section with query instructions
- Lines 21-50: `<question_query_examples>` with detailed examples (duplicates tool docstring examples)
- Lines 52-55: `<tool_use>` section (very basic, redundant)

**Recommendation**: Remove or significantly condense these sections. The tool's docstring and `tool_choice` mechanism are sufficient.

---

### 2. **Repeated Core Instructions**

**Location**: Multiple locations

**Issue**: Core instructions are repeated multiple times:

- **"Answer based ONLY on provided context"**:
  - Line 3: "You answer questions based strictly on provided context"
  - Line 11: "Answer based ONLY on the provided context documents"
  - Line 63: "If the answer is not derivable from the context, you don't know the answer"
  - Line 145: "Keep answers precise and strictly grounded in the provided context"

- **"Never answer without searching first"**:
  - Line 4: "You are focused on searching the knowledge base"
  - Line 8: "**FIRST, before answering ANY question, ALWAYS call retrieve_context tool**"
  - Line 10: "Never answer without searching first"

**Recommendation**: Consolidate into single, clear statements in appropriate sections.

---

### 3. **Formatting Instructions Overlap**

**Location**: Lines 153-158 (`<answer_formatting>`) vs Lines 166-173 (`<code_samples>`)

**Issue**: Both sections mention:
- "Add new lines before and after code blocks" (lines 156, 170)
- Code block formatting (lines 157, 171)

**Recommendation**: Merge code formatting instructions into `<answer_formatting>` and remove redundant `<code_samples>` section, or make `<code_samples>` focus only on code-specific guidance (extraction, language tags).

---

### 4. **Answer Structure Redundancy**

**Location**: Lines 144-151 (`<answer_precision>`) vs Lines 153-158 (`<answer_formatting>`)

**Issue**: Both sections contain overlapping guidance:
- "Be brief" / "concise" (lines 146, 149, 155)
- Structure and clarity (lines 144-150, 154-157)

**Recommendation**: Consolidate into a single `<answer_structure>` section with clear subsections.

---

### 5. **Question Query Examples Section**

**Location**: Lines 21-50 (`<question_query_examples>`)

**Issue**: 
- Contains 3 detailed examples with multiple query variations
- The `retrieve_context` tool docstring (lines 178-191) already provides similar examples and best practices
- Takes up ~30 lines (17% of the prompt)

**Recommendation**: Remove this section entirely. The tool docstring is sufficient, and examples can be learned from tool usage.

---

### 6. **Tool Use Section**

**Location**: Lines 52-55 (`<tool_use>`)

**Issue**: 
- Very basic: "Call retrieve_context tool for information retrieval. Call math tools for calculations."
- Redundant with tool docstrings
- The agent automatically has access to tool descriptions via LangChain

**Recommendation**: Remove entirely. Tool descriptions are provided by LangChain automatically.

---

### 7. **Structured Approach Section**

**Location**: Lines 58-60 (`<structured_approach>`)

**Issue**: 
- Very brief: "Always follow internally: Intent → Plan → Validate → Execute → Result"
- No concrete guidance on how to apply this
- May not add significant value

**Recommendation**: Either expand with concrete guidance or remove if not critical.

---

### 8. **Multi-Perspective Reasoning**

**Location**: Lines 69-79 (`<multi_perspective_reasoning>`)

**Issue**: 
- Complex instruction to "silently consider" from 4 different perspectives
- Instruction to "NEVER expose these roles" suggests trying to hide internal reasoning
- May be over-engineered for practical benefit
- Adds cognitive overhead without clear measurable benefit

**Recommendation**: Consider simplifying or removing if not proven to improve output quality.

---

### 9. **Product Names Lookup Table**

**Location**: Lines 90-106 (`<product_names>`)

**Issue**: 
- Large static mapping table (17 lines)
- Could potentially be handled programmatically via metadata enrichment
- However, this might be necessary for LLM to understand placeholders in source content

**Recommendation**: Keep if placeholders exist in source content, but consider moving to metadata enrichment if possible.

---

### 10. **Content to Search Section**

**Location**: Lines 14-20 (`<content_to_search>`)

**Issue**: 
- Overlaps with `retrieve_context` tool docstring (lines 187-191)
- Instructions about Russian queries, avoiding "Comindware Platform" in queries, etc.
- Tool docstring already covers query best practices

**Recommendation**: Remove or condense to only platform-specific guidance not in tool docstring.

---

## Summary of Redundancies

### High Priority (Remove/Condense)
1. **Lines 21-50**: `<question_query_examples>` - Remove entirely (tool docstring covers this)
2. **Lines 52-55**: `<tool_use>` - Remove entirely (redundant with tool descriptions)
3. **Lines 14-20**: `<content_to_search>` - Condense or remove (overlaps with tool docstring)
4. **Lines 4, 8, 10**: Tool forcing instructions - Condense (tool_choice handles this)

### Medium Priority (Consolidate)
5. **Lines 144-173**: Merge `<answer_precision>`, `<answer_formatting>`, and `<code_samples>` into unified structure
6. **Multiple locations**: Consolidate repeated "answer only from context" instructions
7. **Lines 58-60**: `<structured_approach>` - Expand or remove

### Low Priority (Consider)
8. **Lines 69-79**: `<multi_perspective_reasoning>` - Evaluate if it adds value
9. **Lines 90-106**: `<product_names>` - Keep if needed, but consider programmatic handling

## Estimated Token Savings

If all high-priority redundancies are removed:
- Lines 21-50: ~30 lines (~450 tokens)
- Lines 52-55: ~4 lines (~60 tokens)
- Lines 14-20: ~7 lines (~105 tokens)
- Condensed tool instructions: ~3 lines (~45 tokens)

**Total potential savings: ~660 tokens (~44 lines, ~20% of prompt)**

## Implementation Results (2025-01-28)

**Actual token count after optimization**: 1157 tokens

**Changes made**:
1. ✅ Moved `<question_query_examples>` section to `retrieve_context` tool docstring
2. ✅ Removed `<tool_use>` section (lines 52-55)
3. ✅ Condensed tool usage instructions (removed "ALWAYS", kept essential guidance)
4. ✅ Condensed `<content_to_search>` section (kept only platform-specific guidance)
5. ✅ Consolidated repeated "answer only from context" instructions into single statement
6. ✅ Merged `<answer_precision>`, `<answer_formatting>`, and `<code_samples>` into unified `<answer_structure>`
7. ✅ Simplified `<internal_reasoning>` section (removed `<structured_approach>` and `<multi_perspective_reasoning>`)

**Token reduction**: Estimated ~36% reduction (from ~1817 to 1157 tokens, ~660 tokens saved)

**Files modified**:
- `rag_engine/tools/retrieve_context.py` - Added Russian query decomposition examples to docstring
- `rag_engine/llm/prompts.py` - Optimized system prompt, removed redundancies

## Recommendations

1. **Remove tool usage instructions** that duplicate tool docstrings
2. **Consolidate formatting instructions** into a single section
3. **Remove example sections** - let the tool docstring and usage patterns teach the model
4. **Keep essential constraints**: terminology, citation format, language policy, forbidden topics
5. **Test prompt reduction** to ensure no degradation in agent behavior

## Notes

- The prompt is used in both agent mode (via `agent_factory.py`) and direct mode (via `llm_manager.py`)
- Agent mode uses `tool_choice` for tool forcing, so prompt-based tool instructions are redundant
- Tool docstrings are automatically provided to the LLM by LangChain, making explicit tool instructions in the prompt redundant
- The prompt should focus on **constraints, formatting, and domain-specific rules**, not tool usage patterns
