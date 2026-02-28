# Cacheable System Prompt Implementation Plan

**Date:** 2026-02-17
**Status:** Ready to implement
**Objective:** Make system prompt 100% static and cacheable by moving all dynamic content to user message wrappers.

---

## Current State

```
┌─────────────────────────────────────────────────────────────────┐
│                      SYSTEM PROMPT (Dynamic)                    │
│  1. Base instructions (STATIC)                                  │
│  2. <current_date> - computed at import (DYNAMIC)               │
│  3. SGR suffix (conditionally appended)                         │
│  4. Guardian context [GUARD] ... (conditionally appended)       │
│                       ↓ NOT CACHEABLE ↓                         │
└─────────────────────────────────────────────────────────────────┘
```

## Target State

```
┌─────────────────────────────────────────────────────────────────┐
│                      SYSTEM PROMPT (Static)                     │
│  Base instructions ONLY                                         │
│  NO datetime, NO SGR/SRP, NO guardian                           │
│                       ✓ FULLY CACHEABLE ✓                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   USER MESSAGE WRAPPER (Dynamic)                │
│  <current_date> (fresh per-request)                             │
│  {guardian_context}                                             │
│  {sgr_directive}                                                │
│  {question_wrapper_template}                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: `rag_engine/llm/prompts.py`

### 1.1 Remove datetime from static system prompt
**Lines 10-13** - DELETE:
```python
<current_date>
Current date/time:
{json.dumps(_get_current_datetime_dict(), ensure_ascii=False, separators=(",", ":"))}
<current_date>
```

### 1.2 Add `get_dynamic_context()` function
**After line 134** - ADD:
```python
def get_dynamic_context(
    moderation_context: str | None = None,
    include_sgr: bool = False,
    include_srp: bool = False,
) -> str:
    """Build dynamic context for user message wrapper.
    
    Uses exact same patterns from system prompt - only location changes.
    """
    parts = []
    
    # Same pattern as current system prompt (with fixed closing tag)
    parts.append(
        "<current_date>\n"
        "Current date/time:\n"
        f'{json.dumps(_get_current_datetime_dict(), ensure_ascii=False, separators=(",", ":"))}\n'
        "</current_date>"
    )
    
    if moderation_context:
        parts.append(moderation_context)
    
    if include_sgr:
        parts.append(get_sgr_suffix())
    
    if include_srp:
        parts.append(get_srp_suffix())
    
    return "\n\n".join(parts) + "\n\n"
```

### 1.3 Update user templates
**Lines 163-174** - MODIFY:
```python
USER_QUESTION_TEMPLATE_FIRST = (
    "{dynamic_context}"
    "Найди информацию в базе знаний по следующей теме:\n"
    "{question}\n\n"
    "Ответь на вопрос пользователя, используя эту информацию"
)

USER_QUESTION_TEMPLATE_SUBSEQUENT = (
    "{dynamic_context}"
    "Ответь на вопрос пользователя:\n\n"
    "{question}\n\n"
    "Учти предыдущие сообщения.\n"
    "Если требуется, найди в базе знаний информацию для ответа на вопрос.\n"
)
```

---

## Phase 2: `rag_engine/api/app.py` - Main Agent Flow

### 2.1 Build dynamic context before wrapper
**After line 977** - ADD:
```python
from rag_engine.llm.prompts import get_dynamic_context

dynamic_context = get_dynamic_context(
    moderation_context=moderation_context,
    include_sgr=enable_sgr_planning_flag,
)
```

### 2.2 Inject dynamic context in wrapper
**Lines 979-982** - MODIFY:
```python
if is_first_message:
    wrapped_message = USER_QUESTION_TEMPLATE_FIRST.format(
        dynamic_context=dynamic_context,
        question=message
    )
else:
    wrapped_message = USER_QUESTION_TEMPLATE_SUBSEQUENT.format(
        dynamic_context=dynamic_context,
        question=message
    )
```

### 2.3 Simplify system message construction
**Lines 1090-1102** - REPLACE:
```python
# BEFORE:
base_prompt = get_system_prompt()
guardian_suffix = moderation_context if moderation_context else ""
sgr_suffix = get_sgr_suffix()
system_msg = {
    "role": "system",
    "content": f"{sgr_suffix}\n\n{guardian_suffix}\n\n{base_prompt}"
    if guardian_suffix
    else f"{sgr_suffix}\n\n{base_prompt}",
}

# AFTER:
system_msg = {"role": "system", "content": get_system_prompt()}
messages = [system_msg] + messages
```

---

## Phase 3: `rag_engine/api/app.py` - SRP Flow

### 3.1 Build SRP ephemeral user message
**Lines 2173-2180** - REPLACE:
```python
# BEFORE:
srp_system_prompt = get_srp_suffix() + "\n\n" + get_system_prompt()
if articles:
    srp_system_prompt += "\n\n" + format_sources_list(articles)
srp_messages = _build_agent_messages_from_gradio_history(gradio_history)
srp_messages = [{"role": "system", "content": srp_system_prompt}] + srp_messages

# AFTER:
srp_messages = _build_agent_messages_from_gradio_history(gradio_history)

# Build ephemeral SRP context (not stored in history)
srp_context = get_dynamic_context(include_srp=True)
if articles:
    srp_context += format_sources_list(articles) + "\n\n"

srp_analysis_request = {
    "role": "user",
    "content": srp_context + "Analyze the above assistant answer quality."
}
srp_messages.append(srp_analysis_request)

srp_messages = [{"role": "system", "content": get_system_prompt()}] + srp_messages
```

---

## Phase 4: Tests

| File | Change |
|------|--------|
| `test_llm_prompts.py` | Verify static system prompt (no datetime) |
| `test_sgr_tool.py` | Verify SGR directive in user wrapper |
| `test_srp_tool.py` | Verify SRP as separate user message |

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| System prompt | Dynamic (datetime, SGR, guardian) | **100% static** |
| Datetime | Computed at module import | Computed per-request |
| Guardian context | System prompt | User wrapper |
| SGR directive | System prompt | User wrapper |
| SRP directive + sources | System prompt | Separate ephemeral user message |
| Downstream data | Preserved | **Preserved** (no changes) |

## Files Modified

| File | Net Lines |
|------|-----------|
| `rag_engine/llm/prompts.py` | ~+22 |
| `rag_engine/api/app.py` | ~-5 |
| Tests | ~+15 |

---

## Data Flow Verification

### Structured Data Passed Downstream (via `AgentContext`) - UNCHANGED

| Field | Value |
|-------|-------|
| `sgr_plan` | `{spam_score, user_intent, knowledge_base_search_queries, ...}` |
| `resolution_plan` | `{engineer_intervention_needed, issue_summary, ...}` |
| `final_articles` | `[{kb_id, title, url, metadata: {rerank_score, ...}}]` |
| `final_answer` | Complete answer string |
| `query_traces` | Retrieval trace for confidence calc |
| `diagnostics` | `{model, stream_chunks, tool_results_count, guard_info, ...}` |

### SRP Result Handling - UNCHANGED

- SRP user message: **Ephemeral** (not stored in gradio_history)
- Resolution markdown: Appended to answer (if `engineer_intervention_needed=True`)
- Sources list: Appended to answer
- Resolution plan JSON: `agent_context.resolution_plan` (downstream)

---

## Bug Fix Included

Fix invalid XML closing tag in datetime block:
- Before: `<current_date>` (opening tag used as closing)
- After: `</current_date>` (proper closing tag)
