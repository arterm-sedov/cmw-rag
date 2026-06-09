# Reasoning Content Round-Trip Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix thinking content leak from DeepSeek/OpenRouter reasoning models by preserving `reasoning_content` on all outgoing assistant messages and adding stream-end cleanup.

**Architecture:** Three-point fix: (1) Inject `reasoning_content` on outgoing assistant messages gated on `reasoning_enabled`, (2) Preserve accumulated reasoning from stream back into gradio history for round-trip, (3) Add `_ReasoningCtx.close()` for stream-abort safety.

**Tech Stack:** Python, LangChain, OpenRouter, Gradio, Pytest

**Date**: 2026-06-09

---

## Definitive Findings

**OpenRouter explicitly requires `reasoning_content` round-trip for ALL reasoning-enabled models with tool calls.** Per OpenRouter docs, DeepSeek API docs, and opencode Issue #29618:

- When `reasoning_content` is missing from an assistant message in multi-turn conversations with tool calls, the model falls back to emitting planning text as plain `content` (the leak) — or returns a 400 error.
- **Gate: `llm_reasoning_enabled`** — not per-model prefixes. ALL reasoning models through OpenRouter need this.
- Reference impl: opencode's `transform.ts:217-249` injects `reasoning_content` via `providerOptions.openaiCompatible`.

## Scope

1. Inject `reasoning_content` on all outgoing assistant messages when `reasoning_enabled=True`
2. Preserve `reasoning_content` from `rctx.buffer` back into gradio history for next-turn round-trip
3. Add `_ReasoningCtx.close()` for stream-end/abort dangling reasoning cleanup
4. Add TDD test coverage for all changes

**Out of scope**: Harmony format models, inter-tool text that inherently lacks any delimiter (accepted as model-level limitation per root-cause analysis).

---

## File Change Summary

| File | Change |
|------|--------|
| `rag_engine/api/app.py` | Add `_inject_reasoning_content()`, wire into message build path, preserve `reasoning_content` on final assistant message, add `_find_last_assistant_message()`, add `_ReasoningCtx.close()`, wire into error/abort paths |
| `rag_engine/llm/openrouter_native.py` | Add `reasoning_content` post-processing in `_get_request_body()` |
| `rag_engine/tests/test_api_app.py` | Add ~7 new tests |

---

## TDD Test Suite

### Phase 1A — Tests for `reasoning_content` round-trip preservation

All tests in `rag_engine/tests/test_api_app.py`:

| Test | Behavior to Verify |
|------|--------------------|
| `test_reasoning_content_preserved_in_agent_messages` | When `_build_agent_messages_from_gradio_history` receives a dict with `additional_kwargs.reasoning_content`, the returned message includes `reasoning_content` field on assistant messages |
| `test_reasoning_content_injected_when_reasoning_enabled` | When `reasoning_enabled=True`, all assistant messages get at least empty `reasoning_content: ""` injected |
| `test_reasoning_content_not_injected_when_reasoning_disabled` | When `reasoning_enabled=False`, no `reasoning_content` is injected on assistant messages |
| `test_reasoning_content_from_stream_chunk_goes_to_bubble` | Integration: stream an `AIMessageChunk` with `additional_kwargs.reasoning_content` — verify bubble shows it, answer does not contain it |
| `test_reasoning_content_roundtrip_via_convert_to_openai_messages` | Unit: create BaseMessage with `additional_kwargs["reasoning_content"] = "test"`, call `convert_to_openai_messages()` — verify `reasoning_content` key appears in output dict |

### Phase 1B — Tests for stream-end reasoning flush

| Test | Behavior to Verify |
|------|--------------------|
| `test_reasoning_ctx_close_handles_dangling_in_block` | `_ReasoningCtx(buffer="partial", in_block=True)`. After `close()`, `in_block` is `False` and buffer is preserved |
| `test_reasoning_ctx_close_idempotent` | `close()` twice is a no-op |
| `test_stream_abort_finalizes_reasoning` | Simulate stream abort mid-`<think>` — verify `_ReasoningCtx` is cleaned up and no reasoning leaks into answer |

### Phase 1C — Existing regression guards

Run existing tests and confirm all pass with no behavioral changes.

---

## Implementation

### Task 1 — Inject `reasoning_content` on outgoing messages

**File**: `rag_engine/api/app.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_reasoning_content_injected_when_reasoning_enabled():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "result"},
    ]
    result = _inject_reasoning_content(messages, reasoning_enabled=True)
    assert result[1]["reasoning_content"] == ""  # injected on assistant msg with tool_calls


def test_reasoning_content_not_injected_when_reasoning_disabled():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi", "tool_calls": [{"id": "1"}]},
    ]
    result = _inject_reasoning_content(messages, reasoning_enabled=False)
    assert "reasoning_content" not in result[1]


def test_reasoning_content_preserved_in_agent_messages():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi", "additional_kwargs": {"reasoning_content": "thinking..."}},
    ]
    result = _inject_reasoning_content(messages, reasoning_enabled=True)
    assert result[1]["reasoning_content"] == "thinking..."
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest rag_engine/tests/test_api_app.py -x -k "reasoning_content_injected or reasoning_content_not_injected or reasoning_content_preserved" -v`

Expected: FAIL — `_inject_reasoning_content` not defined

- [ ] **Step 3: Add `_inject_reasoning_content()` function**

After the `_find_last_non_metadata_message_index` function or similar helper area (around line 850):

```python
def _inject_reasoning_content(
    messages: list[dict],
    *,
    reasoning_enabled: bool = False,
) -> list[dict]:
    if not reasoning_enabled:
        return messages
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        if "reasoning_content" in msg:
            continue
        msg["reasoning_content"] = ""
    return messages
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rag_engine/tests/test_api_app.py -x -k "reasoning_content_injected or reasoning_content_not_injected or reasoning_content_preserved" -v`

Expected: PASS

- [ ] **Step 5: Wire into `agent_chat_handler()`**

Near line 1503, after `messages = _build_agent_messages_from_gradio_history(...)` and before `agent.astream()`:

```python
messages = _inject_reasoning_content(
    messages,
    reasoning_enabled=reasoning_enabled,
)
```

### Task 2 — Preserve `reasoning_content` on final assistant message for round-trip

**File**: `rag_engine/api/app.py`

- [ ] **Step 1: Write the failing test**

```python
def test_reasoning_content_preserved_on_final_assistant_message():
    gradio = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer", "metadata": {"title": "..."}},
        {"role": "assistant", "content": "the answer"},
    ]
    result = _preserve_reasoning_on_last_assistant(gradio, "thinking text")
    assert result[2].get("additional_kwargs", {}).get("reasoning_content") == "thinking text"


def test_find_last_assistant_message():
    gradio = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "metadata": {"title": "..."}},
        {"role": "assistant", "content": "real answer"},
    ]
    assert _find_last_assistant_message(gradio) == {"role": "assistant", "content": "real answer"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest rag_engine/tests/test_api_app.py -x -k "preserved_on_final or find_last_assistant" -v`

Expected: FAIL

- [ ] **Step 3: Add `_find_last_assistant_message()` helper**

```python
def _find_last_assistant_message(messages: list[dict]) -> dict | None:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and "metadata" not in msg:
            return msg
    return None
```

- [ ] **Step 4: Add `_preserve_reasoning_on_last_assistant()` function**

```python
def _preserve_reasoning_on_last_assistant(
    gradio_history: list[dict],
    reasoning_text: str,
) -> list[dict]:
    if not reasoning_text.strip():
        return gradio_history
    target = _find_last_assistant_message(gradio_history)
    if target is not None:
        if "additional_kwargs" not in target:
            target["additional_kwargs"] = {}
        target["additional_kwargs"]["reasoning_content"] = reasoning_text.strip()
    return gradio_history
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest rag_engine/tests/test_api_app.py -x -k "preserved_on_final or find_last_assistant" -v`

Expected: PASS

- [ ] **Step 6: Wire into end-of-stream cleanup in `agent_chat_handler()`**

Near lines 2728-2765 (end-of-stream cleanup area, before/after Harmony flush):

```python
# Preserve reasoning_content on final assistant message for next-turn round-trip.
if reasoning_enabled and rctx.buffer.strip():
    _preserve_reasoning_on_last_assistant(gradio_history, rctx.buffer)
```

### Task 3 — `_ReasoningCtx.close()` for stream-end flush

**File**: `rag_engine/api/app.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_reasoning_ctx_close_handles_dangling_in_block():
    rctx = _ReasoningCtx(buffer="partial thinking", in_block=True, inter_tool_text="")
    remaining = rctx.close()
    assert rctx.in_block is False
    assert rctx.inter_tool_text == ""
    # buffer is preserved, remaining returns the dangling text
    assert remaining == "partial thinking"


def test_reasoning_ctx_close_idempotent():
    rctx = _ReasoningCtx(buffer="done", in_block=True, inter_tool_text="")
    first = rctx.close()
    second = rctx.close()
    assert rctx.in_block is False
    assert second == ""  # second call returns empty (already closed)


def test_reasoning_ctx_close_no_dangling():
    rctx = _ReasoningCtx(buffer="", in_block=False, inter_tool_text="")
    remaining = rctx.close()
    assert remaining == ""
    assert rctx.in_block is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest rag_engine/tests/test_api_app.py -x -k "close_handles_dangling or close_idempotent or close_no_dangling" -v`

Expected: FAIL — `close()` doesn't exist on `_ReasoningCtx`

- [ ] **Step 3: Add `close()` method to `_ReasoningCtx`**

```python
@dataclass
class _ReasoningCtx:
    ...
    
    def close(self) -> str:
        remaining = ""
        if self.in_block and self.buffer:
            remaining = self.buffer
        self.in_block = False
        self.inter_tool_text = ""
        return remaining
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rag_engine/tests/test_api_app.py -x -k "close_handles_dangling or close_idempotent or close_no_dangling" -v`

Expected: PASS

- [ ] **Step 5: Wire `close()` into error/cancellation/abort paths**

In `agent_chat_handler()`:

1. `GeneratorExit` handler (around line 2941-2949): `rctx.close()` before re-raise
2. `Exception` handler (around line 2951+): `rctx.close()` 
3. Cancellation path (around line 2658-2678): `rctx.close()` before return
4. Normal completion (before diagnostics at line 2924): `rctx.close()` as safety net

### Task 4 — Verify `OpenRouterNativeFullChatModel` round-trip

**File**: `rag_engine/llm/openrouter_native.py`

- [ ] **Step 1: Write test for `convert_to_openai_messages` round-trip**

In `rag_engine/tests/test_openrouter_native.py`:

```python
def test_reasoning_content_roundtrip_via_convert_to_openai_messages():
    from langchain_core.messages import AIMessage
    from langchain_core.utils import convert_to_openai_messages
    
    msg = AIMessage(
        content="Hello",
        additional_kwargs={"reasoning_content": "thinking text"},
    )
    result = convert_to_openai_messages([msg])
    assert "reasoning_content" in result[0]
    assert result[0]["reasoning_content"] == "thinking text"
```

- [ ] **Step 2: Verify `_get_request_body()` preserves reasoning_content**

Check if `convert_to_openai_messages()` at line 236 preserves `additional_kwargs`. If it does (confirmed by LangChain source), no `openrouter_native.py` changes needed. If not, add:

```python
# After line 236
for oai_msg in oai_messages:
    if oai_msg.get("role") == "assistant" and "reasoning_content" not in oai_msg:
        oai_msg["reasoning_content"] = ""
```

- [ ] **Step 3: Run the round-trip test**

Run: `pytest rag_engine/tests/test_openrouter_native.py -x -k "reasoning_content_roundtrip" -v`

Expected: PASS (or FAIL if `openrouter_native.py` needs the fix)

### Task 5 — Run tests and verify

- [ ] **Step 1: Run new reasoning round-trip tests**

Run: `pytest rag_engine/tests/test_api_app.py -x -k "reasoning_content or find_last_assistant or close_handles_dangling or close_idempotent or close_no_dangling" -v`

Expected: All new tests PASS

- [ ] **Step 2: Run existing reasoning/think-tag tests (no regressions)**

Run: `pytest rag_engine/tests/test_api_app.py -x -k "parse_think_tags or update_or_append or harmony or agent_stream" -v`

Expected: All existing tests PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest -x`

Expected: All tests PASS

- [ ] **Step 4: Run ruff check on modified files**

Run: `ruff check rag_engine/api/app.py rag_engine/llm/openrouter_native.py`

Expected: No errors

- [ ] **Step 5: Manual smoke test**

Run app, chat with GLM 4.5+ (default) with reasoning enabled, observe:
- Inter-tool planning text appears in reasoning bubble, not in chat answer
- Second turn continues correctly without 400 error or leaked thinking

---

## Checkpoints

| Checkpoint | Verification |
|------------|-------------|
| **C1**: Unit tests pass for injection | `pytest rag_engine/tests/test_api_app.py -x -k "reasoning_content_injected or reasoning_content_not_injected or reasoning_content_preserved" -v` |
| **C2**: Unit tests pass for round-trip preservation | `pytest rag_engine/tests/test_api_app.py -x -k "preserved_on_final or find_last_assistant" -v` |
| **C3**: Unit tests pass for `_ReasoningCtx.close()` | `pytest rag_engine/tests/test_api_app.py -x -k "close_handles_dangling or close_idempotent or close_no_dangling" -v` |
| **C4**: Existing tests not broken | `pytest rag_engine/tests/test_api_app.py -x -k "parse_think_tags or update_or_append or harmony or agent_stream" -v` |
| **C5**: Full suite green | `pytest -x` |
| **C6**: Lint clean | `ruff check rag_engine/api/app.py rag_engine/llm/openrouter_native.py` |
| **C7**: Manual smoke test | Two-turn conversation with reasoning-enabled model, thinking in bubble, no 400 error |
