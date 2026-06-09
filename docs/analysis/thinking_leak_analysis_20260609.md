# Inter-tool LLM Thinking Leak — Root-Cause Analysis

**Date**: 2026-06-09
**Status**: Findings documented; not fully fixable in code.

---

## The Problem

Between tool calls, the model emits chain-of-thought / search-planning text (e.g. `сведения о выпуске версия 5.0`, `история версий`, `список изменений версии`) as plain `content` in the streaming response. This text leaks into the chat because:

1. It **lacks any delimiter** — no `<think>` tags, no `reasoning` API field (models emit it as regular assistant content).
2. The LLM response is a **single stream** — there is no event or flag that separates "inter-tool planning" from "final answer."
3. The LangGraph agent yields everything through the same `stream_mode=["updates","messages"]` channel.

## Root Cause

### 1. Model Behavior (Provider-Agnostic)

Not specific to DeepSeek. Claude, GPT-4, Gemini all do this: between tool results and the next tool call, the model emits planning text as regular `content`. The only signal distinguishing it from the final answer is **presence of a subsequent tool call**.

### 2. Streaming Architecture

- `agent.astream(stream_mode=["updates","messages"])` funnels tool-call and answer-phase content through the **same text stream**.
- There is no event that marks "answer phase has begun."
- Text arrives as `content_blocks[].text` in AI messages — indistinguishable from real answers.

### 3. No Reliable Retroactive Detection

A retroactive fix ("if a tool call follows, reclassify prior text as reasoning") has two problems:

| Problem | Detail |
|---------|--------|
| **Multi-message span** | Inter-tool text accumulates across multiple assistant messages interleaved with metadata blocks (search bubbles, disclaimers). Reclassifying only the last plain assistant message leaves earlier ones untouched. |
| **Self-correction ambiguity** | If the model emits text, a tool call follows, text gets reclassified → then model emits more text without another tool call → that's the real answer. But the user has already seen the text appear, disappear, and reappear. Streaming UX is preserved but with visible flickering. |

Purging all plain assistant messages from the current turn via `_turn_base_len` scope addresses the span issue but introduces new edge cases (disclaimer placement, interleaved metadata) disproportionate to the benefit.

### 4. Prompt-Level Mitigation Has Limits

Existing instructions that models ignore:
- _"Hide all your internal reasoning"_
- _"DO NOT output your query decomposition suggestions"_
- _"Always start your answer with **three new lines** followed by H1 heading"_

The model **cannot** control that its inter-tool thinking text emits as `content` — this is baked into how the model is trained. The text arrives in the **same** LLM response as the actual answer, so the heading gets concatenated to leaked text without a newline.

### 5. Summary of Prompt Changes Applied

| Change | Commit |
|--------|--------|
| ИИ spelling rule (never `И`) + heading example with `\n\n\n` | `4bc2c0a` |
| Search tool strategy order: retrieve → fetch → grep, don't over-search | `46185fa` |

## What Cannot Be Fixed (Inherent)

| Issue | Why |
|-------|-----|
| Inter-tool planning text as `content` | Model training artifact — no API field, no delimiter. |
| Heading glued to leaked text | Both arrive in same LLM response; streaming causes concatenation before any signal. |
| `И` → `ИИ` inconsistency | Model struggles mid-stream with character repetition; `И` is a common Russian preposition/adverb, compounding the issue. |

## Attempted & Discarded Fixes

| Fix | Why Discarded |
|-----|---------------|
| Retroactive reclassification on new tool call | Multi-message span bug; edge cases with self-correction; scope creep |
| Heading newline injection via `app.py` | Band-aid; doesn't prevent the underlying leak; unreliable mid-chunk detection |
| `_turn_base_len` scope for assistant message purge | Delicate index math; disclaimer placement broken |

## Recommendation

The thinking leak is an **inherent model behavior** that cannot be fully prevented with current streaming architecture. Prompt-level guidance is the right lever — the two prompt commits cover what's controllable:

1. **ИИ spelling**: model can act on this directly (it controls the characters it emits).
2. **Search strategy**: model can act on this to reduce unnecessary tool chains.
3. **Heading formatting**: mostly cosmetic, reinforced with concrete `\n\n\n` example.

No further code changes recommended. Accept minor leakage as a model-level limitation.
