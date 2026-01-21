# SGR Planning + Structured Agent Callable Implementation (Support Ticket Batch Mode)

## Overview

This change set introduces a schema-guided reasoning (SGR) planning step, a structured agent callable, and supporting UI/trace plumbing to better analyze support tickets and knowledge-base retrieval quality, while keeping the existing streaming UI intact.

## Key Changes

- **SGR planning (spam + intent + subqueries):**
  - Added `SGRPlanResult` / `StructuredAgentResult` schemas (`rag_engine/llm/schemas.py`).
  - Implemented `run_sgr_planning()` (`rag_engine/llm/sgr_planning.py`) using the existing system prompt plus a short SGR clarification, with structured output (spam score, reason, user_intent, subqueries, optional action_plan/clarification).
  - Extended `AgentContext` (`rag_engine/utils/context_tracker.py`) with `sgr_plan`, `query_traces`, `final_answer`, `final_articles`, and `diagnostics` so that all structured data is available without bloating LLM payloads.

- **LLM-driven retrieval trace + confidence:**
  - `RAGRetriever.retrieve()` now attaches raw chunk rerank scores (`rerank_score_raw`) and optional per-query `retrieval_confidence` (`rag_engine/retrieval/confidence.py`).
  - `retrieve_context` tool (`rag_engine/tools/retrieve_context.py`) writes a lightweight per-query trace into `AgentContext.query_traces` capturing query, confidence metrics, ranked articles, and top chunks/snippets.

- **Shared agent core + structured callable:**
  - `agent_chat_handler` remains the single core streaming path but now **yields the final `AgentContext` as its last item**, after all `list[dict]` history chunks.
  - `ask_comindware()` is updated to consume this generator and prefer `context.final_answer` (preserving MCP string API).
  - New `ask_comindware_structured()` callable (`rag_engine/api/app.py`) consumes the same handler and returns `StructuredAgentResult` (SGR plan, per-query trace, final articles, diagnostics).
  - `create_rag_agent()` (`rag_engine/llm/agent_factory.py`) gained opt-in SGR planning (enabled by default) via a `before_model` middleware that:
    - Runs SGR once using the last user message.
    - Stores the plan in `AgentContext.sgr_plan`.
    - Injects a compact SGR-plan system message before the user message using `SGR_PLAN_SYSTEM_MESSAGE_TEMPLATE` (`rag_engine/llm/prompts.py`).
    - Softly suggests clarification when spam_score is high (no hard blocking).

- **Batch XLSX processing now reuses the full agent:**
  - `process_requests_xlsx.py` no longer performs direct retrieval + ad-hoc spam scoring.
  - Each row builds a markdown request (`# H1 + body`), calls `ask_comindware_structured()`, and derives:
    - `Статьи` from `per_query_results` via `format_articles_column_from_trace()`.
    - `Столбец1` (chunks) via `format_chunks_column_from_trace()`.
    - `Ответ на обращение` via `build_answer_column_from_result()` (prepends AI disclaimer + recommended articles + agent answer).
    - `SpamScore` from `result.plan.spam_score`.
  - All column formatters live in `rag_engine/utils/trace_formatters.py` to avoid duplication between batch and potential future tooling.

- **Gradio UI metadata panels (debug/insight only):**
  - The existing chatbot UI still streams via `agent_chat_handler`, but now the wrapper `chat_with_metadata`:
    - Forwards streaming `list[dict]` chunks to the `Chatbot` unchanged.
    - Waits for the final `AgentContext` yield and then updates:
      - Spam badge, retrieval confidence badge, and query-count badge.
      - An “Analysis Summary” accordion (intent, subqueries, action_plan).
      - A “Retrieved Articles” accordion (rank, title, confidence, URL).
  - Badge/label strings are localized via `rag_engine/api/i18n.py` and simple helpers in `rag_engine/api/app.py` (e.g., `format_spam_badge`, `format_confidence_badge`, `format_articles_dataframe`).

## Impact and Guarantees

- **Existing behavior preserved:**
  - Streaming UX for the main support agent is unchanged (still yields `list[dict]` history chunks for Gradio).
  - `ask_comindware()` remains a string-returning MCP-compatible entrypoint, now backed by `agent_chat_handler`’s final `AgentContext.final_answer`.
  - The original user message always stays in the messages list; the SGR plan is added as an extra system message, not a replacement.

- **New capabilities:**
  - Per-turn SGR plan (spam score, user intent, subqueries, optional plan/clarification) available both to the model (as context) and to tooling (via `AgentContext.sgr_plan`).
  - Per-query retrieval trace (confidence + ranked articles + top chunks) available for batch analysis and UI debugging.
  - Support-ticket XLSX processor now uses the **same agentic pipeline** as the UI, ensuring consistent compression/budgeting/citation behavior and producing richer columns for post-hoc analysis and dataset building.

