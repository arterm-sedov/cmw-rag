# Plan: Multiline Chat Input — Match cmw-platform-agent

**Date:** 2026-06-04  
**Branch:** `feat/20260604-multiline-chat-input`  
**Approach:** A (Lean)

## Summary

Copy cmw-platform-agent's `lines=1, max_lines=4` pattern to `rag_engine/api/app.py`.  
Enter still submits. Shift+Enter inserts a newline (browser `<textarea>` default).  
Box grows visually up to 4 lines as content fills.  
No backend, state, test, or contract changes.

## Reference

- cmw-platform-agent: `agent_ng/tabs/chat_tab.py:212-214`
- Commit `72bc59e` (Dec 2025): deliberately changed `lines=2, max_lines=4` → `lines=1, max_lines=1`
  — this tightens back to the better pattern from the reference agent

## Tasks

- [ ] Change `max_lines=1` → `max_lines=4` at `rag_engine/api/app.py:3957`
- [ ] Update comment (lines 3953-3954) to describe new contract: Enter submits, box grows up to 4 lines, Shift+Enter inserts newline
- [ ] Run `ruff check rag_engine/api/app.py`
- [ ] Run `pytest -m "not slow"` to confirm no regression
- [ ] Manual smoke: launch app, type long multiline text, confirm Enter submits and Shift+Enter inserts newline

## Verification

```bash
source .venv/Scripts/Activate.ps1  # or .venv-wsl
ruff check rag_engine/api/app.py
pytest rag_engine/tests/ -m "not slow"
python rag_engine/api/app.py  # manual smoke
```
