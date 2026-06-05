# MCP Tool Contract Fix Plan

## Scope

Fix two MCP contract issues without changing retrieval behavior:

1. Hide the UI-only product-version dropdown handler from MCP/API exposure.
2. Allow `get_knowledge_base_articles` callers to pass `version` as a compatibility alias for `product_version`.

## TDD Tasks

1. Add tests for `get_knowledge_base_articles`.
   - Verify explicit `product_version="v5"` reaches `_retrieve_context_core`.
   - Verify compatibility `version="v5"` reaches `_retrieve_context_core` as `product_version="v5"`.
   - Verify conflicting `product_version` and `version` fails clearly.
2. Add a static contract test for the UI event registration.
   - Verify the `version_selector.change(...)` event uses `api_visibility="private"`.
3. Implement the minimal code changes.
   - Add optional `version` alias to `get_knowledge_base_articles`.
   - Resolve effective version once before validation/use.
   - Mark `_on_version_change` event private.
4. Update MCP documentation.
   - Document `product_version`.
   - Document `version` as a backward-compatible alias.
   - Note UI event handlers are intentionally not MCP tools.

## Checkpoints

1. Tests fail before implementation for the new expected behavior.
2. Code changes are limited to MCP contract behavior.
3. Tests pass for modified behavior.
4. Ruff passes on modified Python files.
5. Git diff is reviewed twice for regressions.
6. Commit and push only after verification.

## Verification Commands

```powershell
.venv\Scripts\Activate.ps1
pytest rag_engine/tests/test_mcp_get_knowledge_base_articles.py --no-cov
ruff check rag_engine/api/app.py rag_engine/tests/test_mcp_get_knowledge_base_articles.py
git diff --check
git status --short
```
