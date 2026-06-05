# Graceful Uvicorn Shutdown Plan

## Goal

Make Ctrl+C process interruption leaner and more graceful for the RAG Gradio/FastAPI server without changing user-facing app behavior.

## Scope

- Add a bounded Uvicorn graceful shutdown timeout to the existing app entrypoint.
- Keep the change non-breaking: same host, same port, same mounted Gradio app, same logging filter.
- Add a focused regression test for the Uvicorn launch contract.

## TDD Tasks

1. Add a test that monkeypatches `uvicorn.run` and verifies the app runner passes:
   - configured host
   - configured port
   - `timeout_graceful_shutdown=3`
2. Extract a minimal side-effect-free helper for launching the mounted ASGI app.
3. Replace the direct `uvicorn.run(...)` entrypoint call with the helper.
4. Run focused lint and tests.

## Checkpoints

- The helper has no side effects except invoking `uvicorn.run`.
- Ctrl+C behavior remains delegated to Uvicorn.
- No request, retrieval, UI, or tool behavior changes.

## Verification Commands

```powershell
ruff check rag_engine/api/app.py rag_engine/tests/test_api_app.py
pytest rag_engine/tests/test_api_app.py::test_run_gradio_uvicorn_uses_bounded_graceful_shutdown
```
