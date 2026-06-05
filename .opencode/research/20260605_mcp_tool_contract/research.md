# MCP Tool Contract Research

## Findings

- Current branch is clean on `main`, tracking `origin/main`.
- `get_knowledge_base_articles` is explicitly exposed as an MCP/API endpoint through `gr.api(...)` in `rag_engine/api/app.py`.
- `_on_version_change` is not an intentional MCP endpoint. It is a Gradio dropdown `.change(...)` event handler for the UI product-version selector.
- Live MCP schema showed `_on_version_change` as a tool only because public Gradio event listeners are converted into API/MCP endpoints.
- Gradio 6 documentation recommends `api_visibility="private"` on event listeners that should be hidden from API docs and Gradio client libraries.
- Gradio MCP documentation recommends pure MCP/API functions via `gr.api(...)`; this matches the existing `get_knowledge_base_articles` and `ask_comindware` registrations.
- Live MCP schema for `get_knowledge_base_articles` exposes `product_version`; a call with `version` fails as an invalid keyword argument.

## Sources

- Local code: `rag_engine/api/app.py`
- Local tests: `rag_engine/tests/test_mcp_get_knowledge_base_articles.py`
- Local docs: `docs/MCP_CONFIGURATION.md`
- Gradio API page docs: https://www.gradio.app/main/guides/view-api-page
- Gradio MCP server docs: https://www.gradio.app/main/guides/building-mcp-server-with-gradio

