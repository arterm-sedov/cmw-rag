from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch


def test_get_knowledge_base_articles_uses_core_with_converted_top_k():
    """get_knowledge_base_articles delegates to async core with parsed top_k."""
    from rag_engine.api.app import get_knowledge_base_articles

    fake_response = json.dumps(
        {
            "articles": [{"kb_id": "123", "title": "T", "url": "U", "content": "C", "metadata": {}}],
            "metadata": {"query": "q", "top_k_requested": 2, "articles_count": 1, "has_results": True},
        }
    )

    async_mock = AsyncMock(return_value=fake_response)

    with patch("rag_engine.api.app._retrieve_context_core", async_mock):
        result = get_knowledge_base_articles("workflow configuration", top_k="2", exclude_kb_ids=None)

    assert result == fake_response
    async_mock.assert_awaited_once_with(
        query="workflow configuration",
        top_k=2,
        exclude_kb_ids=None,
        product_version="v6",
        runtime=None,
    )


def test_get_knowledge_base_articles_accepts_product_version():
    """Explicit product_version selects the requested versioned collection."""
    from rag_engine.api.app import get_knowledge_base_articles

    async_mock = AsyncMock(return_value=json.dumps({"articles": []}))

    with patch("rag_engine.api.app._retrieve_context_core", async_mock):
        get_knowledge_base_articles("workflow configuration", product_version="v5")

    async_mock.assert_awaited_once_with(
        query="workflow configuration",
        top_k=None,
        exclude_kb_ids=None,
        product_version="v5",
        runtime=None,
    )






def test_version_dropdown_change_handler_is_private_api():
    """The UI-only version change event must not be exposed as an MCP tool."""
    source = Path("rag_engine/api/app.py").read_text(encoding="utf-8")
    event_start = source.index("version_selector.change(")
    event_end = source.index("    # Chatbot component", event_start)
    event_block = source[event_start:event_end]

    assert 'api_visibility="private"' in event_block

