from __future__ import annotations

import json
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
        runtime=None,
    )

