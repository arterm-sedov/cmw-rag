from __future__ import annotations

from rag_engine.llm.prompts import SYSTEM_PROMPT


def test_system_prompt_contains_required_instructions():
    # Ensure key constraints and language policy are present in the system prompt
    assert "article.php?id=" in SYSTEM_PROMPT or "kb.comindware.ru" in SYSTEM_PROMPT
    assert "Answer in the same language" in SYSTEM_PROMPT
    assert "Context" not in SYSTEM_PROMPT  # placeholder added at runtime

