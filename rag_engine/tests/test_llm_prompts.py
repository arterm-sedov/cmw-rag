from __future__ import annotations

from rag_engine.llm.prompts import SYSTEM_PROMPT


def test_system_prompt_contains_required_instructions():
    assert "Cite sources" in SYSTEM_PROMPT
    assert "Answer in the same language" in SYSTEM_PROMPT
    assert "Context" not in SYSTEM_PROMPT  # placeholder added at runtime

