from __future__ import annotations

from rag_engine.llm.token_utils import estimate_tokens_for_request


def test_estimate_tokens_for_request_counts_components():
    out = estimate_tokens_for_request(
        system_prompt="sys",
        question="q",
        context="abc",
        max_output_tokens=50,
        overhead=10,
    )
    assert out["output_tokens"] == 50
    assert out["total_tokens"] >= out["output_tokens"]
    assert out["input_tokens"] > 0


