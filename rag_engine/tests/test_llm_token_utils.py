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


def test_estimate_tokens_for_request_uses_fast_path_for_large_strings():
    """Test that very large strings use fast approximation (chars // 4)."""
    # Create a string larger than the threshold (200k chars)
    large_string = "x" * 300_000
    out = estimate_tokens_for_request(
        system_prompt="",
        question="",
        context=large_string,
        max_output_tokens=0,
        overhead=0,
    )
    # Fast path: 300k chars // 4 = 75k tokens
    assert out["input_tokens"] == 75_000
    assert out["total_tokens"] == 75_000


