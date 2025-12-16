"""Integration tests comparing OpenRouter and vLLM behavior for the same model.

These tests are **opt‑in** and hit real LLM endpoints via the OpenAI‑compatible
`ChatOpenAI` client used by `LLMManager`. They are intended for manual
diagnostics when debugging differences between OpenRouter and a vLLM host.

To enable:

    RUN_LLM_INTEGRATION_TESTS=1 python -m pytest rag_engine/tests/test_llm_integration_openrouter_vllm.py -v

Environment requirements:
- OPENROUTER_API_KEY must be set (used by `Settings.openrouter_api_key`)
- VLLM_BASE_URL must point to your vLLM OpenAI‑compatible server
"""
from __future__ import annotations

import os

import pytest

from rag_engine.config.settings import settings
from rag_engine.llm.llm_manager import LLMManager


RUN_LLM_INTEGRATION_TESTS = os.getenv("RUN_LLM_INTEGRATION_TESTS") == "1"


pytestmark = pytest.mark.skipif(
    not RUN_LLM_INTEGRATION_TESTS,
    reason="Set RUN_LLM_INTEGRATION_TESTS=1 to run real OpenRouter/vLLM comparisons.",
)


def _has_required_env() -> bool:
    """Return True if required env for OpenRouter/vLLM tests is present."""
    has_openrouter_key = bool(getattr(settings, "openrouter_api_key", "").strip())
    # vLLM uses an OpenAI‑compatible API; api_key may be unused, base_url must be valid.
    has_vllm_base_url = bool(getattr(settings, "vllm_base_url", "").strip())
    return has_openrouter_key and has_vllm_base_url


@pytest.mark.skipif(
    not _has_required_env(),
    reason="OpenRouter/vLLM env vars not configured (OPENROUTER_API_KEY and VLLM_BASE_URL).",
)
def test_openrouter_and_vllm_same_prompt_non_empty(caplog: pytest.LogCaptureFixture) -> None:
    """Compare OpenRouter and vLLM answers for the same model/prompt.

    This test is primarily diagnostic:
    - Uses `LLMManager` with provider `openrouter` and `vllm`
    - Sends the same short, deterministic question and tiny context
    - Asserts that both answers are non‑empty
    - Logs both raw answers to aid manual inspection when failures occur
    """
    model_name = "openai/gpt-oss-20b"

    # Sanity‑log current endpoints so failures are easier to interpret.
    caplog.set_level("INFO")
    caplog.clear()

    openrouter_mgr = LLMManager(provider="openrouter", model=model_name, temperature=0)
    vllm_mgr = LLMManager(provider="vllm", model=model_name, temperature=0)

    question = "Reply with the single word: Paris."
    # Use minimal context so token budgeting cannot be the cause of empty outputs.
    docs = [type("Doc", (), {"content": "Context for testing OpenRouter vs vLLM."})()]

    openrouter_answer = openrouter_mgr.generate(question, docs, provider="openrouter")
    vllm_answer = vllm_mgr.generate(question, docs, provider="vllm")

    # Log raw answers for manual diagnostics (visible with -s or in CI logs).
    print("\n[OpenRouter answer]\n", openrouter_answer)  # noqa: T201
    print("\n[vLLM answer]\n", vllm_answer)  # noqa: T201

    assert openrouter_answer is not None
    assert vllm_answer is not None
    assert openrouter_answer.strip() != ""
    # This assertion will highlight the "empty vLLM response" failure mode.
    assert vllm_answer.strip() != ""


