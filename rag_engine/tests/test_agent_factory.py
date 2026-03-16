from __future__ import annotations

from typing import Any

import pytest

from rag_engine.llm.model_configs import MODEL_CONFIGS


class DummyChatModel:
    def __init__(self) -> None:
        self.bound_tools: list[Any] | None = None
        self.bound_tool_choice: Any | None = None

    def bind_tools(self, tools: list[Any], tool_choice: Any | None = None) -> DummyChatModel:
        self.bound_tools = tools
        self.bound_tool_choice = tool_choice
        return self


class DummySummarizationMiddleware:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        self.args = args
        self.kwargs = kwargs


def _patch_agent_factory_dependencies(monkeypatch, dummy_model: DummyChatModel) -> None:
    from rag_engine.llm import agent_factory as af_mod

    class FakeLLMManager:
        def __init__(self, provider: str, model: str, temperature: float) -> None:
            self.provider = provider
            self.model = model
            self.temperature = temperature

        def _chat_model(self, provider: str | None = None) -> DummyChatModel:  # noqa: ARG002
            return dummy_model

    def fake_create_agent(*args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        return {"args": args, "kwargs": kwargs}

    monkeypatch.setattr(af_mod, "LLMManager", FakeLLMManager)
    monkeypatch.setattr(
        af_mod, "SummarizationMiddleware", DummySummarizationMiddleware
    )
    monkeypatch.setattr(af_mod, "create_agent", fake_create_agent)


@pytest.mark.parametrize(
    "model_name,force_expected,tool_choice_expected",
    [
        ("openai/gpt-oss-120b", True, {"type": "function", "function": {"name": "retrieve_context"}}),
        ("qwen/qwen3.5-122b-a10b", True, "auto"),
    ],
)
def test_create_rag_agent_respects_model_tool_choice_capabilities(
    monkeypatch, model_name: str, force_expected: bool, tool_choice_expected: Any  # noqa: ANN401
) -> None:
    from rag_engine.llm.agent_factory import create_rag_agent

    assert model_name in MODEL_CONFIGS

    dummy_model = DummyChatModel()
    _patch_agent_factory_dependencies(monkeypatch, dummy_model)

    agent = create_rag_agent(override_model=model_name, force_tool_choice=True)
    assert agent  # Agent object was created

    assert dummy_model.bound_tools is not None and len(dummy_model.bound_tools) > 0
    assert dummy_model.bound_tool_choice == tool_choice_expected

