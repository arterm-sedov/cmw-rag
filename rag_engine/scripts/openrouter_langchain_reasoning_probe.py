from __future__ import annotations

import asyncio
import json
from typing import Any

from dotenv import load_dotenv

from rag_engine.config.settings import settings
from rag_engine.llm.llm_manager import LLMManager


def _pp(title: str, payload: Any) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)
    print()


async def run_reasoning_probe() -> None:
    """Probe raw OpenRouter reasoning as seen by our LangChain harness.

    This bypasses Harmony / <think> parsing and shows exactly what the
    OpenRouter-native client surfaces via LangChain:

    - invoke(): final AIMessage.additional_kwargs["reasoning_content"]
    - astream(): per-chunk AIMessageChunk.additional_kwargs["reasoning_content"]
    """
    load_dotenv()

    model_name = settings.default_model
    manager = LLMManager(
        provider="openrouter",
        model=model_name,
        temperature=settings.llm_temperature,
    )
    chat_model = manager._chat_model(provider="openrouter")

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Смоделируй размышление с включённым reasoning: сначала подробно распиши, "
                "как ты будешь разбирать запрос пользователя по шагам (с указанием "
                "ограничений и плана), потом выведи короткий итоговый ответ.\n\n"
                "Важно: не сокращай и не прячь рассуждения, веди себя так, как будто "
                "reasoning-токены будут видны оператору."
            ),
        }
    ]

    _pp(
        "Reasoning probe config",
        {
            "model": model_name,
            "provider": "openrouter",
            "llm_reasoning_enabled": getattr(settings, "llm_reasoning_enabled", False),
            "llm_reasoning_max_tokens": getattr(settings, "llm_reasoning_max_tokens", None),
            "llm_reasoning_effort": getattr(settings, "llm_reasoning_effort", None),
            "llm_reasoning_exclude_from_response": getattr(
                settings, "llm_reasoning_exclude_from_response", False
            ),
        },
    )

    # --- Non-streaming: invoke() ------------------------------------------------
    response = await chat_model.ainvoke(messages)
    base_ak = getattr(response, "additional_kwargs", {}) or {}

    _pp("invoke(): AIMessage.content", getattr(response, "content", None))
    _pp("invoke(): AIMessage.additional_kwargs", base_ak)
    _pp("invoke(): raw reasoning_content (if any)", base_ak.get("reasoning_content"))

    # --- Streaming: astream() ---------------------------------------------------
    print("\n" + "#" * 90)
    print("STREAMING: per-chunk reasoning_content from OpenRouter via LangChain")
    print("#" * 90)

    chunk_index = 0
    async for chunk in chat_model.astream(messages):
        chunk_index += 1
        role = getattr(chunk, "role", None)
        content = getattr(chunk, "content", None)
        ak = getattr(chunk, "additional_kwargs", {}) or {}
        reasoning = ak.get("reasoning_content")

        preview = content
        if isinstance(content, str):
            preview = content[:160]

        _pp(
            f"astream(): chunk #{chunk_index}",
            {
                "type": type(chunk).__name__,
                "role": role,
                "content_preview": preview,
                "has_reasoning_content": reasoning is not None,
            },
        )
        if reasoning is not None:
            _pp(f"astream(): chunk #{chunk_index} reasoning_content", reasoning)


def main() -> None:
    asyncio.run(run_reasoning_probe())


if __name__ == "__main__":
    main()

