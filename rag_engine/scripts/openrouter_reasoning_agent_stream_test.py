import asyncio
import json
from typing import Any

from dotenv import load_dotenv

from rag_engine.config.settings import settings
from rag_engine.llm.llm_manager import LLMManager, _build_reasoning_extra_body


def pretty_print(title: str, obj: Any) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if isinstance(obj, dict):
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    else:
        print(obj)
    print()


async def run_stream_reasoning_test() -> None:
    """Stream from the same LangChain Chat model the agent uses and inspect reasoning vs content."""
    load_dotenv()

    # Use the same provider/model path as the agent: LLMManager -> _chat_model() with OpenRouter.
    model_name = settings.default_model
    manager = LLMManager(
        provider="openrouter",
        model=model_name,
        temperature=settings.llm_temperature,
    )
    chat_model = manager._chat_model(provider="openrouter")

    pretty_print("Using model", {"model": model_name, "provider": "openrouter"})
    pretty_print("Reasoning extra_body", _build_reasoning_extra_body() or {})

    messages = [
        {
            "role": "user",
            "content": (
                "Ответь очень кратко (1–2 предложения) на вопрос: "
                "\"Как определить последнюю доступную версию Comindware Platform по документации?\" "
                "Если у тебя есть отдельные поля reasoning / reasoning_details, "
                "используй их для внутренних размышлений, а в основном ответе не раскрывай шаги рассуждений."
            ),
        }
    ]

    print("\nStarting ChatOpenAI.astream() with LangChain harness (no tools)...")
    reasoning_chunks: list[str] = []
    text_chunks: list[str] = []

    async for chunk in chat_model.astream(messages):
        # LangChain AIMessageChunk: look for content_blocks first (OpenRouter streaming),
        # then fall back to .reasoning / .content attributes.
        content_blocks = getattr(chunk, "content_blocks", None)
        if content_blocks:
            for block in content_blocks:
                btype = block.get("type")
                if btype == "reasoning":
                    reasoning_text = (
                        block.get("reasoning")
                        or block.get("text")
                        or ""
                    )
                    if reasoning_text:
                        reasoning_chunks.append(str(reasoning_text))
                elif btype == "text" and block.get("text"):
                    text_chunks.append(str(block["text"]))
        else:
            # Non-block streaming: inspect attributes directly (useful for debugging)
            content = getattr(chunk, "content", None)
            reasoning = getattr(chunk, "reasoning", None) or getattr(
                chunk, "reasoning_details", None
            )
            if reasoning:
                reasoning_chunks.append(str(reasoning))
            if content:
                text_chunks.append(str(content))

    final_answer = "".join(text_chunks).strip()
    final_reasoning = "\n".join(reasoning_chunks).strip()

    pretty_print("Final streamed answer (content)", final_answer[:2000])
    pretty_print("Accumulated streamed reasoning", final_reasoning[:2000] or "NOT PRESENT")


def main() -> None:
    asyncio.run(run_stream_reasoning_test())


if __name__ == "__main__":
    main()

