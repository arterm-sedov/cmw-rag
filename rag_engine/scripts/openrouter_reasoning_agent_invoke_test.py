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


def main() -> None:
    """Invoke the same Chat model the agent uses and inspect reasoning fields."""
    load_dotenv()

    model_name = settings.default_model or "openai/gpt-oss-120b"
    manager = LLMManager(
        provider="openrouter",
        model=model_name,
        temperature=settings.llm_temperature,
    )
    chat_model = manager._chat_model(provider="openrouter")

    pretty_print("Using model", {"model": model_name, "provider": "openrouter"})
    pretty_print("Reasoning extra_body", _build_reasoning_extra_body() or {})

    question = (
        "Ответь очень кратко (1–2 предложения) на вопрос: "
        "\"Как определить последнюю доступную версию Comindware Platform по документации?\" "
        "Если у тебя есть отдельные поля reasoning / reasoning_details, "
        "используй их для внутренних размышлений, а в основном ответе не раскрывай шаги рассуждений."
    )

    print("\nCalling ChatOpenAI.invoke() with LangChain harness (no tools)...")
    response = chat_model.invoke(question)

    # Basic shape
    pretty_print("Response type", type(response).__name__)

    # Core fields
    content = getattr(response, "content", None)
    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
    response_metadata = getattr(response, "response_metadata", {}) or {}

    if isinstance(content, str):
        pretty_print("response.content (truncated)", content[:2000])
    else:
        pretty_print("response.content (raw)", content)

    pretty_print("response.additional_kwargs keys", list(additional_kwargs.keys()))
    pretty_print(
        "response_metadata keys",
        list(response_metadata.keys()) if isinstance(response_metadata, dict) else response_metadata,
    )

    # Try common locations where OpenRouter may surface reasoning
    reasoning = {}
    for key in ("reasoning", "reasoning_details"):
        if key in additional_kwargs:
            reasoning[f"additional_kwargs.{key}"] = additional_kwargs[key]
        if isinstance(response_metadata, dict) and key in response_metadata:
            reasoning[f"response_metadata.{key}"] = response_metadata[key]

    if reasoning:
        pretty_print("Extracted reasoning-related payload", reasoning)
    else:
        pretty_print("Extracted reasoning-related payload", "NOT PRESENT")


if __name__ == "__main__":
    main()

