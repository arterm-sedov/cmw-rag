import json
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from rag_engine.config.settings import settings


def pretty_print(title: str, obj: Any) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if isinstance(obj, dict):
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    else:
        print(obj)
    print()


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    """Best-effort conversion of OpenRouter usage object to plain dict."""
    if usage is None:
        return {}
    # Newer OpenAI clients expose Pydantic-style model_dump()
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    # Fallback: build dict from attributes commonly present on usage objects
    fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
        "cost_details",
        "prompt_tokens_details",
        "completion_tokens_details",
    )
    data: dict[str, Any] = {}
    for key in fields:
        if hasattr(usage, key):
            data[key] = getattr(usage, key)
    return data or {"raw": repr(usage)}


def main() -> None:
    """Ad-hoc script to inspect OpenRouter token usage and pricing for a single QA turn.

    Uses the same model and endpoint configuration as the main agent via settings/.env,
    but talks to OpenRouter directly to expose the raw `usage` structure, including
    `cost` and `cost_details.upstream_inference_cost` where available.
    """
    load_dotenv()

    api_key = settings.openrouter_api_key
    base_url = settings.openrouter_base_url
    model_name = settings.default_model

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured in .env / settings.")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    question = (
        "Кратко ответь на вопрос пользователя. "
        "Это тестовый QA-запрос для измерения токенов и стоимости через OpenRouter. "
        "Вопрос: как в Comindware Platform найти актуальную версию документации по RAG-движку?"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Ты помощник для RAG-движка Comindware. Отвечай кратко и по делу. "
                "Этот вызов нужен для измерения usage/cost через OpenRouter."
            ),
        },
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    assistant_message = response.choices[0].message
    pretty_print("Assistant answer", assistant_message.content)

    usage_obj = getattr(response, "usage", None)
    usage = _usage_to_dict(usage_obj)
    pretty_print("Raw usage payload from OpenRouter", usage)

    cost = usage.get("cost")
    cost_details = usage.get("cost_details") or {}
    upstream_cost = None
    if isinstance(cost_details, dict):
        upstream_cost = cost_details.get("upstream_inference_cost")

    print("\n" + "=" * 80)
    print("Cost summary")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Total tokens: {usage.get('total_tokens')}")
    print(f"Prompt tokens: {usage.get('prompt_tokens')}")
    print(f"Completion tokens: {usage.get('completion_tokens')}")
    print(f"OpenRouter credits (usage.cost): {cost}")
    if upstream_cost is not None:
        print(f"Upstream inference cost (usage.cost_details.upstream_inference_cost): {upstream_cost}")
    print()


if __name__ == "__main__":
    main()

