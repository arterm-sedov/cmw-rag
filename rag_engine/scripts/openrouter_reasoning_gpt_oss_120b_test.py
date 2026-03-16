import json
import os
import traceback
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


MODEL = os.getenv("REASONING_TEST_MODEL", "openai/gpt-oss-120b")


def pretty_print(title: str, obj: Any) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if isinstance(obj, dict):
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    else:
        print(obj)
    print()


def make_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment or .env")
    return OpenAI(base_url=base_url, api_key=api_key)


def test_reasoning_with_gpt_oss_120b(client: OpenAI) -> None:
    print("\n--- Reasoning test: openai/gpt-oss-120b ---")
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Ответь очень кратко (1-2 предложения) на вопрос: "
                        "\"Как определить последнюю доступную версию Comindware Platform по документации?\" "
                        "Если у тебя есть отдельное поле для reasoning / thinking, заполни его, "
                        "но в основном ответе не раскрывай пошаговые рассуждения."
                    ),
                }
            ],
            # Pass OpenRouter reasoning config via extra_body to inspect how it is surfaced.
            extra_body={
                "reasoning": {
                    # Use explicit max_tokens so behavior matches our LLM manager logic.
                    "max_tokens": 512,
                    # Keep exclude=False so we can see if reasoning is actually returned.
                    "exclude": False,
                }
            },
        )

        data = resp.model_dump()
        pretty_print("Raw response (truncated top-level)", {k: data.get(k) for k in ("id", "model", "object")})

        choices = data.get("choices", [])
        pretty_print("Number of choices", len(choices))
        if not choices:
            return

        first_choice = choices[0]
        pretty_print("First choice keys", list(first_choice.keys()))

        message = first_choice.get("message", {}) or {}
        pretty_print("message keys", list(message.keys()))

        content = message.get("content", "")
        if isinstance(content, str):
            pretty_print("message.content (truncated)", content[:800])
        else:
            pretty_print("message.content (raw)", content)

        # Try to locate any dedicated reasoning field, if the model/provider exposes it.
        reasoning = message.get("reasoning") or first_choice.get("reasoning")
        if reasoning:
            pretty_print("message.reasoning / choice.reasoning", reasoning)
        else:
            pretty_print("message.reasoning / choice.reasoning", "NOT PRESENT")

    except Exception as exc:  # noqa: BLE001
        pretty_print("Exception during reasoning test", str(exc))
        traceback.print_exc()


def main() -> None:
    client = make_client()
    test_reasoning_with_gpt_oss_120b(client)


if __name__ == "__main__":
    main()

