import json
import os
import traceback
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "qwen/qwen3.5-122b-a10b"


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


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in UTC as an ISO 8601 string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]


def test_simple_chat(client: OpenAI) -> None:
    print("\n--- Test 1: Simple chat, no tools ---")
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        )
        pretty_print("Test 1 raw response", resp.model_dump())
    except Exception as exc:  # noqa: BLE001
        pretty_print("Test 1 exception", str(exc))
        traceback.print_exc()


def test_tools_auto(client: OpenAI) -> None:
    print("\n--- Test 2: Tools with tool_choice='auto' ---")
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Call the tool to get the current time.",
                }
            ],
            tools=TOOLS,
            tool_choice="auto",
        )
        pretty_print("Test 2 raw response", resp.model_dump())
    except Exception as exc:  # noqa: BLE001
        pretty_print("Test 2 exception", str(exc))
        traceback.print_exc()


def test_tools_forced(client: OpenAI) -> None:
    print("\n--- Test 3: Tools with forced tool_choice ---")
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Call the tool to get the current time.",
                }
            ],
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "get_current_time"}},
        )
        pretty_print("Test 3 raw response", resp.model_dump())
    except Exception as exc:  # noqa: BLE001
        pretty_print("Test 3 exception", str(exc))
        traceback.print_exc()


def main() -> None:
    client = make_client()
    test_simple_chat(client)
    test_tools_auto(client)
    test_tools_forced(client)


if __name__ == "__main__":
    main()

