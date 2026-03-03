from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from rag_engine.config.settings import settings


@dataclass
class UsageSample:
    """Normalized snapshot of OpenRouter usage for a single LLM call."""

    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    cost: float
    upstream_cost: float
    raw_usage: dict[str, Any]


class UsageAggregator:
    """Aggregate usage per turn and per conversation in-memory."""

    def __init__(self) -> None:
        self.by_call: list[UsageSample] = []
        self.by_turn: dict[str, list[UsageSample]] = {}
        self.by_conversation: dict[str, list[UsageSample]] = {}

    def add_sample(self, *, conversation_id: str, turn_id: str, sample: UsageSample) -> None:
        self.by_call.append(sample)
        self.by_turn.setdefault(turn_id, []).append(sample)
        self.by_conversation.setdefault(conversation_id, []).append(sample)

    @staticmethod
    def _totals(samples: list[UsageSample]) -> dict[str, float]:
        return {
            "prompt_tokens": sum(s.prompt_tokens for s in samples),
            "completion_tokens": sum(s.completion_tokens for s in samples),
            "total_tokens": sum(s.total_tokens for s in samples),
            "reasoning_tokens": sum(s.reasoning_tokens for s in samples),
            "cached_tokens": sum(s.cached_tokens for s in samples),
            "cache_write_tokens": sum(s.cache_write_tokens for s in samples),
            "cost": float(sum(s.cost for s in samples)),
            "upstream_cost": float(sum(s.upstream_cost for s in samples)),
        }

    def totals_for_turn(self, turn_id: str) -> dict[str, float]:
        return self._totals(self.by_turn.get(turn_id, []))

    def totals_for_conversation(self, conversation_id: str) -> dict[str, float]:
        return self._totals(self.by_conversation.get(conversation_id, []))


def _pp(title: str, obj: Any) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    else:
        print(obj)
    print()


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    # Fallback: best-effort attribute scraping
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


def _sample_from_usage(model: str, usage: dict[str, Any]) -> UsageSample:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

    completion_details = usage.get("completion_tokens_details") or {}
    if not isinstance(completion_details, dict):
        completion_details = {}
    reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)

    prompt_details = usage.get("prompt_tokens_details") or {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    cached_tokens = int(prompt_details.get("cached_tokens") or 0)
    cache_write_tokens = int(prompt_details.get("cache_write_tokens") or 0)

    cost = float(usage.get("cost") or 0.0)
    cost_details = usage.get("cost_details") or {}
    if not isinstance(cost_details, dict):
        cost_details = {}
    upstream_cost = float(cost_details.get("upstream_inference_cost") or 0.0)

    return UsageSample(
        model=model,
        provider="openrouter",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        cost=cost,
        upstream_cost=upstream_cost,
        raw_usage=usage,
    )


def main() -> None:
    """Simulate a short conversation and aggregate OpenRouter usage per turn."""
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

    conversation_id = "demo-conversation"
    prompts = [
        "Кратко опиши, что делает CMW RAG Engine и для чего он используется.",
        "Какие основные компоненты и сервисы входят в типичную архитектуру RAG в Comindware?",
        "Как обычно измерять качество ответов RAG-движка в продакшене?",
    ]

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Ты русскоязычный помощник по Comindware Platform и CMW RAG Engine. "
                "Отвечай кратко (2–3 предложения) и по делу. "
                "Этот диалог используется для измерения usage/cost через OpenRouter."
            ),
        }
    ]

    aggregator = UsageAggregator()

    for idx, user_prompt in enumerate(prompts, start=1):
        turn_id = f"{conversation_id}-turn-{idx}"
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

        assistant_message = response.choices[0].message
        messages.append({"role": "assistant", "content": assistant_message.content or ""})

        usage_dict = _usage_to_dict(getattr(response, "usage", None))
        sample = _sample_from_usage(model_name, usage_dict)
        aggregator.add_sample(conversation_id=conversation_id, turn_id=turn_id, sample=sample)

        _pp(
            f"Turn {idx}: assistant answer (truncated)",
            (assistant_message.content or "")[:800],
        )
        _pp(
            f"Turn {idx}: raw usage",
            usage_dict,
        )
        _pp(
            f"Turn {idx}: aggregated totals",
            aggregator.totals_for_turn(turn_id),
        )

    _pp(
        "Conversation totals",
        aggregator.totals_for_conversation(conversation_id),
    )


if __name__ == "__main__":
    main()

