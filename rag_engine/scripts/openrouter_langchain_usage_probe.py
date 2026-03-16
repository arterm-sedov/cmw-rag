from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from rag_engine.config.settings import settings
from rag_engine.llm.llm_manager import LLMManager


@dataclass
class LangChainUsageSnapshot:
    """What LangChain exposes about usage and metadata for a single call."""

    token_usage: dict[str, Any] | None
    llm_output: dict[str, Any] | None
    generation_info: dict[str, Any] | None
    response_metadata: dict[str, Any] | None
    additional_kwargs: dict[str, Any] | None


class UsageCollectingHandler(BaseCallbackHandler):
    """Capture the final LLMResult for inspection."""

    def __init__(self) -> None:
        self.last_result: LLMResult | None = None

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:  # type: ignore[override]
        self.last_result = response


def _pp(title: str, payload: Any) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)
    print()


def _extract_langchain_usage(handler: UsageCollectingHandler) -> LangChainUsageSnapshot:
    result = handler.last_result
    if result is None:
        return LangChainUsageSnapshot(
            token_usage=None,
            llm_output=None,
            generation_info=None,
            response_metadata=None,
            additional_kwargs=None,
        )

    llm_output = result.llm_output if isinstance(result.llm_output, dict) else None
    token_usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None

    # For ChatOpenAI, generations[0][0].generation_info may contain provider-specific extras.
    generation_info: dict[str, Any] | None = None
    if result.generations and result.generations[0]:
        gen = result.generations[0][0]
        gi = getattr(gen, "generation_info", None)
        if isinstance(gi, dict):
            generation_info = gi

    # For chat models, output is AIMessage.
    response_metadata: dict[str, Any] | None = None
    additional_kwargs: dict[str, Any] | None = None
    if result.generations and result.generations[0]:
        gen = result.generations[0][0]
        msg = getattr(gen, "message", None)
        if msg is not None:
            rm = getattr(msg, "response_metadata", None)
            if isinstance(rm, dict):
                response_metadata = rm
            ak = getattr(msg, "additional_kwargs", None)
            if isinstance(ak, dict):
                additional_kwargs = ak

    return LangChainUsageSnapshot(
        token_usage=token_usage,
        llm_output=llm_output,
        generation_info=generation_info,
        response_metadata=response_metadata,
        additional_kwargs=additional_kwargs,
    )


def _extract_generation_id(snapshot: LangChainUsageSnapshot) -> str | None:
    """Best-effort extraction of OpenRouter generation id from LangChain metadata."""
    # Try response_metadata first.
    rm = snapshot.response_metadata or {}
    if isinstance(rm, dict):
        if "id" in rm and isinstance(rm["id"], str):
            return rm["id"]
        # Sometimes providers nest metadata.
        for key in ("openrouter", "metadata", "raw"):
            nested = rm.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("id"), str):
                return nested["id"]

    # Then additional_kwargs.
    ak = snapshot.additional_kwargs or {}
    if isinstance(ak, dict):
        if "id" in ak and isinstance(ak["id"], str):
            return ak["id"]
        for key in ("openrouter", "metadata", "raw"):
            nested = ak.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("id"), str):
                return nested["id"]

    # Finally, try llm_output/generation_info if they contain a raw response.
    for container in (snapshot.llm_output, snapshot.generation_info):
        if not isinstance(container, dict):
            continue
        if "id" in container and isinstance(container["id"], str):
            return container["id"]
        nested = container.get("raw_response")
        if isinstance(nested, dict) and isinstance(nested.get("id"), str):
            return nested["id"]

    return None


def _fetch_generation_by_id(generation_id: str) -> dict[str, Any] | None:
    """Call OpenRouter /generation to fetch stored usage.

    API: GET https://openrouter.ai/api/v1/generation?id=<generation_id>
    Ref: https://openrouter.ai/docs/api/api-reference/generations/get-generation
    """
    api_key = settings.openrouter_api_key
    base_url = settings.openrouter_base_url.rstrip("/")
    if not api_key:
        return None

    url = f"{base_url}/generation"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.get(url, headers=headers, params={"id": generation_id}, timeout=30)
    if not resp.ok:
        _pp("OpenRouter /generation error", {"status": resp.status_code, "text": resp.text})
        return None
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


def main() -> None:
    """Probe what usage and IDs LangChain preserves for OpenRouter ChatOpenAI."""
    load_dotenv()

    model_name = settings.default_model
    manager = LLMManager(
        provider="openrouter",
        model=model_name,
        temperature=settings.llm_temperature,
    )
    chat_model = manager._chat_model(provider="openrouter")

    handler = UsageCollectingHandler()

    question = (
        "Сделай очень краткое (2–3 предложения) описание того, как CMW RAG Engine "
        "использует Retrieval-Augmented Generation для ответов пользователям."
    )

    _pp(
        "Probe config",
        {
            "model": model_name,
            "provider": "openrouter",
        },
    )

    response = chat_model.invoke(question, config={"callbacks": [handler]})

    _pp("LangChain ChatOpenAI raw response.content", getattr(response, "content", None))

    snapshot = _extract_langchain_usage(handler)

    _pp("LangChain llm_output", snapshot.llm_output or {})
    _pp("LangChain token_usage (if any)", snapshot.token_usage or {})
    _pp("LangChain generation_info (if any)", snapshot.generation_info or {})
    _pp("AIMessage.response_metadata", snapshot.response_metadata or {})
    _pp("AIMessage.additional_kwargs", snapshot.additional_kwargs or {})

    generation_id = _extract_generation_id(snapshot)
    _pp("Resolved OpenRouter generation_id", generation_id or "NOT FOUND")

    if generation_id:
        generation_payload = _fetch_generation_by_id(generation_id)
        _pp("OpenRouter /generation payload", generation_payload or {})


if __name__ == "__main__":
    main()

