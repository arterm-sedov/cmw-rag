from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from rag_engine.config.settings import settings
from rag_engine.llm.llm_manager import LLMManager


@dataclass
class LangChainUsageSnapshot:
    token_usage: dict[str, Any] | None
    llm_output: dict[str, Any] | None
    generation_info: dict[str, Any] | None
    response_metadata: dict[str, Any] | None
    additional_kwargs: dict[str, Any] | None


class UsageCollectingHandler(BaseCallbackHandler):
    """Capture the final LLMResult for streaming tests."""

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

    generation_info: dict[str, Any] | None = None
    if result.generations and result.generations[0]:
        gen = result.generations[0][0]
        gi = getattr(gen, "generation_info", None)
        if isinstance(gi, dict):
            generation_info = gi

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
    rm = snapshot.response_metadata or {}
    if isinstance(rm, dict):
        if isinstance(rm.get("id"), str):
            return rm["id"]
        for key in ("openrouter", "metadata", "raw"):
            nested = rm.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("id"), str):
                return nested["id"]

    ak = snapshot.additional_kwargs or {}
    if isinstance(ak, dict):
        if isinstance(ak.get("id"), str):
            return ak["id"]
        for key in ("openrouter", "metadata", "raw"):
            nested = ak.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("id"), str):
                return nested["id"]

    for container in (snapshot.llm_output, snapshot.generation_info):
        if not isinstance(container, dict):
            continue
        if isinstance(container.get("id"), str):
            return container["id"]
        nested = container.get("raw_response")
        if isinstance(nested, dict) and isinstance(nested.get("id"), str):
            return nested["id"]

    return None


async def run_stream_usage_probe() -> None:
    """Stream from ChatOpenAI via LangChain and inspect usage + generation id."""
    load_dotenv()

    model_name = settings.default_model
    manager = LLMManager(
        provider="openrouter",
        model=model_name,
        temperature=settings.llm_temperature,
    )
    chat_model = manager._chat_model(provider="openrouter")

    handler = UsageCollectingHandler()

    messages = [
        {
            "role": "user",
            "content": (
                "Кратко опиши (2–3 предложения), как CMW RAG Engine использует RAG "
                "для ответов пользователям. Это стриминговый тест usage/generation_id."
            ),
        }
    ]

    _pp(
        "Stream usage probe config",
        {
            "model": model_name,
            "provider": "openrouter",
        },
    )

    last_chunk: Any = None
    async for chunk in chat_model.astream(messages, config={"callbacks": [handler]}):
        last_chunk = chunk
        # Show only lightweight info to avoid flooding output.
        role = getattr(chunk, "role", None)
        content = getattr(chunk, "content", None)
        if isinstance(content, str):
            content_preview = content[:120]
        else:
            content_preview = str(content)[:120]
        _pp(
            "Stream chunk (preview)",
            {
                "type": type(chunk).__name__,
                "role": role,
                "content_preview": content_preview,
            },
        )

    if last_chunk is not None:
        _pp(
            "Final stream chunk (raw repr)",
            repr(last_chunk),
        )

    snapshot = _extract_langchain_usage(handler)

    _pp("LangChain llm_output (stream)", snapshot.llm_output or {})
    _pp("LangChain token_usage (stream, if any)", snapshot.token_usage or {})
    _pp("LangChain generation_info (stream, if any)", snapshot.generation_info or {})
    _pp("AIMessage.response_metadata (stream)", snapshot.response_metadata or {})
    _pp("AIMessage.additional_kwargs (stream)", snapshot.additional_kwargs or {})

    generation_id = _extract_generation_id(snapshot)
    _pp("Resolved OpenRouter generation_id (stream)", generation_id or "NOT FOUND")


def main() -> None:
    asyncio.run(run_stream_usage_probe())


if __name__ == "__main__":
    main()

