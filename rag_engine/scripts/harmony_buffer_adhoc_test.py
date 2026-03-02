"""Ad-hoc test: stream from GPT-OSS via our LangChain harness WITH TOOLS
to provoke the full Harmony flow (analysis → commentary → assistantfinal).

Harmony markers only appear when the model is given tools — the analysis and
commentary channels carry chain-of-thought and tool calls/responses. Without
tools, OpenRouter strips the Harmony formatting and returns clean text.

Uses the same ChatOpenAI + extra_body + bind_tools path as the real agent.
"""

import asyncio
import json
import textwrap
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool

from rag_engine.api.app import _strip_thinking_tags_from_chunk
from rag_engine.api.harmony_parser import HarmonyStreamParser
from rag_engine.api.harmony_parser import split as harmony_split
from rag_engine.config.settings import settings
from rag_engine.llm.llm_manager import LLMManager, _build_reasoning_extra_body


def pretty(title: str, obj: Any) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if isinstance(obj, dict):
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    else:
        text = str(obj)
        print(textwrap.shorten(text, width=3000, placeholder="\n...[truncated]"))
    print()


@tool
def retrieve_context(query: str, top_k: int = 5) -> dict:
    """Search the knowledge base for articles matching the query."""
    return {
        "articles": [
            {
                "kb_id": "5063",
                "title": "Установка ПО в Windows",
                "url": "https://kb.comindware.ru/article.php?id=5063",
                "content": (
                    "Системные требования: Windows Server 2019+, .NET 6, Java 11. "
                    "Скачайте дистрибутив, запустите version_install.ps1, "
                    "затем выполните инициализацию через http://localhost:8080. "
                    "Версия: 5.0.13334.0"
                ),
            },
            {
                "kb_id": "4622",
                "title": "Установка ПО в Linux",
                "url": "https://kb.comindware.ru/article.php?id=4622",
                "content": (
                    "Ubuntu 22.04+, 16 GB RAM. Выполните: "
                    "sudo ./comindware_install.sh --accept-eula"
                ),
            },
        ],
        "metadata": {
            "query": query,
            "top_k_requested": top_k,
            "articles_count": 2,
            "has_results": True,
        },
    }


COMPLEX_QUESTION = (
    "Как установить Comindware Platform? Какая последняя версия? "
    "Используй инструмент поиска чтобы найти информацию."
)

HARMONY_MARKERS = ("assistantfinal", "assistantanalysis", "assistantcommentary")


async def run() -> None:
    load_dotenv()

    model_name = settings.default_model
    manager = LLMManager(
        provider="openrouter",
        model=model_name,
        temperature=settings.llm_temperature,
    )
    chat_model = manager._chat_model(provider="openrouter")

    # Bind tools — this triggers Harmony-format tool calling.
    model_with_tools = chat_model.bind_tools(
        [retrieve_context],
        tool_choice="auto",
    )

    pretty("Config", {
        "model": model_name,
        "reasoning_extra_body": _build_reasoning_extra_body() or {},
        "tools": ["retrieve_context"],
    })

    messages = [{"role": "user", "content": COMPLEX_QUESTION}]

    # --- Multi-turn: stream, handle tool calls, feed results back ---
    raw_chunks: list[str] = []
    reasoning_block_chunks: list[str] = []
    tool_calls_seen: list[dict] = []
    max_turns = 5  # safety cap

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1}: streaming from model ---")
        accumulated_tool_calls: dict[int, dict] = {}
        turn_text_chunks: list[str] = []

        async for chunk in model_with_tools.astream(messages):
            # Collect tool calls
            tc_list = getattr(chunk, "tool_call_chunks", None)
            if tc_list:
                for tc_chunk in tc_list:
                    idx = tc_chunk.get("index", 0)
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "name": tc_chunk.get("name") or "",
                            "args": tc_chunk.get("args") or "",
                            "id": tc_chunk.get("id") or "",
                        }
                    else:
                        entry = accumulated_tool_calls[idx]
                        entry["name"] = entry["name"] or tc_chunk.get("name") or ""
                        entry["args"] += tc_chunk.get("args") or ""
                        entry["id"] = entry["id"] or tc_chunk.get("id") or ""

            # Collect text
            content_blocks = getattr(chunk, "content_blocks", None)
            if content_blocks:
                for block in content_blocks:
                    btype = block.get("type")
                    if btype == "reasoning":
                        r = block.get("reasoning") or block.get("text") or ""
                        if r:
                            reasoning_block_chunks.append(str(r))
                    elif btype == "text" and block.get("text"):
                        turn_text_chunks.append(str(block["text"]))
            else:
                content = getattr(chunk, "content", None)
                if content:
                    turn_text_chunks.append(str(content))

        raw_chunks.extend(turn_text_chunks)
        turn_text = "".join(turn_text_chunks)

        if accumulated_tool_calls:
            # Execute tool calls and add results to messages.
            from langchain_core.messages import AIMessage, ToolMessage

            ai_tool_calls = []
            for _idx, tc in sorted(accumulated_tool_calls.items()):
                ai_tool_calls.append({
                    "name": tc["name"],
                    "args": json.loads(tc["args"]) if tc["args"] else {},
                    "id": tc["id"],
                })
                tool_calls_seen.append(tc)

            messages.append(AIMessage(content=turn_text, tool_calls=ai_tool_calls))

            for tc_info in ai_tool_calls:
                result = retrieve_context.invoke(tc_info["args"])
                result_str = json.dumps(result, ensure_ascii=False)
                messages.append(ToolMessage(content=result_str, tool_call_id=tc_info["id"]))
                print(f"  Tool call: {tc_info['name']}({tc_info['args']}) → {len(result_str)} chars")
        else:
            print(f"  No tool calls — model produced {len(turn_text)} chars of text.")
            break

    raw_text = "".join(raw_chunks)

    pretty("Raw text from LangChain (first 3000 chars)", raw_text[:3000])
    if reasoning_block_chunks:
        pretty("Reasoning from content_blocks", "".join(reasoning_block_chunks)[:1500])
    if tool_calls_seen:
        pretty("Tool calls made", [{"name": tc["name"], "args_len": len(tc["args"])} for tc in tool_calls_seen])

    # --- One-shot split ---
    reasoning_oneshot, final_oneshot = harmony_split(raw_text)
    pretty("One-shot split — reasoning (first 2000 chars)", reasoning_oneshot[:2000])
    pretty("One-shot split — final (first 2000 chars)", final_oneshot[:2000])

    # --- Streaming parser (simulates real streaming in the Gradio loop) ---
    parser = HarmonyStreamParser()
    reasoning_buffer = ""
    in_think_block = False
    all_final = ""

    for chunk_text in raw_chunks:
        chunk_text, reasoning_buffer, in_think_block = _strip_thinking_tags_from_chunk(
            chunk_text, reasoning_buffer, in_think_block
        )
        h_reasoning, h_final = parser.feed(chunk_text)
        if h_reasoning:
            if reasoning_buffer:
                reasoning_buffer += "\n"
            reasoning_buffer += h_reasoning
        all_final += h_final

    # Flush remaining buffer.
    h_reasoning, h_final = parser.flush()
    if h_reasoning:
        if reasoning_buffer:
            reasoning_buffer += "\n"
        reasoning_buffer += h_reasoning
    all_final += h_final

    pretty("Streaming parser — reasoning (first 2000 chars)", reasoning_buffer[:2000])
    pretty("Streaming parser — final answer (first 2000 chars)", all_final[:2000])

    # --- Verification ---
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    leaked = [m for m in HARMONY_MARKERS if m in all_final]
    if leaked:
        print(f"  FAIL: Harmony markers leaked into final answer: {leaked}")
    else:
        print("  PASS: No Harmony channel markers in final answer.")

    if all_final.strip():
        print(f"  PASS: Final answer is non-empty ({len(all_final)} chars).")
    else:
        print("  WARN: Final answer is empty — model may not have emitted assistantfinal.")

    if reasoning_buffer.strip():
        print(f"  PASS: Reasoning captured ({len(reasoning_buffer)} chars).")
    else:
        print("  WARN: No reasoning captured.")

    has_analysis_in_raw = any(m in raw_text for m in HARMONY_MARKERS) or raw_text.lstrip().startswith("analysis")
    if has_analysis_in_raw:
        print("  INFO: Raw text DID contain Harmony markers — splitting was exercised.")
    else:
        print("  INFO: Raw text had NO Harmony markers — model may not have used Harmony format this time.")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
