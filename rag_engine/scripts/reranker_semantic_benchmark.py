"""Reranker Benchmark with Semantic Search using OpenRouter Embeddings."""

import sys
import json
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from chromadb import HttpClient
from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter
from rag_engine.config.settings import settings

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"

QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

# Instructions to test
INSTRUCTIONS = {
    "default_web": "Given a web search query, retrieve relevant passages that answer the query",
    "platform_en": "Find documentation about Comindware Platform features, configurations, and APIs",
    "integration_en": "Find integration guides, API documentation, and configuration examples",
    "code_en": "Find code examples, API references, and technical documentation",
    "detailed_en": "Retrieve Comindware Platform documentation: integration guides, API references, configuration examples",
    "platform_ru": "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации",
    "bilingual_hint": "Find relevant documentation (Russian or English) with code examples and configuration guides",
}

# Realistic questions
TEST_QUESTIONS = [
    # ---- ENGLISH ONLY ----
    {"q": "OData integration setup", "type": "keyword", "lang": "en"},
    {"q": "HTTP POST request", "type": "keyword", "lang": "en"},
    {"q": "Keycloak authentication", "type": "keyword", "lang": "en"},
    {"q": "How to configure OData integration?", "type": "natural", "lang": "en"},
    {"q": "How to send HTTP POST request from script?", "type": "natural", "lang": "en"},
    {"q": "What is a data transfer path?", "type": "natural", "lang": "en"},
    # ---- RUSSIAN ONLY ----
    {"q": "интеграция OData", "type": "keyword", "lang": "ru"},
    {"q": "HTTP POST запрос", "type": "keyword", "lang": "ru"},
    {"q": "настройка Keycloak", "type": "keyword", "lang": "ru"},
    {"q": "Как настроить интеграцию с 1С?", "type": "natural", "lang": "ru"},
    {"q": "Как отправить HTTP запрос из сценария?", "type": "natural", "lang": "ru"},
    {"q": "Что такое путь передачи данных?", "type": "natural", "lang": "ru"},
    # ---- MIXED (Russian + English) ----
    {"q": "OData синхронизация", "type": "keyword", "lang": "mixed"},
    {"q": "REST API подключение", "type": "keyword", "lang": "mixed"},
    {"q": "C# скрипт пример", "type": "keyword", "lang": "mixed"},
    {"q": "Как настроить REST API подключение в Comindware?", "type": "natural", "lang": "mixed"},
    {"q": "Пример C# скрипта для HTTP запроса", "type": "natural", "lang": "mixed"},
    {"q": "Как работает OData интеграция в платформе?", "type": "natural", "lang": "mixed"},
]


def get_embedding_from_openrouter(text: str) -> list[float]:
    """Get embedding from OpenRouter API."""
    api_key = settings.openrouter_api_key
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    # Use Qwen3-Embedding-8B
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "qwen/qwen3-embedding-8b",
        "input": text,
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def search_chromadb(collection, query_embedding: list[float], limit: int = 5):
    """Search ChromaDB with embedding."""
    results = collection.query(
        query_embeddings=[query_embedding], n_results=limit, include=["documents", "metadatas"]
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    docs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append(
            {
                "content": doc[:800],
                "source": meta.get("source_file", "?"),
            }
        )

    return docs


def run_benchmark():
    print("=" * 80)
    print("RERANKER BENCHMARK WITH SEMANTIC SEARCH")
    print("=" * 80)
    print(f"Test questions: {len(TEST_QUESTIONS)}")
    print(f"Using: OpenRouter Qwen3-Embedding-8B + ChromaDB_qwen8b")
    print()

    # Connect to ChromaDB (Qwen3 embeddings collection)
    client = HttpClient(host="localhost", port=8000)
    collection = client.get_collection("mkdocs_kb_qwen8b")
    print(f"ChromaDB (Qwen3 Embeddings): {collection.count} documents")
    print()

    # Create reranker adapter
    config = ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/score",
        reranker_type="llm_reranker",
        default_instruction=INSTRUCTIONS["default_web"],
        formatting=QWEN3_FORMATTING,
    )
    adapter = RerankerAdapter(config)

    # Get embeddings and search for each question
    print("Searching for relevant documents (semantic search)...")
    questions_with_docs = []

    for i, q in enumerate(TEST_QUESTIONS):
        print(f"  [{i + 1}/{len(TEST_QUESTIONS)}] {q['q'][:40]}...", end="", flush=True)

        try:
            # Get embedding from OpenRouter
            embedding = get_embedding_from_openrouter(q["q"])

            # Search ChromaDB
            docs = search_chromadb(collection, embedding, limit=5)

            if docs:
                questions_with_docs.append(
                    {
                        "question": q["q"],
                        "type": q["type"],
                        "lang": q["lang"],
                        "docs": docs,
                    }
                )
                print(f" found {len(docs)} docs")
            else:
                print(" no docs")
        except Exception as e:
            print(f" ERROR: {e}")

    print(f"\nReady to test {len(questions_with_docs)} questions")
    print()

    # Test each instruction
    results = {}
    results_by_lang = {"en": {}, "ru": {}, "mixed": {}}
    results_by_type = {"keyword": {}, "natural": {}}

    for name, instruction in INSTRUCTIONS.items():
        print(f"Testing: {name}")
        all_scores = []
        lang_scores = {"en": [], "ru": [], "mixed": []}
        type_scores = {"keyword": [], "natural": []}

        for q in questions_with_docs:
            candidates = [(d["content"], 0.0) for d in q["docs"]]

            try:
                ranked = adapter.rerank(
                    q["question"],
                    candidates,
                    top_k=len(candidates),
                    instruction=instruction,
                )
                scores = [r[1] for r in ranked]
                all_scores.extend(scores)
                lang_scores[q["lang"]].extend(scores)
                type_scores[q["type"]].extend(scores)
            except Exception as e:
                print(f"  Error: {e}")

        if all_scores:
            results[name] = {
                "avg_score": sum(all_scores) / len(all_scores),
                "instruction": instruction,
            }
            for lang in ["en", "ru", "mixed"]:
                if lang_scores[lang]:
                    results_by_lang[lang][name] = sum(lang_scores[lang]) / len(lang_scores[lang])
            for qtype in ["keyword", "natural"]:
                if type_scores[qtype]:
                    results_by_type[qtype][name] = sum(type_scores[qtype]) / len(type_scores[qtype])

            print(f"  Avg: {results[name]['avg_score']:.4f}")
        print()

    # Final summary
    print("=" * 80)
    print("RESULTS (sorted by avg_score)")
    print("=" * 80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    print(f"\n{'Instruction':<25} {'Avg Score':>10}")
    print("-" * 40)

    for name, data in sorted_results:
        print(f"{name:<25} {data['avg_score']:>10.4f}")

    # By language
    print("\n" + "=" * 80)
    print("BY LANGUAGE")
    print("=" * 80)

    for lang in ["en", "ru", "mixed"]:
        if results_by_lang[lang]:
            print(f"\n{lang.upper()}:")
            sorted_lang = sorted(results_by_lang[lang].items(), key=lambda x: x[1], reverse=True)
            for name, score in sorted_lang[:3]:
                print(f"  {name:<25}: {score:.4f}")

    # By type
    print("\n" + "=" * 80)
    print("BY QUESTION TYPE")
    print("=" * 80)

    for qtype in ["keyword", "natural"]:
        if results_by_type[qtype]:
            print(f"\n{qtype.upper()}:")
            sorted_type = sorted(results_by_type[qtype].items(), key=lambda x: x[1], reverse=True)
            for name, score in sorted_type[:3]:
                print(f"  {name:<25}: {score:.4f}")

    # Best instruction
    best = sorted_results[0]
    print("\n" + "=" * 80)
    print("BEST INSTRUCTION")
    print("=" * 80)
    print(f"Name: {best[0]}")
    print(f"Score: {best[1]['avg_score']:.4f}")
    print(f"Instruction: {best[1]['instruction']}")

    # Category analysis
    en_avg = (
        sum(results_by_lang["en"].values()) / len(results_by_lang["en"])
        if results_by_lang["en"]
        else 0
    )
    ru_avg = (
        sum(results_by_lang["ru"].values()) / len(results_by_lang["ru"])
        if results_by_lang["ru"]
        else 0
    )
    mix_avg = (
        sum(results_by_lang["mixed"].values()) / len(results_by_lang["mixed"])
        if results_by_lang["mixed"]
        else 0
    )

    print("\n" + "=" * 80)
    print("CATEGORY AVERAGES")
    print("=" * 80)
    print(f"English questions:   {en_avg:.4f}")
    print(f"Russian questions:   {ru_avg:.4f}")
    print(f"Mixed questions:     {mix_avg:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"reranker_benchmark_semantic_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": {
                    n: {"avg_score": d["avg_score"], "instruction": d["instruction"]}
                    for n, d in results.items()
                },
                "by_language": results_by_lang,
                "by_type": results_by_type,
                "best": {"name": best[0], "score": best[1]["avg_score"]},
                "timestamp": timestamp,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nResults saved: {results_file}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if best[0].endswith("_en"):
        print("Use ENGLISH instruction (per Qwen3 documentation)")
    elif best[0].endswith("_ru"):
        print("RUSSIAN instruction performed best!")
    else:
        print("Use instruction that performed best above")

    print(f'\ndefault_instruction: "{best[1]["instruction"]}"')


if __name__ == "__main__":
    run_benchmark()
