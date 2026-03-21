"""
Prepare datasets for reranker benchmark.

Pre-fetches documents for all three retrieval methods:
1. RANDOM: Random documents from KB
2. SEMANTIC: Embedding-based semantic search
3. KEYWORD: Keyword/grep matching

Saves datasets once, then benchmarks use the pre-fetched data.
"""

import sys
import json
import random
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from chromadb import HttpClient
from rag_engine.config.settings import settings

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# REALISTIC QUESTIONS - Russian-focused (85% RU, 15% EN/Mixed)
# =============================================================================
TEST_QUESTIONS = [
    # --- RUSSIAN KEYWORDS ---
    {"q": "интеграция 1С", "type": "keyword", "lang": "ru", "topic": "integration"},
    {"q": "OData синхронизация", "type": "keyword", "lang": "ru", "topic": "integration"},
    {"q": "HTTP POST запрос", "type": "keyword", "lang": "ru", "topic": "integration"},
    {"q": "Keycloak настройка", "type": "keyword", "lang": "ru", "topic": "auth"},
    {"q": "ошибка подключения", "type": "keyword", "lang": "ru", "topic": "errors"},
    {"q": "конфигурация атрибутов", "type": "keyword", "lang": "ru", "topic": "config"},
    {"q": "C# скрипт", "type": "keyword", "lang": "ru", "topic": "code"},
    {"q": "сценарий кнопки", "type": "keyword", "lang": "ru", "topic": "scripts"},
    {"q": "шаблон записи", "type": "keyword", "lang": "ru", "topic": "config"},
    {"q": "REST API", "type": "keyword", "lang": "ru", "topic": "api"},
    {"q": "развертывание сервера", "type": "keyword", "lang": "ru", "topic": "infrastructure"},
    {"q": "аутентификация OpenID", "type": "keyword", "lang": "ru", "topic": "auth"},
    {"q": "BPM процессы", "type": "keyword", "lang": "ru", "topic": "platform"},
    {"q": "графовая база данных", "type": "keyword", "lang": "ru", "topic": "platform"},
    {"q": "webhook настройка", "type": "keyword", "lang": "ru", "topic": "integration"},
    {"q": "JSON данные", "type": "keyword", "lang": "ru", "topic": "scripts"},
    {"q": "синхронизация данных", "type": "keyword", "lang": "ru", "topic": "integration"},
    {"q": "подключение базы", "type": "keyword", "lang": "ru", "topic": "infrastructure"},
    {"q": "Low-code платформа", "type": "keyword", "lang": "ru", "topic": "platform"},
    {"q": "атрибут шаблона", "type": "keyword", "lang": "ru", "topic": "config"},
    # --- RUSSIAN NATURAL QUESTIONS ---
    {
        "q": "Как настроить интеграцию с 1С?",
        "type": "natural",
        "lang": "ru",
        "topic": "integration",
    },
    {
        "q": "Как отправить HTTP POST запрос из сценария?",
        "type": "natural",
        "lang": "ru",
        "topic": "scripts",
    },
    {
        "q": "Как подключить аутентификацию через Keycloak?",
        "type": "natural",
        "lang": "ru",
        "topic": "auth",
    },
    {
        "q": "Как исправить ошибку подключения к базе данных?",
        "type": "natural",
        "lang": "ru",
        "topic": "errors",
    },
    {"q": "Как создать шаблон записи?", "type": "natural", "lang": "ru", "topic": "config"},
    {
        "q": "Как использовать C# скрипты в сценариях?",
        "type": "natural",
        "lang": "ru",
        "topic": "scripts",
    },
    {
        "q": "Как настроить синхронизацию данных?",
        "type": "natural",
        "lang": "ru",
        "topic": "integration",
    },
    {
        "q": "Что такое OData и как его использовать?",
        "type": "natural",
        "lang": "ru",
        "topic": "integration",
    },
    {
        "q": "Какие типы HTTP запросов поддерживаются?",
        "type": "natural",
        "lang": "ru",
        "topic": "api",
    },
    {
        "q": "Как развернуть платформу на сервере?",
        "type": "natural",
        "lang": "ru",
        "topic": "infrastructure",
    },
    {
        "q": "Как настроить путь передачи данных?",
        "type": "natural",
        "lang": "ru",
        "topic": "integration",
    },
    {
        "q": "Как обработать JSON ответ сервера?",
        "type": "natural",
        "lang": "ru",
        "topic": "scripts",
    },
    {
        "q": "Какие атрибуты можно использовать в шаблоне?",
        "type": "natural",
        "lang": "ru",
        "topic": "config",
    },
    {
        "q": "Как работает автоматическая синхронизация?",
        "type": "natural",
        "lang": "ru",
        "topic": "integration",
    },
    {"q": "Как создать сценарий на кнопке?", "type": "natural", "lang": "ru", "topic": "scripts"},
    {
        "q": "Пример C# скрипта для HTTP запроса",
        "type": "natural",
        "lang": "ru",
        "topic": "scripts",
    },
    {"q": "Как получить данные из JSON?", "type": "natural", "lang": "ru", "topic": "scripts"},
    {
        "q": "Webhook vs REST API в чем разница?",
        "type": "natural",
        "lang": "ru",
        "topic": "integration",
    },
    # --- MIXED(15%) ---
    {"q": "REST API подключение", "type": "keyword", "lang": "mixed", "topic": "api"},
    {"q": "webhook настройка", "type": "keyword", "lang": "mixed", "topic": "integration"},
    {"q": "JSON данные в сценарии", "type": "keyword", "lang": "mixed", "topic": "scripts"},
    {
        "q": "Как настроить REST API в Comindware?",
        "type": "natural",
        "lang": "mixed",
        "topic": "api",
    },
    {
        "q": "Пример C# скрипта для HTTP запроса",
        "type": "natural",
        "lang": "mixed",
        "topic": "scripts",
    },
    {
        "q": "API для синхронизации данных",
        "type": "natural",
        "lang": "mixed",
        "topic": "integration",
    },
    # --- ENGLISH (15%) ---
    {"q": "OData integration setup", "type": "keyword", "lang": "en", "topic": "integration"},
    {"q": "HTTP POST request example", "type": "keyword", "lang": "en", "topic": "scripts"},
    {
        "q": "How to configure OData integration?",
        "type": "natural",
        "lang": "en",
        "topic": "integration",
    },
    {"q": "How to send HTTP POST request?", "type": "natural", "lang": "en", "topic": "scripts"},
    {"q": "Keycloak authentication setup", "type": "keyword", "lang": "en", "topic": "auth"},
    {"q": "How to create a record template?", "type": "natural", "lang": "en", "topic": "config"},
    {"q": "Platform deployment guide", "type": "keyword", "lang": "en", "topic": "infrastructure"},
    {"q": "BPM process configuration", "type": "keyword", "lang": "en", "topic": "platform"},
]


def get_embedding(text: str) -> list[float]:
    """Get embedding from OpenRouter."""
    api_key = settings.openrouter_api_key
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": "qwen/qwen3-embedding-8b", "input": text}

    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def search_semantic(collection, embedding: list[float], limit: int = 5):
    """Semantic search with embeddings."""
    results = collection.query(
        query_embeddings=[embedding], n_results=limit, include=["documents", "metadatas"]
    )
    if not results["documents"] or not results["documents"][0]:
        return []
    return [
        {"content": d[:1000], "source": m.get("source_file", "?")}
        for d, m in zip(results["documents"][0], results["metadatas"][0])
    ]


def search_keyword(collection, keywords: str, limit: int = 5):
    """Keyword/grep search - simulate BM25 using word matching."""
    words = re.findall(r"\b\w+\b", keywords.lower())
    if len(words) < 1:
        return search_random(collection, limit)

    all_docs = collection.get(limit=2000, include=["documents", "metadatas"])

    if not all_docs["documents"]:
        return []

    scored_docs = []
    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        doc_lower = doc.lower()
        score = sum(1 for w in words if w in doc_lower)
        if score > 0:
            scored_docs.append((score, doc[:1000], meta.get("source_file", "?")))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [{"content": d, "source": s} for _, d, s in scored_docs[:limit]]


def search_random(collection, limit: int = 5):
    """Random document retrieval."""
    count_val = collection.count()  # Call the method
    if count_val == 0:
        return []

    offset = random.randint(0, max(0, count_val - limit - 1))
    results = collection.get(limit=limit, offset=offset, include=["documents", "metadatas"])

    if not results["documents"]:
        return []

    return [
        {"content": d[:1000], "source": m.get("source_file", "?")}
        for d, m in zip(results["documents"], results["metadatas"])
    ]


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"20260321-reranker-dataset-{timestamp}.json"

    print("=" * 80)
    print("PREPARING DATASETS FOR RERANKER BENCHMARK")
    print("=" * 80)
    print(f"Questions: {len(TEST_QUESTIONS)}")
    print()

    # Connect to ChromaDB
    client = HttpClient(host="localhost", port=8000)
    collection = client.get_collection("mkdocs_kb_qwen8b")
    print(f"ChromaDB (Qwen3): {collection.count()} documents")
    print()

    datasets = {"questions": TEST_QUESTIONS, "semantic": [], "keyword": [], "random": []}

    # 1. SEMANTIC SEARCH - requires embeddings
    print("\n[SEMANTIC] Fetching documents with embeddings...")
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"  [{i + 1}/{len(TEST_QUESTIONS)}] {q['q'][:40]}...", end="", flush=True)
        try:
            embedding = get_embedding(q["q"])
            docs = search_semantic(collection, embedding, limit=5)
            datasets["semantic"].append({"q_idx": i, "docs": docs})
            print(f" found {len(docs)}")
        except Exception as e:
            print(f" ERROR: {e}")
            datasets["semantic"].append({"q_idx": i, "docs": []})

    # 2. KEYWORD SEARCH
    print("\n[KEYWORD] Fetching documents with keyword matching...")
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"  [{i + 1}/{len(TEST_QUESTIONS)}] {q['q'][:40]}...", end="", flush=True)
        try:
            docs = search_keyword(collection, q["q"], limit=5)
            datasets["keyword"].append({"q_idx": i, "docs": docs})
            print(f" found {len(docs)}")
        except Exception as e:
            print(f" ERROR: {e}")
            datasets["keyword"].append({"q_idx": i, "docs": []})

    # 3. RANDOM (different for each question)
    print("\n[RANDOM] Fetching random documents...")
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"  [{i + 1}/{len(TEST_QUESTIONS)}] {q['q'][:40]}...", end="", flush=True)
        try:
            docs = search_random(collection, limit=5)
            datasets["random"].append({"q_idx": i, "docs": docs})
            print(f" found {len(docs)}")
        except Exception as e:
            print(f" ERROR: {e}")
            datasets["random"].append({"q_idx": i, "docs": []})

    # Save dataset
    print(f"\nSaving dataset to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(datasets, f, ensure_ascii=False, indent=2)

    print(f"\nDataset saved: {output_file}")
    print(f"  Questions: {len(datasets['questions'])}")
    print(f"  Semantic: {len(datasets['semantic'])} queries")
    print(f"  Keyword: {len(datasets['keyword'])} queries")
    print(f"  Random: {len(datasets['random'])} queries")

    return output_file


if __name__ == "__main__":
    main()
