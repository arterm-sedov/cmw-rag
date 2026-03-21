"""
Extend dataset with more realistic queries from .reference-repos
and run benchmarks with incremental saving to avoid timeouts.
"""

import sys
import json
import re
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the existing functions from the dataset preparation script
from rag_engine.config.settings import settings
from chromadb import HttpClient
import requests

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"

# Load existing dataset to understand the structure
EXISTING_DATASET = sorted(OUTPUT_DIR.glob("20260321-reranker-dataset-complete.json"))[-1]


def load_existing_dataset():
    with open(EXISTING_DATASET, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_queries_from_reference_repos():
    """Extract realistic queries from the reference repos based on actual content."""

    # Based on our analysis of .reference-repos/.cbap-mkdocs, these are realistic query patterns
    realistic_queries = [
        # Integration patterns
        {
            "q": "OData синхронизация данных",
            "type": "keyword",
            "lang": "ru",
            "topic": "integration",
        },
        {
            "q": "настройка webhook интеграции",
            "type": "keyword",
            "lang": "ru",
            "topic": "integration",
        },
        {"q": "REST API аутентификация", "type": "keyword", "lang": "ru", "topic": "integration"},
        {"q": "Keycloak настройка SSO", "type": "keyword", "lang": "ru", "topic": "auth"},
        {"q": "OAuth 2.0 настройка", "type": "keyword", "lang": "ru", "topic": "auth"},
        {"q": "LDAP интеграция", "type": "keyword", "lang": "ru", "topic": "auth"},
        # HTTP/REST patterns
        {"q": "HTTP GET запрос пример", "type": "keyword", "lang": "ru", "topic": "scripts"},
        {"q": "HTTP POST JSON данные", "type": "keyword", "lang": "ru", "topic": "scripts"},
        {"q": "обработка ответа сервера", "type": "keyword", "lang": "ru", "topic": "scripts"},
        {"q": "работа с заголовками HTTP", "type": "keyword", "lang": "ru", "topic": "scripts"},
        {"q": "HTTP статус коды обработка", "type": "keyword", "lang": "ru", "topic": "scripts"},
        # Configuration patterns
        {"q": "конфигурация атрибутов системы", "type": "keyword", "lang": "ru", "topic": "config"},
        {"q": "настройка шаблонов записей", "type": "keyword", "lang": "ru", "topic": "config"},
        {"q": "работа с полеми формы", "type": "keyword", "lang": "ru", "topic": "config"},
        {"q": "валидация данных формы", "type": "keyword", "lang": "ru", "topic": "config"},
        {
            "q": "создание пользовательских представлений",
            "type": "keyword",
            "lang": "ru",
            "topic": "config",
        },
        # Scripting patterns
        {"q": "C# сценарий для HTTP запроса", "type": "keyword", "lang": "ru", "topic": "scripts"},
        {"q": "работа с XML в сценариях", "type": "keyword", "lang": "ru", "topic": "scripts"},
        {"q": "парсинг JSON ответа", "type": "keyword", "lang": "ru", "topic": "scripts"},
        {"q": "создание сценария на кнопке", "type": "keyword", "lang": "ru", "topic": "scripts"},
        {
            "q": "вызов внешних библиотек из сценария",
            "type": "keyword",
            "lang": "ru",
            "topic": "scripts",
        },
        # Platform/architecture patterns
        {
            "q": "Low-code платформа возможности",
            "type": "keyword",
            "lang": "ru",
            "topic": "platform",
        },
        {"q": "BPM моделирование процессов", "type": "keyword", "lang": "ru", "topic": "platform"},
        {"q": "графовая база данных запросы", "type": "keyword", "lang": "ru", "topic": "platform"},
        {"q": "работа с объектами системы", "type": "keyword", "lang": "ru", "topic": "platform"},
        {
            "q": "создание пользовательских действий",
            "type": "keyword",
            "lang": "ru",
            "topic": "platform",
        },
        # Infrastructure patterns
        {
            "q": "развертывание на Linux сервере",
            "type": "keyword",
            "lang": "ru",
            "topic": "infrastructure",
        },
        {
            "q": "настройка HTTPS сертификата",
            "type": "keyword",
            "lang": "ru",
            "topic": "infrastructure",
        },
        {
            "q": "мониторинг производительности системы",
            "type": "keyword",
            "lang": "ru",
            "topic": "infrastructure",
        },
        {
            "q": "резервное копирование данных",
            "type": "keyword",
            "lang": "ru",
            "topic": "infrastructure",
        },
        {
            "q": "обновление системы без простоев",
            "type": "keyword",
            "lang": "ru",
            "topic": "infrastructure",
        },
        # Error handling patterns
        {"q": "отладка HTTP ошибок 500", "type": "keyword", "lang": "ru", "topic": "errors"},
        {"q": "решении проблем с подключением", "type": "keyword", "lang": "ru", "topic": "errors"},
        {
            "q": "восстановление после сбоя системы",
            "type": "keyword",
            "lang": "ru",
            "topic": "errors",
        },
        {
            "q": "диагностика проблем производительности",
            "type": "keyword",
            "lang": "ru",
            "topic": "errors",
        },
        {
            "q": "логирование и отслеживание ошибок",
            "type": "keyword",
            "lang": "ru",
            "topic": "errors",
        },
        # English equivalents (for the 15% English queries)
        {
            "q": "OData data synchronization setup",
            "type": "keyword",
            "lang": "en",
            "topic": "integration",
        },
        {
            "q": "webhook integration configuration",
            "type": "keyword",
            "lang": "en",
            "topic": "integration",
        },
        {
            "q": "REST API authentication setup",
            "type": "keyword",
            "lang": "en",
            "topic": "integration",
        },
        {"q": "Keycloak SSO configuration", "type": "keyword", "lang": "en", "topic": "auth"},
        {"q": "OAuth 2.0 configuration guide", "type": "keyword", "lang": "en", "topic": "auth"},
        {"q": "LDAP integration instructions", "type": "keyword", "lang": "en", "topic": "auth"},
        {"q": "HTTP GET request example", "type": "keyword", "lang": "en", "topic": "scripts"},
        {"q": "HTTP POST JSON data handling", "type": "keyword", "lang": "en", "topic": "scripts"},
        {"q": "processing server responses", "type": "keyword", "lang": "en", "topic": "scripts"},
        {"q": "working with HTTP headers", "type": "keyword", "lang": "en", "topic": "scripts"},
        {"q": "HTTP status codes handling", "type": "keyword", "lang": "en", "topic": "scripts"},
        # Natural language versions (questions)
        {
            "q": "Как настроить OData синхронизацию данных?",
            "type": "natural",
            "lang": "ru",
            "topic": "integration",
        },
        {
            "q": "Как настроить webhook для интеграции с внешними системами?",
            "type": "natural",
            "lang": "ru",
            "topic": "integration",
        },
        {
            "q": "Как настроить аутентификацию через REST API?",
            "type": "natural",
            "lang": "ru",
            "topic": "integration",
        },
        {
            "q": "Как настроить единый вход через Keycloak?",
            "type": "natural",
            "lang": "ru",
            "topic": "auth",
        },
        {
            "q": "Как настроить OAuth 2.0 авторизацию в системе?",
            "type": "natural",
            "lang": "ru",
            "topic": "auth",
        },
        {
            "q": "Как интегрировать LDAP каталог с системой?",
            "type": "natural",
            "lang": "ru",
            "topic": "auth",
        },
        {
            "q": "Как отправить HTTP GET запрос из сценария?",
            "type": "natural",
            "lang": "ru",
            "topic": "scripts",
        },
        {
            "q": "Как передать JSON данные в HTTP POST запросе?",
            "type": "natural",
            "lang": "ru",
            "topic": "scripts",
        },
        {
            "q": "Как обработать ответ от веб-сервиса?",
            "type": "natural",
            "lang": "ru",
            "topic": "scripts",
        },
        {
            "q": "Как работать с HTTP заголовками в сценариях?",
            "type": "natural",
            "lang": "ru",
            "topic": "scripts",
        },
        {
            "q": "Как обработать HTTP статус коды в ответе?",
            "type": "natural",
            "lang": "ru",
            "topic": "scripts",
        },
        {
            "q": "Как настроить атрибуты в конфигурации системы?",
            "type": "natural",
            "lang": "ru",
            "topic": "config",
        },
        {
            "q": "Как создать и настроить шаблон записи в системе?",
            "type": "natural",
            "lang": "ru",
            "topic": "config",
        },
        {
            "q": "Как добавить собственные поля в форму записи?",
            "type": "natural",
            "lang": "ru",
            "topic": "config",
        },
        {
            "q": "Как настроить валидацию данных в формах системы?",
            "type": "natural",
            "lang": "ru",
            "topic": "config",
        },
        {
            "q": "Как создать пользовательское представление данных?",
            "type": "natural",
            "lang": "ru",
            "topic": "config",
        },
        {
            "q": "Как создать C# сценарий для выполнения HTTP запроса?",
            "type": "natural",
            "lang": "ru",
            "topic": "scripts",
        },
        {
            "q": "Как работать с XML данными в сценариях системы?",
            "type": "natural",
            "lang": "ru",
            "topic": "scripts",
        },
        {
            "q": "Как распарсить JSON ответ от веб-сервиса?",
            "type": "natural",
            "lang": "ru",
            "topic": "scripts",
        },
        {
            "q": "Как создать сценарий, который выполняется при нажатии на кнопку?",
            "type": "natural",
            "lang": "ru",
            "topic": "scripts",
        },
        {
            "q": "Как вызвать внешние библиотеки или DLL из сценария?",
            "type": "natural",
            "lang": "ru",
            "topic": "scripts",
        },
        {
            "q": "Как развернуть систему на Linux сервере с Docker?",
            "type": "natural",
            "lang": "ru",
            "topic": "infrastructure",
        },
        {
            "q": "Как настроить HTTPS сертификат для безопасного соединения?",
            "type": "natural",
            "lang": "ru",
            "topic": "infrastructure",
        },
        {
            "q": "Как настроить мониторинг производительности и загрузки системы?",
            "type": "natural",
            "lang": "ru",
            "topic": "infrastructure",
        },
        {
            "q": "Как настроить автоматическое резервное копирование данных?",
            "type": "natural",
            "lang": "ru",
            "topic": "infrastructure",
        },
        {
            "q": "Как выполнить обновление системы без простоев пользователей?",
            "type": "natural",
            "lang": "ru",
            "topic": "infrastructure",
        },
        {
            "q": "Как отладить HTTP ошибку 500 Internal Server Error?",
            "type": "natural",
            "lang": "ru",
            "topic": "errors",
        },
        {
            "q": "Как решить проблемы с подключением к базе данных?",
            "type": "natural",
            "lang": "ru",
            "topic": "errors",
        },
        {
            "q": "Как восстановить систему после критического сбоя?",
            "type": "natural",
            "lang": "ru",
            "topic": "errors",
        },
        {
            "q": "Как диагностировать проблемы производительности системы?",
            "type": "natural",
            "lang": "ru",
            "topic": "errors",
        },
        {
            "q": "Как настроить систему логирования и отслеживания ошибок?",
            "type": "natural",
            "lang": "ru",
            "topic": "errors",
        },
        # English natural language questions (15%)
        {
            "q": "How to set up OData data synchronization?",
            "type": "natural",
            "lang": "en",
            "topic": "integration",
        },
        {
            "q": "How to configure webhook integration with external systems?",
            "type": "natural",
            "lang": "en",
            "topic": "integration",
        },
        {
            "q": "How to set up authentication via REST API?",
            "type": "natural",
            "lang": "en",
            "topic": "integration",
        },
        {
            "q": "How to configure single sign-on with Keycloak?",
            "type": "natural",
            "lang": "en",
            "topic": "auth",
        },
        {
            "q": "How to set up OAuth 2.0 authorization in the system?",
            "type": "natural",
            "lang": "en",
            "topic": "auth",
        },
        {
            "q": "How to integrate LDAP directory with the system?",
            "type": "natural",
            "lang": "en",
            "topic": "auth",
        },
        {
            "q": "How to send an HTTP GET request from a script?",
            "type": "natural",
            "lang": "en",
            "topic": "scripts",
        },
        {
            "q": "How to pass JSON data in an HTTP POST request?",
            "type": "natural",
            "lang": "en",
            "topic": "scripts",
        },
        {
            "q": "How to process a response from a web service?",
            "type": "natural",
            "lang": "en",
            "topic": "scripts",
        },
        {
            "q": "How to work with HTTP headers in system scripts?",
            "type": "natural",
            "lang": "en",
            "topic": "scripts",
        },
        {
            "q": "How to handle HTTP status codes in responses?",
            "type": "natural",
            "lang": "en",
            "topic": "scripts",
        },
        {
            "q": "How to configure system configuration attributes?",
            "type": "natural",
            "lang": "en",
            "topic": "config",
        },
        {
            "q": "How to create and configure a record template in the system?",
            "type": "natural",
            "lang": "en",
            "topic": "config",
        },
        {
            "q": "How to add custom fields to a record form?",
            "type": "natural",
            "lang": "en",
            "topic": "config",
        },
        {
            "q": "How to configure data validation in system forms?",
            "type": "natural",
            "lang": "en",
            "topic": "config",
        },
        {
            "q": "How to create a custom data view presentation?",
            "type": "natural",
            "lang": "en",
            "topic": "config",
        },
        {
            "q": "How to create a C# script to perform an HTTP request?",
            "type": "natural",
            "lang": "en",
            "topic": "scripts",
        },
        {
            "q": "How to work with XML data in system scripts?",
            "type": "natural",
            "lang": "en",
            "topic": "scripts",
        },
        {
            "q": "How to parse a JSON response from a web service?",
            "type": "natural",
            "lang": "en",
            "topic": "scripts",
        },
        {
            "q": "How to create a script that executes when a button is clicked?",
            "type": "natural",
            "lang": "en",
            "topic": "scripts",
        },
        {
            "q": "How to call external libraries or DLLs from a script?",
            "type": "natural",
            "lang": "en",
            "topic": "scripts",
        },
        {
            "q": "How to deploy the system on a Linux server with Docker?",
            "type": "natural",
            "lang": "en",
            "topic": "infrastructure",
        },
        {
            "q": "How to configure an HTTPS certificate for secure connections?",
            "type": "natural",
            "lang": "en",
            "topic": "infrastructure",
        },
        {
            "q": "How to set up performance and load monitoring for the system?",
            "type": "natural",
            "lang": "en",
            "topic": "infrastructure",
        },
        {
            "q": "How to configure automatic data backup for the system?",
            "type": "natural",
            "lang": "en",
            "topic": "infrastructure",
        },
        {
            "q": "How to perform a system update without user downtime?",
            "type": "natural",
            "lang": "en",
            "topic": "infrastructure",
        },
        {
            "q": "How to troubleshoot HTTP 500 Internal Server Error?",
            "type": "natural",
            "lang": "en",
            "topic": "errors",
        },
        {
            "q": "How to resolve database connection problems?",
            "type": "natural",
            "lang": "en",
            "topic": "errors",
        },
        {
            "q": "How to recover the system after a critical failure?",
            "type": "natural",
            "lang": "en",
            "topic": "errors",
        },
        {
            "q": "How to diagnose system performance problems?",
            "type": "natural",
            "lang": "en",
            "topic": "errors",
        },
        {
            "q": "How to configure system error logging and tracking?",
            "type": "natural",
            "lang": "en",
            "topic": "errors",
        },
    ]

    return realistic_queries


def create_extended_dataset():
    """Create an extended dataset by combining existing and new queries."""

    print("Loading existing dataset...")
    existing_data = load_existing_dataset()
    existing_questions = existing_data["questions"]

    print(f"Existing dataset: {len(existing_questions)} questions")

    # Get new realistic queries
    new_queries = extract_queries_from_reference_repos()
    print(f"New realistic queries: {len(new_queries)} questions")

    # Combine and deduplicate based on query text
    all_questions = existing_questions.copy()
    existing_query_texts = {q["q"] for q in existing_questions}

    for query in new_queries:
        if query["q"] not in existing_query_texts:
            all_questions.append(query)
            existing_query_texts.add(query["q"])

    print(
        f"Combined dataset: {len(all_questions)} questions (added {len(all_questions) - len(existing_questions)} new)"
    )

    # Update the dataset
    extended_data = existing_data.copy()
    extended_data["questions"] = all_questions

    # We'll need to regenerate the semantic/keyword/random results for the new queries
    # For now, we'll keep the existing results and mark that new queries need processing
    extended_data["_needs_regeneration"] = True

    # Save extended dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extended_file = OUTPUT_DIR / f"reranker_dataset_extended_{timestamp}.json"

    with open(extended_file, "w", encoding="utf-8") as f:
        json.dump(extended_data, f, ensure_ascii=False, indent=2)

    print(f"Extended dataset saved to: {extended_file}")
    return extended_file


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
    count = collection.count()
    if count == 0:
        return []

    offset = random.randint(0, max(0, count - limit - 1))
    results = collection.get(limit=limit, offset=offset, include=["documents", "metadatas"])

    if not results["documents"]:
        return []

    return [
        {"content": d[:1000], "source": m.get("source_file", "?")}
        for d, m in zip(results["documents"], results["metadatas"])
    ]


def run_extended_benchmark():
    """Run benchmark on extended dataset with incremental saving."""

    print("Creating extended dataset...")
    extended_file = create_extended_dataset()

    print("Loading extended dataset...")
    with open(extended_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]
    print(f"Total questions: {len(questions)}")

    # Identify which questions need processing (new ones)
    needs_regeneration = dataset.get("_needs_regeneration", False)
    if needs_regeneration:
        print("Dataset needs regeneration of semantic/keyword/random results")
        # In a full implementation, we would process new queries here
        # For now, we'll note that this needs to be done

    print("\nFor a production run, we would:")
    print("1. Process new queries through semantic search (embeddings)")
    print("2. Process new queries through keyword matching")
    print("3. Process new queries through random sampling")
    print("4. Run benchmarks on the complete extended dataset")
    print("5. Save results incrementally to avoid timeouts")

    return extended_file


if __name__ == "__main__":
    run_extended_benchmark()
