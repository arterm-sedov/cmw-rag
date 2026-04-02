"""Realistic Reranker Benchmark with Proper Test Questions.

Two types of queries:
1. KEYWORD queries - what retriever gets (short, keyword-based)
2. NATURAL queries - what user asks knowledge base (full questions)

Based on actual Comindware Platform KB content.
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from chromadb import HttpClient
from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter

# Configuration
BATCH_SIZE = 15
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"

QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

# Instructions to test - ENGLISH (per Qwen3 docs)
INSTRUCTIONS = {
    # Default/baseline
    "default_web": "Given a web search query, retrieve relevant passages that answer the query",
    # Platform-specific (best from previous tests)
    "platform_en": "Find documentation about Comindware Platform features, configurations, and APIs",
    # Task-focused
    "integration_en": "Find integration guides, API documentation, and configuration examples",
    "troubleshooting_en": "Find troubleshooting guides, error solutions, and problem resolutions",
    "setup_en": "Find setup guides, installation instructions, and configuration steps",
    # Code-aware
    "code_en": "Find code examples, API references, and technical documentation",
    # Bilingual note (still English instruction)
    "bilingual_hint_en": "Find relevant documentation (Russian or English) with code examples and configuration guides",
    # Detailed platform
    "detailed_en": "Retrieve Comindware Platform documentation: integration guides, API references, configuration examples",
    # Russian (testing multilingual capability)
    "platform_ru": "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации",
}

# Realistic test questions - English, Russian, Mixed
# Two types: (1) keyword queries for retriever (2) natural language questions
TEST_QUESTIONS = [
    # ═══════════════════════════════════════════════════════════════════════════
    # ENGLISH ONLY
    # ═══════════════════════════════════════════════════════════════════════════
    # -- English Keywords --
    {"q": "OData integration setup", "type": "keyword", "lang": "en"},
    {"q": "HTTP POST request", "type": "keyword", "lang": "en"},
    {"q": "Keycloak authentication", "type": "keyword", "lang": "en"},
    {"q": "REST API connection", "type": "keyword", "lang": "en"},
    {"q": "C# script example", "type": "keyword", "lang": "en"},
    {"q": "JSON response parsing", "type": "keyword", "lang": "en"},
    {"q": "webhook configuration", "type": "keyword", "lang": "en"},
    {"q": "data synchronization", "type": "keyword", "lang": "en"},
    {"q": "record template", "type": "keyword", "lang": "en"},
    {"q": "attribute mapping", "type": "keyword", "lang": "en"},
    # -- English Natural Questions --
    {"q": "How to configure OData integration?", "type": "natural", "lang": "en"},
    {"q": "How to send HTTP POST request from script?", "type": "natural", "lang": "en"},
    {"q": "How to set up Keycloak authentication?", "type": "natural", "lang": "en"},
    {"q": "What is a data transfer path?", "type": "natural", "lang": "en"},
    {"q": "How to parse JSON response from server?", "type": "natural", "lang": "en"},
    {"q": "How to create record template for integration?", "type": "natural", "lang": "en"},
    {"q": "What HTTP request types are supported?", "type": "natural", "lang": "en"},
    {"q": "How to use C# scripts in scenarios?", "type": "natural", "lang": "en"},
    {"q": "How to configure data synchronization?", "type": "natural", "lang": "en"},
    {"q": "How to save server response to attributes?", "type": "natural", "lang": "en"},
    # ═══════════════════════════════════════════════════════════════════════════
    # RUSSIAN ONLY
    # ═══════════════════════════════════════════════════════════════════════════
    # -- Russian Keywords --
    {"q": "интеграция OData", "type": "keyword", "lang": "ru"},
    {"q": "HTTP POST запрос", "type": "keyword", "lang": "ru"},
    {"q": "настройка Keycloak", "type": "keyword", "lang": "ru"},
    {"q": "подключение REST API", "type": "keyword", "lang": "ru"},
    {"q": "сценарий C#", "type": "keyword", "lang": "ru"},
    {"q": "синхронизация данных", "type": "keyword", "lang": "ru"},
    {"q": "шаблон записи", "type": "keyword", "lang": "ru"},
    {"q": "атрибуты сообщения", "type": "keyword", "lang": "ru"},
    {"q": "путь передачи данных", "type": "keyword", "lang": "ru"},
    {"q": "глобальная конфигурация", "type": "keyword", "lang": "ru"},
    # -- Russian Natural Questions --
    {"q": "Как настроить интеграцию с 1С?", "type": "natural", "lang": "ru"},
    {"q": "Как отправить HTTP запрос из сценария?", "type": "natural", "lang": "ru"},
    {"q": "Как подключить аутентификацию Keycloak?", "type": "natural", "lang": "ru"},
    {"q": "Что такое путь передачи данных?", "type": "natural", "lang": "ru"},
    {"q": "Как обработать JSON ответ сервера?", "type": "natural", "lang": "ru"},
    {"q": "Как создать шаблон записи для интеграции?", "type": "natural", "lang": "ru"},
    {"q": "Какие типы HTTP запросов поддерживаются?", "type": "natural", "lang": "ru"},
    {"q": "Как использовать C# скрипты в сценариях?", "type": "natural", "lang": "ru"},
    {"q": "Как настроить синхронизацию данных?", "type": "natural", "lang": "ru"},
    {"q": "Как сохранить данные в атрибуты?", "type": "natural", "lang": "ru"},
    # ═══════════════════════════════════════════════════════════════════════════
    # MIXED (Russian + English technical terms)
    # ═══════════════════════════════════════════════════════════════════════════
    # -- Mixed Keywords --
    {"q": "OData синхронизация", "type": "keyword", "lang": "mixed"},
    {"q": "REST API подключение", "type": "keyword", "lang": "mixed"},
    {"q": "Keycloak аутентификация", "type": "keyword", "lang": "mixed"},
    {"q": "webhook настройка", "type": "keyword", "lang": "mixed"},
    {"q": "C# скрипт пример", "type": "keyword", "lang": "mixed"},
    {"q": "JSON данные", "type": "keyword", "lang": "mixed"},
    {"q": "API конфигурация", "type": "keyword", "lang": "mixed"},
    {"q": "BPM система", "type": "keyword", "lang": "mixed"},
    {"q": "Low-code платформа", "type": "keyword", "lang": "mixed"},
    {"q": "графовая база данных", "type": "keyword", "lang": "mixed"},
    # -- Mixed Natural Questions --
    {"q": "Как настроить REST API подключение в Comindware?", "type": "natural", "lang": "mixed"},
    {"q": "Пример C# скрипта для HTTP запроса", "type": "natural", "lang": "mixed"},
    {"q": "Как получить данные из JSON ответа?", "type": "natural", "lang": "mixed"},
    {"q": "Как работает OData интеграция в платформе?", "type": "natural", "lang": "mixed"},
    {"q": "Настройка webhook для внешних систем", "type": "natural", "lang": "mixed"},
    {"q": "Как использовать REST API в Comindware Platform?", "type": "natural", "lang": "mixed"},
    {"q": "Интеграция с 1С через OData протокол", "type": "natural", "lang": "mixed"},
    {"q": "Как настроить HTTP POST запрос?", "type": "natural", "lang": "mixed"},
    {"q": "Конфигурация атрибутов для JSON", "type": "natural", "lang": "mixed"},
    {"q": "API для синхронизации данных", "type": "natural", "lang": "mixed"},
]


def get_all_documents(collection, limit=500):
    """Get all documents from ChromaDB (no embedding query needed)."""
    docs = collection.get(limit=limit, include=["documents", "metadatas"])

    if not docs["documents"]:
        return []

    samples = []
    for doc, meta in zip(docs["documents"], docs["metadatas"]):
        samples.append(
            {
                "content": doc[:800],
                "source": meta.get("source_file", "?"),
            }
        )

    return samples


def run_benchmark():
    print("=" * 80)
    print("REALISTIC RERANKER BENCHMARK")
    print("=" * 80)
    print(f"Test questions: {len(TEST_QUESTIONS)}")

    # Count by language
    en_kw = sum(1 for q in TEST_QUESTIONS if q["lang"] == "en" and q["type"] == "keyword")
    en_nat = sum(1 for q in TEST_QUESTIONS if q["lang"] == "en" and q["type"] == "natural")
    ru_kw = sum(1 for q in TEST_QUESTIONS if q["lang"] == "ru" and q["type"] == "keyword")
    ru_nat = sum(1 for q in TEST_QUESTIONS if q["lang"] == "ru" and q["type"] == "natural")
    mix_kw = sum(1 for q in TEST_QUESTIONS if q["lang"] == "mixed" and q["type"] == "keyword")
    mix_nat = sum(1 for q in TEST_QUESTIONS if q["lang"] == "mixed" and q["type"] == "natural")

    print(f"\nQuestions by language and type:")
    print(f"  EN (English only):     {en_kw} keyword, {en_nat} natural")
    print(f"  RU (Russian only):     {ru_kw} keyword, {ru_nat} natural")
    print(f"  MIX (Russian+English): {mix_kw} keyword, {mix_nat} natural")
    print()

    # Connect to ChromaDB
    client = HttpClient(host="localhost", port=8000)

    # Use mkdocs_kb (FRIDA, 10685 docs) for document pool
    collection = client.get_collection("mkdocs_kb")
    print(f"ChromaDB (mkdocs_kb): {collection.count} documents")

    # Get document pool
    print("\nLoading document pool...")
    doc_pool = get_all_documents(collection, limit=500)
    print(f"Loaded {len(doc_pool)} documents")

    # Create adapter
    config = ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/score",
        reranker_type="llm_reranker",
        default_instruction=INSTRUCTIONS["default_web"],
        formatting=QWEN3_FORMATTING,
    )
    adapter = RerankerAdapter(config)

    # Assign random documents to questions
    print("\nAssigning documents to questions...")
    random.shuffle(doc_pool)

    questions_with_docs = []
    for i, q in enumerate(TEST_QUESTIONS):
        # Use different slice for each question to get variety
        start = (i * 5) % (len(doc_pool) - 10)
        docs = doc_pool[start : start + 5]
        questions_with_docs.append(
            {
                "question": q["q"],
                "type": q["type"],
                "lang": q["lang"],
                "docs": docs,
            }
        )

    print(f"Ready to test {len(questions_with_docs)} questions")

    # Test each instruction
    results = {}
    results_by_lang = {"en": {}, "ru": {}, "mixed": {}}
    results_by_type = {"keyword": {}, "natural": {}}

    for name, instruction in INSTRUCTIONS.items():
        print(f"\nTesting: {name}")
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
                "count": len(all_scores),
                "instruction": instruction,
            }
            for lang in ["en", "ru", "mixed"]:
                if lang_scores[lang]:
                    results_by_lang[lang][name] = sum(lang_scores[lang]) / len(lang_scores[lang])
            for qtype in ["keyword", "natural"]:
                if type_scores[qtype]:
                    results_by_type[qtype][name] = sum(type_scores[qtype]) / len(type_scores[qtype])

            print(f"  Avg: {results[name]['avg_score']:.4f}")

    # Analyze by question type
    print("\n" + "=" * 80)
    print("RESULTS BY QUESTION TYPE")
    print("=" * 80)

    keyword_questions = [q for q in questions_with_docs if q["type"] == "keyword"]
    natural_questions = [q for q in questions_with_docs if q["type"] == "natural"]

    for qtype, questions in [("keyword", keyword_questions), ("natural", natural_questions)]:
        if not questions:
            continue
        print(f"\n{qtype.upper()} QUESTIONS ({len(questions)}):")

        for name, instruction in INSTRUCTIONS.items():
            scores = []
            for q in questions:
                candidates = [(d["content"], 0.0) for d in q["docs"]]
                try:
                    ranked = adapter.rerank(
                        q["question"], candidates, top_k=len(candidates), instruction=instruction
                    )
                    scores.extend([r[1] for r in ranked])
                except:
                    pass

            if scores:
                avg = sum(scores) / len(scores)
                print(f"  {name:<25}: {avg:.4f}")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS (sorted by avg_score)")
    print("=" * 80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    print(f"\n{'Instruction':<25} {'Avg Score':>10} {'Count':>8}")
    print("-" * 50)

    for name, data in sorted_results:
        print(f"{name:<25} {data['avg_score']:>10.4f} {data['count']:>8}")

    # Best instruction
    best = sorted_results[0]
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\nBest instruction: {best[0]}")
    print(f"Score: {best[1]['avg_score']:.4f}")
    print(f"Instruction: {best[1]['instruction']}")

    # Category analysis
    en_scores = [
        d["avg_score"]
        for n, d in results.items()
        if not n.endswith("_ru") and not n.startswith("bilingual")
    ]
    ru_scores = [d["avg_score"] for n, d in results.items() if n.endswith("_ru")]
    bi_scores = [d["avg_score"] for n, d in results.items() if "bilingual" in n or "hint" in n]

    print("\nCategory averages:")
    if en_scores:
        print(f"  English:   {sum(en_scores) / len(en_scores):.4f}")
    if ru_scores:
        print(f"  Russian:   {sum(ru_scores) / len(ru_scores):.4f}")
    if bi_scores:
        print(f"  Bilingual: {sum(bi_scores) / len(bi_scores):.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save test dataset
    dataset_file = output_dir / f"realistic_test_dataset_{timestamp}.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "questions": questions_with_docs,
                "created": timestamp,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nDataset saved: {dataset_file}")

    # Save results
    results_file = output_dir / f"realistic_benchmark_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": {
                    n: {"avg_score": d["avg_score"], "instruction": d["instruction"]}
                    for n, d in results.items()
                },
                "best": {
                    "name": best[0],
                    "score": best[1]["avg_score"],
                    "instruction": best[1]["instruction"],
                },
                "timestamp": timestamp,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Results saved: {results_file}")

    return results


if __name__ == "__main__":
    run_benchmark()
