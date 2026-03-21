"""
Reranker Instruction Benchmark - Hybrid Search Comparison

Tests three retrieval methods:
1. RANDOM: Documents fetched randomly from KB
2. SEMANTIC: Documents from semantic search (embeddings)
3. KEYWORD: Documents from keyword/grep search (BM25-like)

Question types:
- 85% Russian (realistic for support agent)
- 15% English/Mixed

Saves results incrementally to avoid timeouts.
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
from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter
from rag_engine.config.settings import settings

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

# =============================================================================
# INSTRUCTIONS TO TEST - Best from previous benchmarks + variations
# =============================================================================
INSTRUCTIONS = {
    # --- BASELINE (Qwen3 default - proven best with semantic search) ---
    "baseline_default": "Given a web search query, retrieve relevant passages that answer the query",
    # --- ENGLISH WITH CONTEXT (best from semantic benchmark) ---
    "en_context_ru_docs": "Find relevant documentation. Documents are primarily in Russian with code snippets in English",
    "en_context_mixed": "Retrieve technical documentation (Russian text, English code examples) that answers the query",
    "en_context_platform": "Find Comindware Platform documentation in Russian/English with configuration examples and code",
    # --- PURE RUSSIAN (natural forRussian queries) ---
    "ru_platform": "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации",
    "ru_integration": "Найдите руководства по интеграции, документацию API и примеры конфигураций",
    "ru_troubleshooting": "Найдите руководства по устранению неполадок, решения ошибок и проблемы",
    # --- BILINGUAL (EN+RU duplicated) ---
    "bilingual_platform": "Find Comindware Platform documentation. Найдите документацию по платформе Comindware",
    "bilingual_integration": "Find integration guides. Найдите руководства по интеграции",
    "bilingual_context": "Retrieve technical documentation (Russian/English). Получите техническую документацию (русский/английский)",
    # --- INSTRUCTION VARIATIONS ---
    "en_web_search": "Given a web search query, retrieve relevant passages that answer the query from documentation",
    "en_find_docs": "Find documents that help answer the question from technical documentation",
    "en_retrieve_russian": "Retrieve relevant passages from Russian technical documentation with code examples",
    "ru_find_docs": "Найдите документы, которые помогают ответить на вопрос из технической документации",
    "ru_retrieve_platform": "Получите релевантные отрывки из документации Comindware Platform",
}

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
    # --- MIXED ---
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
]

# State file for incremental saves
STATE_FILE = None


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
        {"content": d[:800], "source": m.get("source_file", "?")}
        for d, m in zip(results["documents"][0], results["metadatas"][0])
    ]


def search_keyword(collection, keywords: str, limit: int = 5):
    """Keyword/grep search - simulate BM25 using word matching."""
    # Extract keywords (2-5 words)
    words = re.findall(r"\b\w+\b", keywords.lower())
    if len(words) < 1:
        return search_random(collection, limit)

    # Get all documents and search for keyword matches
    # ChromaDB doesn't have native keyword search, so we use contains filtering
    # This simulates hybrid search behavior
    all_docs = collection.get(limit=1000, include=["documents", "metadatas"])

    if not all_docs["documents"]:
        return []

    scored_docs = []
    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        doc_lower = doc.lower()
        # Count keyword matches
        score = sum(1 for w in words if w in doc_lower)
        if score > 0:
            scored_docs.append((score, doc[:800], meta.get("source_file", "?")))

    # Sort by score and return top
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [{"content": d, "source": s} for _, d, s in scored_docs[:limit]]


def search_random(collection, limit: int = 5):
    """Random document retrieval."""
    count = collection.count
    if count == 0:
        return []

    # Get random offset
    offset = random.randint(0, max(0, count - limit - 1))
    results = collection.get(limit=limit, offset=offset, include=["documents", "metadatas"])

    if not results["documents"]:
        return []

    return [
        {"content": d[:800], "source": m.get("source_file", "?")}
        for d, m in zip(results["documents"], results["metadatas"])
    ]


def save_state(state: dict):
    """Save state to file."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_state() -> dict:
    """Load state from file."""
    if STATE_FILE and STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"questions_with_docs": {}, "results": {}, "processed": []}


def run_benchmark():
    global STATE_FILE

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATE_FILE = OUTPUT_DIR / f"reranker_hybrid_state_{timestamp}.json"

    print("=" * 80)
    print("RERANKER HYBRID BENCHMARK -Random vs Semantic vs Keyword")
    print("=" * 80)
    print(f"Instructions: {len(INSTRUCTIONS)}")
    print(f"Questions: {len(TEST_QUESTIONS)}")
    print(f"  Russian: {sum(1 for q in TEST_QUESTIONS if q['lang'] == 'ru')}")
    print(f"  Mixed: {sum(1 for q in TEST_QUESTIONS if q['lang'] == 'mixed')}")
    print(f"  English: {sum(1 for q in TEST_QUESTIONS if q['lang'] == 'en')}")
    print()

    # Load previous state
    state = load_state()

    # Connect to ChromaDB
    client = HttpClient(host="localhost", port=8000)
    collection = client.get_collection("mkdocs_kb_qwen8b")
    print(f"ChromaDB (Qwen3): {collection.count} documents")
    print()

    # Process each retrieval method
    retrieval_methods = {
        "semantic": search_semantic,
        "random": lambda c, e, l: search_random(c, l),
        "keyword": search_keyword,
    }

    for method_name, search_func in retrieval_methods.items():
        if method_name == "semantic":
            # Need embeddings for semantic search
            if method_name not in state.get("questions_with_docs", {}):
                print(f"\n[{method_name.upper()}] Getting embeddings and searching...")
                questions_with_docs = []

                for i, q in enumerate(TEST_QUESTIONS):
                    print(f"  [{i + 1}/{len(TEST_QUESTIONS)}] {q['q'][:40]}...", end="", flush=True)
                    try:
                        embedding = get_embedding(q["q"])
                        docs = search_func(collection, embedding, limit=5)
                        questions_with_docs.append({**q, "docs": docs})
                        print(f" found {len(docs)}")
                    except Exception as e:
                        print(f" ERROR: {e}")

                state.setdefault("questions_with_docs", {})[method_name] = questions_with_docs
                save_state(state)
        elif method_name not in state.get("questions_with_docs", {}):
            print(f"\n[{method_name.upper()}] Fetching documents...")
            questions_with_docs = []

            for i, q in enumerate(TEST_QUESTIONS):
                print(f"  [{i + 1}/{len(TEST_QUESTIONS)}] {q['q'][:40]}...", end="", flush=True)
                try:
                    if method_name == "random":
                        docs = search_random(collection, limit=5)
                    else:  # keyword
                        docs = search_keyword(collection, q["q"], limit=5)
                    questions_with_docs.append({**q, "docs": docs})
                    print(f" found {len(docs)}")
                except Exception as e:
                    print(f" ERROR: {e}")

            state.setdefault("questions_with_docs", {})[method_name] = questions_with_docs
            save_state(state)

    # Get embeddings for semantic (if not done yet)
    if "semantic" not in state.get("questions_with_docs", {}):
        print("\n[SEMANTIC] Getting embeddings...")
        questions_with_docs = []
        for i, q in enumerate(TEST_QUESTIONS):
            print(f"  [{i + 1}/{len(TEST_QUESTIONS)}] {q['q'][:40]}...", end="", flush=True)
            try:
                embedding = get_embedding(q["q"])
                docs = search_semantic(collection, embedding, limit=5)
                questions_with_docs.append({**q, "docs": docs})
                print(f" found {len(docs)}")
            except Exception as e:
                print(f" ERROR: {e}")
        state["questions_with_docs"]["semantic"] = questions_with_docs
        save_state(state)

    # Create reranker adapter
    config = ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/score",
        reranker_type="llm_reranker",
        default_instruction=INSTRUCTIONS["baseline_default"],
        formatting=QWEN3_FORMATTING,
    )
    adapter = RerankerAdapter(config)

    # Test each retrieval method and instruction combination
    print("\nTesting instruction/retrieval combinations...")
    print("-" * 80)

    for method_name, questions_with_docs in state["questions_with_docs"].items():
        print(f"\n{'=' * 80}")
        print(f"RETRIEVAL METHOD: {method_name.upper()}")
        print(f"{'=' * 80}")

        for name, instruction in INSTRUCTIONS.items():
            result_key = f"{method_name}_{name}"
            if result_key in state.get("processed", []):
                print(f"[SKIP] {result_key} (already done)")
                continue

            print(f"\n[{result_key}]")
            print(f"  Instruction: {instruction[:60]}...")

            all_scores = []
            scores_by_lang = {"en": [], "ru": [], "mixed": []}
            scores_by_type = {"keyword": [], "natural": []}

            for q in questions_with_docs:
                candidates = [(d["content"], 0.0) for d in q["docs"]]
                try:
                    ranked = adapter.rerank(
                        q["q"], candidates, top_k=len(candidates), instruction=instruction
                    )
                    scores = [r[1] for r in ranked]
                    all_scores.extend(scores)
                    scores_by_lang[q["lang"]].extend(scores)
                    scores_by_type[q["type"]].extend(scores)
                except Exception as e:
                    print(f"  Error: {e}")

            if all_scores:
                state["results"][result_key] = {
                    "method": method_name,
                    "instruction": instruction,
                    "avg_score": sum(all_scores) / len(all_scores),
                    "count": len(all_scores),
                    "by_lang": {k: sum(v) / len(v) if v else 0 for k, v in scores_by_lang.items()},
                    "by_type": {k: sum(v) / len(v) if v else 0 for k, v in scores_by_type.items()},
                }
                print(f"  Avg: {state['results'][result_key]['avg_score']:.4f}")
                print(
                    f"  By lang: EN={scores_by_lang['en'] and sum(scores_by_lang['en']) / len(scores_by_lang['en']):.4f}, "
                    f"RU={scores_by_lang['ru'] and sum(scores_by_lang['ru']) / len(scores_by_lang['ru']):.4f}, "
                    f"MIX={scores_by_lang['mixed'] and sum(scores_by_lang['mixed']) / len(scores_by_lang['mixed']):.4f}"
                )

            state.setdefault("processed", []).append(result_key)
            save_state(state)

    # Generate report
    generate_report(state, timestamp)
    print(f"\nReport saved to: {OUTPUT_DIR / f'rereranker_hybrid_report_{timestamp}.md'}")


def generate_report(state: dict, timestamp: str):
    """Generate comprehensive comparison report."""
    results = state["results"]

    # Group by retrieval method
    by_method = {}
    for key, data in results.items():
        method = data.get("method", "unknown")
        if method not in by_method:
            by_method[method] = []
        by_method[method].append((key, data))

    # Sort each method by score
    for method in by_method:
        by_method[method].sort(key=lambda x: x[1]["avg_score"], reverse=True)

    # Generate markdown report
    report = f"""# Reranker Hybrid Search Benchmark Report

**Date:** {timestamp}
**Questions:** {len(TEST_QUESTIONS)} (85% Russian, 15% English/Mixed)
**Instructions Tested:** {len(INSTRUCTIONS)}
**Retrieval Methods:** Random, Semantic, Keyword

---

## Executive Summary

Compared three retrieval methods for reranker instruction optimization:
1. **RANDOM**: Documents fetched randomly (baseline, no relevance)
2. **SEMANTIC**: Embedding-based semantic search (Qwen3-Embeddings)
3. **KEYWORD**: Keyword/grep matching (simulated hybrid search)

---

## Best Instruction per Retrieval Method

"""

    for method in ["semantic", "keyword", "random"]:
        if method in by_method and by_method[method]:
            best = by_method[method][0]
            report += f"### {method.upper()}\n"
            report += f"- **Best:** `{best[0]}`\n"
            report += f"  - Score: {best[1]['avg_score']:.4f}\n"
            report += f"  - Instruction: {best[1]['instruction'][:60]}...\n\n"

    report += "---\n\n## Detailed Results by Method\n\n"

    for method in ["semantic", "keyword", "random"]:
        if method not in by_method:
            continue

        report += f"### {method.upper()}\n\n"
        report += "| Rank | Instruction | Avg Score | EN Score | RU Score | MIX Score |\n"
        report += "|------|-------------|-----------|----------|----------|-----------|\n"

        for i, (key, data) in enumerate(by_method[method], 1):
            by_lang = data.get("by_lang", {})
            inst_name = key.replace(f"{method}_", "")
            report += f"| {i} | `{inst_name}` | {data['avg_score']:.4f} | {by_lang.get('en', 0):.4f} | {by_lang.get('ru', 0):.4f} | {by_lang.get('mixed', 0):.4f} |\n"

        report += "\n"

    # Cross-method comparison
    report += "---\n\n## Cross-Method Comparison\n\n"
    report += "Best performing instruction for each combination:\n\n"
    report += "| Instruction | Semantic | Keyword | Random |\n"
    report += "|-------------|----------|---------|--------|\n"

    for inst_name in INSTRUCTIONS.keys():
        scores_row = []
        for method in ["semantic", "keyword", "random"]:
            key = f"{method}_{inst_name}"
            if key in results:
                scores_row.append(f"{results[key]['avg_score']:.4f}")
            else:
                scores_row.append("-")
        report += f"| `{inst_name}` | {scores_row[0]} | {scores_row[1]} | {scores_row[2]} |\n"

    # Recommendations
    report += "\n---\n\n## Recommendations\n\n"

    # Best overall
    all_results = [(k, v) for k, v in results.items()]
    all_results.sort(key=lambda x: x[1]["avg_score"], reverse=True)
    best_overall = all_results[0]

    report += f"### Best Overall (All Methods)\n"
    report += f"- **{best_overall[0]}**: {best_overall[1]['avg_score']:.4f}\n"
    report += f"- Instruction: `{best_overall[1]['instruction']}`\n\n"

    # Best for semantic (realistic RAG usage)
    if "semantic" in by_method and by_method["semantic"]:
        best_semantic = by_method["semantic"][0]
        report += f"### Recommended for RAG with Semantic Search\n"
        report += f"```yaml\n"
        report += f'default_instruction: "{best_semantic[1]["instruction"]}"\n'
        report += f"```\n\n"

    # Low performers
    report += "---\n\n## Low Performers (Baseline Reference)\n\n"
    report += "These instructions underperformed across methods:\n\n"

    for key, data in all_results[-5:]:
        report += f"- `{key}`: {data['avg_score']:.4f}\n"
        report += f"  - {data['instruction'][:70]}...\n\n"

    # Methodology
    report += f"""---

## Methodology

1. **Retrieval Methods:**
   - RANDOM: Random documents from ChromaDB (baseline)
   - SEMANTIC: Qwen3-Embedding-8B via OpenRouter
   - KEYWORD: Word matching (2-5 keywords from query)

2. **Documents:** ChromaDB collection with {state.get("questions_with_docs", {}).get("semantic", []).__len__() if state.get("questions_with_docs", {}).get("semantic") else "N/A"} Russian/English technical docs

3. **Reranker:** Qwen3-Reranker-0.6B via mosec server (port 7998)

4. **Questions:** {len(TEST_QUESTIONS)} total
   - Russian: {sum(1 for q in TEST_QUESTIONS if q["lang"] == "ru")}
   - Mixed: {sum(1 for q in TEST_QUESTIONS if q["lang"] == "mixed")}
   - English: {sum(1 for q in TEST_QUESTIONS if q["lang"] == "en")}

---

## Raw Results

```json
{json.dumps(results, indent=2, ensure_ascii=False)}
```
"""

    # Save report
    report_file = OUTPUT_DIR / f"reranker_hybrid_report_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    run_benchmark()
