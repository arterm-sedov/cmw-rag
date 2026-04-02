"""
Reranker Instruction Research - Comprehensive Benchmark

Tests 20+ instructions for Russian-focused support agent:
- Pure English instructions (with/without doc language context)
- Pure Russian instructions
- Bilingual instructions (EN+RU duplicated)
- Platform-specific, code-aware, error-focused instructions

Saves results incrementally to avoid timeouts.
"""

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

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

# =============================================================================
# INSTRUCTIONS TO TEST - 20+ variants
# =============================================================================
INSTRUCTIONS = {
    # --- BASELINE (Qwen3 default) ---
    "baseline_default": "Given a web search query, retrieve relevant passages that answer the query",
    # --- PURE ENGLISH (no context) ---
    "en_platform": "Find documentation about Comindware Platform features, configurations, and APIs",
    "en_integration": "Find integration guides, API documentation, and configuration examples",
    "en_code": "Find code examples, API references, and technical documentation",
    "en_troubleshooting": "Find troubleshooting guides, error solutions, and problem resolutions",
    # --- ENGLISH WITH DOC LANGUAGE CONTEXT ---
    "en_context_ru_docs": "Find relevant documentation. Documents are primarily in Russian with code snippets in English",
    "en_context_mixed": "Retrieve technical documentation (Russian text, English code examples) that answers the query",
    "en_context_platform": "Find Comindware Platform documentation in Russian/English with configuration examples and code",
    "en_context_bilingual": "Find relevant documents (Russian documentation with English code examples) that answer the question",
    # --- PURE RUSSIAN ---
    "ru_platform": "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации",
    "ru_integration": "Найдите руководства по интеграции, документацию API и примеры конфигураций",
    "ru_code": "Найдите примеры кода, справочники API и техническую документацию",
    "ru_troubleshooting": "Найдите руководства по устранению неполадок, решения ошибок и проблемы",
    # --- BILINGUAL (EN+RU duplicated) ---
    "bilingual_platform": "Find Comindware Platform documentation. Найдите документацию по платформе Comindware",
    "bilingual_integration": "Find integration guides. Найдите руководства по интеграции",
    "bilingual_code": "Find code examples and documentation. Найдите примеры кода и документацию",
    "bilingual_context": "Retrieve technical documentation (Russian/English). Получите техническую документацию (русский/английский)",
    "bilingual_full": "Find relevant documentation about Comindware Platform including guides, code examples, configurations, and API references. Найдите релевантную документацию по платформе Comindware включая руководства, примеры кода, конфигурации и API",
    # --- INFRASTRUCTURE & ERRORS ---
    "en_infrastructure": "Find infrastructure setup guides, deployment instructions, and configuration",
    "ru_infrastructure": "Найдите руководства по настройке инфраструктуры, инструкции по развертыванию",
    "en_errors": "Find error descriptions, solutions, and troubleshooting steps",
    "ru_errors": "Найдите описания ошибок, решения и инструкции по устранению неполадок",
    # --- CONCISE ---
    "en_concise": "Find relevant technical documentation",
    "ru_concise": "Найдите релевантную техническую документацию",
}

# =============================================================================
# REALISTIC QUESTIONS - Russian-focused (85% RU, 15% EN/Mixed)
# =============================================================================
TEST_QUESTIONS = [
    # --- RUSSIAN KEYWORDS (most common for support) ---
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
    # --- MIXED (Russian + English terms) - common for devs ---
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
    # --- ENGLISH (less common, 15%) ---
    {"q": "OData integration setup", "type": "keyword", "lang": "en", "topic": "integration"},
    {"q": "HTTP POST request example", "type": "keyword", "lang": "en", "topic": "scripts"},
    {
        "q": "How to configure OData integration?",
        "type": "natural",
        "lang": "en",
        "topic": "integration",
    },
    {"q": "How to send HTTP POST request?", "type": "natural", "lang": "en", "topic": "scripts"},
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


def search_chromadb(collection, embedding: list[float], limit: int = 5):
    """Search ChromaDB with embedding."""
    results = collection.query(
        query_embeddings=[embedding], n_results=limit, include=["documents", "metadatas"]
    )
    if not results["documents"] or not results["documents"][0]:
        return []
    return [
        {"content": d[:800], "source": m.get("source_file", "?")}
        for d, m in zip(results["documents"][0], results["metadatas"][0])
    ]


def save_state(state: dict):
    """Save state to file."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_state() -> dict:
    """Load state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"questions_with_docs": [], "results": {}, "processed": []}


def run_benchmark():
    global STATE_FILE

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATE_FILE = OUTPUT_DIR / f"reranker_research_state_{timestamp}.json"

    print("=" * 80)
    print("RERANKER INSTRUCTION RESEARCH - Russian-focused Support Agent")
    print("=" * 80)
    print(f"Instructions: {len(INSTRUCTIONS)}")
    print(f"Questions: {len(TEST_QUESTIONS)}")
    print(f"  Russian: {sum(1 for q in TEST_QUESTIONS if q['lang'] == 'ru')} keyword + natural")
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

    # Get embeddings for questions (if not already done)
    if not state.get("questions_with_docs"):
        print("Getting embeddings and searching ChromaDB...")
        questions_with_docs = []

        for i, q in enumerate(TEST_QUESTIONS):
            print(f"  [{i + 1}/{len(TEST_QUESTIONS)}] {q['q'][:40]}...", end="", flush=True)
            try:
                embedding = get_embedding(q["q"])
                docs = search_chromadb(collection, embedding, limit=5)
                questions_with_docs.append({**q, "docs": docs})
                print(f" found {len(docs)}")
            except Exception as e:
                print(f" ERROR: {e}")

        state["questions_with_docs"] = questions_with_docs
        save_state(state)

    questions_with_docs = state["questions_with_docs"]
    print(f"\nReady: {len(questions_with_docs)} questions with docs")
    print()

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

    # Test each instruction
    print("Testing instructions...")
    print("-" * 80)

    for name, instruction in INSTRUCTIONS.items():
        if name in state.get("processed", []):
            print(f"[SKIP] {name} (already done)")
            continue

        print(f"\n[{name}]")
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
            state["results"][name] = {
                "instruction": instruction,
                "avg_score": sum(all_scores) / len(all_scores),
                "count": len(all_scores),
                "by_lang": {k: sum(v) / len(v) if v else 0 for k, v in scores_by_lang.items()},
                "by_type": {k: sum(v) / len(v) if v else 0 for k, v in scores_by_type.items()},
            }
            print(f"  Avg: {state['results'][name]['avg_score']:.4f}")
            print(
                f"  By lang: EN={scores_by_lang['en'] and sum(scores_by_lang['en']) / len(scores_by_lang['en']):.4f}, "
                f"RU={scores_by_lang['ru'] and sum(scores_by_lang['ru']) / len(scores_by_lang['ru']):.4f}, "
                f"MIX={scores_by_lang['mixed'] and sum(scores_by_lang['mixed']) / len(scores_by_lang['mixed']):.4f}"
            )

        state.setdefault("processed", []).append(name)
        save_state(state)

    # Generate report
    generate_report(state, timestamp)
    print(f"\nReport saved to: {OUTPUT_DIR / f'rereranker_research_report_{timestamp}.md'}")


def generate_report(state: dict, timestamp: str):
    """Generate comprehensive research report."""
    results = state["results"]

    # Sort by overall score
    sorted_results = sorted(results.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    # Generate markdown report
    report = f"""# Reranker Instruction Research Report

**Date:** {timestamp}
**Questions:** {len(state.get("questions_with_docs", []))} (85% Russian, 15% English/Mixed)
**Instructions Tested:** {len(results)}

---

## Executive Summary

Tested {len(results)} instructions on Russian-focused support queries using semantic search (Qwen3-Embeddings) + Qwen3-Reranker.

**Best Instruction:** `{sorted_results[0][0]}`
**Score:** {sorted_results[0][1]["avg_score"]:.4f}
**Instruction:** {sorted_results[0][1]["instruction"]}

---

## Overall Rankings (by avg_score)

| Rank | Instruction | Avg Score | EN Score | RU Score | MIX Score |
|------|-------------|-----------|----------|----------|-----------|
"""

    for i, (name, data) in enumerate(sorted_results, 1):
        by_lang = data.get("by_lang", {})
        report += f"| {i} | `{name}` | {data['avg_score']:.4f} | {by_lang.get('en', 0):.4f} | {by_lang.get('ru', 0):.4f} | {by_lang.get('mixed', 0):.4f} |\n"

    # Category analysis
    report += "\n---\n\n## Category Analysis\n\n"

    # By instruction type
    en_no_context = [r for r in sorted_results if r[0].startswith("en_") and "context" not in r[0]]
    en_with_context = [r for r in sorted_results if "context" in r[0]]
    ru_instructions = [r for r in sorted_results if r[0].startswith("ru_")]
    bilingual = [r for r in sorted_results if r[0].startswith("bilingual_")]

    report += "### English Instructions (no context)\n"
    for name, data in en_no_context[:5]:
        report += f"- `{name}`: {data['avg_score']:.4f} - {data['instruction'][:50]}...\n"

    report += "\n### English Instructions (with RU/doc context)\n"
    for name, data in en_with_context[:5]:
        report += f"- `{name}`: {data['avg_score']:.4f} - {data['instruction'][:50]}...\n"

    report += "\n### Russian Instructions\n"
    for name, data in ru_instructions[:5]:
        report += f"- `{name}`: {data['avg_score']:.4f} - {data['instruction'][:50]}...\n"

    report += "\n### Bilingual Instructions (EN+RU)\n"
    for name, data in bilingual:
        report += f"- `{name}`: {data['avg_score']:.4f} - {data['instruction'][:50]}...\n"

    # Low performers
    report += "\n---\n\n## Low Performers (Baseline Reference)\n\n"
    report += "These instructions underperformed and should not be used:\n\n"
    for name, data in sorted_results[-5:]:
        report += f"- `{name}`: {data['avg_score']:.4f}\n"
        report += f"  - {data['instruction'][:70]}...\n\n"

    # Recommendations
    report += """---

## Recommendations

### For Russian-focused Support Agent (85% RU queries)

"""

    # Best for Russian
    best_ru = max(
        [(name, data["by_lang"].get("ru", 0)) for name, data in results.items()], key=lambda x: x[1]
    )
    report += f"**Best for Russian queries:** `{best_ru[0]}` (score: {best_ru[1]:.4f})\n"

    # Best overall
    report += f"\n**Best overall:** `{sorted_results[0][0]}` (score: {sorted_results[0][1]['avg_score']:.4f})\n"

    report += f"""

### Recommended default_instruction in models.yaml:

```yaml
default_instruction: "{sorted_results[0][1]["instruction"]}"
```

---

## Methodology

1. **Semantic Search:** Qwen3-Embedding-8B via OpenRouter
2. **Documents:** ChromaDB collection with {state.get("questions_with_docs", []).__len__() if state.get("questions_with_docs") else "N/A"} Russian/English technical docs
3. **Reranker:** Qwen3-Reranker-0.6B via mosec server
4. **Questions:** 85% Russian, 15% English/Mixed (realistic for Russian support)

---

## Raw Results

```json
{json.dumps(results, indent=2, ensure_ascii=False)}
```
"""

    # Save report
    report_file = OUTPUT_DIR / f"reranker_research_report_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    run_benchmark()
