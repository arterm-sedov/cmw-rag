"""
Reranker Instruction Benchmark - Uses Pre-fetched Datasets

Benchmarks different instructions on pre-fetched documents from:
1. SEMANTIC search (Qwen3-Embeddings)
2. KEYWORD search (BM25-like word matching)
3. RANDOM baseline

No network calls during benchmark - uses saved dataset.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find the most recent complete dataset
DATASET_FILE = sorted(OUTPUT_DIR.glob("20260321-reranker-dataset-complete.json"))[-1]

QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

# =============================================================================
# INSTRUCTIONS TO TEST -Following Qwen3 docs: "write instructions in English"
# =============================================================================
INSTRUCTIONS = {
    # --- BASELINE (Qwen3 default) ---
    "baseline_default": "Given a web search query, retrieve relevant passages that answer the query",
    # --- ENGLISH WITH DOC LANGUAGE CONTEXT (Qwen3 recommends English) ---
    "en_context_ru_docs": "Find relevant documentation. Documents are primarily in Russian with code snippets in English",
    "en_context_mixed": "Retrieve technical documentation (Russian text, English code examples) that answers the query",
    "en_context_platform": "Find Comindware Platform documentation in Russian/English with configuration examples and code",
    "en_context_bilingual": "Find relevant documents (Russian documentation with English code examples) that answer the question",
    "en_context_code": "Find documentation with code examples. Documents contain Russian text and English/Russian code",
    # --- PURE ENGLISH (no context) ---
    "en_platform": "Find documentation about Comindware Platform features, configurations, and APIs",
    "en_integration": "Find integration guides, API documentation, and configuration examples",
    "en_code": "Find code examples, API references, and technical documentation",
    "en_troubleshooting": "Find troubleshooting guides, error solutions, and problem resolutions",
    "en_concise": "Find relevant technical documentation",
    # --- PURE RUSSIAN---
    "ru_platform": "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации",
    "ru_integration": "Найдите руководства по интеграции, документацию API и примеры конфигураций",
    "ru_code": "Найдите примеры кода, справочники API и техническую документацию",
    "ru_troubleshooting": "Найдите руководства по устранению неполадок, решения ошибок и проблемы",
    "ru_concise": "Найдите релевантную техническую документацию",
    # --- BILINGUAL (EN+RU duplicated literally) ---
    "bilingual_platform": "Find Comindware Platform documentation. Найдите документацию по платформе Comindware",
    "bilingual_integration": "Find integration guides. Найдите руководства по интеграции",
    "bilingual_code": "Find code examples and documentation. Найдите примеры кода и документацию",
    "bilingual_context": "Retrieve technical documentation (Russian/English). Получите техническую документацию (русский/английский)",
    "bilingual_full": "Find relevant documentation about Comindware Platform including guides, code examples, configurations, and API references. Найдите релевантную документацию по платформе Comindware включая руководства, примеры кода, конфигурации и API",
    # --- INSTRUCTION VARIATIONS ---
    "en_web_search": "Given a web search query, retrieve relevant passages that answer the query from documentation",
    "en_find_docs": "Find documents that help answer the question from technical documentation",
    "en_retrieve_russian": "Retrieve relevant passages from Russian technical documentation with code examples",
    "ru_find_docs": "Найдите документы, которые помогают ответить на вопрос из технической документации",
    "ru_retrieve_platform": "Получите релевантные отрывки из документации Comindware Platform",
}


def load_dataset():
    """Load pre-fetched dataset."""
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    state_file = OUTPUT_DIR / f"reranker_benchmark_results_{timestamp}.json"

    print("=" * 80)
    print("RERANKER INSTRUCTION BENCHMARK - Pre-fetched Dataset")
    print("=" * 80)
    print(f"Dataset: {DATASET_FILE.name}")
    print(f"Instructions: {len(INSTRUCTIONS)}")
    print()

    # Load dataset
    dataset = load_dataset()
    questions = dataset["questions"]
    print(f"Questions: {len(questions)}")
    print(f"  Russian: {sum(1 for q in questions if q['lang'] == 'ru')}")
    print(f"  Mixed: {sum(1 for q in questions if q['lang'] == 'mixed')}")
    print(f"  English: {sum(1 for q in questions if q['lang'] == 'en')}")
    print(f"  Keyword: {sum(1 for q in questions if q['type'] == 'keyword')}")
    print(f"  Natural: {sum(1 for q in questions if q['type'] == 'natural')}")
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

    # Results storage
    results = {
        "metadata": {
            "timestamp": timestamp,
            "questions": len(questions),
            "instructions": len(INSTRUCTIONS),
        }
    }

    # Test each retrieval method
    for method in ["semantic", "keyword", "random"]:
        print(f"\n{'=' * 80}")
        print(f"RETRIEVAL METHOD: {method.upper()}")
        print(f"{'=' * 80}")

        method_data = dataset[method]
        results[method] = {}

        for name, instruction in INSTRUCTIONS.items():
            print(f"\n[{method}] {name}")
            print(f"  Instruction: {instruction[:60]}...")

            all_scores = []
            scores_by_lang = {"en": [], "ru": [], "mixed": []}
            scores_by_type = {"keyword": [], "natural": []}
            scores_by_topic = {}

            for q_idx, q_data in enumerate(method_data):
                q = questions[q_data["q_idx"]]
                docs = q_data["docs"]

                if not docs:
                    continue

                candidates = [(d["content"], 0.0) for d in docs]
                try:
                    ranked = adapter.rerank(
                        q["q"], candidates, top_k=len(candidates), instruction=instruction
                    )
                    scores = [r[1] for r in ranked]
                    all_scores.extend(scores)
                    scores_by_lang[q["lang"]].extend(scores)
                    scores_by_type[q["type"]].extend(scores)

                    topic = q.get("topic", "other")
                    if topic not in scores_by_topic:
                        scores_by_topic[topic] = []
                    scores_by_topic[topic].extend(scores)
                except Exception as e:
                    print(f"  Error on question {q_idx}: {e}")

            if all_scores:
                avg_score = sum(all_scores) / len(all_scores)
                results[method][name] = {
                    "instruction": instruction,
                    "avg_score": avg_score,
                    "count": len(all_scores),
                    "by_lang": {k: sum(v) / len(v) if v else 0 for k, v in scores_by_lang.items()},
                    "by_type": {k: sum(v) / len(v) if v else 0 for k, v in scores_by_type.items()},
                    "by_topic": {
                        k: sum(v) / len(v) if v else 0 for k, v in scores_by_topic.items()
                    },
                }
                print(f"  Avg: {avg_score:.4f}")
                print(
                    f"  By lang: EN={results[method][name]['by_lang']['en']:.4f}, "
                    f"RU={results[method][name]['by_lang']['ru']:.4f}, "
                    f"MIX={results[method][name]['by_lang']['mixed']:.4f}"
                )
                print(
                    f"  By type: Keyword={results[method][name]['by_type']['keyword']:.4f}, "
                    f"Natural={results[method][name]['by_type']['natural']:.4f}"
                )

        # Save after each method
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Generate report
    generate_report(results, timestamp)
    print(f"\nReport saved to: {OUTPUT_DIR / f'reranker_benchmark_report_{timestamp}.md'}")


def generate_report(results: dict, timestamp: str):
    """Generate comprehensive comparison report."""

    # Sort each method by score
    sorted_by_method = {}
    for method in ["semantic", "keyword", "random"]:
        if method in results:
            sorted_by_method[method] = sorted(
                results[method].items(), key=lambda x: x[1]["avg_score"], reverse=True
            )

    # Generate markdown report
    report = f"""# Reranker Instruction Benchmark Report

**Date:** {timestamp}
**Questions:** {results["metadata"]["questions"]}
**Instructions Tested:** {results["metadata"]["instructions"]}

---

## Executive Summary

Benchmarked {results["metadata"]["instructions"]} instructions across three retrieval methods:
1. **SEMANTIC**: Embedding-based search (Qwen3-Embeddings)
2. **KEYWORD**: BM25-like word matching
3. **RANDOM**: Baseline (random documents)

---

## Best Instruction per Retrieval Method

"""

    for method in ["semantic", "keyword", "random"]:
        if method in sorted_by_method and sorted_by_method[method]:
            best = sorted_by_method[method][0]
            report += f"### {method.upper()}\n"
            report += f"- **Best:** `{best[0]}`\n"
            report += f"  - Score: {best[1]['avg_score']:.4f}\n"
            report += f"  - By lang: EN={best[1]['by_lang']['en']:.4f}, RU={best[1]['by_lang']['ru']:.4f}\n"
            report += f"  - Instruction: {best[1]['instruction'][:80]}...\n\n"

    report += "---\n\n## Detailed Results\n\n"

    for method in ["semantic", "keyword", "random"]:
        if method not in sorted_by_method:
            continue

        report += f"### {method.upper()}\n\n"
        report += "| Rank | Instruction | Avg Score | EN Score | RU Score | MIX Score | Keyword | Natural |\n"
        report += "|------|-------------|-----------|----------|----------|-----------|---------|--------|\n"

        for i, (name, data) in enumerate(sorted_by_method[method], 1):
            by_lang = data.get("by_lang", {})
            by_type = data.get("by_type", {})
            report += f"| {i} | `{name}` | {data['avg_score']:.4f} | {by_lang.get('en', 0):.4f} | {by_lang.get('ru', 0):.4f} | {by_lang.get('mixed', 0):.4f} | {by_type.get('keyword', 0):.4f} | {by_type.get('natural', 0):.4f} |\n"

        report += "\n"

    # Cross-method comparison
    report += "---\n\n## Cross-Method Comparison\n\n"
    report += "| Instruction | Semantic | Keyword | Random | Best For |\n"
    report += "|-------------|----------|---------|--------|----------|\n"

    for inst_name in INSTRUCTIONS.keys():
        scores_row = []
        for method in ["semantic", "keyword", "random"]:
            if method in results and inst_name in results[method]:
                scores_row.append(f"{results[method][inst_name]['avg_score']:.4f}")
            else:
                scores_row.append("-")

        # Determine best method for this instruction
        valid_scores = {}
        for i, method in enumerate(["semantic", "keyword", "random"]):
            if scores_row[i] != "-":
                valid_scores[method] = float(scores_row[i])

        best_method = max(valid_scores, key=valid_scores.get) if valid_scores else "-"

        report += f"| `{inst_name}` | {scores_row[0]} | {scores_row[1]} | {scores_row[2]} | {best_method} |\n"

    # Category analysis
    report += "\n---\n\n## Instruction Category Analysis\n\n"

    categories = {
        "Baseline": ["baseline_default"],
        "English + Context": [k for k in INSTRUCTIONS if k.startswith("en_context_")],
        "English (no context)": [
            k
            for k in INSTRUCTIONS
            if k.startswith("en_")
            and "context" not in k
            and "web" not in k
            and "find" not in k
            and "retrieve" not in k
        ],
        "Russian": [k for k in INSTRUCTIONS if k.startswith("ru_")],
        "Bilingual": [k for k in INSTRUCTIONS if k.startswith("bilingual_")],
    }

    for method in ["semantic", "keyword", "random"]:
        if method not in sorted_by_method:
            continue

        report += f"### {method.upper()} - Category Performance\n\n"

        for cat_name, cat_insts in categories.items():
            cat_scores = []
            for inst_name in cat_insts:
                if inst_name in results[method]:
                    cat_scores.append(results[method][inst_name]["avg_score"])

            if cat_scores:
                avg_cat = sum(cat_scores) / len(cat_scores)
                report += (
                    f"- **{cat_name}**: {avg_cat:.4f} (avg of {len(cat_scores)} instructions)\n"
                )

        report += "\n"

    # Recommendations
    report += "---\n\n## Recommendations\n\n"

    # Best overall (semantic is most realistic for RAG)
    if "semantic" in sorted_by_method and sorted_by_method["semantic"]:
        best_semantic = sorted_by_method["semantic"][0]
        report += f"### Recommended for RAG with Semantic Search\n\n"
        report += f"**Best instruction:** `{best_semantic[0]}`\n\n"
        report += f"```yaml\n"
        report += f'default_instruction: "{best_semantic[1]["instruction"]}"\n'
        report += f"```\n\n"
        report += f"- **Score:** {best_semantic[1]['avg_score']:.4f}\n"
        report += f"- **Russian queries:** {best_semantic[1]['by_lang']['ru']:.4f}\n"
        report += f"- **English queries:** {best_semantic[1]['by_lang']['en']:.4f}\n\n"

    # Best for keyword search
    if "keyword" in sorted_by_method and sorted_by_method["keyword"]:
        best_keyword = sorted_by_method["keyword"][0]
        report += f"### Best for Keyword/Hybrid Search\n\n"
        report += (
            f"**Instruction:** `{best_keyword[0]}` (score: {best_keyword[1]['avg_score']:.4f})\n\n"
        )

    # Qwen3 recommendation
    report += "### Notes\n\n"
    report += "Per Qwen3-Reranker documentation:\n"
    report += '> "In multilingual contexts, we advise users to write their instructions in English, as most instructions utilized during the model training process were originally written in English."\n\n'
    report += "This benchmark tests both English and Russian instructions to validate this recommendation.\n"

    # Low performers
    report += "---\n\n## Low Performers\n\n"
    report += "Instructions that underperformed across all methods:\n\n"

    all_scores = {}
    for method in ["semantic", "keyword", "random"]:
        if method in results:
            for inst_name, data in results[method].items():
                if inst_name not in all_scores:
                    all_scores[inst_name] = []
                all_scores[inst_name].append(data["avg_score"])

    avg_all = {k: sum(v) / len(v) for k, v in all_scores.items()}
    sorted_low = sorted(avg_all.items(), key=lambda x: x[1])[:5]

    for inst_name, score in sorted_low:
        report += f"- `{inst_name}`: avg {score:.4f}\n"
        report += f"  - {INSTRUCTIONS.get(inst_name, '')[:70]}...\n\n"

    # Save report
    report_file = OUTPUT_DIR / f"reranker_benchmark_report_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    run_benchmark()
