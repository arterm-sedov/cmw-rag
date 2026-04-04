"""
Continue Reranker Benchmark - Complete remaining tests

Runs keyword and random tests with better timeout handling.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"

# Find latest state file
STATE_FILE = sorted(OUTPUT_DIR.glob("reranker_fast_results_*.json"))[-1]
DATASET_FILE = sorted(OUTPUT_DIR.glob("20260321-reranker-dataset-complete.json"))[-1]

QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

INSTRUCTIONS = {
    "baseline_default": "Given a web search query, retrieve relevant passages that answer the query",
    "en_context_ru_docs": "Find relevant documentation. Documents are primarily in Russian with code snippets in English",
    "en_context_mixed": "Retrieve technical documentation (Russian text, English code examples) that answers the query",
    "en_context_platform": "Find Comindware Platform documentation in Russian/English with configuration examples and code",
    "en_platform": "Find documentation about Comindware Platform features, configurations, and APIs",
    "en_integration": "Find integration guides, API documentation, and configuration examples",
    "en_concise": "Find relevant technical documentation",
    "ru_platform": "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации",
    "ru_integration": "Найдите руководства по интеграции, документацию API и примеры конфигураций",
    "ru_concise": "Найдите релевантную техническую документацию",
    "bilingual_platform": "Find Comindware Platform documentation. Найдите документацию по платформе Comindware",
    "bilingual_integration": "Find integration guides. Найдите руководства по интеграции",
    "bilingual_context": "Retrieve technical documentation (Russian/English). Получите техническую документацию (русский/английский)",
}


def run_benchmark():
    print(f"Loading state from: {STATE_FILE}")
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    results = state.get("results", {})
    processed = set(state.get("processed", []))

    print(f"Loading dataset from: {DATASET_FILE}")
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]
    print(f"Questions: {len(questions)}")
    print(f"Already processed: {len(processed)}")

    config = ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/score",
        reranker_type="llm_reranker",
        default_instruction=INSTRUCTIONS["baseline_default"],
        formatting=QWEN3_FORMATTING,
        timeout=30.0,
        max_retries=2,
    )
    adapter = RerankerAdapter(config)

    # Run remaining methods
    for method in ["keyword", "random"]:
        if method not in results:
            results[method] = {}

        method_data = dataset[method]
        print(f"\n{'=' * 70}\n[{method.upper()}]\n{'=' * 70}")

        for name, instruction in INSTRUCTIONS.items():
            key = f"{method}_{name}"
            if key in processed:
                print(f"[SKIP] {key}")
                continue

            print(f"\n[{key}]")
            print(f"  Inst: {instruction[:50]}...")

            all_scores = []
            scores_by_lang = {"en": [], "ru": [], "mixed": []}
            scores_by_type = {"keyword": [], "natural": []}
            errors = 0

            for q_data in method_data:
                q = questions[q_data["q_idx"]]
                if not q_data["docs"]:
                    continue

                candidates = [(d["content"], 0.0) for d in q_data["docs"]]
                try:
                    ranked = adapter.rerank(
                        q["q"], candidates, top_k=len(candidates), instruction=instruction
                    )
                    scores = [r[1] for r in ranked]
                    all_scores.extend(scores)
                    scores_by_lang[q["lang"]].extend(scores)
                    scores_by_type[q["type"]].extend(scores)
                except Exception as e:
                    errors += 1
                    if errors > 10:
                        print(f"  Too many errors, stopping")
                        break
                    continue

            if all_scores:
                avg = sum(all_scores) / len(all_scores)
                results[method][name] = {
                    "instruction": instruction,
                    "avg_score": avg,
                    "count": len(all_scores),
                    "by_lang": {k: sum(v) / len(v) if v else 0 for k, v in scores_by_lang.items()},
                    "by_type": {k: sum(v) / len(v) if v else 0 for k, v in scores_by_type.items()},
                }
                print(
                    f"  Avg: {avg:.4f} | EN: {results[method][name]['by_lang']['en']:.4f} | RU: {results[method][name]['by_lang']['ru']:.4f}"
                )

            processed.add(key)

            # Save after each instruction
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {"results": results, "processed": list(processed)},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generate_report(results, timestamp)
    print(f"\nReport: {OUTPUT_DIR / f'reranker_benchmark_final_{timestamp}.md'}")


def generate_report(results: dict, timestamp: str):
    sorted_by_method = {}
    for method in ["semantic", "keyword", "random"]:
        if method in results:
            method_results = {
                k: v for k, v in results[method].items() if isinstance(v, dict) and "avg_score" in v
            }
            sorted_by_method[method] = sorted(
                method_results.items(), key=lambda x: x[1]["avg_score"], reverse=True
            )

    report = f"""# Reranker Instruction Benchmark Report

**Date:** {timestamp}

---

## Best Instruction per Retrieval Method

"""

    for method in ["semantic", "keyword", "random"]:
        if method in sorted_by_method and sorted_by_method[method]:
            best = sorted_by_method[method][0]
            report += f"### {method.upper()}\n"
            report += f"- **Best:** `{best[0]}` ({best[1]['avg_score']:.4f})\n"
            report += f"- RU: {best[1]['by_lang']['ru']:.4f} | EN: {best[1]['by_lang']['en']:.4f}\n"
            report += f"- `{best[1]['instruction']}`\n\n"

    report += "---\n\n## Full Results\n\n"

    for method in ["semantic", "keyword", "random"]:
        if method not in sorted_by_method or not sorted_by_method[method]:
            continue

        report += f"### {method.upper()}\n\n"
        report += "| Rank | Instruction | Score | EN | RU | MIX |\n|------|-------------|-------|-----|-----|-----|\n"

        for i, (name, data) in enumerate(sorted_by_method[method], 1):
            report += f"| {i} | `{name}` | {data['avg_score']:.4f} | {data['by_lang']['en']:.4f} | {data['by_lang']['ru']:.4f} | {data['by_lang']['mixed']:.4f} |\n"
        report += "\n"

    # Cross-method comparison
    report += "---\n\n## Cross-Method Comparison\n\n"
    report += "| Instruction | Semantic | Keyword | Random |\n|-------------|----------|---------|--------|\n"

    for inst_name in INSTRUCTIONS.keys():
        scores = []
        for method in ["semantic", "keyword", "random"]:
            if (
                method in results
                and inst_name in results[method]
                and isinstance(results[method][inst_name], dict)
            ):
                scores.append(f"{results[method][inst_name]['avg_score']:.4f}")
            else:
                scores.append("-")
        report += f"| `{inst_name}` | {scores[0]} | {scores[1]} | {scores[2]} |\n"

    # Recommendation
    report += "\n---\n\n## Recommendation\n\n"
    if "semantic" in sorted_by_method and sorted_by_method["semantic"]:
        best = sorted_by_method["semantic"][0]
        report += f'**Recommended for RAG with Semantic Search:**\n\n```yaml\ndefault_instruction: "{best[1]["instruction"]}"\n```\n\n'
        report += f"- **Score:** {best[1]['avg_score']:.4f}\n"
        report += f"- **Russian:** {best[1]['by_lang']['ru']:.4f}\n"
        report += f"- **English:** {best[1]['by_lang']['en']:.4f}\n"

    # Save
    report_file = OUTPUT_DIR / f"reranker_benchmark_final_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    run_benchmark()
