"""
Fast Reranker Benchmark - Key Instructions Only

Tests essential instructions on pre-fetched datasets with incremental saves.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"
DATASET_FILE = sorted(OUTPUT_DIR.glob("20260321-reranker-dataset-complete.json"))[-1]

QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

# Key instructions to test
INSTRUCTIONS = {
    # Baseline
    "baseline_default": "Given a web search query, retrieve relevant passages that answer the query",
    # English with context (recommended by Qwen3)
    "en_context_ru_docs": "Find relevant documentation. Documents are primarily in Russian with code snippets in English",
    "en_context_mixed": "Retrieve technical documentation (Russian text, English code examples) that answers the query",
    "en_context_platform": "Find Comindware Platform documentation in Russian/English with configuration examples and code",
    # Pure English
    "en_platform": "Find documentation about Comindware Platform features, configurations, and APIs",
    "en_integration": "Find integration guides, API documentation, and configuration examples",
    "en_concise": "Find relevant technical documentation",
    # Pure Russian
    "ru_platform": "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации",
    "ru_integration": "Найдите руководства по интеграции, документацию API и примеры конфигураций",
    "ru_concise": "Найдите релевантную техническую документацию",
    # Bilingual
    "bilingual_platform": "Find Comindware Platform documentation. Найдите документацию по платформе Comindware",
    "bilingual_integration": "Find integration guides. Найдите руководства по интеграции",
    "bilingual_context": "Retrieve technical documentation (Russian/English). Получите техническую документацию (русский/английский)",
}


def load_dataset():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    state_file = OUTPUT_DIR / f"reranker_fast_results_{timestamp}.json"

    print("=" * 70)
    print("FAST RERANKER BENCHMARK")
    print("=" * 70)
    print(f"Dataset: {DATASET_FILE.name}")
    print(f"Instructions: {len(INSTRUCTIONS)}")

    dataset = load_dataset()
    questions = dataset["questions"]
    print(f"Questions: {len(questions)}")
    print(f"  RU: {sum(1 for q in questions if q['lang'] == 'ru')}")
    print(f"  MIX: {sum(1 for q in questions if q['lang'] == 'mixed')}")
    print(f"  EN: {sum(1 for q in questions if q['lang'] == 'en')}")

    config = ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/score",
        reranker_type="llm_reranker",
        default_instruction=INSTRUCTIONS["baseline_default"],
        formatting=QWEN3_FORMATTING,
    )
    adapter = RerankerAdapter(config)

    results = {
        "metadata": {
            "timestamp": timestamp,
            "questions": len(questions),
            "instructions": len(INSTRUCTIONS),
        }
    }
    processed = set()

    # Check for existing state
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
            results = state.get("results", results)
            processed = set(state.get("processed", []))
            print(f"Resuming from {len(processed)} already processed")

    # Test each retrieval method
    for method in ["semantic", "keyword", "random"]:
        print(f"\n{'=' * 70}\n[{method.upper()}]\n{'=' * 70}")

        if method not in results:
            results[method] = {}
        method_data = dataset[method]

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
                    pass

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
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"results": results, "processed": list(processed)},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    # Generate report
    generate_report(results, timestamp)
    print(f"\nReport: {OUTPUT_DIR / f'reranker_fast_report_{timestamp}.md'}")


def generate_report(results: dict, timestamp: str):
    sorted_by_method = {}
    for method in ["semantic", "keyword", "random"]:
        if method in results:
            sorted_by_method[method] = sorted(
                results[method].items(), key=lambda x: x[1]["avg_score"], reverse=True
            )

    report = f"""# Reranker Instruction Benchmark Report

**Date:** {timestamp}
**Questions:** {results["metadata"]["questions"]}
**Instructions:** {results["metadata"]["instructions"]}

---

## Best Instruction per Method

"""

    for method in ["semantic", "keyword", "random"]:
        if method in sorted_by_method and sorted_by_method[method]:
            best = sorted_by_method[method][0]
            report += f"### {method.upper()}\n"
            report += f"- **Best:** `{best[0]}` ({best[1]['avg_score']:.4f})\n"
            report += f"- RU: {best[1]['by_lang']['ru']:.4f} | EN: {best[1]['by_lang']['en']:.4f} | MIX: {best[1]['by_lang']['mixed']:.4f}\n"
            report += f"- `{best[1]['instruction']}`\n\n"

    report += "---\n\n## Full Results\n\n"
    report += "### SEMANTIC\n\n"
    if "semantic" in sorted_by_method:
        report += "| Rank | Instruction | Score | EN | RU | MIX |\n|------|-------------|-------|-----|-----|-----|\n"
        for i, (name, data) in enumerate(sorted_by_method["semantic"], 1):
            report += f"| {i} | `{name}` | {data['avg_score']:.4f} | {data['by_lang']['en']:.4f} | {data['by_lang']['ru']:.4f} | {data['by_lang']['mixed']:.4f} |\n"

    report += "\n### KEYWORD\n\n"
    if "keyword" in sorted_by_method:
        report += "| Rank | Instruction | Score | EN | RU | MIX |\n|------|-------------|-------|-----|-----|-----|\n"
        for i, (name, data) in enumerate(sorted_by_method["keyword"], 1):
            report += f"| {i} | `{name}` | {data['avg_score']:.4f} | {data['by_lang']['en']:.4f} | {data['by_lang']['ru']:.4f} | {data['by_lang']['mixed']:.4f} |\n"

    report += "\n### RANDOM\n\n"
    if "random" in sorted_by_method:
        report += "| Rank | Instruction | Score | EN | RU | MIX |\n|------|-------------|-------|-----|-----|-----|\n"
        for i, (name, data) in enumerate(sorted_by_method["random"], 1):
            report += f"| {i} | `{name}` | {data['avg_score']:.4f} | {data['by_lang']['en']:.4f} | {data['by_lang']['ru']:.4f} | {data['by_lang']['mixed']:.4f} |\n"

    # Recommendation
    report += "\n---\n\n## Recommendation\n\n"
    if "semantic" in sorted_by_method and sorted_by_method["semantic"]:
        best = sorted_by_method["semantic"][0]
        report += f'**Recommended for RAG with Semantic Search:**\n\n```yaml\ndefault_instruction: "{best[1]["instruction"]}"\n```\n\n'
        report += f"Score: {best[1]['avg_score']:.4f} | Russian: {best[1]['by_lang']['ru']:.4f} | English: {best[1]['by_lang']['en']:.4f}\n"

    # Save
    report_file = OUTPUT_DIR / f"reranker_fast_report_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    run_benchmark()
