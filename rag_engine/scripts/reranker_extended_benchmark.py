"""
Extended Reranker Benchmark - Round 2

Tests refined instructions:
1. Informal Russian ("Найди" vs "Найдите")
2. Qwen-like format ("Дан поисковый запрос. Найди...")
3. Focused on best performers from Round 1
4. Completes KEYWORD and RANDOM methods

Also tests embedding instruction variants.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"
STATE_FILE = sorted(OUTPUT_DIR.glob("reranker_fast_results_*.json"))[-1]
DATASET_FILE = sorted(OUTPUT_DIR.glob("20260321-reranker-dataset-complete.json"))[-1]

QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

# =============================================================================
# EXTENDED INSTRUCTIONS - Round 2
# Focus on best performers + variants
# =============================================================================
INSTRUCTIONS_V2 = {
    # --- BEST FROM ROUND 1 ---
    "v1_best_ru_concise": "Найдите релевантную техническую документацию",  # Best in Round 1
    "v1_best_baseline": "Given a web search query, retrieve relevant passages that answer the query",  # Runner-up
    "v1_best_en_concise": "Find relevant technical documentation",
    "v1_best_ru_integration": "Найдите руководства по интеграции, документацию API и примеры конфигураций",
    # --- INFORMAL RUSSIAN ("Найди" vs "Найдите") ---
    "ru_informal_concise": "Найди релевантную техническую документацию",
    "ru_informal_docs": "Найди техническую документацию, подходящую под запрос",
    "ru_informal_integration": "Найди руководства по интеграции и API",
    "ru_informal_platform": "Найди документацию по платформе Comindware",
    # --- QWEN-LIKE FORMAT (inspired by default) ---
    "qwen_like_ru": "Дан поисковый запрос. Найди релевантную техническую документацию, подходящую под запрос",
    "qwen_like_ru_short": "Дан запрос. Найди техническую документацию",
    "qwen_like_ru_platform": "Дан поисковый запрос. Найди документацию по платформе Comindware: руководства, примеры кода, API",
    # --- ENGLISH OPTIMIZED ---
    "en_optimized": "Given a search query about Comindware Platform, retrieve relevant technical documentation",
    "en_short": "Find documentation for the query",
    "en_instructive": "Retrieve relevant passages that answer the question from technical documentation",
    # --- BILINGUAL INFORMAL ---
    "bilingual_informal": "Find relevant documentation. Найди релевантную документацию",
    "bilingual_informal_platform": "Find platform docs. Найди документацию по платформе",
    # --- CONTEXT-AWARE (mentioning corpus language) ---
    "context_ru_docs": "Find relevant documentation. Documents are in Russian with English code examples",
    "context_ru_short": "Релевантная документация на русском с примерами кода",
    # --- FOCUSED VARIANTS ---
    "ru_code_aware": "Найди техническую документацию с примерами кода",
    "ru_solution_aware": "Найди документацию для решения проблемы",
}


def load_state():
    """Load previous benchmark results."""
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset():
    """Load pre-fetched dataset."""
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_extended_benchmark():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"reranker_extended_results_{timestamp}.json"

    print("=" * 70)
    print("EXTENDED RERANKER BENCHMARK - ROUND 2")
    print("=" * 70)
    print(f"Previous best: ru_concise (0.5533)")
    print(f"New instructions: {len(INSTRUCTIONS_V2)}")
    print()

    # Load dataset
    dataset = load_dataset()
    questions = dataset["questions"]
    print(f"Questions: {len(questions)}")

    # Load previous state for methods already tested
    prev_state = load_state()
    prev_results = prev_state.get("results", {})

    # Create reranker adapter
    config = ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/score",
        reranker_type="llm_reranker",
        default_instruction=INSTRUCTIONS_V2["v1_best_baseline"],
        formatting=QWEN3_FORMATTING,
    )
    adapter = RerankerAdapter(config)

    results = {
        "round1": prev_results,  # Keep Round 1 results
        "round2": {},
        "metadata": {
            "timestamp": timestamp,
            "questions": len(questions),
            "instructions_round2": len(INSTRUCTIONS_V2),
        },
    }

    processed = set()

    # Test each retrieval method
    for method in ["semantic", "keyword", "random"]:
        if method not in dataset:
            print(f"[SKIP] {method} - not in dataset")
            continue

        print(f"\n{'=' * 70}\n[{method.upper()}] ROUND 2\n{'=' * 70}")

        method_data = dataset[method]
        results["round2"][method] = {}

        for name, instruction in INSTRUCTIONS_V2.items():
            # Skip if already tested in Round 1 (same instruction)
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
                    if errors > 5:
                        break

            if all_scores:
                avg = sum(all_scores) / len(all_scores)
                results["round2"][method][name] = {
                    "instruction": instruction,
                    "avg_score": avg,
                    "count": len(all_scores),
                    "by_lang": {k: sum(v) / len(v) if v else 0 for k, v in scores_by_lang.items()},
                    "by_type": {k: sum(v) / len(v) if v else 0 for k, v in scores_by_type.items()},
                }
                print(
                    f"  Avg: {avg:.4f} | EN: {results['round2'][method][name]['by_lang']['en']:.4f} | RU: {results['round2'][method][name]['by_lang']['ru']:.4f}"
                )

            processed.add(key)

            # Save after each instruction
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # Generate cumulative report
    generate_cumulative_report(results, timestamp)
    print(f"\nReport: {OUTPUT_DIR / f'reranker_cumulative_report_{timestamp}.md'}")


def generate_cumulative_report(results: dict, timestamp: str):
    """Generate cumulative report combining Round 1 and Round 2."""

    # Round 1 results
    round1 = results.get("round1", {})
    round2_semantic = results.get("round2", {}).get("semantic", {})
    round2_keyword = results.get("round2", {}).get("keyword", {})
    round2_random = results.get("round2", {}).get("random", {})

    # Combine all semantic results
    all_semantic = {}
    for method in ["semantic"]:
        if method in round1:
            for name, data in round1[method].items():
                if isinstance(data, dict) and "avg_score" in data:
                    all_semantic[f"r1_{name}"] = data
        if method in round2_semantic:
            for name, data in round2_semantic.items():
                if isinstance(data, dict) and "avg_score" in data:
                    all_semantic[f"r2_{name}"] = data

    sorted_semantic = sorted(all_semantic.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    report = f"""# Cumulative Reranker Benchmark Report

**Date:** {timestamp}
**Round 1:** {len(round1.get("semantic", {}))} instructions (SEMANTIC complete, KEYWORD partial)
**Round 2:** {len(INSTRUCTIONS_V2)} new instruction variants

---

## Executive Summary

### SEMANTIC SEARCH - Top 10 (Combined R1 + R2)

| Rank | Instruction | Score | EN | RU | MIX | Type |
|------|-------------|-------|-----|-----|-----|------|
"""

    for i, (name, data) in enumerate(sorted_semantic[:10], 1):
        r = "R2" if name.startswith("r2_") else "R1"
        inst_name = name[3:] if name.startswith(("r1_", "r2_")) else name
        inst_type = (
            "Russian"
            if "ru" in inst_name.lower() or "найди" in data["instruction"].lower()
            else ("Bilingual" if "bilingual" in inst_name.lower() else "English")
        )
        report += f"| {i} | `{inst_name}` ({r}) | {data['avg_score']:.4f} | {data['by_lang']['en']:.4f} | {data['by_lang']['ru']:.4f} | {data['by_lang']['mixed']:.4f} | {inst_type} |\n"

    # Round 2 specific results
    if round2_semantic:
        report += "\n---\n\n## Round 2 - New Variants\n\n"
        report += "### Informal Russian (Найди vs Найдите)\n\n"
        informal = [(k, v) for k, v in round2_semantic.items() if "informal" in k]
        for name, data in sorted(informal, key=lambda x: x[1]["avg_score"], reverse=True):
            report += f"- `{name}`: {data['avg_score']:.4f} - {data['instruction'][:50]}...\n"

        report += "\n### Qwen-like Format\n\n"
        qwen_like = [(k, v) for k, v in round2_semantic.items() if "qwen_like" in k]
        for name, data in sorted(qwen_like, key=lambda x: x[1]["avg_score"], reverse=True):
            report += f"- `{name}`: {data['avg_score']:.4f} - {data['instruction'][:50]}...\n"

    # Recommendations
    report += "\n---\n\n## Recommendations\n\n"

    if sorted_semantic:
        best = sorted_semantic[0]
        inst_name = best[0][3:] if best[0].startswith(("r1_", "r2_")) else best[0]
        report += f"### Best Overall (SEMANTIC)\n\n**Instruction:** `{inst_name}`\n```\n{best[1]['instruction']}\n```\n\n"
        report += f"- **Score:** {best[1]['avg_score']:.4f}\n"
        report += f"- **Russian:** {best[1]['by_lang']['ru']:.4f}\n"
        report += f"- **English:** {best[1]['by_lang']['en']:.4f}\n\n"

        # Check if informal outperforms formal
        formal_score = round1.get("semantic", {}).get("ru_concise", {}).get("avg_score", 0)
        informal_score = round2_semantic.get("ru_informal_concise", {}).get("avg_score", 0)
        if informal_score and formal_score:
            if informal_score > formal_score:
                report += f"### Informal Russian Outperforms!\n\n"
                report += f"- **Informal** (`ru_informal_concise`): {informal_score:.4f}\n"
                report += f"- **Formal** (`ru_concise`): {formal_score:.4f}\n"
                report += f"- **Gain:** +{(informal_score - formal_score) * 100:.2f}%\n\n"

    report += '### For models.yaml\n\n```yaml\ndefault_instruction: "Найди релевантную техническую документацию"\n# Or use Qwen3 default for broader compatibility:\n# default_instruction: "Given a web search query, retrieve relevant passages that answer the query"\n```\n'

    # Save
    report_file = OUTPUT_DIR / f"reranker_cumulative_report_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    run_extended_benchmark()
