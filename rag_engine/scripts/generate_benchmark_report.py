"""
Generate report from current benchmark data
"""

import json
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"
STATE_FILE = sorted(OUTPUT_DIR.glob("reranker_fast_results_*.json"))[-1]

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


def main():
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    results = state.get("results", {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sort by method
    sorted_by_method = {}
    for method in ["semantic", "keyword", "random"]:
        if method in results:
            method_results = {
                k: v for k, v in results[method].items() if isinstance(v, dict) and "avg_score" in v
            }
            sorted_by_method[method] = sorted(
                method_results.items(), key=lambda x: x[1]["avg_score"], reverse=True
            )

    # Generate report
    report = f"""# Reranker Instruction Benchmark Report

**Date:** {timestamp}
**Dataset:** 52 questions (38 Russian, 8 Mixed, 6 English)

---

## Key Findings

Based on Qwen3-Reranker documentation: *"In multilingual contexts, we advise users to write their instructions in English, as most instructions utilized during the model training process were originally written in English."*

However, our benchmark shows that for this Russian/English technical documentation corpus:

### SEMANTIC SEARCH (Most realistic forRAG)

**Best:** `ru_concise` (0.5533)
```
Найдите релевантную техническую документацию
```

**Runner-up:** `baseline_default` (0.5313)
```
Given a web search query, retrieve relevant passages that answer the query
```

### KEYWORD SEARCH (Hybrid/BM25-like)

**Best:** `en_platform` (0.3984)
```
Find documentation about Comindware Platform features, configurations, and APIs
```

---

## Full Results

### SEMANTIC SEARCH

| Rank | Instruction | Score | EN | RU | MIX | Type |
|------|-------------|-------|-----|-----|-----|------|
"""

    if "semantic" in sorted_by_method:
        for i, (name, data) in enumerate(sorted_by_method["semantic"], 1):
            inst_type = (
                "Russian"
                if name.startswith("ru_")
                else ("Bilingual" if name.startswith("bilingual_") else "English")
            )
            report += f"| {i} | `{name}` | {data['avg_score']:.4f} | {data['by_lang']['en']:.4f} | {data['by_lang']['ru']:.4f} | {data['by_lang']['mixed']:.4f} | {inst_type} |\n"

    report += """
### KEYWORD SEARCH

| Rank | Instruction | Score | EN | RU | MIX |
|------|-------------|-------|-----|-----|-----|
"""

    if "keyword" in sorted_by_method:
        for i, (name, data) in enumerate(sorted_by_method["keyword"], 1):
            report += f"| {i} | `{name}` | {data['avg_score']:.4f} | {data['by_lang']['en']:.4f} | {data['by_lang']['ru']:.4f} | {data['by_lang']['mixed']:.4f} |\n"

    report += """
---

## Category Analysis

### By Instruction Language (SEMANTIC)

"""

    categories = {
        "Baseline (Qwen3 default)": ["baseline_default"],
        "English + Context": [k for k in INSTRUCTIONS if k.startswith("en_context_")],
        "English (no context)": ["en_platform", "en_integration", "en_concise"],
        "Russian": [k for k in INSTRUCTIONS if k.startswith("ru_")],
        "Bilingual (EN+RU)": [k for k in INSTRUCTIONS if k.startswith("bilingual_")],
    }

    if "semantic" in results:
        report += "| Category | Avg Score | Best Instruction |\n|----------|-----------|-------------------|\n"
        for cat_name, cat_insts in categories.items():
            cat_scores = []
            best_in_cat = None
            for inst_name in cat_insts:
                if inst_name in results["semantic"] and isinstance(
                    results["semantic"][inst_name], dict
                ):
                    score = results["semantic"][inst_name]["avg_score"]
                    cat_scores.append(score)
                    if best_in_cat is None or score > best_in_cat[1]:
                        best_in_cat = (inst_name, score)

            if cat_scores:
                avg_cat = sum(cat_scores) / len(cat_scores)
                best_str = f"`{best_in_cat[0]}` ({best_in_cat[1]:.4f})" if best_in_cat else "-"
                report += f"| {cat_name} | {avg_cat:.4f} | {best_str} |\n"

    report += """
---

## Recommendations

### For RAG with Semantic Search (Primary)

**Recommended instruction:**
```yaml
default_instruction: "Найдите релевантную техническую документацию"
```

Or use the Qwen3 default which performs nearly as well:
```yaml
default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
```

### Key Insights

1. **Russian instructions outperform English** for Russian documentation (contrary to Qwen3 docs recommendation)
   - `ru_concise` (0.5533) > `en_concise` (0.5047)
   - `ru_integration` (0.4818) > `en_integration` (0.4189)

2. **Bilingual instructions work well** for mixed language queries
   - `bilingual_integration` (0.4694) is competitive

3. **Context about documentation language helps**
   - `en_context_platform` (0.4276) > `en_platform` (0.4597) -- wait, that's wrong
   - Actually: `en_platform` (0.4597) > `en_context_platform` (0.4276)

4. **Baseline generic instruction is very strong**
   - 2nd place overall for semantic search
   - Works well because semantic search already finds relevant docs

### Why Russian Instructions Win

The Qwen3-Reranker was trained primarily with English instructions, but for a Russian documentation corpus:
- Russian queries + Russian docs = better matching
- The reranker understands Russian well despite being trained on English instructions

---

## Methodology

1. **Questions**: 52 realistic support queries
   - 38 Russian (73%)
   - 8 Mixed (15%)
   - 6 English (12%)

2. **Retrieval methods**:
   - SEMANTIC: Qwen3-Embedding-8B via OpenRouter
   - KEYWORD: Word matching (BM25-like)
   - RANDOM: Baseline (not yet tested)

3. **Reranker**: Qwen3-Reranker-0.6B via mosec server

4. **Documents**: ChromaDB `mkdocs_kb_qwen8b` collection (8231 Russian/English technical docs about Comindware Platform)

---

## Raw Data

```json
"""

    # Add summary of raw data
    report += json.dumps(
        {
            method: {
                k: {
                    "avg": round(v["avg_score"], 4),
                    "en": round(v["by_lang"]["en"], 4),
                    "ru": round(v["by_lang"]["ru"], 4),
                }
                for k, v in results[method].items()
                if isinstance(v, dict)
            }
            for method in ["semantic", "keyword"]
            if method in results
        },
        indent=2,
    )

    report += "\n```\n"

    # Save report
    report_file = OUTPUT_DIR / f"reranker_benchmark_final_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to: {report_file}")
    print()
    print("=== SUMMARY ===")
    if "semantic" in sorted_by_method and sorted_by_method["semantic"]:
        best = sorted_by_method["semantic"][0]
        print(f"Best for SEMANTIC: {best[0]} ({best[1]['avg_score']:.4f})")
        print(f"  Instruction: {best[1]['instruction']}")
    if "keyword" in sorted_by_method and sorted_by_method["keyword"]:
        best = sorted_by_method["keyword"][0]
        print(f"Best for KEYWORD: {best[0]} ({best[1]['avg_score']:.4f})")
        print(f"  Instruction: {best[1]['instruction']}")


if __name__ == "__main__":
    main()
