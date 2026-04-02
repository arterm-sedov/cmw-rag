"""
Embedding vs Reranker Instruction Investigation

Tests whether embedding and reranker instructions should be:
1. THE SAME (unified approach)
2. TAILORED DIFFERENTLY (optimized separately)

Current cmw-rag:
- Embedding sends raw query text (NO instruction)
- Reranker uses instruction from models.yaml

Qwen3 Recommendation:
- Embedding: Uses `Instruct: {task}\nQuery: {query}` for queries
- Reranker: Uses `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter
from rag_engine.config.settings import settings

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"
DATASET_FILE = sorted(OUTPUT_DIR.glob("20260321-reranker-dataset-complete.json"))[-1]

QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

# Best instructions from previous benchmarks
BEST_INSTRUCTIONS = {
    "ru_informal_concise": "Найди релевантную техническую документацию",
    "ru_concise": "Найдите релевантную техническую документацию",
    "baseline_default": "Given a web search query, retrieve relevant passages that answer the query",
    "en_concise": "Find relevant technical documentation",
    "qwen_like_ru": "Дан поисковый запрос. Найди релевантную техническую документацию, подходящую под запрос",
}

# Embedding instructions (for Qwen3-Embedding)
EMBEDDING_INSTRUCTIONS = {
    "none": None,  # No instruction (current behavior)
    "baseline": "Given a web search query, retrieve relevant passages that answer the query",
    "ru_informal": "Найди релевантную техническую документацию",
    "ru_formal": "Найдите релевантную техническую документацию",
    "ru_context": "Релевантная техническая документация на русском с примерами кода",
}


def get_embedding_with_instruction(text: str, instruction: str | None = None) -> list[float]:
    """Get embedding from OpenRouter with optional instruction."""
    api_key = settings.openrouter_api_key
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    # Format query with instruction if provided (Qwen3-Embedding format)
    if instruction:
        formatted_query = f"Instruct: {instruction}\nQuery: {text}"
    else:
        formatted_query = text

    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": "qwen/qwen3-embedding-8b", "input": formatted_query}

    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def run_embedding_reranker_investigation():
    """Test combinations of embedding and reranker instructions."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"20260321-embedding-reranker-investigation.json"

    print("=" * 70)
    print("EMBEDDING vs RERANKER INSTRUCTION INVESTIGATION")
    print("=" * 70)
    print(f"Embedding variants: {len(EMBEDDING_INSTRUCTIONS)}")
    print(f"Reranker variants: {len(BEST_INSTRUCTIONS)}")
    print()

    # Load dataset
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]
    print(f"Questions: {len(questions)}")

    # Connect to ChromaDB
    from chromadb import HttpClient

    client = HttpClient(host="localhost", port=8000)
    collection = client.get_collection("mkdocs_kb_qwen8b")
    print(f"ChromaDB: {collection.count()} documents")

    # Create reranker
    config = ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/score",
        reranker_type="llm_reranker",
        default_instruction=BEST_INSTRUCTIONS["baseline_default"],
        formatting=QWEN3_FORMATTING,
    )
    adapter = RerankerAdapter(config)

    results = {
        "metadata": {
            "timestamp": timestamp,
            "questions": len(questions),
            "embedding_variants": len(EMBEDDING_INSTRUCTIONS),
            "reranker_variants": len(BEST_INSTRUCTIONS),
        },
        "combinations": {},
    }

    # Test combinations
    combinations_to_test = [
        # (embed_inst, rerank_inst, name)
        (None, "baseline_default", "current_default"),  # Current: no embed inst, default rerank
        ("baseline", "baseline_default", "both_baseline"),  # Both use baseline
        ("ru_informal", "ru_informal_concise", "both_ru_informal"),  # Both use informal Russian
        ("ru_formal", "ru_concise", "both_ru_formal"),  # Both use formal Russian
        (None, "ru_informal_concise", "embed_none_rerank_ru_informal"),  # No embed, RU rerank
        ("ru_informal", "baseline_default", "embed_ru_rerank_en"),  # RU embed, EN rerank
        ("baseline", "ru_informal_concise", "embed_en_rerank_ru"),  # EN embed, RU rerank
        (
            "ru_context",
            "ru_informal_concise",
            "embed_context_rerank_ru",
        ),  # Context embed, RU rerank
    ]

    processed = set()

    for embed_inst, rerank_inst, combo_name in combinations_to_test:
        if combo_name in processed:
            continue

        print(f"\n[{combo_name}]")
        print(f"  Embedding: {embed_inst or 'none'}")
        print(f"  Reranker: {rerank_inst}")

        all_scores = []
        scores_by_lang = {"en": [], "ru": [], "mixed": []}

        for i, q_data in enumerate(dataset["semantic"]):
            q = questions[q_data["q_idx"]]

            # Get embedding with instruction
            embed_inst_text = EMBEDDING_INSTRUCTIONS.get(embed_inst) if embed_inst else None
            try:
                embedding = get_embedding_with_instruction(q["q"], embed_inst_text)
                # Search ChromaDB
                chroma_results = collection.query(
                    query_embeddings=[embedding], n_results=5, include=["documents", "metadatas"]
                )
                if not chroma_results["documents"] or not chroma_results["documents"][0]:
                    continue

                docs = [
                    {"content": d[:1000], "source": m.get("source_file", "?")}
                    for d, m in zip(chroma_results["documents"][0], chroma_results["metadatas"][0])
                ]
            except Exception as e:
                print(f"  Error getting embedding: {e}")
                continue

            # Rerank with instruction
            rerank_inst_text = BEST_INSTRUCTIONS[rerank_inst]
            candidates = [(d["content"], 0.0) for d in docs]
            try:
                ranked = adapter.rerank(
                    q["q"], candidates, top_k=len(candidates), instruction=rerank_inst_text
                )
                scores = [r[1] for r in ranked]
                all_scores.extend(scores)
                scores_by_lang[q["lang"]].extend(scores)
            except Exception as e:
                print(f"  Error reranking: {e}")
                continue

        if all_scores:
            avg = sum(all_scores) / len(all_scores)
            results["combinations"][combo_name] = {
                "embedding": embed_inst or "none",
                "reranker": rerank_inst,
                "avg_score": avg,
                "count": len(all_scores),
                "by_lang": {k: sum(v) / len(v) if v else 0 for k, v in scores_by_lang.items()},
            }
            print(f"  Avg: {avg:.4f}")
            print(
                f"  EN: {results['combinations'][combo_name]['by_lang']['en']:.4f} | "
                f"RU: {results['combinations'][combo_name]['by_lang']['ru']:.4f}"
            )

        processed.add(combo_name)

        # Save after each combination
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Generate report
    generate_investigation_report(results, timestamp)
    print(f"\nReport: {OUTPUT_DIR / f'20260321-embedding-reranker-investigation-report.md'}")


def generate_investigation_report(results: dict, timestamp: str):
    """Generate investigation report."""

    combos = results.get("combinations", {})
    sorted_combos = sorted(combos.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    report = f"""# Embedding vs Reranker Instruction Investigation

**Date:** {timestamp}
**Combinations Tested:** {len(combos)}

---

## Key Question

Should embedding and reranker instructions be:
1. **THE SAME** (unified approach)
2. **TAILORED DIFFERENTLY** (optimized separately)

---

## Findings

### Best Combination

"""

    if sorted_combos:
        best = sorted_combos[0]
        report += f"**{best[0]}**\n"
        report += f"- Embedding: `{best[1]['embedding']}`\n"
        report += f"- Reranker: `{best[1]['reranker']}`\n"
        report += f"- **Score: {best[1]['avg_score']:.4f}**\n"
        report += (
            f"- Russian: {best[1]['by_lang']['ru']:.4f} | English: {best[1]['by_lang']['en']:.4f}\n"
        )

    report += """
### All Combinations

| Rank | Combination | Embedding | Reranker | Score | EN | RU |
|------|-------------|-----------|----------|-------|-----|-----|
"""

    for i, (name, data) in enumerate(sorted_combos, 1):
        report += f"| {i} | `{name}` | {data['embedding']} | {data['reranker']} | {data['avg_score']:.4f} | {data['by_lang']['en']:.4f} | {data['by_lang']['ru']:.4f} |\n"

    report += """
---

## Analysis

### Same vs Different Instructions

"""

    # Compare same vs different
    same_scores = []
    diff_scores = []
    for name, data in sorted_combos:
        if data["embedding"] == "none" or data["embedding"] == data["reranker"]:
            same_scores.append((name, data["avg_score"]))
        else:
            diff_scores.append((name, data["avg_score"]))

    if same_scores and diff_scores:
        avg_same = sum(s for _, s in same_scores) / len(same_scores)
        avg_diff = sum(s for _, s in diff_scores) / len(diff_scores)

        report += f"- **Same/Tailored Instructions:** avg {avg_same:.4f}\n"
        for name, score in same_scores:
            report += f"  - `{name}`: {score:.4f}\n"
        report += f"\n- **Different Instructions:** avg {avg_diff:.4f}\n"
        for name, score in diff_scores:
            report += f"  - `{name}`: {score:.4f}\n"

        if avg_same > avg_diff:
            report += "\n**Conclusion:** Same instructions for embedding and reranker perform **better**.\n"
        else:
            report += "\n**Conclusion:** Different instructions for embedding and reranker perform **better**.\n"

    report += """
---

## Recommendation

```yaml
# For Qwen3-Embedding-8B
default_instruction: "Найди релевантную техническую документацию"

# For Qwen3-Reranker-0.6B
default_instruction: "Найди релевантную техническую документацию"
```

Or consider using the Qwen3 default for broader compatibility:

```yaml
default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
```

---

## Methodology

1. **Embedding:** Qwen3-Embedding-8B via OpenRouter with formatted query
2. **Search:** ChromaDB `mkdocs_kb_qwen8b` collection
3. **Reranker:** Qwen3-Reranker-0.6B via mosec server
4. **Questions:** 52 Russian/English technical support queries
"""

    # Save
    report_file = OUTPUT_DIR / f"20260321-embedding-reranker-investigation-report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    run_embedding_reranker_investigation()
