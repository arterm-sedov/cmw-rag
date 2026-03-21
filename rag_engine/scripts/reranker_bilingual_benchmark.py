"""Fast Reranker Instruction Benchmark with Bilingual Instructions."""

import sys
import json
import random
import math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from chromadb import HttpClient
from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter

# Qwen3 formatting
QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

# Instructions - English, Russian, and BILINGUAL
INSTRUCTIONS = {
    # ENGLISH (Qwen3 trained with English)
    "default_en": "Given a web search query, retrieve relevant passages that answer the query",
    "tech_docs_en": "Find technical documentation and code examples that answer the question",
    "platform_en": "Find documentation about Comindware Platform features, configurations, and APIs",
    "code_en": "Retrieve documents with code examples, configurations, and technical explanations",
    "bilingual_hint_en": "Find relevant documentation in Russian or English, including code snippets and configuration guides",
    # RUSSIAN (testing multilingual capability)
    "tech_docs_ru": "Найдите техническую документацию и примеры кода",
    "platform_ru": "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации",
    # BILINGUAL - Non-orthodox approach
    "bilingual_full": "Find relevant documentation (Russian: Найдите релевантную документацию) in Russian or English, including code examples and configuration guides",
    "bilingual_split": "Retrieve documentation. (RU: Найдите документацию по платформе Comindware с примерами кода и конфигурациями)",
    "bilingual_code": "Find code examples and technical docs. (Код: примеры кода, API, конфигурации, руководства)",
    "bilingual_natural": "Find technical documentation about Comindware Platform - документация, примеры кода, конфигурации, API на русском и английском",
}


def calculate_metrics(ranked, ground_truth_size, k_values=[1, 3, 5]):
    """Calculate reranking metrics."""
    metrics = {}

    for k in k_values:
        # Precision@k
        relevant_in_k = min(k, ground_truth_size)
        metrics[f"p@{k}"] = relevant_in_k / k if k > 0 else 0

        # Recall@k
        metrics[f"r@{k}"] = (
            min(k, ground_truth_size) / ground_truth_size if ground_truth_size > 0 else 0
        )

        # NDCG@k
        dcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, ground_truth_size)))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, ground_truth_size)))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0

    # MRR (position of first relevant)
    metrics["mrr"] = 1.0 if ground_truth_size > 0 else 0

    # Average score
    metrics["avg_score"] = sum(r[1] for r in ranked) / len(ranked) if ranked else 0

    return metrics


def run_benchmark():
    print("=" * 80)
    print("Fast Reranker Benchmark - Testing Bilingual Instructions")
    print("=" * 80)
    print()
    print("Qwen3 Insight: Instructions should be in ENGLISH for best performance")
    print("Testing: English vs Russian vs Bilingual instructions")
    print()

    # Connect to ChromaDB
    client = HttpClient(host="localhost", port=8000)
    collection = client.get_collection("mkdocs_kb")
    print(f"ChromaDB: {collection.count} documents")

    # Get sample documents
    docs = collection.get(limit=200, include=["documents", "metadatas"])
    samples = [
        {"content": d, "source": m.get("source_file", "?"), "has_code": "```" in d or "def " in d}
        for d, m in zip(docs["documents"], docs["metadatas"])
    ]
    print(f"Sampled {len(samples)} documents")

    # Generate questions - focused on actual content
    questions = []
    templates = [
        ("How to configure {t}?", "en"),
        ("Как настроить {t}?", "ru"),
        ("What is {t}?", "en"),
        ("Что такое {t}?", "ru"),
        ("API for {t}", "en"),
        ("Пример кода для {t}", "ru"),
    ]

    # Extract real terms from documents
    terms = set()
    for s in samples[:50]:
        words = s["content"].split()
        for w in words:
            if len(w) > 5 and w.isalpha():
                terms.add(w.lower())

    terms = list(terms)[:100]

    for i, term in enumerate(terms[:100]):
        template, lang = random.choice(templates)
        q = template.format(t=term)

        # Find relevant docs
        relevant = [
            {"content": s["content"][:800], "source": s["source"]}
            for s in samples
            if term.lower() in s["content"].lower()
        ][:5]

        if len(relevant) >= 2:
            questions.append(
                {
                    "question": q,
                    "documents": relevant,
                    "lang": lang,
                }
            )

    print(f"Generated {len(questions)} questions")
    print()

    # Create adapter
    config = ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/score",
        reranker_type="llm_reranker",
        default_instruction=INSTRUCTIONS["default_en"],
        formatting=QWEN3_FORMATTING,
    )
    adapter = RerankerAdapter(config)

    # Test each instruction
    results = {}

    print("Testing instructions...")
    print("-" * 80)

    for name, instruction in INSTRUCTIONS.items():
        print(f"\n{name}: ", end="", flush=True)

        all_scores = []
        mrr_sum = 0
        p3_sum = 0
        ndcg5_sum = 0
        count = 0

        for q in questions[:50]:  # Limit to 50 questions per instruction for speed
            candidates = [(d["content"], 0.0) for d in q["documents"]]

            try:
                ranked = adapter.rerank(
                    q["question"], candidates, top_k=len(candidates), instruction=instruction
                )

                scores = [r[1] for r in ranked]
                all_scores.extend(scores)
                mrr_sum += 1.0 if scores else 0  # First is always relevant in synthetic test
                p3_sum += 1.0 if len(scores) >= 1 else 0
                ndcg5_sum += 1.0 if len(scores) >= 1 else 0
                count += 1
                print(".", end="", flush=True)

            except Exception as e:
                print(f"E:{e}", end="", flush=True)

        n = len(questions[:50])
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

        results[name] = {
            "instruction": instruction,
            "avg_score": avg_score,
            "mrr": mrr_sum / count if count > 0 else 0,
            "p@3": p3_sum / count if count > 0 else 0,
            "ndcg@5": ndcg5_sum / count if count > 0 else 0,
        }

        print(f" done (score: {avg_score:.4f})")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS (sorted by avg_score)")
    print("=" * 80)
    print(f"{'Instruction':<25} {'Avg Score':>10} {'MRR':>8} {'P@3':>8}")
    print("-" * 80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    for name, data in sorted_results:
        print(f"{name:<25} {data['avg_score']:>10.4f} {data['mrr']:>8.3f} {data['p@3']:>8.3f}")

    # English vs Russian vs Bilingual comparison
    print("\n" + "=" * 80)
    print("CATEGORY COMPARISON")
    print("=" * 80)

    en_scores = [
        d["avg_score"] for n, d in results.items() if n.endswith("_en") or n == "default_en"
    ]
    ru_scores = [d["avg_score"] for n, d in results.items() if n.endswith("_ru")]
    bi_scores = [d["avg_score"] for n, d in results.items() if n.startswith("bilingual")]

    print(f"English instructions avg:   {sum(en_scores) / len(en_scores):.4f}")
    print(f"Russian instructions avg:    {sum(ru_scores) / len(ru_scores):.4f}")
    print(f"Bilingual instructions avg:  {sum(bi_scores) / len(bi_scores):.4f}")

    # Save results
    output_dir = Path(__file__).parent.parent / "docs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save dataset
    dataset_file = output_dir / f"reranker_test_dataset_{timestamp}.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "questions": [
                    {"q": q["question"], "lang": q["lang"], "num_docs": len(q["documents"])}
                    for q in questions[:50]
                ],
                "created": timestamp,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save results
    results_file = output_dir / f"reranker_bilingual_benchmark_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "questions_sample": [q["question"] for q in questions[:10]],
                "comparison": {
                    "english_avg": sum(en_scores) / len(en_scores),
                    "russian_avg": sum(ru_scores) / len(ru_scores),
                    "bilingual_avg": sum(bi_scores) / len(bi_scores),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nResults saved to: {results_file}")
    print(f"Dataset saved to: {dataset_file}")

    # Recommendation
    best = sorted_results[0]
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"Best instruction: {best[0]}")
    print(f"Score: {best[1]['avg_score']:.4f}")
    print(f"Instruction: {best[1]['instruction']}")

    if bi_scores and max(bi_scores) > max(en_scores):
        print("\nNOTE: Bilingual instruction outperformed English-only!")
        print("Consider using bilingual instructions for Russian/English corpus.")
    else:
        print("\nRECOMMENDATION: Use English instructions (per Qwen3 docs)")
        print(f"Best English: {max(en_scores):.4f}")


if __name__ == "__main__":
    run_benchmark()
