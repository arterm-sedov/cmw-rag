"""Comprehensive Reranker Instruction Benchmark.

This script:
1. Generates 100+ synthetic questions from ChromaDB
2. Tests English and Russian instructions on Qwen3-Reranker
3. Calculates MRR, NDCG@k, Precision@k, Recall@k
4. Saves test dataset for reuse

Key insight from Qwen3-Reranker docs:
"In multilingual contexts, we advise users to write their instructions in English,
as most instructions utilized during the model training process were originally written in English."
"""

import os
import sys
import json
import random
import math
from pathlib import Path
from datetime import datetime
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from chromadb import HttpClient
from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter

# =============================================================================
# INSTRUCTIONS TO TEST
# =============================================================================
# Qwen3-Reranker was trained with English instructions - use English even for non-English content

INSTRUCTIONS = {
    # Baseline
    "default_web_search": "Given a web search query, retrieve relevant passages that answer the query",
    # English instructions for Russian/English technical corpus
    "tech_docs_en": "Find technical documentation and code examples that answer the question",
    "platform_specific_en": "Find documentation about Comindware Platform features, configurations, and APIs",
    "code_aware_en": "Retrieve documents with code examples, configurations, and technical explanations",
    "bilingual_en": "Find relevant documentation in Russian or English, including code snippets",
    "qa_focused_en": "Find answers to technical questions about enterprise platform features",
    # Comprehensive English instructions
    "comprehensive_en": "Retrieve technical documentation about Comindware Platform: guides, code examples, configurations, and API references",
    # Task-focused English instructions
    "integration_focused": "Find integration guides, API documentation, and configuration examples",
    "task_oriented": "Find step-by-step guides, tutorials, and how-to documentation",
    "troubleshooting": "Find troubleshooting guides, error solutions, and problem resolutions",
    # Russian instructions (testing Qwen3's multilingual capability)
    "tech_docs_ru": "Найдите техническую документацию и примеры кода",
    "platform_ru": "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации и API",
    "bilingual_ru": "Найдите релевантную документацию на русском или английском языке",
    # Short/concise instructions
    "concise_en": "Find relevant technical documentation",
    "ultra_short": "Retrieve documents",
}

# =============================================================================
# QWEN3 FORMATTING
# =============================================================================
QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

# =============================================================================
# CHROMADB DOCUMENT SAMPLING
# =============================================================================


def get_diverse_documents(collection, limit=500):
    """Get diverse sample of documents from different kbIds."""
    docs = collection.get(limit=limit, include=["documents", "metadatas"])

    samples = []
    kb_ids_seen = set()

    for doc, meta in zip(docs["documents"], docs["metadatas"]):
        kbId = meta.get("kbId", "unknown")
        source = meta.get("source_file", "unknown")

        # Track diversity
        if kbId not in kb_ids_seen:
            kb_ids_seen.add(kbId)

        # Skip very short documents
        if len(doc) < 100:
            continue

        samples.append(
            {
                "content": doc,
                "kbId": kbId,
                "source": source,
                "has_code": "```" in doc
                or "code" in doc.lower()
                or "def " in doc
                or "class " in doc
                or "{" in doc,
            }
        )

    print(f"Sampled {len(samples)} documents from {len(kb_ids_seen)} different kbIds")
    return samples


def generate_quality_questions(samples, num_questions=150):
    """Generate high-quality synthetic questions."""
    questions = []

    # Question templates - varied types
    templates = [
        # How-to questions
        ("How do I {action}?", "action"),
        ("Как {action}?", "action"),
        ("How to {action}?", "action"),
        # What isquestions
        ("What is {concept}?", "concept"),
        ("Что такое {concept}?", "concept"),
        ("What are {concept}?", "concept"),
        # Configuration questions
        ("How to configure {feature}?", "feature"),
        ("Как настроить {feature}?", "feature"),
        ("Configuration of {feature}", "feature"),
        # Integration questions
        ("How to integrate {system}?", "system"),
        ("Как интегрировать {system}?", "system"),
        # API questions
        ("API for {feature}", "feature"),
        ("{feature} API documentation", "feature"),
        # Troubleshooting
        ("How to fix {problem}?", "problem"),
        ("Ошибка {problem}", "problem"),
        # Code examples
        ("Code example for {feature}", "feature"),
        ("Пример кода для {feature}", "feature"),
    ]

    # Extract key terms from documents
    key_terms_russian = set()
    key_terms_english = set()
    code_terms = set()

    for sample in samples[:100]:
        content = sample["content"]
        words = content.split()

        for word in words:
            # Russian terms
            if any(c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in word):
                if len(word) > 4 and word.isalpha():
                    key_terms_russian.add(word.lower())
            # English terms
            elif len(word) > 4 and word.isalpha():
                key_terms_english.add(word.lower())

        # Code-related terms
        if "```" in content or "def " in content:
            code_terms.add(sample["source"].split("/")[-1].replace(".md", ""))

    key_terms_russian = list(key_terms_russian)[:50]
    key_terms_english = list(key_terms_english)[:50]
    code_terms = list(code_terms)[:30]

    # Generate questions
    for i in range(num_questions):
        template, key = random.choice(templates)

        # Choose appropriate term based on question language
        if any(c in template for c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"):
            # Russian question - use Russian term
            term = random.choice(key_terms_russian) if key_terms_russian else "элемент"
        else:
            # English question - can use either
            term = (
                random.choice(key_terms_english + key_terms_russian[:25])
                if (key_terms_english or key_terms_russian)
                else "element"
            )

        try:
            question = template.format(**{key: term})
        except KeyError:
            question = template.format(action=term, feature=term)

        # Find relevant documents (ground truth)
        relevant_docs = []
        for sample in samples:
            # Check if term appears in document
            if term.lower() in sample["content"].lower():
                # Calculate relevance score based on:
                # - Term frequency
                # - Position (earlier = more relevant)
                # - Document has code
                term_count = sample["content"].lower().count(term.lower())
                position = sample["content"].lower().find(term.lower())
                has_code_bonus = 0.1 if sample["has_code"] else 0

                relevance = (term_count * 0.3) + (1 / (position + 1) * 0.5) + has_code_bonus

                relevant_docs.append(
                    {
                        "content": sample["content"][:1000],  # Truncate for API
                        "source": sample["source"],
                        "relevance": min(relevance, 1.0),  # Cap at 1.0
                    }
                )

                if len(relevant_docs) >= 10:
                    break

        # Sort by relevance and take top 5 as ground truth
        relevant_docs.sort(key=lambda x: x["relevance"], reverse=True)
        ground_truth = relevant_docs[:5]

        if len(ground_truth) >= 2:  # Need at least 2 docs for meaningful ranking
            questions.append(
                {
                    "question": question,
                    "documents": ground_truth,
                    "term": term,
                    "expected_top_k": [i for i in range(min(3, len(ground_truth)))],
                }
            )

    print(f"Generated {len(questions)} quality questions")
    return questions


# =============================================================================
# RERANKING METRICS
# =============================================================================


def calculate_mrr(results, ground_truth_indices):
    """Mean Reciprocal Rank - position of first relevant item."""
    for i, (doc, score) in enumerate(results):
        if i in ground_truth_indices:
            return 1.0 / (i + 1)
    return 0.0


def calculate_precision_at_k(results, ground_truth_indices, k):
    """Precision@k - fraction of relevant items in top k."""
    top_k = results[:k]
    relevant_in_top_k = sum(1 for i, (doc, score) in enumerate(top_k) if i in ground_truth_indices)
    return relevant_in_top_k / min(k, len(top_k)) if top_k else 0.0


def calculate_recall_at_k(results, ground_truth_indices, k):
    """Recall@k - fraction of relevant items found in top k."""
    top_k_indices = set(range(min(k, len(results))))
    relevant_found = len(ground_truth_indices & top_k_indices)
    return relevant_found / len(ground_truth_indices) if ground_truth_indices else 0.0


def calculate_ndcg_at_k(results, ground_truth_indices, k):
    """Normalized Discounted Cumulative Gain @ k."""
    dcg = 0.0
    for i, (doc, score) in enumerate(results[:k]):
        if i in ground_truth_indices:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(ground_truth_indices))))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def calculate_f1_at_k(precision, recall):
    """F1@k - harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# =============================================================================
# BENCHMARKRUNNER
# =============================================================================


def run_benchmark(adapter, questions, instruction_name, instruction):
    """Run benchmark for a single instruction."""
    results = []
    metrics_sum = {
        "mrr": 0,
        "precision@1": 0,
        "precision@3": 0,
        "precision@5": 0,
        "recall@3": 0,
        "recall@5": 0,
        "ndcg@3": 0,
        "ndcg@5": 0,
        "avg_score": 0,
    }

    for q in questions:
        candidates = [(doc["content"], doc["relevance"]) for doc in q["documents"]]

        try:
            ranked = adapter.rerank(
                q["question"],
                candidates,
                top_k=len(candidates),
                instruction=instruction,
            )

            # Ground truth: top docs by original relevance
            ground_truth = set(q["expected_top_k"])

            # Calculate metrics
            mrr = calculate_mrr(ranked, ground_truth)
            p1 = calculate_precision_at_k(ranked, ground_truth, 1)
            p3 = calculate_precision_at_k(ranked, ground_truth, 3)
            p5 = calculate_precision_at_k(ranked, ground_truth, 5)
            r3 = calculate_recall_at_k(ranked, ground_truth, 3)
            r5 = calculate_recall_at_k(ranked, ground_truth, 5)
            ndcg3 = calculate_ndcg_at_k(ranked, ground_truth, 3)
            ndcg5 = calculate_ndcg_at_k(ranked, ground_truth, 5)
            avg_score = sum(r[1] for r in ranked) / len(ranked) if ranked else 0

            results.append(
                {
                    "question": q["question"],
                    "mrr": mrr,
                    "precision@1": p1,
                    "precision@3": p3,
                    "precision@5": p5,
                    "recall@3": r3,
                    "recall@5": r5,
                    "ndcg@3": ndcg3,
                    "ndcg@5": ndcg5,
                    "avg_score": avg_score,
                }
            )

            metrics_sum["mrr"] += mrr
            metrics_sum["precision@1"] += p1
            metrics_sum["precision@3"] += p3
            metrics_sum["precision@5"] += p5
            metrics_sum["recall@3"] += r3
            metrics_sum["recall@5"] += r5
            metrics_sum["ndcg@3"] += ndcg3
            metrics_sum["ndcg@5"] += ndcg5
            metrics_sum["avg_score"] += avg_score

        except Exception as e:
            print(f"Error processing question: {e}")

    n = len(results)
    if n == 0:
        return {}, []

    avg_metrics = {k: v / n for k, v in metrics_sum.items()}

    return avg_metrics, results


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 80)
    print("Qwen3 Reranker Comprehensive Instruction Benchmark")
    print("=" * 80)
    print()
    print("Key Insight from Qwen3-Reranker docs:")
    print("'In multilingual contexts, we advise users to write their instructions")
    print("in English, as most instructions used during training were in English.'")
    print()

    # Connect to ChromaDB
    print("Connecting to ChromaDB...")
    client = HttpClient(host="localhost", port=8000)
    collection = client.get_collection("mkdocs_kb")
    print(f"Collection: {collection.name}, Count: {collection.count}")
    print()

    # Get diverse documents
    print("Sampling diverse documents...")
    samples = get_diverse_documents(collection, limit=500)
    print()

    # Generate questions
    print("Generating quality questions...")
    questions = generate_quality_questions(samples, num_questions=150)

    if len(questions) < 50:
        print(f"ERROR: Only generated {len(questions)} questions!")
        return

    print(f"\nSample questions:")
    for q in questions[:10]:
        print(f"- {q['question']}")
    print()

    # Create reranker adapter
    config = ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/score",
        reranker_type="llm_reranker",
        default_instruction=INSTRUCTIONS["default_web_search"],
        formatting=QWEN3_FORMATTING,
    )
    adapter = RerankerAdapter(config)

    # Test all instructions
    print("=" * 80)
    print("Testing instructions...")
    print("=" * 80)
    print()

    all_results = {}

    for name, instruction in INSTRUCTIONS.items():
        print(f"Testing: {name}")
        print(f"  Instruction: {instruction[:70]}...")

        avg_metrics, detail_results = run_benchmark(adapter, questions, name, instruction)

        if avg_metrics:
            all_results[name] = {
                "instruction": instruction,
                "metrics": avg_metrics,
                "num_questions": len(detail_results),
            }

            print(f"  MRR: {avg_metrics['mrr']:.4f}")
            print(f"P@3: {avg_metrics['precision@3']:.4f}, P@5: {avg_metrics['precision@5']:.4f}")
            print(f"R@3: {avg_metrics['recall@3']:.4f}, R@5: {avg_metrics['recall@5']:.4f}")
            print(f"  NDCG@3: {avg_metrics['ndcg@3']:.4f}, NDCG@5: {avg_metrics['ndcg@5']:.4f}")
            print(f"  Avg Score: {avg_metrics['avg_score']:.4f}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY (sorted by NDCG@5)")
    print("=" * 80)
    print()

    sorted_results = sorted(
        all_results.items(), key=lambda x: x[1]["metrics"]["ndcg@5"], reverse=True
    )

    print(
        f"{'Instruction':<40} {'MRR':>6} {'P@3':>6} {'P@5':>6} {'R@5':>6} {'NDCG@5':>8} {'Avg':>6}"
    )
    print("-" * 80)

    for name, data in sorted_results:
        m = data["metrics"]
        print(
            f"{name:<40} {m['mrr']:>6.3f} {m['precision@3']:>6.3f} {m['precision@5']:>6.3f} {m['recall@5']:>6.3f} {m['ndcg@5']:>8.3f} {m['avg_score']:>6.3f}"
        )

    # Save results
    output_dir = Path(__file__).parent.parent / "docs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save test dataset
    dataset_file = output_dir / f"reranker_test_dataset_{timestamp}.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "questions": [
                    {
                        "question": q["question"],
                        "documents": [
                            {
                                "content": d["content"][:500],
                                "source": d["source"],
                                "relevance": d["relevance"],
                            }
                            for d in q["documents"]
                        ],
                        "term": q["term"],
                    }
                    for q in questions
                ],
                "metadata": {
                    "num_questions": len(questions),
                    "num_docs_per_question": len(questions[0]["documents"]) if questions else 0,
                    "created": timestamp,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nTest dataset saved to: {dataset_file}")

    # Save benchmark results
    results_file = output_dir / f"reranker_comprehensive_benchmark_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "all_results": all_results,
                "questions_summary": {
                    "total": len(questions),
                    "sample": [q["question"] for q in questions[:10]],
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Results saved to: {results_file}")

    # Print recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    best_ndcg = sorted_results[0]
    best_mrr = max(all_results.items(), key=lambda x: x[1]["metrics"]["mrr"])
    best_p5 = max(all_results.items(), key=lambda x: x[1]["metrics"]["precision@5"])

    print(f"Best NDCG@5: {best_ndcg[0]}")
    print(f"  {best_ndcg[1]['instruction']}")
    print()
    print(f"Best MRR: {best_mrr[0]}")
    print(f"  {best_mrr[1]['instruction']}")
    print()
    print(f"Best Precision@5: {best_p5[0]}")
    print(f"  {best_p5[1]['instruction']}")
    print()

    # Check English vs Russian
    en_results = [
        (n, d)
        for n, d in sorted_results
        if n.endswith(("_en", "_focused", "_oriented", "_short")) or n == "default_web_search"
    ]
    ru_results = [(n, d) for n, d in sorted_results if n.endswith("_ru")]

    if en_results and ru_results:
        avg_en_ndcg = sum(d["metrics"]["ndcg@5"] for _, d in en_results) / len(en_results)
        avg_ru_ndcg = sum(d["metrics"]["ndcg@5"] for _, d in ru_results) / len(ru_results)

        print("English vs Russian Instructions:")
        print(f"  English avg NDCG@5: {avg_en_ndcg:.4f}")
        print(f"  Russian avg NDCG@5: {avg_ru_ndcg:.4f}")
        print(f"  Difference: {avg_en_ndcg - avg_ru_ndcg:+.4f}")
        print()

        if avg_en_ndcg > avg_ru_ndcg:
            print("RECOMMENDATION: Use ENGLISH instructions (as per Qwen3 documentation)")
        else:
            print("NOTE: Russian instructions performed better - consider testing more")


if __name__ == "__main__":
    main()
