"""Reranker Instruction Benchmark Script.

This script:
1. Samples documents from ChromaDB
2. Generates synthetic questions
3. Tests different Qwen3 instructions
4. Reports ranking quality metrics
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chromadb import HttpClient
from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter

# Default instruction from models.yaml
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# Alternative instructions to test - incorporating language awareness and code references
INSTRUCTIONS = {
    # Baseline
    "default": DEFAULT_INSTRUCTION,
    # Best from previous run
    "technical_qa": "Find technical documentation and answers to questions about enterprise platform features",
    # Language-aware instructions
    "ru_en_tech": "Найдите техническую документацию на русском или английском языке, отвечающую на вопрос",
    "bilingual_tech": "Retrieve technical documentation in Russian or English that answers the query, including code examples",
    "ru_primacy": "Приоритизируйте русскоязычную документацию и примеры кода для технических вопросов",
    # Code-aware instructions
    "code_examples": "Find documentation with relevant code examples, configurations, and technical explanations",
    "code_ru_en": "Найдите документы с примерами кода, конфигурациями и техническими объяснениями на русском или английском",
    # Combined best practices
    "comprehensive": "Retrieve technical documentation about Comindware Platform with code examples, configurations, and API references in Russian or English",
    "platform_ru": "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации и API на русском или английском",
    # Concise variants
    "concise": "Retrieve documents with answers to the query",
    "concise_ru_en": "Найдите релевантные документы (RU/EN) с ответами на запрос",
}


def get_sample_documents(collection, limit=100):
    """Get random sample of documents from ChromaDB."""
    docs = collection.get(limit=limit, include=["documents", "metadatas"])

    samples = []
    for doc, meta in zip(docs["documents"], docs["metadatas"]):
        samples.append(
            {
                "content": doc,
                "kbId": meta.get("kbId", "unknown"),
                "source": meta.get("source_file", "unknown"),
            }
        )

    return samples


def generate_synthetic_questions(samples, num_questions=50):
    """Generate synthetic questions based on document content."""
    questions = []

    # Technical documentation about Comindware Platform (Russian)
    templates = [
        ("Как {action} в Comindware Platform?", "action"),
        ("Что такое {concept}?", "concept"),
        ("Как настроить {feature}?", "feature"),
        ("Какие есть {type} в системе?", "type"),
        ("Как интегрировать {system}?", "system"),
        ("Что означает {term}?", "term"),
        ("Как создать {object}?", "object"),
        ("Где найти {info}?", "info"),
        ("Как работает {process}?", "process"),
        ("Какие преимущества у {feature}?", "feature"),
    ]

    # Extract key terms from documents
    key_terms = set()
    for sample in samples[:50]:
        content = sample["content"].lower()
        # Extract Russian and English terms
        words = content.split()
        for word in words:
            if len(word) > 4 and word.isalpha():
                if any(c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in word):
                    key_terms.add(word)

    key_terms = list(key_terms)[:100]

    # Generate questions
    for i in range(min(num_questions, len(key_terms))):
        template, key = random.choice(templates)
        term = key_terms[i % len(key_terms)] if key_terms else "элемент"

        question = template.format(**{key: term})

        # Find relevant documents for this question
        relevant_docs = []
        for sample in samples:
            if term.lower() in sample["content"].lower():
                relevant_docs.append(
                    {
                        "content": sample["content"][:500],
                        "source": sample["source"],
                    }
                )
                if len(relevant_docs) >= 5:
                    break

        if relevant_docs:
            questions.append(
                {
                    "question": question,
                    "documents": relevant_docs[:3],
                    "expected_topic": term,
                }
            )

    return questions


def test_instruction(adapter, questions, instruction_name, instruction):
    """Test reranker with a specific instruction."""
    results = []

    for q in questions:
        candidates = [(doc["content"], 0.0) for doc in q["documents"]]

        try:
            ranked = adapter.rerank(
                q["question"],
                candidates,
                top_k=len(candidates),
                instruction=instruction,
            )

            # Check if relevant docs are ranked higher
            scores = [r[1] for r in ranked]

            results.append(
                {
                    "question": q["question"],
                    "scores": scores,
                    "top_doc_source": q["documents"][0]["source"] if q["documents"] else None,
                }
            )
        except Exception as e:
            print(f"Error: {e}")

    return results


def calculate_metrics(results):
    """Calculate ranking quality metrics."""
    if not results:
        return {"avg_score": 0, "score_variance": 0}

    all_scores = []
    for r in results:
        all_scores.extend(r["scores"])

    return {
        "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        "score_variance": sum((s - sum(all_scores) / len(all_scores)) ** 2 for s in all_scores)
        / len(all_scores)
        if all_scores
        else 0,
        "num_questions": len(results),
    }


def main():
    print("=" * 80)
    print("Qwen3 Reranker Instruction Benchmark")
    print("=" * 80)
    print()

    # Connect to ChromaDB
    print("Connecting to ChromaDB...")
    client = HttpClient(host="localhost", port=8000)
    collection = client.get_collection("mkdocs_kb")
    print(f"Collection: {collection.name}, Count: {collection.count()}")
    print()

    # Get sample documents
    print("Sampling documents...")
    samples = get_sample_documents(collection, limit=200)
    print(f"Sampled {len(samples)} documents")
    print()

    # Generate questions
    print("Generating synthetic questions...")
    questions = generate_synthetic_questions(samples, num_questions=50)
    print(f"Generated {len(questions)} questions")

    if not questions:
        print("ERROR: No questions generated!")
        return

    print("\nSample questions:")
    for q in questions[:5]:
        print(f"  - {q['question']}")
    print()

    # Create reranker adapter
    config = ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/score",
        reranker_type="llm_reranker",
        default_instruction=DEFAULT_INSTRUCTION,
        formatting=RerankerFormatting(
            query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
            doc_template="<Document>: {doc}{suffix}",
            prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
            suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
        ),
    )
    adapter = RerankerAdapter(config)

    # Test different instructions
    print("=" * 80)
    print("Testing instructions...")
    print("=" * 80)
    print()

    all_results = {}

    for name, instruction in INSTRUCTIONS.items():
        print(f"Testing: {name}")
        print(f"  Instruction: {instruction[:60]}...")

        results = test_instruction(adapter, questions, name, instruction)
        metrics = calculate_metrics(results)

        all_results[name] = {
            "instruction": instruction,
            "metrics": metrics,
        }

        print(f"  Avg Score: {metrics['avg_score']:.4f}")
        print(f"  Variance: {metrics['score_variance']:.4f}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Sort by average score
    sorted_results = sorted(
        all_results.items(), key=lambda x: x[1]["metrics"]["avg_score"], reverse=True
    )

    for name, data in sorted_results:
        print(
            f"{name:20s} - Avg: {data['metrics']['avg_score']:.4f}, Var: {data['metrics']['score_variance']:.4f}"
        )

    # Save results
    output_dir = Path(__file__).parent.parent / "docs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir
        / f"reranker_instruction_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "questions": questions,
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
