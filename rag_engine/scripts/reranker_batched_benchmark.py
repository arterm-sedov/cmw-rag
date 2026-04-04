"""Batched Reranker Benchmark with incremental result saving.

Processes questions in batches, saves after each batch, handles interruptions gracefully.
"""

import sys
import json
import random
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from chromadb import HttpClient
from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter

# Configuration
BATCH_SIZE = 10  # Questions per batch
TOTAL_QUESTIONS = 100  # Target total questions
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "analysis"

QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

INSTRUCTIONS = {
    # ENGLISH (Qwen3 trained with English)
    "default_web": "Given a web search query, retrieve relevant passages that answer the query",
    "tech_docs": "Find technical documentation and code examples that answer the question",
    "platform": "Find documentation about Comindware Platform features, configurations, and APIs",
    "code_snippets": "Retrieve documents with code examples, configurations, and technical explanations",
    "detailed": "Retrieve Comindware Platform documentation: integration guides, API references, configuration examples",
    # RUSSIAN
    "tech_docs_ru": "Найдите техническую документацию и примеры кода",
    "platform_ru": "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации",
    # BILINGUAL
    "bilingual_full": "Find relevant documentation (Russian: Найдите релевантную документацию) in Russian or English",
    "bilingual_natural": "Find technical documentation - документация, примеры кода, API на русском и английском",
}


def load_or_create_state(timestamp):
    """Load existing state or create new."""
    state_file = OUTPUT_DIR / f"benchmark_state_{timestamp}.json"

    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)

    return {
        "results": {name: {"scores": [], "count": 0} for name in INSTRUCTIONS},
        "questions_processed": 0,
        "questions": [],
        "timestamp": timestamp,
    }


def save_state(state, timestamp):
    """Save state to disk."""
    state_file = OUTPUT_DIR / f"benchmark_state_{timestamp}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def generate_questions(samples, num_questions):
    """Generate diverse questions from samples."""
    questions = []

    # Extract terms
    terms = set()
    for s in samples[:150]:
        for w in s["content"].split():
            if len(w) > 5 and w.isalpha():
                terms.add(w.lower())
    terms = list(terms)[:200]

    # Question templates
    templates = [
        ("How to configure {t}?", "en"),
        ("What is {t}?", "en"),
        ("API documentation for {t}", "en"),
        ("Как настроить {t}?", "ru"),
        ("Что такое {t}?", "ru"),
        ("Пример кода для {t}", "ru"),
    ]

    for term in random.sample(terms, min(num_questions, len(terms))):
        template, lang = random.choice(templates)
        q = template.format(t=term)

        relevant = [s for s in samples if term.lower() in s["content"].lower()][:5]
        if len(relevant) >= 2:
            questions.append(
                {
                    "question": q,
                    "docs": [
                        {"content": r["content"][:600], "source": r["source"]} for r in relevant
                    ],
                    "term": term,
                    "lang": lang,
                }
            )

    return questions


def process_batch(adapter, state, batch_questions, batch_num):
    """Process a batch of questions."""
    print(f"\nBatch {batch_num}: Processing {len(batch_questions)} questions...")

    batch_results = {name: [] for name in INSTRUCTIONS}

    for i, q in enumerate(batch_questions):
        print(f"  Q{i + 1}/{len(batch_questions)}: {q['question'][:50]}...", end="", flush=True)

        candidates = [(d["content"], 0.0) for d in q["docs"]]

        for name, instruction in INSTRUCTIONS.items():
            try:
                ranked = adapter.rerank(
                    q["question"], candidates, top_k=len(candidates), instruction=instruction
                )
                scores = [r[1] for r in ranked]
                batch_results[name].extend(scores)
            except Exception as e:
                print(f" [{name}: error]", end="", flush=True)

        print(" ✓", flush=True)

    # Update state
    for name, scores in batch_results.items():
        state["results"][name]["scores"].extend(scores)
        state["results"][name]["count"] += len(batch_questions)

    state["questions_processed"] += len(batch_questions)

    # Save after each batch
    save_state(state, state["timestamp"])

    # Print batch summary
    print(f"\n  Batch {batch_num} results:")
    for name in INSTRUCTIONS:
        if batch_results[name]:
            avg = sum(batch_results[name]) / len(batch_results[name])
            print(f"    {name:<20}: {avg:.4f}")

    return state


def compute_metrics(state):
    """Compute final metrics."""
    metrics = {}

    for name, data in state["results"].items():
        scores = data["scores"]
        if scores:
            avg = sum(scores) / len(scores)
            variance = sum((s - avg) ** 2 for s in scores) / len(scores)
            metrics[name] = {
                "avg_score": avg,
                "variance": variance,
                "count": len(scores),
                "min": min(scores),
                "max": max(scores),
            }

    return metrics


def print_final_report(state):
    """Print final report."""
    metrics = compute_metrics(state)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    print(f"\n{'Instruction':<25} {'Avg':>8} {'Var':>8} {'Min':>8} {'Max':>8} {'Count':>8}")
    print("-" * 80)

    for name, m in sorted_metrics:
        print(
            f"{name:<25} {m['avg_score']:>8.4f} {m['variance']:>8.4f} {m['min']:>8.4f} {m['max']:>8.4f} {m['count']:>8}"
        )

    # Category analysis
    en_scores = [
        m["avg_score"]
        for n, m in metrics.items()
        if not n.endswith("_ru") and not n.startswith("bilingual")
    ]
    ru_scores = [m["avg_score"] for n, m in metrics.items() if n.endswith("_ru")]
    bi_scores = [m["avg_score"] for n, m in metrics.items() if n.startswith("bilingual")]

    print("\n" + "=" * 80)
    print("CATEGORY ANALYSIS")
    print("=" * 80)

    if en_scores:
        print(f"English instructions avg:   {sum(en_scores) / len(en_scores):.4f}")
    if ru_scores:
        print(f"Russian instructions avg:    {sum(ru_scores) / len(ru_scores):.4f}")
    if bi_scores:
        print(f"Bilingual instructions avg:  {sum(bi_scores) / len(bi_scores):.4f}")

    # Best
    best = sorted_metrics[0]
    print("\n" + "=" * 80)
    print("BEST INSTRUCTION")
    print("=" * 80)
    print(f"Name: {best[0]}")
    print(f"Score: {best[1]['avg_score']:.4f}")
    print(f"Instruction: {INSTRUCTIONS[best[0]]}")

    return metrics


def main():
    print("=" * 80)
    print("BATCHED RERANKER INSTRUCTION BENCHMARK")
    print("=" * 80)
    print(f"Batch size: {BATCH_SIZE} questions")
    print(f"Total questions: {TOTAL_QUESTIONS}")
    print(f"Instructions: {len(INSTRUCTIONS)}")
    print()

    # Initialize
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    state = load_or_create_state(timestamp)

    # Connect to ChromaDB
    print("Connecting to ChromaDB...")
    client = HttpClient(host="localhost", port=8000)
    collection = client.get_collection("mkdocs_kb")
    print(f"Collection: {collection.name}")

    # Get samples
    docs = collection.get(limit=300, include=["documents", "metadatas"])
    samples = [
        {"content": d, "source": m.get("source_file", "?")}
        for d, m in zip(docs["documents"], docs["metadatas"])
    ]
    print(f"Sampled {len(samples)} documents")

    # Generate questions if needed
    if not state["questions"]:
        print(f"Generating {TOTAL_QUESTIONS} questions...")
        state["questions"] = generate_questions(samples, TOTAL_QUESTIONS)
        print(f"Generated {len(state['questions'])} questions")
        save_state(state, timestamp)

    questions = state["questions"]
    processed = state["questions_processed"]
    remaining = len(questions) - processed

    print(f"Questions: {len(questions)} total, {processed} processed, {remaining} remaining")
    print()

    # Create adapter
    config = ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/score",
        reranker_type="llm_reranker",
        default_instruction=INSTRUCTIONS["default_web"],
        formatting=QWEN3_FORMATTING,
    )
    adapter = RerankerAdapter(config)

    # Process remainingquestions in batches
    batch_num = processed // BATCH_SIZE + 1

    while processed < len(questions):
        batch_questions = questions[processed : processed + BATCH_SIZE]

        try:
            state = process_batch(adapter, state, batch_questions, batch_num)
            processed = state["questions_processed"]
            batch_num += 1

            # Small pause between batches
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\nInterrupted! Progress saved to state file.")
            print(f"Processed {processed}/{len(questions)} questions")
            returnstate
        except Exception as e:
            print(f"\n\nError in batch {batch_num}: {e}")
            print("Progress saved. You can resume by running again.")
            return state

    # Final report
    metrics = print_final_report(state)

    # Save final results
    results_file = OUTPUT_DIR / f"reranker_benchmark_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "instructions": INSTRUCTIONS,
                "total_questions": len(questions),
                "timestamp": timestamp,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nResults saved to: {results_file}")

    # Recommendation
    best_name = max(metrics.items(), key=lambda x: x[1]["avg_score"])[0]
    print("\n" + "=" * 80)
    print("RECOMMENDATION FOR models.yaml")
    print("=" * 80)
    print(f'\ndefault_instruction: "{INSTRUCTIONS[best_name]}"')

    if best_name.endswith("_ru"):
        print("\nNOTE: Russian instruction performed best.")
        print("However, Qwen3 docs recommend English instructions for multilingual content.")
    elif best_name.startswith("bilingual"):
        print("\nNOTE: Bilingual instruction performed well.")
    else:
        print("\nRECOMMENDATION: Use English instruction (per Qwen3 documentation)")


if __name__ == "__main__":
    main()
