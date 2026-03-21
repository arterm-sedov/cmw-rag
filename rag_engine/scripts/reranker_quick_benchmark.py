"""Quick Reranker Benchmark - 10 questions per instruction."""

import sys
import json
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from chromadb import HttpClient
from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting
from rag_engine.retrieval.reranker import RerankerAdapter

QWEN3_FORMATTING = RerankerFormatting(
    query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
    doc_template="<Document>: {doc}{suffix}",
    prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
    suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
)

INSTRUCTIONS = {
    # ENGLISH (Qwen3 recommendation)
    "default_en": "Given a web search query, retrieve relevant passages that answer the query",
    "tech_docs_en": "Find technical documentation and code examples that answer the question",
    "platform_en": "Find documentation about Comindware Platform features, configurations, and APIs",
    "code_en": "Retrieve documents with code examples, configurations, and technical explanations",
    "platform_detailed_en": "Retrieve Comindware Platform documentation: integration guides, API references, configuration examples, and code snippets in Russian or English",
    # RUSSIAN
    "tech_docs_ru": "Найдите техническую документацию и примеры кода",
    "platform_ru": "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации",
    # BILINGUAL
    "bilingual_full": "Find relevant documentation (Russian: Найдите релевантную документацию) in Russian or English, including code examples and configuration guides",
    "bilingual_code": "Find code examples and technical docs. (Код: примеры кода, API, конфигурации, руководства)",
    "bilingual_natural": "Find technical documentation about Comindware Platform - документация, примеры кода, конфигурации, API на русском и английском",
}


def run_benchmark():
    print("Reranker Instruction Benchmark (10 questions per instruction)")
    print("=" * 60)

    client = HttpClient(host="localhost", port=8000)
    collection = client.get_collection("mkdocs_kb")

    docs = collection.get(limit=200, include=["documents", "metadatas"])
    samples = [
        {"content": d, "source": m.get("source_file", "?")}
        for d, m in zip(docs["documents"], docs["metadatas"])
    ]

    # Generate questions
    questions = []
    terms = set()
    for s in samples[:100]:
        for w in s["content"].split():
            if len(w) > 5 and w.isalpha():
                terms.add(w.lower())
    terms = list(terms)[:100]

    templates = [
        ("How to configure {t}?", "en"),
        ("What is {t}?", "en"),
        ("Как настроить {t}?", "ru"),
        ("Пример {t}", "ru"),
    ]

    for term in terms[:30]:
        template, lang = random.choice(templates)
        q = template.format(t=term)
        relevant = [s for s in samples if term.lower() in s["content"].lower()][:5]
        if len(relevant) >= 2:
            questions.append({"q": q, "docs": relevant, "lang": lang})

    print(f"Generated {len(questions)} questions\n")

    config = ServerRerankerConfig(
        type="server",
        provider="mosec",
        endpoint="http://localhost:7998/v1/score",
        reranker_type="llm_reranker",
        default_instruction=INSTRUCTIONS["default_en"],
        formatting=QWEN3_FORMATTING,
    )
    adapter = RerankerAdapter(config)

    results = {}

    for name, instruction in INSTRUCTIONS.items():
        scores = []
        for q in questions[:10]:
            candidates = [(d["content"][:600], 0.0) for d in q["docs"]]
            try:
                ranked = adapter.rerank(
                    q["q"], candidates, top_k=len(candidates), instruction=instruction
                )
                scores.extend([r[1] for r in ranked])
            except:
                pass

        avg = sum(scores) / len(scores) if scores else 0
        results[name] = {"avg_score": avg, "instruction": instruction}
        print(f"{name:<25} {avg:.4f}")

    # Save
    output_dir = Path(__file__).parent.parent / "docs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(
        output_dir / f"reranker_quick_benchmark_{timestamp}.json", "w", encoding="utf-8"
    ) as f:
        json.dump({"results": results, "timestamp": timestamp}, f, ensure_ascii=False, indent=2)

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    en_scores = [d["avg_score"] for n, d in results.items() if n.endswith("_en")]
    ru_scores = [d["avg_score"] for n, d in results.items() if n.endswith("_ru")]
    bi_scores = [d["avg_score"] for n, d in results.items() if n.startswith("bilingual")]

    print(f"English avg:   {sum(en_scores) / len(en_scores):.4f}")
    print(f"Russian avg:   {sum(ru_scores) / len(ru_scores):.4f}")
    print(f"Bilingual avg: {sum(bi_scores) / len(bi_scores):.4f}")

    best = max(results.items(), key=lambda x: x[1]["avg_score"])
    print(f"\nBest: {best[0]} ({best[1]['avg_score']:.4f})")
    print(f"Instruction: {best[1]['instruction']}")

    if best[0].endswith("_en"):
        print("\nRECOMMENDATION: Use English instructions (per Qwen3 docs)")
    elif best[0].endswith("_ru"):
        print("\nNOTE: Russian instruction won - but Qwen3 recommends English")
    else:
        print("\nNOTE: Bilingual instruction performed best")


if __name__ == "__main__":
    run_benchmark()
