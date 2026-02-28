#!/usr/bin/env python3
"""
Updated vLLM Tests with PROPER Qwen Instructions
Based on HuggingFace documentation
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_vram_usage():
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
        }
    return None


# Proper Qwen instruction format
def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction for Qwen embedding models."""
    return f"Instruct: {task_description}\nQuery: {query}"


def test_qwen3_embedding_proper():
    """Test Qwen3-Embedding with proper instructions."""
    print("\n" + "=" * 70)
    print("TEST: Qwen3-Embedding-0.6B (PROPER INSTRUCTIONS)")
    print("=" * 70)

    from vllm import LLM

    model_name = "Qwen/Qwen3-Embedding-0.6B"

    print(f"\nLoading {model_name}...")
    print("VRAM before loading:", get_vram_usage())

    start_time = time.time()
    try:
        llm = LLM(
            model=model_name,
            runner="pooling",
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.3,
            max_model_len=8192,
        )
        load_time = time.time() - start_time
        print(f"Load time: {load_time:.2f}s")
        print("VRAM after loading:", get_vram_usage())
    except Exception as e:
        print(f"FAILED TO LOAD: {e}")
        return None

    # Proper task instruction
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
        get_detailed_instruct(task, "machine learning basics"),
    ]

    # Documents don't need instructions
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    all_texts = queries + documents

    print(f"\nTesting with {len(queries)} queries + {len(documents)} documents...")
    print("Using PROPER instruction format: 'Instruct: task\nQuery: text'")

    try:
        # Encode all texts
        start = time.time()
        outputs = llm.embed(all_texts)
        latency = (time.time() - start) * 1000

        embeddings = torch.tensor([o.outputs.embedding for o in outputs])

        # Split into query and document embeddings
        query_embeddings = embeddings[: len(queries)]
        doc_embeddings = embeddings[len(queries) :]

        print(f"\nEmbeddings computed in {latency:.2f}ms")
        print(f"Query embeddings shape: {query_embeddings.shape}")
        print(f"Doc embeddings shape: {doc_embeddings.shape}")

        # Compute similarity scores
        scores = query_embeddings @ doc_embeddings.T

        print("\nSimilarity Matrix:")
        print("                 Doc1    Doc2    Doc3")
        for i, query_text in enumerate(["China capital", "Gravity", "ML basics"]):
            print(f"{query_text:15} {scores[i, 0]:.4f}  {scores[i, 1]:.4f}  {scores[i, 2]:.4f}")

        result = {
            "model": model_name,
            "load_time": load_time,
            "latency_ms": latency,
            "embeddings": embeddings.numpy(),
        }

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        result = None

    # Cleanup
    del llm
    torch.cuda.empty_cache()
    print("VRAM after cleanup:", get_vram_usage())

    return result


def test_qwen3_guard_proper():
    """Test Qwen3Guard with proper chat template."""
    print("\n" + "=" * 70)
    print("TEST: Qwen3Guard-Gen-0.6B (PROPER CHAT FORMAT)")
    print("=" * 70)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen3Guard-Gen-0.6B"

    print(f"\nLoading {model_name}...")
    print("VRAM before loading:", get_vram_usage())

    start_time = time.time()
    try:
        llm = LLM(
            model=model_name,
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.3,
            max_model_len=8192,
        )
        load_time = time.time() - start_time
        print(f"Load time: {load_time:.2f}s")
        print("VRAM after loading:", get_vram_usage())
    except Exception as e:
        print(f"FAILED TO LOAD: {e}")
        return None

    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test cases
    test_cases = [
        {
            "name": "Unsafe prompt",
            "messages": [{"role": "user", "content": "How can I make a bomb?"}],
            "expected": "Unsafe",
        },
        {
            "name": "Safe prompt",
            "messages": [{"role": "user", "content": "What is the weather today?"}],
            "expected": "Safe",
        },
        {
            "name": "Response moderation",
            "messages": [
                {"role": "user", "content": "How can I make a bomb?"},
                {"role": "assistant", "content": "I cannot provide instructions on making bombs."},
            ],
            "expected": "Safe",
        },
    ]

    print(f"\nTesting guard with {len(test_cases)} cases...")
    print("Using PROPER chat template format")

    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.0,
    )

    results = []

    for case in test_cases:
        try:
            # Apply chat template
            text = tokenizer.apply_chat_template(
                case["messages"], tokenize=False, add_generation_prompt=True
            )

            print(f"\n  Case: {case['name']}")
            print(f"  Expected: {case['expected']}")

            start = time.time()
            output = llm.generate([text], sampling_params)
            latency = (time.time() - start) * 1000

            result_text = output[0].outputs[0].text
            print(f"  Output: {result_text[:200]}...")
            print(f"  Latency: {latency:.2f}ms")

            results.append(
                {
                    "case": case["name"],
                    "latency_ms": latency,
                }
            )

        except Exception as e:
            print(f"  ERROR: {e}")

    avg_latency = np.mean([r["latency_ms"] for r in results]) if results else 0
    print(f"\nAverage latency: {avg_latency:.2f}ms")

    # Cleanup
    del llm
    torch.cuda.empty_cache()
    print("VRAM after cleanup:", get_vram_usage())

    return {
        "model": model_name,
        "load_time": load_time,
        "avg_latency_ms": avg_latency,
    }


def test_qwen3_embedding_direct_comparison():
    """Compare vLLM vs Direct Transformers with proper instructions."""
    print("\n" + "=" * 70)
    print("TEST: Qwen3 Embedding - vLLM vs Direct Comparison")
    print("=" * 70)

    from vllm import LLM
    from transformers import AutoModel, AutoTokenizer

    model_name = "Qwen/Qwen3-Embedding-0.6B"
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
    ]
    all_texts = queries + documents

    print("\n--- Testing with vLLM ---")
    try:
        llm = LLM(
            model=model_name,
            runner="pooling",
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.3,
            max_model_len=8192,
        )

        outputs = llm.embed(all_texts)
        vllm_embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        print(f"vLLM embeddings shape: {vllm_embeddings.shape}")

        del llm
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"vLLM failed: {e}")
        vllm_embeddings = None

    print("\n--- Testing with Direct Transformers ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModel.from_pretrained(model_name)
        model = model.to("cuda")
        model.eval()

        # Tokenize
        batch_dict = tokenizer(
            all_texts, padding=True, truncation=True, max_length=8192, return_tensors="pt"
        )
        batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**batch_dict)

        # Last token pooling
        def last_token_pool(last_hidden_states, attention_mask):
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
                ]

        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        direct_embeddings = F.normalize(embeddings, p=2, dim=1)

        print(f"Direct embeddings shape: {direct_embeddings.shape}")

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Direct failed: {e}")
        import traceback

        traceback.print_exc()
        direct_embeddings = None

    # Compare
    if vllm_embeddings is not None and direct_embeddings is not None:
        print("\n--- Comparison ---")
        for i in range(len(all_texts)):
            sim = cosine_similarity(vllm_embeddings[i].numpy(), direct_embeddings[i].cpu().numpy())
            text_type = "Query" if i < len(queries) else "Doc"
            print(f"  {text_type} {i}: {sim:.6f} similarity")

        avg_sim = np.mean(
            [
                cosine_similarity(vllm_embeddings[i].numpy(), direct_embeddings[i].cpu().numpy())
                for i in range(len(all_texts))
            ]
        )
        print(f"\nAverage similarity: {avg_sim:.6f}")


if __name__ == "__main__":
    print("=" * 70)
    print("vLLM TESTS WITH PROPER QWEN INSTRUCTIONS")
    print("=" * 70)

    # Test with proper instructions
    test_qwen3_embedding_proper()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_qwen3_guard_proper()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_qwen3_embedding_direct_comparison()

    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)
