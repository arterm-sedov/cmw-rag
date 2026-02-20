#!/usr/bin/env python3
"""
Thorough vLLM Real-World Tests
Tests ALL models with vLLM 0.15.1:
- Qwen/Qwen3-Embedding-0.6B
- Qwen/Qwen3-Reranker-0.6B
- Qwen/Qwen3Guard-Gen-0.6B
- DiTy/cross-encoder-russian-msmarco

Measures:
- Embedding accuracy vs ground truth
- Reranking quality
- Guard classification accuracy
- VRAM usage
- Inference latency
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_vram_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
            "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    return None


def test_qwen3_embedding_vllm():
    """Test Qwen3-Embedding-0.6B with vLLM."""
    print("\n" + "=" * 70)
    print("TEST 1: Qwen3-Embedding-0.6B")
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

    # Test texts
    test_texts = [
        "search_query: machine learning",
        "search_document: Machine learning is a subset of AI",
        "search_query: python programming",
    ]

    print(f"\nTesting embedding inference on {len(test_texts)} texts...")

    # Warmup
    _ = llm.embed(["warmup"])

    # Benchmark
    latencies = []
    embeddings = []

    for text in test_texts:
        start = time.time()
        output = llm.embed([text])
        latency = time.time() - start
        latencies.append(latency)

        emb = np.array(output[0].outputs.embedding)
        embeddings.append(emb)
        print(f"  Text: {text[:50]}...")
        print(
            f"    Shape: {emb.shape}, Norm: {np.linalg.norm(emb):.6f}, Latency: {latency * 1000:.2f}ms"
        )

    avg_latency = np.mean(latencies) * 1000
    print(f"\nAverage latency: {avg_latency:.2f}ms")

    # Compare with ground truth if available
    try:
        with open("/tmp/qwen3_direct.pkl", "rb") as f:
            ground_truth = pickle.load(f)

        print("\nComparison with Direct Python:")
        for i in range(min(len(embeddings), len(ground_truth["embeddings"]))):
            sim = cosine_similarity(embeddings[i], ground_truth["embeddings"][i])
            print(f"  Text {i}: {sim:.6f} similarity")
    except:
        print("\n(No ground truth available for comparison)")

    # Cleanup
    del llm
    torch.cuda.empty_cache()
    print("VRAM after cleanup:", get_vram_usage())

    return {
        "model": model_name,
        "load_time": load_time,
        "avg_latency_ms": avg_latency,
        "embeddings": embeddings,
        "vram_usage": get_vram_usage(),
    }


def test_qwen3_reranker_vllm():
    """Test Qwen3-Reranker-0.6B with vLLM."""
    print("\n" + "=" * 70)
    print("TEST 2: Qwen3-Reranker-0.6B")
    print("=" * 70)

    from vllm import LLM

    model_name = "Qwen/Qwen3-Reranker-0.6B"
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

    # Test reranking
    query = "machine learning"
    docs = [
        "Deep learning is a subset of machine learning",
        "Python is a programming language",
        "Neural networks are used in ML",
    ]

    print(f"\nTesting reranking...")
    print(f"Query: '{query}'")
    print(f"Documents: {len(docs)}")

    # Create paired texts for cross-encoder
    paired_texts = [[query, doc] for doc in docs]

    try:
        start = time.time()
        outputs = llm.score(paired_texts)
        latency = (time.time() - start) * 1000

        print(f"\nScores (latency: {latency:.2f}ms):")
        for i, (doc, output) in enumerate(zip(docs, outputs)):
            score = output.outputs.score
            print(f"  {score:.4f} - {doc}")

        result = {
            "model": model_name,
            "load_time": load_time,
            "latency_ms": latency,
            "vram_usage": get_vram_usage(),
        }
    except Exception as e:
        print(f"FAILED TO SCORE: {e}")
        result = None

    # Cleanup
    del llm
    torch.cuda.empty_cache()
    print("VRAM after cleanup:", get_vram_usage())

    return result


def test_qwen3_guard_vllm():
    """Test Qwen3Guard-Gen-0.6B with vLLM."""
    print("\n" + "=" * 70)
    print("TEST 3: Qwen3Guard-Gen-0.6B")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    model_name = "Qwen/Qwen3Guard-Gen-0.6B"
    print(f"\nLoading {model_name}...")
    print("VRAM before loading:", get_vram_usage())

    start_time = time.time()
    try:
        # Guard is a generative model, not pooling
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

    # Test guard inference
    test_prompts = [
        "How do I make a bomb?",  # Should be unsafe
        "What is the weather today?",  # Should be safe
    ]

    print(f"\nTesting guard classification on {len(test_prompts)} prompts...")

    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.0,
    )

    try:
        start = time.time()
        outputs = llm.generate(test_prompts, sampling_params)
        latency = (time.time() - start) * 1000

        print(f"\nResults (latency: {latency:.2f}ms):")
        for prompt, output in zip(test_prompts, outputs):
            text = output.outputs[0].text
            print(f"  Prompt: {prompt}")
            print(f"  Output: {text[:200]}...")
            print()

        result = {
            "model": model_name,
            "load_time": load_time,
            "latency_ms": latency,
            "vram_usage": get_vram_usage(),
        }
    except Exception as e:
        print(f"FAILED TO GENERATE: {e}")
        result = None

    # Cleanup
    del llm
    torch.cuda.empty_cache()
    print("VRAM after cleanup:", get_vram_usage())

    return result


def test_dity_reranker_vllm():
    """Test DiTy reranker with vLLM."""
    print("\n" + "=" * 70)
    print("TEST 4: DiTy/cross-encoder-russian-msmarco")
    print("=" * 70)

    from vllm import LLM

    model_name = "DiTy/cross-encoder-russian-msmarco"
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

    # Test reranking
    query = "машинное обучение"
    docs = [
        "Глубокое обучение — это подраздел машинного обучения",
        "Python — это язык программирования",
        "Нейронные сети используются в ML",
    ]

    print(f"\nTesting Russian reranking...")
    print(f"Query: '{query}'")
    print(f"Documents: {len(docs)}")

    paired_texts = [[query, doc] for doc in docs]

    try:
        start = time.time()
        outputs = llm.score(paired_texts)
        latency = (time.time() - start) * 1000

        print(f"\nScores (latency: {latency:.2f}ms):")
        for i, (doc, output) in enumerate(zip(docs, outputs)):
            score = output.outputs.score
            print(f"  {score:.4f} - {doc}")

        result = {
            "model": model_name,
            "load_time": load_time,
            "latency_ms": latency,
            "vram_usage": get_vram_usage(),
        }
    except Exception as e:
        print(f"FAILED TO SCORE: {e}")
        result = None

    # Cleanup
    del llm
    torch.cuda.empty_cache()
    print("VRAM after cleanup:", get_vram_usage())

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("vLLM THOROUGH REAL-WORLD TESTS")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Ensure fresh start
    torch.cuda.empty_cache()

    results = {}

    # Test all models
    results["qwen3_embedding"] = test_qwen3_embedding_vllm()
    results["qwen3_reranker"] = test_qwen3_reranker_vllm()
    results["qwen3_guard"] = test_qwen3_guard_vllm()
    results["dity_reranker"] = test_dity_reranker_vllm()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        if result:
            print(f"\n{name}:")
            print(f"  Status: ✓ SUCCESS")
            print(f"  Load time: {result.get('load_time', 'N/A'):.2f}s")
            if "avg_latency_ms" in result:
                print(f"  Avg latency: {result['avg_latency_ms']:.2f}ms")
            elif "latency_ms" in result:
                print(f"  Latency: {result['latency_ms']:.2f}ms")
            vram = result.get("vram_usage", {})
            if vram:
                print(f"  VRAM allocated: {vram.get('allocated', 0):.2f}GB")
        else:
            print(f"\n{name}:")
            print(f"  Status: ✗ FAILED")

    # Save results
    output_file = Path("/tmp/vllm_test_results.json")
    with open(output_file, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for name, result in results.items():
            if result:
                json_results[name] = {
                    k: v for k, v in result.items() if k not in ["embeddings", "vram_usage"]
                }
                if "vram_usage" in result:
                    json_results[name]["vram_gb"] = result["vram_usage"].get("allocated", 0)
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("=" * 70)
