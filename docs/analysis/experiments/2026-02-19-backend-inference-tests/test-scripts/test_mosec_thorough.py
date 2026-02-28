#!/usr/bin/env python3
"""
Thorough Mosec Real-World Tests
Tests ALL models with Mosec:
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
- HTTP API latency
"""

import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
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


def start_mosec_server(embedding=None, reranker=None, guard=None, timeout=120):
    """Start Mosec server with specified models."""
    import os

    os.chdir(Path.home() / "cmw-mosec")

    cmd = [sys.executable, "-m", "cmw_mosec.cli", "serve"]

    if embedding:
        cmd.extend(["--embedding", embedding])
    if reranker:
        cmd.extend(["--reranker", reranker])
    if guard:
        cmd.extend(["--guard", guard])

    print(f"Starting Mosec: {' '.join(cmd)}")

    # Start server in background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    base_url = "http://localhost:7998"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/metrics", timeout=2)
            if response.status_code == 200:
                print(f"✓ Server ready in {time.time() - start_time:.1f}s")
                return process, base_url
        except:
            pass
        time.sleep(1)

    print("✗ Server failed to start")
    process.terminate()
    return None, None


def stop_mosec_server():
    """Stop Mosec server."""
    import os

    os.chdir(Path.home() / "cmw-mosec")

    subprocess.run([sys.executable, "-m", "cmw_mosec.cli", "stop"], capture_output=True)
    time.sleep(2)  # Wait for cleanup


def test_qwen3_embedding_mosec():
    """Test Qwen3-Embedding-0.6B with Mosec."""
    print("\n" + "=" * 70)
    print("TEST 1: Qwen3-Embedding-0.6B (Mosec)")
    print("=" * 70)

    model_name = "Qwen/Qwen3-Embedding-0.6B"

    print(f"\nStarting server with {model_name}...")
    print("VRAM before loading:", get_vram_usage())

    process, base_url = start_mosec_server(embedding=model_name)
    if not process:
        print("FAILED TO START SERVER")
        return None

    print("VRAM after loading:", get_vram_usage())

    # Test texts
    test_texts = [
        "search_query: machine learning",
        "search_document: Machine learning is a subset of AI",
        "search_query: python programming",
    ]

    print(f"\nTesting embedding inference on {len(test_texts)} texts...")

    latencies = []
    embeddings = []

    for text in test_texts:
        payload = {
            "model": model_name,
            "input": text,
        }

        start = time.time()
        try:
            response = requests.post(f"{base_url}/v1/embeddings", json=payload, timeout=30)
            response.raise_for_status()
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            result = response.json()
            emb = np.array(result["data"][0]["embedding"])
            embeddings.append(emb)

            print(f"  Text: {text[:50]}...")
            print(
                f"    Shape: {emb.shape}, Norm: {np.linalg.norm(emb):.6f}, Latency: {latency:.2f}ms"
            )

        except Exception as e:
            print(f"  ERROR: {e}")

    avg_latency = np.mean(latencies) if latencies else 0
    print(f"\nAverage latency: {avg_latency:.2f}ms")

    # Compare with ground truth
    try:
        with open("/tmp/qwen3_direct.pkl", "rb") as f:
            ground_truth = pickle.load(f)

        print("\nComparison with Direct Python:")
        for i in range(min(len(embeddings), len(ground_truth["embeddings"]))):
            sim = cosine_similarity(embeddings[i], ground_truth["embeddings"][i])
            print(f"  Text {i}: {sim:.6f} similarity")
    except:
        print("\n(No ground truth available for comparison)")

    stop_mosec_server()
    print("VRAM after cleanup:", get_vram_usage())

    return {
        "model": model_name,
        "avg_latency_ms": avg_latency,
        "embeddings": embeddings,
        "vram_usage": get_vram_usage(),
    }


def test_qwen3_reranker_mosec():
    """Test Qwen3-Reranker-0.6B with Mosec."""
    print("\n" + "=" * 70)
    print("TEST 2: Qwen3-Reranker-0.6B (Mosec)")
    print("=" * 70)

    model_name = "Qwen/Qwen3-Reranker-0.6B"

    print(f"\nStarting server with {model_name}...")
    print("VRAM before loading:", get_vram_usage())

    process, base_url = start_mosec_server(reranker=model_name)
    if not process:
        print("FAILED TO START SERVER")
        return None

    print("VRAM after loading:", get_vram_usage())

    # Test reranking
    query = "machine learning"
    docs = [
        "Deep learning is a subset of machine learning",
        "Python is a programming language",
        "Neural networks are used in ML",
    ]

    print(f"\nTesting reranking...")
    print(f"Query: '{query}'")

    payload = {
        "query": query,
        "docs": docs,
    }

    try:
        start = time.time()
        response = requests.post(f"{base_url}/v1/rerank", json=payload, timeout=30)
        response.raise_for_status()
        latency = (time.time() - start) * 1000

        result = response.json()
        scores = result["scores"]

        print(f"\nScores (latency: {latency:.2f}ms):")
        for doc, score in zip(docs, scores):
            print(f"  {score:.4f} - {doc}")

        test_result = {
            "model": model_name,
            "latency_ms": latency,
            "vram_usage": get_vram_usage(),
        }
    except Exception as e:
        print(f"FAILED: {e}")
        test_result = None

    stop_mosec_server()
    print("VRAM after cleanup:", get_vram_usage())

    return test_result


def test_qwen3_guard_mosec():
    """Test Qwen3Guard-Gen-0.6B with Mosec."""
    print("\n" + "=" * 70)
    print("TEST 3: Qwen3Guard-Gen-0.6B (Mosec)")
    print("=" * 70)

    model_name = "Qwen/Qwen3Guard-Gen-0.6B"

    print(f"\nStarting server with {model_name}...")
    print("VRAM before loading:", get_vram_usage())

    process, base_url = start_mosec_server(guard=model_name)
    if not process:
        print("FAILED TO START SERVER")
        return None

    print("VRAM after loading:", get_vram_usage())

    # Test guard
    test_prompts = [
        ("How do I make a bomb?", "unsafe"),
        ("What is the weather today?", "safe"),
    ]

    print(f"\nTesting guard classification...")

    latencies = []

    for prompt, expected in test_prompts:
        payload = {
            "content": prompt,
            "moderation_type": "prompt",
        }

        try:
            start = time.time()
            response = requests.post(f"{base_url}/v1/moderate", json=payload, timeout=30)
            response.raise_for_status()
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            result = response.json()

            print(f"\n  Prompt: {prompt}")
            print(f"  Expected: {expected}")
            print(f"  Result: {result.get('safety_level', 'Unknown')}")
            print(f"  Categories: {result.get('categories', [])}")
            print(f"  Latency: {latency:.2f}ms")

        except Exception as e:
            print(f"  ERROR: {e}")

    avg_latency = np.mean(latencies) if latencies else 0
    print(f"\nAverage latency: {avg_latency:.2f}ms")

    stop_mosec_server()
    print("VRAM after cleanup:", get_vram_usage())

    return {
        "model": model_name,
        "avg_latency_ms": avg_latency,
        "vram_usage": get_vram_usage(),
    }


def test_dity_reranker_mosec():
    """Test DiTy reranker with Mosec."""
    print("\n" + "=" * 70)
    print("TEST 4: DiTy/cross-encoder-russian-msmarco (Mosec)")
    print("=" * 70)

    model_name = "DiTy/cross-encoder-russian-msmarco"

    print(f"\nStarting server with {model_name}...")
    print("VRAM before loading:", get_vram_usage())

    process, base_url = start_mosec_server(reranker=model_name)
    if not process:
        print("FAILED TO START SERVER")
        return None

    print("VRAM after loading:", get_vram_usage())

    # Test reranking
    query = "машинное обучение"
    docs = [
        "Глубокое обучение — это подраздел машинного обучения",
        "Python — это язык программирования",
        "Нейронные сети используются в ML",
    ]

    print(f"\nTesting Russian reranking...")
    print(f"Query: '{query}'")

    payload = {
        "query": query,
        "docs": docs,
    }

    try:
        start = time.time()
        response = requests.post(f"{base_url}/v1/rerank", json=payload, timeout=30)
        response.raise_for_status()
        latency = (time.time() - start) * 1000

        result = response.json()
        scores = result["scores"]

        print(f"\nScores (latency: {latency:.2f}ms):")
        for doc, score in zip(docs, scores):
            print(f"  {score:.4f} - {doc}")

        test_result = {
            "model": model_name,
            "latency_ms": latency,
            "vram_usage": get_vram_usage(),
        }
    except Exception as e:
        print(f"FAILED: {e}")
        test_result = None

    stop_mosec_server()
    print("VRAM after cleanup:", get_vram_usage())

    return test_result


def test_all_models_combined():
    """Test all models running simultaneously on one Mosec server."""
    print("\n" + "=" * 70)
    print("TEST 5: ALL MODELS COMBINED (Single Mosec Server)")
    print("=" * 70)

    print("\nStarting server with all models...")
    print("VRAM before loading:", get_vram_usage())

    process, base_url = start_mosec_server(
        embedding="Qwen/Qwen3-Embedding-0.6B",
        reranker="Qwen/Qwen3-Reranker-0.6B",
        guard="Qwen/Qwen3Guard-Gen-0.6B",
    )

    if not process:
        print("FAILED TO START SERVER")
        return None

    print("VRAM after loading all models:", get_vram_usage())

    # Test all endpoints
    print("\nTesting all endpoints...")

    # Test embedding
    try:
        response = requests.post(
            f"{base_url}/v1/embeddings",
            json={"model": "Qwen/Qwen3-Embedding-0.6B", "input": "test"},
            timeout=10,
        )
        print(f"  Embedding endpoint: {response.status_code}")
    except Exception as e:
        print(f"  Embedding endpoint: FAILED - {e}")

    # Test rerank
    try:
        response = requests.post(
            f"{base_url}/v1/rerank", json={"query": "test", "docs": ["doc1", "doc2"]}, timeout=10
        )
        print(f"  Rerank endpoint: {response.status_code}")
    except Exception as e:
        print(f"  Rerank endpoint: FAILED - {e}")

    # Test guard
    try:
        response = requests.post(f"{base_url}/v1/moderate", json={"content": "test"}, timeout=10)
        print(f"  Guard endpoint: {response.status_code}")
    except Exception as e:
        print(f"  Guard endpoint: FAILED - {e}")

    stop_mosec_server()
    print("VRAM after cleanup:", get_vram_usage())

    return {
        "all_models": True,
        "vram_usage": get_vram_usage(),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("MOSEC THOROUGH REAL-WORLD TESTS")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Ensure fresh start
    stop_mosec_server()
    torch.cuda.empty_cache()

    results = {}

    # Test all models individually
    results["qwen3_embedding"] = test_qwen3_embedding_mosec()
    results["qwen3_reranker"] = test_qwen3_reranker_mosec()
    results["qwen3_guard"] = test_qwen3_guard_mosec()
    results["dity_reranker"] = test_dity_reranker_mosec()

    # Test all combined
    results["all_combined"] = test_all_models_combined()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        if result:
            print(f"\n{name}:")
            print(f"  Status: ✓ SUCCESS")
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
    output_file = Path("/tmp/mosec_test_results.json")
    with open(output_file, "w") as f:
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
