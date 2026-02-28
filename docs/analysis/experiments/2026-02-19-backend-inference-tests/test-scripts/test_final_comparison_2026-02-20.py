#!/usr/bin/env python3
"""
COMPREHENSIVE FINAL TEST - vLLM vs Fixed Mosec
Tests performance and accuracy of both backends with Qwen3-Embedding-0.6B

Updated: 2026-02-20 with fixed Mosec (last-token pooling support)
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
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_vram_usage():
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
        }
    return {"allocated": 0, "reserved": 0}


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


class TestResults:
    def __init__(self):
        self.results = {}

    def add(self, model, backend, status, metrics):
        key = f"{model}_{backend}"
        self.results[key] = {
            "model": model,
            "backend": backend,
            "status": status,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def print_summary(self):
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)

        models = {}
        for key, data in self.results.items():
            model = data["model"]
            if model not in models:
                models[model] = []
            models[model].append(data)

        for model, backends in models.items():
            print(f"\n{model}:")
            for b in backends:
                status_icon = "✓" if b["status"] == "SUCCESS" else "✗"
                print(f"  {status_icon} {b['backend']:15} - {b['status']}")
                if b["metrics"]:
                    for k, v in b["metrics"].items():
                        if isinstance(v, float):
                            print(f"      {k}: {v:.4f}")
                        else:
                            print(f"      {k}: {v}")


results = TestResults()


################################################################################
# DIRECT TRANSFORMERS (Ground Truth)
################################################################################


def test_direct_baseline():
    """Test with Direct Transformers for comparison."""
    print("\n" + "=" * 80)
    print("BASELINE: Direct Transformers")
    print("=" * 80)

    model_name = "Qwen/Qwen3-Embedding-0.6B"

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModel.from_pretrained(model_name)
    model = model.to("cuda")
    model.eval()

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

    batch_dict = tokenizer(
        all_texts, padding=True, truncation=True, max_length=8192, return_tensors="pt"
    )
    batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}

    # Warmup
    with torch.no_grad():
        _ = model(**batch_dict)

    # Benchmark
    latencies = []
    for _ in range(5):
        start = time.time()
        with torch.no_grad():
            outputs = model(**batch_dict)
        latencies.append((time.time() - start) * 1000)

    avg_latency = np.mean(latencies)

    # Last token pooling
    def last_token_pool(last_hidden_states, attention_mask):
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    query_emb = embeddings[:2].cpu().numpy()
    doc_emb = embeddings[2:].cpu().numpy()
    scores = query_emb @ doc_emb.T

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Average latency: {avg_latency:.2f}ms (over 5 runs)")
    print(f"VRAM usage: {get_vram_usage()['allocated']:.2f}GB")
    print(f"Similarity scores:")
    print(f"  China query: {scores[0]}")
    print(f"  Gravity query: {scores[1]}")

    # Save for comparison
    with open("/tmp/qwen3_direct_baseline.pkl", "wb") as f:
        pickle.dump(embeddings.cpu().numpy(), f)

    del model
    torch.cuda.empty_cache()

    results.add(
        "Qwen3-Embedding",
        "Direct",
        "SUCCESS",
        {
            "latency_ms": avg_latency,
            "scores": scores.tolist(),
            "vram_gb": get_vram_usage()["allocated"],
        },
    )

    return embeddings.cpu().numpy()


################################################################################
# VLLM TESTS
################################################################################


def test_vllm_performance():
    """Test vLLM with performance metrics."""
    print("\n" + "=" * 80)
    print("VLLM: Performance Test (8GB VRAM cap)")
    print("=" * 80)

    try:
        from vllm import LLM

        model_name = "Qwen/Qwen3-Embedding-0.6B"

        print(f"\nLoading {model_name} with 8GB VRAM limit...")
        print("Settings: gpu_memory_utilization=0.17, enforce_eager=True")

        start_time = time.time()
        llm = LLM(
            model=model_name,
            runner="pooling",
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.17,
            max_model_len=8192,
            enforce_eager=True,
        )
        load_time = time.time() - start_time

        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"VRAM usage: {get_vram_usage()['allocated']:.2f}GB")

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

        # Warmup
        _ = llm.embed(["warmup"])

        # Benchmark
        print(f"\nBenchmarking {len(all_texts)} texts over 5 runs...")
        latencies = []
        for _ in range(5):
            start = time.time()
            outputs = llm.embed(all_texts)
            latencies.append((time.time() - start) * 1000)

        avg_latency = np.mean(latencies)

        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        query_emb = embeddings[:2].numpy()
        doc_emb = embeddings[2:].numpy()
        scores = query_emb @ doc_emb.T

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Average latency: {avg_latency:.2f}ms (over 5 runs)")
        print(f"Similarity scores:")
        print(f"  China query: {scores[0]}")
        print(f"  Gravity query: {scores[1]}")

        # Compare with direct
        try:
            with open("/tmp/qwen3_direct_baseline.pkl", "rb") as f:
                direct_emb = pickle.load(f)

            print("\nComparison with Direct Transformers:")
            similarities = []
            for i in range(len(embeddings)):
                sim = cosine_similarity(embeddings[i].numpy(), direct_emb[i])
                similarities.append(sim)
                text_type = "Query" if i < 2 else "Doc"
                print(f"  {text_type} {i}: {sim:.6f} similarity")

            avg_sim = np.mean(similarities)
            print(f"\nAverage similarity: {avg_sim:.6f}")
        except Exception as e:
            print(f"Could not compare: {e}")
            avg_sim = None

        del llm
        torch.cuda.empty_cache()

        results.add(
            "Qwen3-Embedding",
            "vLLM",
            "SUCCESS",
            {
                "load_time_s": load_time,
                "latency_ms": avg_latency,
                "scores": scores.tolist(),
                "similarity_vs_direct": avg_sim,
                "vram_gb": get_vram_usage()["allocated"],
            },
        )

        return embeddings.numpy()

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        results.add("Qwen3-Embedding", "vLLM", "FAILED", {"error": str(e)})
        return None


################################################################################
# MOSEC TESTS (Fixed with last-token pooling)
################################################################################


def start_mosec(embedding=None, timeout=120):
    """Start Mosec server."""
    import os

    os.chdir(Path.home() / "cmw-mosec")

    cmd = [sys.executable, "-m", "cmw_mosec.cli", "serve"]
    if embedding:
        cmd.extend(["--embedding", embedding])

    print(f"Starting Mosec: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

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


def stop_mosec():
    """Stop Mosec server."""
    import os

    os.chdir(Path.home() / "cmw-mosec")
    subprocess.run([sys.executable, "-m", "cmw_mosec.cli", "stop"], capture_output=True)
    time.sleep(2)


def test_mosec_fixed():
    """Test fixed Mosec with last-token pooling."""
    print("\n" + "=" * 80)
    print("MOSEC: Fixed with Last-Token Pooling")
    print("=" * 80)

    try:
        stop_mosec()
        torch.cuda.empty_cache()

        model_name = "Qwen/Qwen3-Embedding-0.6B"

        print(f"\nStarting server with {model_name}...")
        print("Expected pooling: last_token (from models.yaml)")

        process, base_url = start_mosec(embedding=model_name)
        if not process:
            raise Exception("Failed to start server")

        print(f"VRAM usage: {get_vram_usage()['allocated']:.2f}GB")

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

        # Warmup
        payload = {"model": model_name, "input": "warmup"}
        requests.post(f"{base_url}/v1/embeddings", json=payload, timeout=30)

        # Benchmark
        print(f"\nBenchmarking {len(all_texts)} texts over 5 runs...")
        latencies = []
        embeddings_list = []

        for _ in range(5):
            batch_embeddings = []
            batch_start = time.time()

            for text in all_texts:
                payload = {"model": model_name, "input": text}
                response = requests.post(f"{base_url}/v1/embeddings", json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                emb = np.array(result["data"][0]["embedding"])
                batch_embeddings.append(emb)

            latencies.append((time.time() - batch_start) * 1000)
            embeddings_list = batch_embeddings

        avg_latency = np.mean(latencies)
        embeddings = np.array(embeddings_list)

        query_emb = embeddings[:2]
        doc_emb = embeddings[2:]
        scores = query_emb @ doc_emb.T

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Average latency: {avg_latency:.2f}ms (over 5 runs)")
        print(f"Similarity scores:")
        print(f"  China query: {scores[0]}")
        print(f"  Gravity query: {scores[1]}")

        # Compare with direct
        try:
            with open("/tmp/qwen3_direct_baseline.pkl", "rb") as f:
                direct_emb = pickle.load(f)

            print("\nComparison with Direct Transformers:")
            similarities = []
            for i in range(len(embeddings)):
                sim = cosine_similarity(embeddings[i], direct_emb[i])
                similarities.append(sim)
                text_type = "Query" if i < 2 else "Doc"
                print(f"  {text_type} {i}: {sim:.6f} similarity")

            avg_sim = np.mean(similarities)
            print(f"\nAverage similarity: {avg_sim:.6f}")
        except Exception as e:
            print(f"Could not compare: {e}")
            avg_sim = None

        stop_mosec()
        torch.cuda.empty_cache()

        results.add(
            "Qwen3-Embedding",
            "Mosec-Fixed",
            "SUCCESS",
            {
                "latency_ms": avg_latency,
                "scores": scores.tolist(),
                "similarity_vs_direct": avg_sim,
                "vram_gb": get_vram_usage()["allocated"],
            },
        )

        return embeddings

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        stop_mosec()
        results.add("Qwen3-Embedding", "Mosec-Fixed", "FAILED", {"error": str(e)})
        return None


################################################################################
# MAIN
################################################################################

if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE TEST - vLLM vs Fixed Mosec")
    print("Qwen3-Embedding-0.6B Performance & Accuracy")
    print("=" * 80)
    print(f"Date: 2026-02-20")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Ensure clean state
    stop_mosec()
    torch.cuda.empty_cache()

    # Test all backends
    direct_emb = test_direct_baseline()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vllm_emb = test_vllm_performance()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mosec_emb = test_mosec_fixed()

    # Final summary
    results.print_summary()

    # Save results
    with open("/tmp/final_test_results_2026-02-20.json", "w") as f:
        json.dump(results.results, f, indent=2, default=str)

    print(f"\n✓ Results saved to /tmp/final_test_results_2026-02-20.json")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
