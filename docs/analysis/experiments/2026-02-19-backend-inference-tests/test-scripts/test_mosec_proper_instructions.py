#!/usr/bin/env python3
"""
Updated Mosec Tests with PROPER Qwen Instructions
Based on HuggingFace documentation
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


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_vram_usage():
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
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


def stop_mosec_server():
    """Stop Mosec server."""
    import os

    os.chdir(Path.home() / "cmw-mosec")

    subprocess.run([sys.executable, "-m", "cmw_mosec.cli", "stop"], capture_output=True)
    time.sleep(2)


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction for Qwen embedding models."""
    return f"Instruct: {task_description}\nQuery: {query}"


def test_qwen3_embedding_proper():
    """Test Qwen3-Embedding with proper instructions via Mosec."""
    print("\n" + "=" * 70)
    print("TEST: Qwen3-Embedding-0.6B via Mosec (PROPER INSTRUCTIONS)")
    print("=" * 70)

    model_name = "Qwen/Qwen3-Embedding-0.6B"

    print(f"\nStarting server with {model_name}...")
    print("VRAM before loading:", get_vram_usage())

    process, base_url = start_mosec_server(embedding=model_name)
    if not process:
        print("FAILED TO START SERVER")
        return None

    print("VRAM after loading:", get_vram_usage())

    # Proper task instruction
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
        get_detailed_instruct(task, "machine learning basics"),
    ]

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    all_texts = queries + documents

    print(f"\nTesting with {len(queries)} queries + {len(documents)} documents...")
    print("Using PROPER instruction format: 'Instruct: task\nQuery: text'")

    latencies = []
    embeddings = []

    for text in all_texts:
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

        except Exception as e:
            print(f"  ERROR: {e}")

    if embeddings:
        embeddings = np.array(embeddings)
        query_embeddings = embeddings[: len(queries)]
        doc_embeddings = embeddings[len(queries) :]

        print(f"\nEmbeddings computed (avg latency: {np.mean(latencies):.2f}ms)")
        print(f"Query embeddings shape: {query_embeddings.shape}")
        print(f"Doc embeddings shape: {doc_embeddings.shape}")

        # Compute similarity scores
        scores = query_embeddings @ doc_embeddings.T

        print("\nSimilarity Matrix:")
        print("                 Doc1    Doc2    Doc3")
        for i, query_text in enumerate(["China capital", "Gravity", "ML basics"]):
            print(f"{query_text:15} {scores[i, 0]:.4f}  {scores[i, 1]:.4f}  {scores[i, 2]:.4f}")

    stop_mosec_server()
    print("VRAM after cleanup:", get_vram_usage())

    return {
        "model": model_name,
        "avg_latency_ms": np.mean(latencies) if latencies else 0,
        "embeddings": embeddings,
    }


def test_qwen3_embedding_direct():
    """Test Qwen3-Embedding directly with proper instructions for comparison."""
    print("\n" + "=" * 70)
    print("TEST: Qwen3-Embedding-0.6B Direct Transformers (PROPER INSTRUCTIONS)")
    print("=" * 70)

    from transformers import AutoModel, AutoTokenizer

    model_name = "Qwen/Qwen3-Embedding-0.6B"

    print(f"\nLoading {model_name}...")
    print("VRAM before loading:", get_vram_usage())

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModel.from_pretrained(model_name)
        model = model.to("cuda")
        model.eval()

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

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
        "Machine learning is a subset of artificial intelligence.",
    ]

    all_texts = queries + documents

    print(f"\nTesting with {len(queries)} queries + {len(documents)} documents...")

    try:
        # Tokenize
        batch_dict = tokenizer(
            all_texts, padding=True, truncation=True, max_length=8192, return_tensors="pt"
        )
        batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}

        start = time.time()
        with torch.no_grad():
            outputs = model(**batch_dict)
        latency = (time.time() - start) * 1000

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
        embeddings = F.normalize(embeddings, p=2, dim=1)

        print(f"\nEmbeddings computed in {latency:.2f}ms")
        print(f"Embeddings shape: {embeddings.shape}")

        # Split and compute similarity
        query_embeddings = embeddings[: len(queries)]
        doc_embeddings = embeddings[len(queries) :]

        scores = (query_embeddings @ doc_embeddings.T).cpu().numpy()

        print("\nSimilarity Matrix:")
        print("                 Doc1    Doc2    Doc3")
        for i, query_text in enumerate(["China capital", "Gravity", "ML basics"]):
            print(f"{query_text:15} {scores[i, 0]:.4f}  {scores[i, 1]:.4f}  {scores[i, 2]:.4f}")

        result = {
            "model": model_name,
            "latency_ms": latency,
            "embeddings": embeddings.cpu().numpy(),
        }

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        result = None

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print("VRAM after cleanup:", get_vram_usage())

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("MOSEC TESTS WITH PROPER QWEN INSTRUCTIONS")
    print("=" * 70)

    # Ensure fresh start
    stop_mosec_server()
    torch.cuda.empty_cache()

    # Test Mosec with proper instructions
    mosec_results = test_qwen3_embedding_proper()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test Direct with proper instructions
    direct_results = test_qwen3_embedding_direct()

    # Compare if both succeeded
    if (
        mosec_results
        and "embeddings" in mosec_results
        and direct_results
        and "embeddings" in direct_results
    ):
        print("\n" + "=" * 70)
        print("COMPARISON: Mosec vs Direct")
        print("=" * 70)

        mosec_emb = mosec_results["embeddings"]
        direct_emb = direct_results["embeddings"]

        for i in range(len(mosec_emb)):
            sim = cosine_similarity(mosec_emb[i], direct_emb[i])
            text_type = "Query" if i < 3 else "Doc"
            print(f"  {text_type} {i}: {sim:.6f} similarity")

        avg_sim = np.mean(
            [cosine_similarity(mosec_emb[i], direct_emb[i]) for i in range(len(mosec_emb))]
        )
        print(f"\nAverage similarity: {avg_sim:.6f}")

    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)
