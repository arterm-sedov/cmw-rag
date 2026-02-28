#!/usr/bin/env python3
"""
Test Mosec Server Embeddings
Compares Mosec server embeddings against direct Python ground truth
"""

import pickle
import sys
import time

import numpy as np
import requests


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_mosec_frida():
    """Test FRIDA embeddings via Mosec server."""
    print("\n" + "=" * 60)
    print("Testing FRIDA via Mosec Server")
    print("=" * 60)

    base_url = "http://localhost:7998"

    # Check if server is running
    try:
        response = requests.get(f"{base_url}/metrics", timeout=5)
        print(f"Server status: {response.status_code}")
    except requests.RequestException as e:
        print(f"Server not available at {base_url}: {e}")
        print("Start Mosec server first:")
        print("  cd ~/cmw-mosec && python -m cmw_mosec.cli start --embedding ai-forever/FRIDA")
        return None

    test_texts = [
        "search_query: как работает искусственный интеллект",
        "search_query: machine learning basics",
    ]

    print(f"\nGetting embeddings for {len(test_texts)} texts...")

    mosec_embeddings = []
    for text in test_texts:
        payload = {
            "model": "ai-forever/FRIDA",
            "input": text,
        }

        try:
            response = requests.post(f"{base_url}/v1/embeddings", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            emb = np.array(result["data"][0]["embedding"])
            mosec_embeddings.append(emb)
            print(f"  Shape: {emb.shape}, Norm: {np.linalg.norm(emb):.6f}")

        except Exception as e:
            print(f"Error: {e}")
            return None

    # Compare with ground truth
    try:
        with open("/tmp/frida_direct.pkl", "rb") as f:
            ground_truth_direct = pickle.load(f)

        with open("/tmp/frida_st.pkl", "rb") as f:
            ground_truth_st = pickle.load(f)

        print("\nComparison with Direct Transformers (CLS pooling):")
        for i in range(len(mosec_embeddings)):
            sim = cosine_similarity(mosec_embeddings[i], ground_truth_direct["cls_embeddings"][i])
            print(f"  Text {i}: {sim:.6f}")

        print("\nComparison with Sentence-Transformers:")
        for i in range(len(mosec_embeddings)):
            sim = cosine_similarity(mosec_embeddings[i], ground_truth_st["embeddings"][i])
            print(f"  Text {i}: {sim:.6f}")

    except FileNotFoundError as e:
        print(f"Ground truth not found: {e}")

    return mosec_embeddings


def test_mosec_reranker():
    """Test DiTy reranker via Mosec server."""
    print("\n" + "=" * 60)
    print("Testing DiTy Reranker via Mosec Server")
    print("=" * 60)

    base_url = "http://localhost:7998"

    query = "machine learning"
    docs = [
        "ML is a subset of AI",
        "Python is a programming language",
        "Deep learning is part of machine learning",
    ]

    payload = {
        "query": query,
        "docs": docs,
    }

    try:
        response = requests.post(f"{base_url}/v1/rerank", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        print(f"Query: '{query}'")
        print("Scores:")
        for i, (doc, score) in enumerate(zip(docs, result["scores"])):
            print(f"  {i}: {score:.4f} - {doc}")

        return result["scores"]

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    print("Mosec Server Tests")
    print("==================")

    frida_results = test_mosec_frida()
    reranker_results = test_mosec_reranker()

    print("\n" + "=" * 60)
    print("Mosec Tests Complete")
    print("=" * 60)
