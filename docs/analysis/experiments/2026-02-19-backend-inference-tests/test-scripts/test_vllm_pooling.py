#!/usr/bin/env python3
"""
vLLM Pooling Test
Tests vLLM's embedding capabilities and compares with ground truth

vLLM 0.15.0+ supports pooling models via:
- --task embed (for embeddings)
- --task score (for reranking)
- --task classify (for classification)

Default pooling: LAST token for embeddings
"""

import pickle
from pathlib import Path

import numpy as np
import torch


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_vllm_frida():
    """Test FRIDA with vLLM offline inference."""
    print("\n" + "=" * 60)
    print("Testing FRIDA with vLLM")
    print("=" * 60)

    try:
        from vllm import LLM
        from vllm.sampling_params import SamplingParams
    except ImportError as e:
        print(f"vLLM not available: {e}")
        return None

    model_name = "ai-forever/FRIDA"

    print(f"Loading {model_name} with vLLM...")
    print("Note: vLLM uses LAST token pooling by default for embeddings")

    try:
        # vLLM for embeddings uses runner="pooling"
        # See: https://docs.vllm.ai/en/latest/models/pooling_models/
        llm = LLM(
            model=model_name,
            runner="pooling",
            trust_remote_code=True,
            dtype="float16",
        )
    except Exception as e:
        print(f"Error loading with vLLM: {e}")
        print("FRIDA may not be natively supported by vLLM")
        return None

    test_texts = [
        "search_query: как работает искусственный интеллект",
        "search_query: machine learning basics",
    ]

    print("\nGetting embeddings...")
    try:
        outputs = llm.embed(test_texts)

        vllm_embeddings = []
        for i, output in enumerate(outputs):
            emb = output.outputs.embedding
            vllm_embeddings.append(np.array(emb))
            print(f"  Text {i}: Shape={len(emb)}, Norm={np.linalg.norm(emb):.6f}")

        # Compare with ground truth
        try:
            with open("/tmp/frida_direct.pkl", "rb") as f:
                ground_truth = pickle.load(f)

            print("\nComparison with Direct Transformers (CLS pooling):")
            for i in range(len(vllm_embeddings)):
                sim = cosine_similarity(vllm_embeddings[i], ground_truth["cls_embeddings"][i])
                print(f"  Text {i}: {sim:.6f}")
        except FileNotFoundError:
            print("Ground truth not found, run test_direct_embeddings.py first")

        return vllm_embeddings

    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None


def test_vllm_qwen3():
    """Test Qwen3-Embedding with vLLM."""
    print("\n" + "=" * 60)
    print("Testing Qwen3-Embedding-0.6B with vLLM")
    print("=" * 60)

    try:
        from vllm import LLM
    except ImportError:
        print("vLLM not available")
        return None

    model_name = "Qwen/Qwen3-Embedding-0.6B"

    print(f"Loading {model_name} with vLLM...")

    try:
        llm = LLM(
            model=model_name,
            runner="pooling",
            trust_remote_code=True,
            dtype="float16",
        )
    except Exception as e:
        print(f"Error loading with vLLM: {e}")
        return None

    test_texts = [
        "search_query: как работает искусственный интеллект",
        "search_query: machine learning basics",
    ]

    print("\nGetting embeddings...")
    try:
        outputs = llm.embed(test_texts)

        vllm_embeddings = []
        for i, output in enumerate(outputs):
            emb = output.outputs.embedding
            vllm_embeddings.append(np.array(emb))
            print(f"  Text {i}: Shape={len(emb)}, Norm={np.linalg.norm(emb):.6f}")

        # Compare with ground truth
        try:
            with open("/tmp/qwen3_direct.pkl", "rb") as f:
                ground_truth = pickle.load(f)

            print("\nComparison with Direct Transformers (last token):")
            for i in range(len(vllm_embeddings)):
                sim = cosine_similarity(vllm_embeddings[i], ground_truth["embeddings"][i])
                print(f"  Text {i}: {sim:.6f}")
        except FileNotFoundError:
            print("Ground truth not found")

        return vllm_embeddings

    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None


def test_vllm_reranker():
    """Test reranking with vLLM."""
    print("\n" + "=" * 60)
    print("Testing Reranker with vLLM (score task)")
    print("=" * 60)

    try:
        from vllm import LLM
    except ImportError:
        print("vLLM not available")
        return None

    # Test DiTy reranker
    model_name = "DiTy/cross-encoder-russian-msmarco"

    print(f"Testing {model_name}...")
    print("Note: Cross-encoder rerankers may not be supported by vLLM pooling")

    try:
        llm = LLM(
            model=model_name,
            runner="pooling",
            trust_remote_code=True,
            dtype="float16",
        )
        print("Model loaded successfully with task='score'")

        # Test scoring
        query = "machine learning"
        docs = ["ML is a subset of AI", "Python is a programming language"]

        # vLLM score task expects specific format
        print(f"\nScoring query: '{query}'")
        print(f"Documents: {docs}")
        print("(Scoring test would go here - format TBD)")

    except Exception as e:
        print(f"Error: {e}")
        print("Cross-encoder rerankers likely not supported")
        return None


def check_vllm_pooling_config():
    """Check vLLM pooling configuration options."""
    print("\n" + "=" * 60)
    print("vLLM Pooling Configuration")
    print("=" * 60)

    try:
        from vllm import LLM
        from vllm.pooling_params import PoolingParams
        import inspect

        print("Available pooling options:")
        sig = inspect.signature(PoolingParams.__init__)
        params = [p for p in sig.parameters.keys() if p != "self"]
        print(f"  PoolingParams params: {params}")

    except ImportError as e:
        print(f"Cannot import vLLM pooling: {e}")


if __name__ == "__main__":
    print("vLLM Pooling Tests")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Check vLLM pooling config first
    check_vllm_pooling_config()

    # Test FRIDA
    frida_results = test_vllm_frida()

    # Clear cache between tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test Qwen3
    qwen3_results = test_vllm_qwen3()

    # Test reranker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reranker_results = test_vllm_reranker()

    print("\n" + "=" * 60)
    print("vLLM Tests Complete")
    print("=" * 60)
