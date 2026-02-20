#!/usr/bin/env python3
"""
Direct Python embedding tests - Ground Truth
Tests embeddings using pure transformers/sentence-transformers (no servers)

Models tested:
- ai-forever/FRIDA (T5-based, requires CLS pooling)
- Qwen/Qwen3-Embedding-0.6B (Qwen architecture)

Output: Saved embeddings for comparison with server backends
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# Test texts
TEST_QUERIES = [
    "search_query: как работает искусственный интеллект",
    "search_query: machine learning basics",
    "search_query: python programming language",
]

TEST_DOCS = [
    "search_document: Искусственный интеллект (ИИ) — это область компьютерных наук, которая занимается созданием систем, способных выполнять задачи, требующие человеческого интеллекта.",
    "search_document: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
    "search_document: Python is a high-level, interpreted programming language known for its readability and versatility.",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_frida_direct():
    """Test FRIDA with direct transformers (AutoModel)."""
    print("\n" + "=" * 60)
    print("Testing FRIDA - Direct Transformers (AutoModel)")
    print("=" * 60)

    model_name = "ai-forever/FRIDA"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print(f"Model type: {type(model).__name__}")
    print(f"Has encoder attribute: {hasattr(model, 'encoder')}")

    def cls_pooling(model_output):
        """CLS pooling - use first token."""
        return model_output[0][:, 0, :]

    def mean_pooling(model_output, attention_mask):
        """Mean pooling."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embeddings(texts, use_cls=True):
        """Get embeddings with specified pooling."""
        if isinstance(texts, str):
            texts = [texts]

        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            if hasattr(model, "encoder"):
                outputs = model.encoder(**inputs)
            else:
                outputs = model(**inputs)

        if use_cls:
            embeddings = cls_pooling(outputs)
        else:
            embeddings = mean_pooling(outputs, inputs["attention_mask"])

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    # Test both pooling methods
    print("\nTesting CLS pooling (recommended for FRIDA):")
    cls_embeddings = []
    for text in TEST_QUERIES[:2]:
        emb = get_embeddings(text, use_cls=True)
        cls_embeddings.append(emb[0])
        print(f"  Shape: {emb.shape}, Norm: {np.linalg.norm(emb[0]):.6f}")

    print("\nTesting mean pooling:")
    mean_embeddings = []
    for text in TEST_QUERIES[:2]:
        emb = get_embeddings(text, use_cls=False)
        mean_embeddings.append(emb[0])
        print(f"  Shape: {emb.shape}, Norm: {np.linalg.norm(emb[0]):.6f}")

    # Compare CLS vs mean
    sim = cosine_similarity(cls_embeddings[0], mean_embeddings[0])
    print(f"\nCLS vs Mean similarity: {sim:.6f}")

    # Save reference embeddings
    results = {
        "model": model_name,
        "method": "transformers_automodel",
        "device": device,
        "cls_embeddings": cls_embeddings,
        "mean_embeddings": mean_embeddings,
        "test_texts": TEST_QUERIES[:2],
    }

    output_file = Path("/tmp/frida_direct.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved to {output_file}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def test_frida_sentence_transformers():
    """Test FRIDA with sentence-transformers."""
    print("\n" + "=" * 60)
    print("Testing FRIDA - Sentence Transformers")
    print("=" * 60)

    model_name = "ai-forever/FRIDA"

    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model loaded on {model.device}")

    # Get embeddings
    print("\nGetting embeddings...")
    embeddings = model.encode(TEST_QUERIES[:2], normalize_embeddings=True)

    for i, emb in enumerate(embeddings):
        print(f"  Text {i}: Shape={emb.shape}, Norm={np.linalg.norm(emb):.6f}")

    # Save reference
    results = {
        "model": model_name,
        "method": "sentence_transformers",
        "device": str(model.device),
        "embeddings": embeddings,
        "test_texts": TEST_QUERIES[:2],
    }

    output_file = Path("/tmp/frida_st.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved to {output_file}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def test_qwen3_embedding_direct():
    """Test Qwen3-Embedding with direct transformers."""
    print("\n" + "=" * 60)
    print("Testing Qwen3-Embedding-0.6B - Direct Transformers")
    print("=" * 60)

    model_name = "Qwen/Qwen3-Embedding-0.6B"

    print(f"Loading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Model may not be downloaded yet.")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

    def get_embeddings(texts):
        """Get embeddings (try different pooling methods)."""
        if isinstance(texts, str):
            texts = [texts]

        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Try last token pooling (common for Qwen)
        # Get the last non-padding token for each sequence
        attention_mask = inputs["attention_mask"]
        last_hidden = outputs.last_hidden_state

        embeddings = []
        for i in range(len(texts)):
            # Find last non-padding token
            seq_len = attention_mask[i].sum().item()
            # Use last token
            last_token_emb = last_hidden[i, seq_len - 1, :]
            embeddings.append(last_token_emb)

        embeddings = torch.stack(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    print("\nGetting embeddings (last token pooling)...")
    embeddings = []
    for text in TEST_QUERIES[:2]:
        emb = get_embeddings(text)
        embeddings.append(emb[0])
        print(f"  Shape: {emb.shape}, Norm: {np.linalg.norm(emb[0]):.6f}")

    # Save reference
    results = {
        "model": model_name,
        "method": "transformers_automodel_lasttoken",
        "device": device,
        "embeddings": embeddings,
        "test_texts": TEST_QUERIES[:2],
    }

    output_file = Path("/tmp/qwen3_direct.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved to {output_file}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def compare_embeddings():
    """Compare all saved embeddings."""
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    # Load FRIDA results
    try:
        with open("/tmp/frida_direct.pkl", "rb") as f:
            frida_direct = pickle.load(f)
        with open("/tmp/frida_st.pkl", "rb") as f:
            frida_st = pickle.load(f)

        print("\nFRIDA - Direct vs Sentence-Transformers:")
        for i in range(len(frida_direct["cls_embeddings"])):
            sim = cosine_similarity(frida_direct["cls_embeddings"][i], frida_st["embeddings"][i])
            print(f"  Text {i}: {sim:.6f}")
    except FileNotFoundError:
        print("FRIDA results not found")


if __name__ == "__main__":
    print("Direct Python Embedding Tests - Ground Truth")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Run tests
    frida_direct_results = test_frida_direct()
    frida_st_results = test_frida_sentence_transformers()
    qwen3_results = test_qwen3_embedding_direct()

    # Compare
    compare_embeddings()

    print("\n" + "=" * 60)
    print("Tests complete. Reference embeddings saved to /tmp/")
    print("=" * 60)
