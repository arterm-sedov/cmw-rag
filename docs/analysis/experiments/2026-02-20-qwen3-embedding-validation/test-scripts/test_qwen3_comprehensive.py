#!/usr/bin/env python3
"""
Comprehensive Qwen3 Embedding Test Suite
Tests all sizes (0.6B, 4B, 8B) across all providers with Direct GPU baseline.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.spatial.distance import cosine

# Add rag_engine to path
sys.path.insert(0, "/home/asedov/cmw-rag")


@dataclass
class TestResult:
    """Test result container."""

    provider: str
    model_size: str
    model_path: str
    status: str
    dimensions: Optional[int] = None
    load_time_sec: Optional[float] = None
    latency_ms: Optional[float] = None
    cosine_similarity: Optional[float] = None
    mean_relative_error: Optional[float] = None
    error_message: Optional[str] = None
    embedding_sample: Optional[List[float]] = None
    test_text: Optional[str] = None


class Qwen3EmbeddingTester:
    """Comprehensive tester for Qwen3 embeddings."""

    # Test configuration
    TEST_TEXTS = [
        "What is machine learning?",
        "Natural language processing applications",
        "Deep learning neural networks",
        "Artificial intelligence in healthcare",
    ]

    INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

    MODELS = {
        "0.6B": {
            "hf_path": "Qwen/Qwen3-Embedding-0.6B",
            "openrouter_id": None,  # Not available on OpenRouter
            "dimensions": 1024,
        },
        "4B": {
            "hf_path": "Qwen/Qwen3-Embedding-4B",
            "openrouter_id": "qwen/qwen3-embedding-4b",
            "dimensions": 2560,
        },
        "8B": {
            "hf_path": "Qwen/Qwen3-Embedding-8B",
            "openrouter_id": "qwen/qwen3-embedding-8b",
            "dimensions": 4096,
        },
    }

    def __init__(self):
        self.results: List[TestResult] = []
        self.baselines: Dict[str, np.ndarray] = {}  # model_size -> embedding
        self.output_dir = Path(__file__).parent / "data" / "qwen3_comprehensive_tests"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_query(self, text: str) -> str:
        """Format text with Qwen3 instruction format."""
        return f"Instruct: {self.INSTRUCTION}\nQuery: {text}"

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(1 - cosine(emb1, emb2))

    def relative_error(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute mean relative error."""
        norm1 = emb1 / np.linalg.norm(emb1)
        norm2 = emb2 / np.linalg.norm(emb2)
        return float(np.mean(np.abs(norm1 - norm2)))

    # ==================== DIRECT TRANSFORMERS ====================

    def test_direct(self, size: str) -> TestResult:
        """Test with Direct Transformers (GPU baseline)."""
        model_info = self.MODELS[size]
        model_path = model_info["hf_path"]
        expected_dim = model_info["dimensions"]

        print(f"\n{'=' * 60}")
        print(f"Direct Transformers: Qwen3-Embedding-{size}")
        print(f"{'=' * 60}")

        try:
            from transformers import AutoModel, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Device: {device}")

            # Load model
            print("Loading model...")
            start_time = time.time()

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            model.eval()

            load_time = time.time() - start_time
            print(f"✓ Model loaded in {load_time:.2f}s")

            # Test with first text
            test_text = self.TEST_TEXTS[0]
            formatted_text = self.format_query(test_text)

            print(f"\nTesting: '{test_text[:50]}...'")
            print(f"Formatted: '{formatted_text[:70]}...'")

            # Tokenize
            inputs = tokenizer(
                formatted_text, return_tensors="pt", truncation=True, max_length=8192
            ).to(device)

            # Forward pass
            start = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state

                # Last-token pooling
                attention_mask = inputs["attention_mask"]
                last_idx = attention_mask.sum(dim=1) - 1
                embedding = hidden_states[0, last_idx[0], :]
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

            latency = (time.time() - start) * 1000
            embedding_np = embedding.cpu().numpy()

            # Store baseline
            self.baselines[size] = embedding_np

            # Cleanup
            del model, tokenizer
            torch.cuda.empty_cache()

            result = TestResult(
                provider="direct",
                model_size=size,
                model_path=model_path,
                status="SUCCESS",
                dimensions=len(embedding_np),
                load_time_sec=load_time,
                latency_ms=latency,
                test_text=test_text,
                embedding_sample=embedding_np[:5].tolist(),
            )

            print(f"✓ Success: {len(embedding_np)}d, {latency:.1f}ms")
            print(f"  Sample: {embedding_np[:5]}")

            return result

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()

            return TestResult(
                provider="direct",
                model_size=size,
                model_path=model_path,
                status="FAILED",
                error_message=str(e),
            )

    # ==================== vLLM ====================

    def test_vllm(self, size: str, gpu_util: float = 0.25) -> TestResult:
        """Test with vLLM."""
        model_info = self.MODELS[size]
        model_path = model_info["hf_path"]
        expected_dim = model_info["dimensions"]

        print(f"\n{'=' * 60}")
        print(f"vLLM: Qwen3-Embedding-{size}")
        print(f"GPU Utilization: {gpu_util * 100:.0f}%")
        print(f"{'=' * 60}")

        try:
            from vllm import LLM

            # Load model
            print("Loading model...")
            start_time = time.time()

            llm = LLM(
                model=model_path,
                gpu_memory_utilization=gpu_util,
                enforce_eager=True,
                max_model_len=8192,
                trust_remote_code=True,
            )

            load_time = time.time() - start_time
            print(f"✓ Model loaded in {load_time:.2f}s")

            # Test with first text
            test_text = self.TEST_TEXTS[0]
            formatted_text = self.format_query(test_text)

            print(f"\nTesting: '{test_text[:50]}...'")
            print(f"Formatted: '{formatted_text[:70]}...'")

            # Get embedding
            start = time.time()
            output = llm.embed(formatted_text)
            latency = (time.time() - start) * 1000

            embedding = output[0].outputs.embedding
            embedding_np = np.array(embedding)

            # Compare to baseline if available
            similarity = None
            rel_error = None
            if size in self.baselines:
                baseline = self.baselines[size]
                if len(embedding_np) == len(baseline):
                    similarity = self.cosine_similarity(embedding_np, baseline)
                    rel_error = self.relative_error(embedding_np, baseline)
                    print(f"  Cosine Similarity to Direct: {similarity:.8f}")
                    print(f"  Mean Relative Error: {rel_error:.8f}")

            # Cleanup
            del llm
            torch.cuda.empty_cache()

            result = TestResult(
                provider="vllm",
                model_size=size,
                model_path=model_path,
                status="SUCCESS",
                dimensions=len(embedding_np),
                load_time_sec=load_time,
                latency_ms=latency,
                cosine_similarity=similarity,
                mean_relative_error=rel_error,
                test_text=test_text,
                embedding_sample=embedding_np[:5].tolist(),
            )

            print(f"✓ Success: {len(embedding_np)}d, {latency:.1f}ms")
            print(f"  Sample: {embedding_np[:5]}")

            return result

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()

            return TestResult(
                provider="vllm",
                model_size=size,
                model_path=model_path,
                status="FAILED",
                error_message=str(e),
            )

    # ==================== OpenRouter ====================

    def test_openrouter(self, size: str) -> TestResult:
        """Test with OpenRouter API."""
        model_info = self.MODELS[size]
        openrouter_id = model_info["openrouter_id"]
        model_path = model_info["hf_path"]

        if not openrouter_id:
            print(f"\n{'=' * 60}")
            print(f"OpenRouter: Qwen3-Embedding-{size}")
            print(f"{'=' * 60}")
            print(f"✗ Skipped: Not available on OpenRouter")
            return TestResult(
                provider="openrouter",
                model_size=size,
                model_path=model_path,
                status="SKIPPED",
                error_message="Not available on OpenRouter",
            )

        print(f"\n{'=' * 60}")
        print(f"OpenRouter: {openrouter_id}")
        print(f"{'=' * 60}")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("✗ Error: OPENROUTER_API_KEY not set")
            return TestResult(
                provider="openrouter",
                model_size=size,
                model_path=model_path,
                status="FAILED",
                error_message="OPENROUTER_API_KEY not set",
            )

        try:
            from openai import OpenAI

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            # Test with first text
            test_text = self.TEST_TEXTS[0]
            formatted_text = self.format_query(test_text)

            print(f"Testing: '{test_text[:50]}...'")
            print(f"Formatted: '{formatted_text[:70]}...'")

            # Get embedding
            start = time.time()
            response = client.embeddings.create(
                model=openrouter_id,
                input=formatted_text,
                encoding_format="float",
                extra_headers={
                    "HTTP-Referer": "https://localhost",
                    "X-Title": "CMW-RAG Test",
                },
            )
            latency = (time.time() - start) * 1000

            embedding = response.data[0].embedding
            embedding_np = np.array(embedding)

            # Compare to baseline if available
            similarity = None
            rel_error = None
            if size in self.baselines:
                baseline = self.baselines[size]
                if len(embedding_np) == len(baseline):
                    similarity = self.cosine_similarity(embedding_np, baseline)
                    rel_error = self.relative_error(embedding_np, baseline)
                    print(f"  Cosine Similarity to Direct: {similarity:.8f}")
                    print(f"  Mean Relative Error: {rel_error:.8f}")

            result = TestResult(
                provider="openrouter",
                model_size=size,
                model_path=openrouter_id,
                status="SUCCESS",
                dimensions=len(embedding_np),
                latency_ms=latency,
                cosine_similarity=similarity,
                mean_relative_error=rel_error,
                test_text=test_text,
                embedding_sample=embedding_np[:5].tolist(),
            )

            print(f"✓ Success: {len(embedding_np)}d, {latency:.1f}ms")
            print(f"  Sample: {embedding_np[:5]}")

            return result

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()

            return TestResult(
                provider="openrouter",
                model_size=size,
                model_path=openrouter_id,
                status="FAILED",
                error_message=str(e),
            )

    # ==================== MAIN TEST RUNNER ====================

    def run_all_tests(self):
        """Run comprehensive tests for all models and providers."""
        print("=" * 60)
        print("COMPREHENSIVE QWEN3 EMBEDDING TEST SUITE")
        print("=" * 60)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"Test Texts: {len(self.TEST_TEXTS)}")
        print()

        # Test each model size
        for size in ["0.6B", "4B", "8B"]:
            print(f"\n{'#' * 60}")
            print(f"# Testing Model Size: {size}")
            print(f"{'#' * 60}")

            # 1. Direct Transformers (baseline)
            direct_result = self.test_direct(size)
            self.results.append(direct_result)

            if direct_result.status == "SUCCESS":
                # 2. vLLM
                time.sleep(2)  # Allow GPU memory to settle
                torch.cuda.empty_cache()

                # Adjust GPU util based on model size
                gpu_util = 0.20 if size == "8B" else 0.25
                vllm_result = self.test_vllm(size, gpu_util=gpu_util)
                self.results.append(vllm_result)

                # 3. OpenRouter
                time.sleep(1)
                openrouter_result = self.test_openrouter(size)
                self.results.append(openrouter_result)
            else:
                print(f"\n⚠️  Skipping vLLM and OpenRouter tests - Direct failed")

        # Save results
        self.save_results()
        self.print_summary()

    def save_results(self):
        """Save test results to JSON."""
        output_file = self.output_dir / f"test_results_{time.strftime('%Y%m%d_%H%M%S')}.json"

        # Convert results to dict
        results_dict = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_texts": self.TEST_TEXTS,
            "instruction": self.INSTRUCTION,
            "results": [asdict(r) for r in self.results],
        }

        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        # Group by provider and model
        summary = {}
        for result in self.results:
            key = f"{result.provider}-{result.model_size}"
            summary[key] = result

        # Print table
        print(
            f"\n{'Provider':<15} {'Size':<8} {'Status':<10} {'Dims':<8} {'Latency':<12} {'Similarity':<12}"
        )
        print("-" * 70)

        for provider in ["direct", "vllm", "openrouter"]:
            for size in ["0.6B", "4B", "8B"]:
                key = f"{provider}-{size}"
                if key in summary:
                    r = summary[key]
                    status = r.status
                    dims = r.dimensions if r.dimensions else "-"
                    latency = f"{r.latency_ms:.1f}ms" if r.latency_ms else "-"
                    sim = f"{r.cosine_similarity:.6f}" if r.cosine_similarity else "-"
                    print(
                        f"{provider:<15} {size:<8} {status:<10} {dims:<8} {latency:<12} {sim:<12}"
                    )

        # Count successes
        successes = sum(1 for r in self.results if r.status == "SUCCESS")
        total = len(self.results)

        print(f"\nTotal: {successes}/{total} tests passed")

        # Check accuracy
        print("\n" + "=" * 60)
        print("ACCURACY VALIDATION (vs Direct baseline)")
        print("=" * 60)

        for size in ["0.6B", "4B", "8B"]:
            vllm_key = f"vllm-{size}"
            openrouter_key = f"openrouter-{size}"

            if vllm_key in summary and summary[vllm_key].cosine_similarity:
                sim = summary[vllm_key].cosine_similarity
                status = "✓ PASS" if sim >= 0.99999 else "✗ FAIL"
                print(f"{status} vLLM {size}: {sim:.8f} (threshold: 0.99999)")

            if openrouter_key in summary and summary[openrouter_key].cosine_similarity:
                sim = summary[openrouter_key].cosine_similarity
                status = "✓ PASS" if sim >= 0.99999 else "✗ FAIL"
                print(f"{status} OpenRouter {size}: {sim:.8f} (threshold: 0.99999)")


if __name__ == "__main__":
    tester = Qwen3EmbeddingTester()
    tester.run_all_tests()
