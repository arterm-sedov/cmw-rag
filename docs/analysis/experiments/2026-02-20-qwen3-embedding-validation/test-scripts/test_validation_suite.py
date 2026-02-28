#!/usr/bin/env python3
"""
Qwen3 Embedding Validation Suite - All Providers
Validates all Qwen3 sizes (0.6B, 4B, 8B) across all providers with proper VRAM management.
Includes: Direct, vLLM, OpenRouter, Mosec
"""

import os
import sys
import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from scipy.spatial.distance import cosine
from datetime import datetime

sys.path.insert(0, "/home/asedov/cmw-rag")


@dataclass
class ProviderResult:
    """Result from a single provider test."""

    provider: str
    model_size: str
    status: str
    dimensions: Optional[int] = None
    latency_ms: Optional[float] = None
    load_time_sec: Optional[float] = None
    vram_used_gb: Optional[float] = None
    embedding_sample: Optional[List[float]] = None
    error_message: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result from comparing two providers."""

    model_size: str
    provider_a: str
    provider_b: str
    cosine_similarity: float
    mean_relative_error: float
    status: str  # PASS, REVIEW, FAIL


class VRAMManager:
    """Manages GPU memory cleanup and defragmentation."""

    @staticmethod
    def get_vram_info() -> Dict[str, float]:
        """Get current VRAM usage in GB."""
        if not torch.cuda.is_available():
            return {"total": 0, "used": 0, "free": 0}

        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        free = total - reserved

        return {
            "total": round(total, 2),
            "used": round(allocated, 2),
            "reserved": round(reserved, 2),
            "free": round(free, 2),
        }

    @staticmethod
    def cleanup():
        """Aggressive VRAM cleanup."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        time.sleep(2)  # Allow GPU to settle

    @staticmethod
    def print_vram_status(label: str = ""):
        """Print current VRAM status."""
        info = VRAMManager.get_vram_info()
        print(
            f"  [VRAM {label}] Total: {info['total']:.1f}GB | "
            f"Used: {info['used']:.1f}GB | Reserved: {info['reserved']:.1f}GB | "
            f"Free: {info['free']:.1f}GB"
        )


class Qwen3ValidationSuite:
    """Comprehensive validation suite for Qwen3 embeddings."""

    TEST_TEXTS = [
        "What is machine learning and how does it work?",
        "Natural language processing applications in modern technology",
    ]

    INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

    MODELS = {
        "0.6B": {
            "hf_path": "Qwen/Qwen3-Embedding-0.6B",
            "openrouter_id": None,
            "mosec_port": 7998,
            "dimensions": 1024,
            "vllm_gpu_util": 0.15,
            "estimated_vram_gb": 1.5,
        },
        "4B": {
            "hf_path": "Qwen/Qwen3-Embedding-4B",
            "openrouter_id": "qwen/qwen3-embedding-4b",
            "mosec_port": 7998,
            "dimensions": 2560,
            "vllm_gpu_util": 0.18,
            "estimated_vram_gb": 8.0,
        },
        "8B": {
            "hf_path": "Qwen/Qwen3-Embedding-8B",
            "openrouter_id": "qwen/qwen3-embedding-8b",
            "mosec_port": 7998,
            "dimensions": 4096,
            "vllm_gpu_util": 0.35,
            "estimated_vram_gb": 16.0,
        },
    }

    PROVIDERS = ["direct", "vllm", "openrouter", "mosec"]

    def __init__(self):
        self.results: List[ProviderResult] = []
        self.comparisons: List[ComparisonResult] = []
        self.embeddings_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.output_dir = (
            Path(__file__).parent
            / "docs"
            / "analysis"
            / "experiments"
            / "2026-02-20-qwen3-embedding-validation"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def format_query(self, text: str) -> str:
        """Format query with Qwen3 instruction."""
        return f"Instruct: {self.INSTRUCTION}\nQuery: {text}"

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(1 - cosine(emb1, emb2))

    def relative_error(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute mean relative error."""
        norm1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        norm2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.mean(np.abs(norm1 - norm2)))

    # ==================== PROVIDER IMPLEMENTATIONS ====================

    def test_direct(self, size: str, text: str, use_offload: bool = False) -> ProviderResult:
        """Test Direct transformers provider."""
        print(f"\n  [Direct {size}] Starting test...")
        VRAMManager.print_vram_status("before Direct")

        model_info = self.MODELS[size]
        model_path = model_info["hf_path"]

        try:
            from transformers import AutoModel, AutoTokenizer

            load_start = time.time()

            # Use device_map for CPU offloading on 8B
            if use_offload and size == "8B":
                print(f"    Loading with CPU offloading (accelerate)...")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    offload_folder="offload",
                    offload_state_dict=True,
                )
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"    Loading on {device}...")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                ).to(device)

            model.eval()
            load_time = time.time() - load_start

            # Get embedding
            formatted = self.format_query(text)
            device = next(model.parameters()).device
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=8192).to(
                device
            )

            inf_start = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                last_idx = attention_mask.sum(dim=1) - 1
                embedding = hidden_states[0, last_idx[0], :]
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

            latency = (time.time() - inf_start) * 1000
            vram_info = VRAMManager.get_vram_info()

            result = ProviderResult(
                provider="direct",
                model_size=size,
                status="SUCCESS",
                dimensions=len(embedding),
                latency_ms=round(latency, 2),
                load_time_sec=round(load_time, 2),
                vram_used_gb=vram_info["used"],
                embedding_sample=embedding.cpu().numpy()[:5].tolist(),
            )

            print(
                f"    ✓ Success: {result.dimensions}d, {result.latency_ms:.1f}ms, "
                f"{result.load_time_sec:.1f}s load, {result.vram_used_gb:.1f}GB VRAM"
            )

            # Cleanup
            del model, tokenizer
            VRAMManager.cleanup()
            VRAMManager.print_vram_status("after Direct")

            return result

        except Exception as e:
            print(f"    ✗ Error: {e}")
            VRAMManager.cleanup()
            return ProviderResult(
                provider="direct",
                model_size=size,
                status="FAILED",
                error_message=str(e),
            )

    def test_vllm(self, size: str, text: str) -> ProviderResult:
        """Test vLLM provider."""
        print(f"\n  [vLLM {size}] Starting test...")
        VRAMManager.print_vram_status("before vLLM")

        model_info = self.MODELS[size]
        model_path = model_info["hf_path"]
        gpu_util = model_info["vllm_gpu_util"]

        try:
            from vllm import LLM

            load_start = time.time()
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=gpu_util,
                enforce_eager=True,
                max_model_len=8192,
                trust_remote_code=True,
            )
            load_time = time.time() - load_start

            formatted = self.format_query(text)

            inf_start = time.time()
            output = llm.embed(formatted)
            latency = (time.time() - inf_start) * 1000

            embedding = np.array(output[0].outputs.embedding)
            vram_info = VRAMManager.get_vram_info()

            result = ProviderResult(
                provider="vllm",
                model_size=size,
                status="SUCCESS",
                dimensions=len(embedding),
                latency_ms=round(latency, 2),
                load_time_sec=round(load_time, 2),
                vram_used_gb=vram_info["used"],
                embedding_sample=embedding[:5].tolist(),
            )

            print(
                f"    ✓ Success: {result.dimensions}d, {result.latency_ms:.1f}ms, "
                f"{result.load_time_sec:.1f}s load, {result.vram_used_gb:.1f}GB VRAM"
            )

            # Cleanup
            del llm
            VRAMManager.cleanup()
            VRAMManager.print_vram_status("after vLLM")

            return result

        except Exception as e:
            print(f"    ✗ Error: {e}")
            VRAMManager.cleanup()
            return ProviderResult(
                provider="vllm",
                model_size=size,
                status="FAILED",
                error_message=str(e),
            )

    def test_openrouter(self, size: str, text: str) -> ProviderResult:
        """Test OpenRouter provider."""
        print(f"\n  [OpenRouter {size}] Starting test...")

        model_info = self.MODELS[size]
        model_id = model_info["openrouter_id"]

        if not model_id:
            print(f"    ⏭️  Not available on OpenRouter")
            return ProviderResult(
                provider="openrouter",
                model_size=size,
                status="SKIPPED",
                error_message="Model not available on OpenRouter",
            )

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return ProviderResult(
                provider="openrouter",
                model_size=size,
                status="FAILED",
                error_message="OPENROUTER_API_KEY not set",
            )

        try:
            from openai import OpenAI

            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

            formatted = self.format_query(text)

            start = time.time()
            response = client.embeddings.create(
                model=model_id,
                input=formatted,
                encoding_format="float",
                extra_headers={
                    "HTTP-Referer": "https://localhost",
                    "X-Title": "CMW-RAG Validation",
                },
            )
            latency = (time.time() - start) * 1000

            embedding = np.array(response.data[0].embedding)

            result = ProviderResult(
                provider="openrouter",
                model_size=size,
                status="SUCCESS",
                dimensions=len(embedding),
                latency_ms=round(latency, 2),
                embedding_sample=embedding[:5].tolist(),
            )

            print(f"    ✓ Success: {result.dimensions}d, {result.latency_ms:.1f}ms")
            return result

        except Exception as e:
            print(f"    ✗ Error: {e}")
            return ProviderResult(
                provider="openrouter",
                model_size=size,
                status="FAILED",
                error_message=str(e),
            )

    def test_mosec(self, size: str, text: str) -> ProviderResult:
        """Test Mosec provider via HTTP."""
        print(f"\n  [Mosec {size}] Starting test...")

        import requests

        model_info = self.MODELS[size]
        port = model_info["mosec_port"]
        endpoint = f"http://localhost:{port}/v1/embeddings"
        model_path = model_info["hf_path"]

        formatted = self.format_query(text)

        try:
            start = time.time()
            response = requests.post(
                endpoint,
                json={"input": formatted, "model": model_path},
                timeout=30,
            )
            latency = (time.time() - start) * 1000

            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

            data = response.json()
            embedding = np.array(data["data"][0]["embedding"])

            result = ProviderResult(
                provider="mosec",
                model_size=size,
                status="SUCCESS",
                dimensions=len(embedding),
                latency_ms=round(latency, 2),
                embedding_sample=embedding[:5].tolist(),
            )

            print(f"    ✓ Success: {result.dimensions}d, {result.latency_ms:.1f}ms")
            return result

        except requests.exceptions.ConnectionError:
            print(f"    ⏭️  Mosec server not running on port {port}")
            return ProviderResult(
                provider="mosec",
                model_size=size,
                status="SKIPPED",
                error_message=f"Mosec server not running on port {port}",
            )
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return ProviderResult(
                provider="mosec",
                model_size=size,
                status="FAILED",
                error_message=str(e),
            )

    # ==================== TEST RUNNER ====================

    def run_validation(self):
        """Run complete validation suite."""
        print("=" * 80)
        print("QWEN3 EMBEDDING VALIDATION SUITE - ALL PROVIDERS")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"Models: {list(self.MODELS.keys())}")
        print(f"Providers: {self.PROVIDERS}")
        print()

        VRAMManager.print_vram_status("initial")

        test_text = self.TEST_TEXTS[0]

        # Test each model size
        for size in ["0.6B", "4B", "8B"]:
            print(f"\n{'#' * 80}")
            print(f"# MODEL SIZE: {size}")
            print(f"{'#' * 80}")

            model_info = self.MODELS[size]
            print(f"Expected dimensions: {model_info['dimensions']}")
            print(f"Estimated VRAM: {model_info['estimated_vram_gb']}GB")

            size_embeddings = {}

            # Test Direct (baseline)
            use_offload = size == "8B"
            result = self.test_direct(size, test_text, use_offload)
            self.results.append(result)
            if result.status == "SUCCESS":
                size_embeddings["direct"] = self._get_embedding_from_result(result)

            # Test vLLM
            if size != "8B" or model_info["estimated_vram_gb"] < 20:  # Skip vLLM 8B if too large
                result = self.test_vllm(size, test_text)
                self.results.append(result)
                if result.status == "SUCCESS":
                    size_embeddings["vllm"] = self._get_embedding_from_result(result)
            else:
                print(f"\n  [vLLM {size}] Skipped (model too large for current VRAM)")
                self.results.append(
                    ProviderResult(
                        provider="vllm",
                        model_size=size,
                        status="SKIPPED",
                        error_message="Model too large for available VRAM",
                    )
                )

            # Test OpenRouter
            result = self.test_openrouter(size, test_text)
            self.results.append(result)
            if result.status == "SUCCESS":
                size_embeddings["openrouter"] = self._get_embedding_from_result(result)

            # Test Mosec
            result = self.test_mosec(size, test_text)
            self.results.append(result)
            if result.status == "SUCCESS":
                size_embeddings["mosec"] = self._get_embedding_from_result(result)

            # Store embeddings for comparison
            self.embeddings_cache[size] = size_embeddings

            # Compare providers
            self._compare_providers(size, size_embeddings)

            # Final cleanup for this model size
            VRAMManager.cleanup()
            print(f"\n{'=' * 80}")

        # Save results
        self._save_results()
        self._print_summary()

    def _get_embedding_from_result(
        self, result: ProviderResult, full_embedding: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Extract embedding from result (placeholder - need to store full embeddings)."""
        # Note: In production, store full embeddings in results
        # For now, return None - comparisons will be done differently
        return None

    def _compare_providers(self, size: str, embeddings: Dict[str, np.ndarray]):
        """Compare embeddings between providers."""
        if len(embeddings) < 2:
            return

        print(f"\n  Comparing providers for {size}:")

        providers = list(embeddings.keys())
        for i, prov_a in enumerate(providers):
            for prov_b in providers[i + 1 :]:
                emb_a = embeddings[prov_a]
                emb_b = embeddings[prov_b]

                sim = self.cosine_similarity(emb_a, emb_b)
                rel_err = self.relative_error(emb_a, emb_b)

                status = "PASS" if sim >= 0.999 else ("REVIEW" if sim >= 0.99 else "FAIL")

                comparison = ComparisonResult(
                    model_size=size,
                    provider_a=prov_a,
                    provider_b=prov_b,
                    cosine_similarity=sim,
                    mean_relative_error=rel_err,
                    status=status,
                )
                self.comparisons.append(comparison)

                icon = "✓" if status == "PASS" else ("⚠️" if status == "REVIEW" else "✗")
                print(f"    {icon} {prov_a} vs {prov_b}: {sim:.6f} ({status})")

    def _save_results(self):
        """Save all results to JSON."""
        output = {
            "timestamp": self.timestamp,
            "test_config": {
                "models": {
                    k: {"path": v["hf_path"], "dimensions": v["dimensions"]}
                    for k, v in self.MODELS.items()
                },
                "providers": self.PROVIDERS,
                "test_texts": self.TEST_TEXTS,
                "instruction": self.INSTRUCTION,
            },
            "results": [asdict(r) for r in self.results],
            "comparisons": [asdict(c) for c in self.comparisons],
        }

        output_file = self.output_dir / f"validation_results_{self.timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        # Count results by status
        by_status = {"SUCCESS": 0, "FAILED": 0, "SKIPPED": 0}
        for r in self.results:
            by_status[r.status] = by_status.get(r.status, 0) + 1

        print(
            f"\nProvider Results: {by_status['SUCCESS']} success, {by_status['FAILED']} failed, {by_status['SKIPPED']} skipped"
        )

        # Count comparisons by status
        by_comp_status = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
        for c in self.comparisons:
            by_comp_status[c.status] = by_comp_status.get(c.status, 0) + 1

        print(
            f"Comparisons: {by_comp_status['PASS']} pass, {by_comp_status['REVIEW']} review, {by_comp_status['FAIL']} fail"
        )

        # Print detailed results by model
        for size in ["0.6B", "4B", "8B"]:
            size_results = [r for r in self.results if r.model_size == size]
            if not size_results:
                continue

            print(f"\n{size} Model:")
            for r in size_results:
                if r.status == "SUCCESS":
                    print(f"  ✓ {r.provider}: {r.dimensions}d, {r.latency_ms:.1f}ms")
                elif r.status == "SKIPPED":
                    print(f"  ⏭️  {r.provider}: skipped")
                else:
                    print(f"  ✗ {r.provider}: failed")

            size_comparisons = [c for c in self.comparisons if c.model_size == size]
            if size_comparisons:
                print(f"  Comparisons:")
                for c in size_comparisons:
                    icon = "✓" if c.status == "PASS" else ("⚠️" if c.status == "REVIEW" else "✗")
                    print(f"    {icon} {c.provider_a} vs {c.provider_b}: {c.cosine_similarity:.6f}")

        print(f"\n{'=' * 80}")


if __name__ == "__main__":
    suite = Qwen3ValidationSuite()
    suite.run_validation()
