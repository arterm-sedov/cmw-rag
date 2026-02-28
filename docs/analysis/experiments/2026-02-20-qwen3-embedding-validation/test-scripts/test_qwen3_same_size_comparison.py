#!/usr/bin/env python3
"""
Same-Size Qwen3 Embedding Comparison Across All Providers
Compares 0.6B↔0.6B, 4B↔4B, 8B↔8B with CPU offloading support for 8B
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from scipy.spatial.distance import cosine

# Add rag_engine to path
sys.path.insert(0, "/home/asedov/cmw-rag")


@dataclass
class ComparisonResult:
    """Comparison result between two providers."""

    model_size: str
    provider_a: str
    provider_b: str
    status: str
    cosine_similarity: Optional[float] = None
    mean_relative_error: Optional[float] = None
    dimensions_match: bool = False
    error_message: Optional[str] = None


class SameSizeComparator:
    """Compare embeddings from same model size across providers."""

    TEST_TEXTS = [
        "What is machine learning and how does it work?",
        "Natural language processing applications in industry",
        "Deep learning neural networks architectures explained",
        "Artificial intelligence in healthcare diagnosis",
    ]

    INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

    MODELS = {
        "0.6B": {
            "hf_path": "Qwen/Qwen3-Embedding-0.6B",
            "openrouter_id": None,
            "dimensions": 1024,
            "vllm_gpu_util": 0.20,
        },
        "4B": {
            "hf_path": "Qwen/Qwen3-Embedding-4B",
            "openrouter_id": "qwen/qwen3-embedding-4b",
            "dimensions": 2560,
            "vllm_gpu_util": 0.20,
        },
        "8B": {
            "hf_path": "Qwen/Qwen3-Embedding-8B",
            "openrouter_id": "qwen/qwen3-embedding-8b",
            "dimensions": 4096,
            "vllm_gpu_util": 0.15,  # Lower for 8B
        },
    }

    def __init__(self):
        self.results: List[ComparisonResult] = []
        self.embeddings_cache: Dict[
            str, Dict[str, np.ndarray]
        ] = {}  # size -> {provider: embedding}
        self.output_dir = Path(__file__).parent / "data" / "qwen3_same_size_comparison"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_query(self, text: str) -> str:
        """Format text with Qwen3 instruction format."""
        return f"Instruct: {self.INSTRUCTION}\nQuery: {text}"

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(1 - cosine(emb1, emb2))

    def relative_error(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute mean relative error."""
        norm1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        norm2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.mean(np.abs(norm1 - norm2)))

    def clear_gpu(self):
        """Clear GPU memory."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)

    # ==================== PROVIDER IMPLEMENTATIONS ====================

    def get_direct_embedding(
        self, size: str, text: str, use_cpu_offload: bool = False
    ) -> Optional[np.ndarray]:
        """Get embedding from Direct Transformers."""
        from transformers import AutoModel, AutoTokenizer

        model_info = self.MODELS[size]
        model_path = model_info["hf_path"]

        # For 8B with CPU offloading, use device_map='auto'
        if use_cpu_offload and size == "8B":
            print(f"  [Direct {size}] Loading with CPU offloading...")
            device = "auto"
            device_map = "auto"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device_map = None

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            if device_map == "auto":
                # CPU offloading for large models
                model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    offload_folder="offload",
                )
            else:
                model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                ).to(device)

            model.eval()

            # Tokenize
            formatted = self.format_query(text)
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
            ).to(model.device if hasattr(model, "device") else next(model.parameters()).device)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state

                # Last-token pooling
                attention_mask = inputs["attention_mask"]
                last_idx = attention_mask.sum(dim=1) - 1
                embedding = hidden_states[0, last_idx[0], :]
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

            result = embedding.cpu().numpy()

            # Cleanup
            del model, tokenizer
            self.clear_gpu()

            return result

        except Exception as e:
            print(f"  ✗ Direct {size} error: {e}")
            return None

    def get_vllm_embedding(self, size: str, text: str) -> Optional[np.ndarray]:
        """Get embedding from vLLM."""
        from vllm import LLM

        model_info = self.MODELS[size]
        model_path = model_info["hf_path"]
        gpu_util = model_info["vllm_gpu_util"]

        try:
            # Load model with appropriate GPU utilization
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=gpu_util,
                enforce_eager=True,
                max_model_len=8192,
                trust_remote_code=True,
            )

            # Get embedding
            formatted = self.format_query(text)
            output = llm.embed(formatted)
            embedding = np.array(output[0].outputs.embedding)

            # Cleanup
            del llm
            self.clear_gpu()

            return embedding

        except Exception as e:
            print(f"  ✗ vLLM {size} error: {e}")
            return None

    def get_openrouter_embedding(self, size: str, text: str) -> Optional[np.ndarray]:
        """Get embedding from OpenRouter."""
        from openai import OpenAI

        model_info = self.MODELS[size]
        model_id = model_info["openrouter_id"]

        if not model_id:
            print(f"  ⏭️ OpenRouter {size} not available")
            return None

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print(f"  ✗ OpenRouter API key not set")
            return None

        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            formatted = self.format_query(text)
            response = client.embeddings.create(
                model=model_id,
                input=formatted,
                encoding_format="float",
                extra_headers={
                    "HTTP-Referer": "https://localhost",
                    "X-Title": "CMW-RAG Test",
                },
            )

            embedding = np.array(response.data[0].embedding)
            return embedding

        except Exception as e:
            print(f"  ✗ OpenRouter {size} error: {e}")
            return None

    # ==================== COMPARISON LOGIC ====================

    def compare_providers(self, size: str, text_idx: int):
        """Compare all providers for a given model size."""
        test_text = self.TEST_TEXTS[text_idx]

        print(f"\n{'=' * 70}")
        print(f"Comparing {size} Model - Test Text {text_idx + 1}/{len(self.TEST_TEXTS)}")
        print(f"Text: '{test_text[:60]}...'")
        print(f"{'=' * 70}")

        embeddings = {}
        model_info = self.MODELS[size]
        expected_dim = model_info["dimensions"]

        # 1. Direct Transformers (with CPU offloading for 8B)
        print(f"\n1. Direct Transformers ({size})...")
        use_cpu_offload = size == "8B"
        direct_emb = self.get_direct_embedding(size, test_text, use_cpu_offload)
        if direct_emb is not None:
            embeddings["direct"] = direct_emb
            print(f"   ✓ Direct: {len(direct_emb)}d")

        # 2. vLLM
        if direct_emb is not None:
            print(f"\n2. vLLM ({size})...")
            vllm_emb = self.get_vllm_embedding(size, test_text)
            if vllm_emb is not None:
                embeddings["vllm"] = vllm_emb
                print(f"   ✓ vLLM: {len(vllm_emb)}d")

        # 3. OpenRouter
        print(f"\n3. OpenRouter ({size})...")
        openrouter_emb = self.get_openrouter_embedding(size, test_text)
        if openrouter_emb is not None:
            embeddings["openrouter"] = openrouter_emb
            print(f"   ✓ OpenRouter: {len(openrouter_emb)}d")

        # Compare all pairs
        print(f"\n{'-' * 70}")
        print(f"Comparison Results ({size}):")
        print(f"{'-' * 70}")

        providers = list(embeddings.keys())
        for i, prov_a in enumerate(providers):
            for prov_b in providers[i + 1 :]:
                emb_a = embeddings[prov_a]
                emb_b = embeddings[prov_b]

                if len(emb_a) != len(emb_b):
                    result = ComparisonResult(
                        model_size=size,
                        provider_a=prov_a,
                        provider_b=prov_b,
                        status="DIMENSION_MISMATCH",
                        dimensions_match=False,
                        error_message=f"Dimensions: {len(emb_a)} vs {len(emb_b)}",
                    )
                else:
                    sim = self.cosine_similarity(emb_a, emb_b)
                    rel_err = self.relative_error(emb_a, emb_b)

                    status = "PASS" if sim >= 0.999 else "REVIEW"

                    result = ComparisonResult(
                        model_size=size,
                        provider_a=prov_a,
                        provider_b=prov_b,
                        status=status,
                        cosine_similarity=sim,
                        mean_relative_error=rel_err,
                        dimensions_match=True,
                    )

                    print(f"  {prov_a} vs {prov_b}:")
                    print(f"    Cosine Similarity: {sim:.8f}")
                    print(f"    Mean Relative Error: {rel_err:.8f}")
                    print(f"    Status: {'✓' if status == 'PASS' else '⚠️'} {status}")

                self.results.append(result)

        # Store in cache
        self.embeddings_cache[f"{size}_{text_idx}"] = embeddings

        return len(embeddings) >= 2

    def run_comparison(self):
        """Run complete same-size comparison."""
        print("=" * 70)
        print("SAME-SIZE QWEN3 EMBEDDING COMPARISON")
        print("=" * 70)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"Test Texts: {len(self.TEST_TEXTS)}")
        print()

        # Compare each model size
        for size in ["0.6B", "4B", "8B"]:
            print(f"\n{'#' * 70}")
            print(f"# MODEL SIZE: {size}")
            print(f"{'#' * 70}")

            # Test with multiple texts for robustness
            success_count = 0
            for text_idx in range(min(2, len(self.TEST_TEXTS))):  # Test first 2 texts
                if self.compare_providers(size, text_idx):
                    success_count += 1

            print(f"\n{'=' * 70}")
            print(f"{size} Summary: {success_count}/2 texts successfully compared")
            print(f"{'=' * 70}")

        # Save and summarize
        self.save_results()
        self.print_summary()

    def save_results(self):
        """Save comparison results."""
        output_file = (
            self.output_dir / f"same_size_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )

        results_dict = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_texts": self.TEST_TEXTS,
            "results": [asdict(r) for r in self.results],
        }

        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    def print_summary(self):
        """Print comparison summary."""
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)

        # Group by model size
        by_size: Dict[str, List[ComparisonResult]] = {}
        for r in self.results:
            if r.model_size not in by_size:
                by_size[r.model_size] = []
            by_size[r.model_size].append(r)

        for size in ["0.6B", "4B", "8B"]:
            if size in by_size:
                print(f"\n{size} Model:")
                print("-" * 70)

                for r in by_size[size]:
                    if r.status == "PASS":
                        print(f"  ✓ {r.provider_a} vs {r.provider_b}: {r.cosine_similarity:.8f}")
                    elif r.status == "REVIEW":
                        print(
                            f"  ⚠️ {r.provider_a} vs {r.provider_b}: {r.cosine_similarity:.8f} (below 0.999)"
                        )
                    elif r.status == "DIMENSION_MISMATCH":
                        print(f"  ✗ {r.provider_a} vs {r.provider_b}: Dimension mismatch")
                    else:
                        print(f"  ✗ {r.provider_a} vs {r.provider_b}: {r.status}")

        # Overall stats
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        review = sum(1 for r in self.results if r.status == "REVIEW")
        failed = total - passed - review

        print(f"\n{'=' * 70}")
        print(
            f"Overall: {passed} PASS, {review} REVIEW, {failed} FAILED / {total} total comparisons"
        )
        print(f"{'=' * 70}")


if __name__ == "__main__":
    comparator = SameSizeComparator()
    comparator.run_comparison()
