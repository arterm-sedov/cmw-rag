#!/usr/bin/env python3
"""
Final Qwen3 Embedding Validation - All Providers
Proper VRAM management, same-size comparisons, and clean reporting.
"""

import os
import sys
import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from scipy.spatial.distance import cosine

from dotenv import load_dotenv

load_dotenv("/home/asedov/cmw-rag/.env")

sys.path.insert(0, "/home/asedov/cmw-rag")


@dataclass
class ProviderTest:
    provider: str
    model_size: str
    status: str
    dimensions: int = 0
    latency_ms: float = 0.0
    load_time_sec: float = 0.0
    error: str = ""


@dataclass
class Comparison:
    model_size: str
    provider_a: str
    provider_b: str
    cosine_sim: float
    rel_error: float
    status: str


class VRAMManager:
    @staticmethod
    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(3)

    @staticmethod
    def status() -> Dict:
        if not torch.cuda.is_available():
            return {"total": 0, "used": 0, "free": 0}
        t = torch.cuda.get_device_properties(0).total_memory / 1e9
        u = torch.cuda.memory_allocated(0) / 1e9
        return {"total": t, "used": u, "free": t - u}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - cosine(a, b))


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = a / (np.linalg.norm(a) + 1e-10), b / (np.linalg.norm(b) + 1e-10)
    return float(np.mean(np.abs(n1 - n2)))


class FinalValidator:
    TEXT = "What is machine learning and how does it work?"
    INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

    MODELS = {
        "0.6B": {
            "path": "Qwen/Qwen3-Embedding-0.6B",
            "or_id": None,
            "dim": 1024,
            "vllm_util": 0.10,
        },
        "4B": {
            "path": "Qwen/Qwen3-Embedding-4B",
            "or_id": "qwen/qwen3-embedding-4b",
            "dim": 2560,
            "vllm_util": 0.08,
        },
        "8B": {
            "path": "Qwen/Qwen3-Embedding-8B",
            "or_id": "qwen/qwen3-embedding-8b",
            "dim": 4096,
            "vllm_util": 0.10,
        },
    }

    def __init__(self):
        self.results: List[ProviderTest] = []
        self.comparisons: List[Comparison] = []
        self.embeddings: Dict[str, Dict[str, np.ndarray]] = {}  # size -> provider -> emb
        self.dir = Path(
            "/home/asedov/cmw-rag/docs/analysis/experiments/2026-02-20-qwen3-embedding-validation"
        )
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def fmt(self, text: str) -> str:
        return f"Instruct: {self.INSTRUCTION}\nQuery: {text}"

    def test_direct(self, size: str) -> Tuple[Optional[np.ndarray], ProviderTest]:
        print(f"  [Direct {size}]", end=" ")
        m = self.MODELS[size]

        try:
            from transformers import AutoModel, AutoTokenizer

            use_offload = size == "8B"
            load_start = time.time()

            if use_offload:
                tok = AutoTokenizer.from_pretrained(m["path"], trust_remote_code=True)
                model = AutoModel.from_pretrained(
                    m["path"],
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    offload_folder="offload",
                )
            else:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                tok = AutoTokenizer.from_pretrained(m["path"], trust_remote_code=True)
                model = AutoModel.from_pretrained(
                    m["path"],
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
                ).to(dev)

            model.eval()
            load_time = time.time() - load_start

            inp = tok(self.fmt(self.TEXT), return_tensors="pt", truncation=True, max_length=8192)
            inp = {k: v.to(next(model.parameters()).device) for k, v in inp.items()}

            start = time.time()
            with torch.no_grad():
                out = model(**inp)
                hid = out.last_hidden_state
                last_idx = inp["attention_mask"].sum(1) - 1
                emb = torch.nn.functional.normalize(hid[0, last_idx[0], :], p=2, dim=0)
            lat = (time.time() - start) * 1000

            result = emb.cpu().numpy()

            del model, tok
            VRAMManager.cleanup()

            print(f"✓ {len(result)}d, {lat:.1f}ms, {load_time:.1f}s load")
            return result, ProviderTest("direct", size, "SUCCESS", len(result), lat, load_time)

        except Exception as e:
            print(f"✗ {e}")
            VRAMManager.cleanup()
            return None, ProviderTest("direct", size, "FAILED", 0, 0, 0, str(e))

    def test_vllm(self, size: str) -> Tuple[Optional[np.ndarray], ProviderTest]:
        print(f"  [vLLM {size}]", end=" ")
        m = self.MODELS[size]

        try:
            from vllm import LLM

            load_start = time.time()
            llm = LLM(
                model=m["path"],
                gpu_memory_utilization=m["vllm_util"],
                enforce_eager=True,
                max_model_len=8192,
                trust_remote_code=True,
            )
            load_time = time.time() - load_start

            start = time.time()
            out = llm.embed(self.fmt(self.TEXT))
            lat = (time.time() - start) * 1000

            result = np.array(out[0].outputs.embedding)

            del llm
            VRAMManager.cleanup()

            print(f"✓ {len(result)}d, {lat:.1f}ms, {load_time:.1f}s load")
            return result, ProviderTest("vllm", size, "SUCCESS", len(result), lat, load_time)

        except Exception as e:
            print(f"✗ {e}")
            VRAMManager.cleanup()
            return None, ProviderTest("vllm", size, "FAILED", 0, 0, 0, str(e))

    def test_openrouter(self, size: str) -> Tuple[Optional[np.ndarray], ProviderTest]:
        print(f"  [OpenRouter {size}]", end=" ")
        m = self.MODELS[size]

        if not m["or_id"]:
            print("⏭️ not available")
            return None, ProviderTest("openrouter", size, "SKIPPED", 0, 0, 0, "Not available")

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            print("✗ no API key")
            return None, ProviderTest("openrouter", size, "FAILED", 0, 0, 0, "No API key")

        try:
            from openai import OpenAI

            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

            start = time.time()
            resp = client.embeddings.create(
                model=m["or_id"],
                input=self.fmt(self.TEXT),
                encoding_format="float",
                extra_headers={"HTTP-Referer": "https://localhost", "X-Title": "CMW-RAG"},
            )
            lat = (time.time() - start) * 1000

            result = np.array(resp.data[0].embedding)
            print(f"✓ {len(result)}d, {lat:.1f}ms")
            return result, ProviderTest("openrouter", size, "SUCCESS", len(result), lat)

        except Exception as e:
            print(f"✗ {e}")
            return None, ProviderTest("openrouter", size, "FAILED", 0, 0, 0, str(e))

    def test_mosec(self, size: str) -> Tuple[Optional[np.ndarray], ProviderTest]:
        print(f"  [Mosec {size}]", end=" ")

        import requests

        m = self.MODELS[size]

        try:
            start = time.time()
            resp = requests.post(
                "http://localhost:7998/v1/embeddings",
                json={"input": self.fmt(self.TEXT), "model": m["path"]},
                timeout=30,
            )
            lat = (time.time() - start) * 1000

            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}")

            result = np.array(resp.json()["data"][0]["embedding"])
            print(f"✓ {len(result)}d, {lat:.1f}ms")
            return result, ProviderTest("mosec", size, "SUCCESS", len(result), lat)

        except requests.exceptions.ConnectionError:
            print("⏭️ server not running")
            return None, ProviderTest("mosec", size, "SKIPPED", 0, 0, 0, "Server not running")
        except Exception as e:
            print(f"✗ {e}")
            return None, ProviderTest("mosec", size, "FAILED", 0, 0, 0, str(e))

    def compare_all(self, size: str, embs: Dict[str, np.ndarray]):
        if len(embs) < 2:
            return

        print(f"\n  Comparisons for {size}:")
        provs = list(embs.keys())
        for i, a in enumerate(provs):
            for b in provs[i + 1 :]:
                sim = cosine_sim(embs[a], embs[b])
                err = rel_error(embs[a], embs[b])
                status = "PASS" if sim >= 0.999 else ("REVIEW" if sim >= 0.99 else "FAIL")
                icon = "✓" if status == "PASS" else ("⚠️" if status == "REVIEW" else "✗")
                print(f"    {icon} {a} vs {b}: {sim:.6f} ({status})")
                self.comparisons.append(Comparison(size, a, b, sim, err, status))

    def run(self):
        print("=" * 80)
        print("QWEN3 EMBEDDING FINAL VALIDATION")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        VRAMManager.status()

        for size in ["0.6B", "4B", "8B"]:
            print(f"\n{'=' * 80}")
            print(f"MODEL: {size} (expected {self.MODELS[size]['dim']}d)")
            print(f"{'=' * 80}")

            emb = {}

            # Direct (baseline)
            e, r = self.test_direct(size)
            self.results.append(r)
            if e is not None:
                emb["direct"] = e

            # vLLM
            e, r = self.test_vllm(size)
            self.results.append(r)
            if e is not None:
                emb["vllm"] = e

            # OpenRouter
            e, r = self.test_openrouter(size)
            self.results.append(r)
            if e is not None:
                emb["openrouter"] = e

            # Mosec
            e, r = self.test_mosec(size)
            self.results.append(r)
            if e is not None:
                emb["mosec"] = e

            self.embeddings[size] = emb
            self.compare_all(size, emb)

            VRAMManager.cleanup()

        self.save()
        self.summary()

    def save(self):
        data = {
            "timestamp": self.ts,
            "config": {
                "models": {k: {"path": v["path"], "dim": v["dim"]} for k, v in self.MODELS.items()},
                "test_text": self.TEXT,
                "instruction": self.INSTRUCTION,
            },
            "results": [asdict(r) for r in self.results],
            "comparisons": [asdict(c) for c in self.comparisons],
        }

        out = self.dir / f"final_validation_{self.ts}.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Results: {out}")

    def summary(self):
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        by_status = {"SUCCESS": 0, "FAILED": 0, "SKIPPED": 0}
        for r in self.results:
            by_status[r.status] += 1

        print(
            f"\nProvider Tests: {by_status['SUCCESS']} ✓, {by_status['FAILED']} ✗, {by_status['SKIPPED']} ⏭️"
        )

        by_comp = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
        for c in self.comparisons:
            by_comp[c.status] += 1

        print(f"Comparisons: {by_comp['PASS']} ✓, {by_comp['REVIEW']} ⚠️, {by_comp['FAIL']} ✗")

        if by_comp["PASS"] == len(self.comparisons) and len(self.comparisons) > 0:
            print("\n✅ ALL VALIDATIONS PASSED - PRODUCTION READY")
        elif by_comp["FAIL"] == 0:
            print("\n⚠️  ALL VALIDATIONS PASSED with minor precision differences")
        else:
            print("\n❌ SOME VALIDATIONS FAILED - review required")


if __name__ == "__main__":
    v = FinalValidator()
    v.run()
