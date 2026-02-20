#!/usr/bin/env python3
"""Qwen3-Embedding-4B Comprehensive Backend Test

Tests all backends with proper VRAM management between tests.
"""

import gc
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import requests
from dotenv import load_dotenv

load_dotenv("/home/asedov/cmw-rag/.env")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
MODEL_SIZE = "4B"
EXPECTED_DIM = 2560

TEST_TEXTS = [
    "What is machine learning and how does it work?",
    "Natural language processing applications in modern AI",
    "Deep learning neural networks explained",
    "Artificial intelligence in healthcare diagnostics",
    "The future of autonomous vehicles technology",
]


def get_instruction(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"


TASK = "Given a web search query, retrieve relevant passages that answer the query"


def format_vram(msg: str = "") -> str:
    if not torch.cuda.is_available():
        return "N/A (CPU)"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"{allocated:.2f}GB / {reserved:.2f}GB"


def cleanup_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)


def get_vram_total() -> float:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


print(f"\n{'=' * 60}")
print(f"QWEN3-EMBEDDING-4B COMPREHENSIVE BACKEND TEST")
print(f"{'=' * 60}")
print(f"Time: {datetime.now()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Total VRAM: {get_vram_total():.1f}GB")
print(f"Expected dim: {EXPECTED_DIM}")
print(f"{'=' * 60}\n")

results = {
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "model": MODEL_NAME,
    "model_size": MODEL_SIZE,
    "expected_dim": EXPECTED_DIM,
    "test_texts": TEST_TEXTS,
    "backends": {},
}


# ============================================================
# TEST 1: Direct Transformers
# ============================================================
print(f"\n[1/4] Testing Direct Transformers...")
cleanup_vram()

import transformers
import torch.nn.functional as F


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


try:
    load_start = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    load_time = time.time() - load_start

    print(f"  Load time: {load_time:.1f}s")
    print(f"  VRAM: {format_vram()}")

    embeddings = []
    latencies = []
    for text in TEST_TEXTS:
        formatted = get_instruction(TASK, text)
        encoded = tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True, max_length=8192
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            output = model(**encoded)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        latency = (time.time() - start) * 1000

        emb = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        embeddings.append(emb.cpu().numpy())
        latencies.append(latency)

    direct_embeddings = embeddings[0]
    avg_latency = np.mean(latencies)

    results["backends"]["direct"] = {
        "status": "SUCCESS",
        "load_time_sec": load_time,
        "latency_ms": avg_latency,
        "dimensions": embeddings[0].shape[1],
        "vram_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        "sample": embeddings[0][0, :5].tolist(),
    }
    print(f"  ✓ SUCCESS: {avg_latency:.1f}ms, dim={embeddings[0].shape[1]}, VRAM={format_vram()}")

    del model, tokenizer
    cleanup_vram()

except Exception as e:
    results["backends"]["direct"] = {"status": "FAILED", "error": str(e)}
    print(f"  ✗ FAILED: {e}")


# ============================================================
# TEST 2: vLLM
# ============================================================
print(f"\n[2/4] Testing vLLM...")
cleanup_vram()

try:
    from vllm import LLM

    load_start = time.time()
    llm = LLM(
        model=MODEL_NAME,
        runner="pooling",
        convert="embed",
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.15,
        enforce_eager=True,
        max_model_len=8192,
        disable_log_stats=True,
    )
    load_time = time.time() - load_start

    print(f"  Load time: {load_time:.1f}s")
    print(f"  VRAM: {format_vram()}")

    input_texts = [get_instruction(TASK, text) for text in TEST_TEXTS]

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    outputs = llm.embed(input_texts)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    latency = (time.time() - start) * 1000

    vllm_embeddings = [o.outputs.embedding for o in outputs]
    avg_latency = latency / len(TEST_TEXTS)

    results["backends"]["vllm"] = {
        "status": "SUCCESS",
        "load_time_sec": load_time,
        "latency_ms": avg_latency,
        "dimensions": len(vllm_embeddings[0]),
        "vram_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        "sample": vllm_embeddings[0][:5],
    }
    print(f"  ✓ SUCCESS: {avg_latency:.1f}ms, dim={len(vllm_embeddings[0])}, VRAM={format_vram()}")

    del llm
    cleanup_vram()

except Exception as e:
    results["backends"]["vllm"] = {"status": "FAILED", "error": str(e)}
    print(f"  ✗ FAILED: {e}")


# ============================================================
# TEST 3: OpenRouter
# ============================================================
print(f"\n[3/4] Testing OpenRouter...")
cleanup_vram()

try:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    input_texts = [get_instruction(TASK, text) for text in TEST_TEXTS]

    start = time.time()
    response = client.embeddings.create(
        model="qwen/qwen3-embedding-4b",
        input=input_texts,
    )
    latency = time.time() - start

    or_embeddings = [d.embedding for d in response.data]
    avg_latency = latency / len(TEST_TEXTS) * 1000

    results["backends"]["openrouter"] = {
        "status": "SUCCESS",
        "latency_ms": avg_latency,
        "dimensions": len(or_embeddings[0]),
        "sample": or_embeddings[0][:5],
    }
    print(f"  ✓ SUCCESS: {avg_latency:.1f}ms, dim={len(or_embeddings[0])}")

except Exception as e:
    results["backends"]["openrouter"] = {"status": "FAILED", "error": str(e)}
    print(f"  ✗ FAILED: {e}")


# ============================================================
# TEST 4: Mosec
# ============================================================
print(f"\n[4/4] Testing Mosec...")
cleanup_vram()

mosec_process = None

try:
    import subprocess

    print("  Starting Mosec server with Qwen3-Embedding-4B...")
    mosec_process = subprocess.Popen(
        ["cmw-mosec", "serve", "--embedding", "Qwen/Qwen3-Embedding-4B"],
        cwd=os.path.expanduser("~/cmw-mosec"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(30)

    input_texts = [get_instruction(TASK, text) for text in TEST_TEXTS]

    start = time.time()
    resp = requests.post(
        "http://localhost:7998/v1/embeddings",
        json={"input": input_texts, "model": MODEL_NAME},
        timeout=60,
    )
    latency = time.time() - start

    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    mosec_embeddings = [d["embedding"] for d in data["data"]]
    avg_latency = latency / len(TEST_TEXTS) * 1000

    results["backends"]["mosec"] = {
        "status": "SUCCESS",
        "latency_ms": avg_latency,
        "dimensions": len(mosec_embeddings[0]),
        "sample": mosec_embeddings[0][:5],
    }
    print(f"  ✓ SUCCESS: {avg_latency:.1f}ms, dim={len(mosec_embeddings[0])}")

except Exception as e:
    results["backends"]["mosec"] = {"status": "FAILED", "error": str(e)}
    print(f"  ✗ FAILED: {e}")

finally:
    if mosec_process:
        print("  Stopping Mosec server...")
        mosec_process.terminate()
        mosec_process.wait(timeout=10)

cleanup_vram()


# ============================================================
# COMPARISONS
# ============================================================
print(f"\n{'=' * 60}")
print("CROSS-BACKEND COMPARISONS")
print(f"{'=' * 60}\n")

comparisons = []


def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if "direct" in results["backends"] and results["backends"]["direct"]["status"] == "SUCCESS":
    direct_sample = results["backends"]["direct"]["sample"]

    if "vllm" in results["backends"] and results["backends"]["vllm"]["status"] == "SUCCESS":
        vllm_sample = results["backends"]["vllm"]["sample"]
        sim = cosine_sim(direct_sample, vllm_sample)
        comparisons.append({"direct_vs_vllm": sim})
        print(f"  Direct vs vLLM: {sim * 100:.4f}%")

    if (
        "openrouter" in results["backends"]
        and results["backends"]["openrouter"]["status"] == "SUCCESS"
    ):
        or_sample = results["backends"]["openrouter"]["sample"]
        sim = cosine_sim(direct_sample, or_sample)
        comparisons.append({"direct_vs_openrouter": sim})
        print(f"  Direct vs OpenRouter: {sim * 100:.4f}%")

    if "mosec" in results["backends"] and results["backends"]["mosec"]["status"] == "SUCCESS":
        mosec_sample = results["backends"]["mosec"]["sample"]
        sim = cosine_sim(direct_sample, mosec_sample)
        comparisons.append({"direct_vs_mosec": sim})
        print(f"  Direct vs Mosec: {sim * 100:.4f}%")

results["comparisons"] = comparisons


# ============================================================
# SAVE RESULTS
# ============================================================
output_file = f"/home/asedov/cmw-rag/docs/analysis/experiments/2026-02-20-qwen3-embedding-validation/data/2026-02-20-4b-backend-test.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print(f"RESULTS SAVED: {output_file}")
print(f"{'=' * 60}\n")


# ============================================================
# SUMMARY
# ============================================================
print(f"{'=' * 60}")
print("FINAL SUMMARY")
print(f"{'=' * 60}\n")

print(f"| Backend   | Status   | Latency | Dim   | VRAM    |")
print(f"|-----------|----------|---------|-------|---------|")

for backend, data in results["backends"].items():
    status = "✓" if data["status"] == "SUCCESS" else "✗"
    lat = f"{data.get('latency_ms', 0):.1f}ms" if data["status"] == "SUCCESS" else "N/A"
    dim = data.get("dimensions", "N/A")
    vram = f"{data.get('vram_gb', 0):.2f}GB" if data.get("vram_gb", 0) > 0 else "N/A"
    print(f"| {backend:9} | {status} {data['status']:6} | {lat:7} | {dim:5} | {vram:7} |")

print(f"\n{'=' * 60}")
