#!/usr/bin/env python3
"""Comprehensive Qwen3-Embedding-4B Backend Test with averaged latencies."""

import gc
import json
import os
import time
from datetime import datetime

import torch
import requests
from dotenv import load_dotenv

load_dotenv("/home/asedov/cmw-rag/.env")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
EXPECTED_DIM = 2560
NUM_RUNS = 10

TEST_TEXTS = [
    "What is machine learning?",
    "Natural language processing",
    "Deep learning neural networks",
    "Artificial intelligence",
    "Computer vision",
]


def get_instruction(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"


TASK = "Given a web search query, retrieve relevant passages that answer the query"


def cleanup_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)


def format_vram() -> str:
    if not torch.cuda.is_available():
        return "N/A"
    return f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"


print(f"\n{'=' * 60}")
print(f"QWEN3-4B COMPREHENSIVE BACKEND TEST")
print(f"{'=' * 60}")
print(f"Runs: {NUM_RUNS}")
print(f"Initial VRAM: {format_vram()}")
print(f"{'=' * 60}\n")

results = {
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "model": MODEL_NAME,
    "num_runs": NUM_RUNS,
    "backends": {},
}

# ============================================================
# TEST 1: Direct Transformers
# ============================================================
print("[1/4] Direct Transformers (10 runs)...")
cleanup_vram()

import transformers
import torch.nn.functional as F


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


try:
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to("cuda")
    model.eval()

    latencies = []
    for i in range(NUM_RUNS):
        texts = [get_instruction(TASK, t) for t in TEST_TEXTS]
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=8192
        )
        encoded = {k: v.cuda() for k, v in encoded.items()}

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output = model(**encoded)
        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000 / len(TEST_TEXTS))

    emb = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
    emb = F.normalize(emb, p=2, dim=1)

    results["backends"]["direct"] = {
        "status": "SUCCESS",
        "latency_avg_ms": round(sum(latencies) / len(latencies), 2),
        "latency_min_ms": round(min(latencies), 2),
        "latency_max_ms": round(max(latencies), 2),
        "latencies_ms": [round(l, 2) for l in latencies],
        "dimensions": EXPECTED_DIM,
        "vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
    }
    print(
        f"  ✓ Avg: {results['backends']['direct']['latency_avg_ms']:.1f}ms, "
        f"Min: {results['backends']['direct']['latency_min_ms']:.1f}ms, "
        f"Max: {results['backends']['direct']['latency_max_ms']:.1f}ms"
    )

    del model, tokenizer
    cleanup_vram()

except Exception as e:
    results["backends"]["direct"] = {"status": "FAILED", "error": str(e)[:100]}
    print(f"  ✗ FAILED: {e}")


# ============================================================
# TEST 2: vLLM (find minimum VRAM)
# ============================================================
print("\n[2/4] vLLM - Finding minimum VRAM...")

vllm_utilizations = [0.30, 0.25, 0.22, 0.20, 0.18, 0.16, 0.14]
vllm_result = None

for util in vllm_utilizations:
    print(f"  Testing gpu_memory_utilization={util}...", end=" ")
    cleanup_vram()

    try:
        from vllm import LLM

        llm = LLM(
            model=MODEL_NAME,
            runner="pooling",
            convert="embed",
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=util,
            enforce_eager=True,
            max_model_len=4096,
            disable_log_stats=True,
        )

        vram_used = torch.cuda.memory_allocated() / 1024**3

        latencies = []
        for i in range(NUM_RUNS):
            texts = [get_instruction(TASK, t) for t in TEST_TEXTS]
            torch.cuda.synchronize()
            start = time.time()
            outputs = llm.embed(texts)
            torch.cuda.synchronize()
            latencies.append((time.time() - start) * 1000 / len(TEST_TEXTS))

        vllm_result = {
            "status": "SUCCESS",
            "gpu_memory_utilization": util,
            "vram_gb": round(vram_used, 2),
            "latency_avg_ms": round(sum(latencies) / len(latencies), 2),
            "latency_min_ms": round(min(latencies), 2),
            "latency_max_ms": round(max(latencies), 2),
            "latencies_ms": [round(l, 2) for l in latencies],
            "dimensions": EXPECTED_DIM,
        }
        print(f"✓ {vllm_result['latency_avg_ms']:.1f}ms, VRAM: {vram_used:.2f}GB")

        del llm
        cleanup_vram()
        break

    except Exception as e:
        print(f"✗ {str(e)[:60]}")
        if "llm" in locals():
            del llm
        cleanup_vram()
        continue

results["backends"]["vllm"] = vllm_result or {
    "status": "FAILED",
    "error": "All configurations failed",
}


# ============================================================
# TEST 3: OpenRouter
# ============================================================
print("\n[3/4] OpenRouter (10 runs)...")
cleanup_vram()

try:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
    )

    latencies = []
    for i in range(NUM_RUNS):
        texts = [get_instruction(TASK, t) for t in TEST_TEXTS]

        start = time.time()
        response = client.embeddings.create(
            model="qwen/qwen3-embedding-4b",
            input=texts,
        )
        latencies.append((time.time() - start) * 1000 / len(TEST_TEXTS))

    or_embeddings = [d.embedding for d in response.data]

    results["backends"]["openrouter"] = {
        "status": "SUCCESS",
        "latency_avg_ms": round(sum(latencies) / len(latencies), 2),
        "latency_min_ms": round(min(latencies), 2),
        "latency_max_ms": round(max(latencies), 2),
        "latencies_ms": [round(l, 2) for l in latencies],
        "dimensions": len(or_embeddings[0]),
    }
    print(
        f"  ✓ Avg: {results['backends']['openrouter']['latency_avg_ms']:.1f}ms, "
        f"Min: {results['backends']['openrouter']['latency_min_ms']:.1f}ms, "
        f"Max: {results['backends']['openrouter']['latency_max_ms']:.1f}ms"
    )

except Exception as e:
    results["backends"]["openrouter"] = {"status": "FAILED", "error": str(e)[:100]}
    print(f"  ✗ FAILED: {e}")


# ============================================================
# TEST 4: Mosec
# ============================================================
print("\n[4/4] Mosec (10 runs)...")
cleanup_vram()

mosec_process = None
try:
    import subprocess

    print("  Starting Mosec server...")
    mosec_process = subprocess.Popen(
        ["cmw-mosec", "serve", "--embedding", "Qwen/Qwen3-Embedding-4B"],
        cwd=os.path.expanduser("~/cmw-mosec"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(30)

    latencies = []
    for i in range(NUM_RUNS):
        texts = [get_instruction(TASK, t) for t in TEST_TEXTS]

        start = time.time()
        resp = requests.post(
            "http://localhost:7998/v1/embeddings",
            json={"input": texts, "model": MODEL_NAME},
            timeout=60,
        )
        latencies.append((time.time() - start) * 1000 / len(TEST_TEXTS))

    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}")

    data = resp.json()
    mosec_embeddings = [d["embedding"] for d in data["data"]]

    results["backends"]["mosec"] = {
        "status": "SUCCESS",
        "latency_avg_ms": round(sum(latencies) / len(latencies), 2),
        "latency_min_ms": round(min(latencies), 2),
        "latency_max_ms": round(max(latencies), 2),
        "latencies_ms": [round(l, 2) for l in latencies],
        "dimensions": len(mosec_embeddings[0]),
    }
    print(
        f"  ✓ Avg: {results['backends']['mosec']['latency_avg_ms']:.1f}ms, "
        f"Min: {results['backends']['mosec']['latency_min_ms']:.1f}ms, "
        f"Max: {results['backends']['mosec']['latency_max_ms']:.1f}ms"
    )

except Exception as e:
    results["backends"]["mosec"] = {"status": "FAILED", "error": str(e)[:100]}
    print(f"  ✗ FAILED: {e}")

finally:
    if mosec_process:
        print("  Stopping Mosec...")
        mosec_process.terminate()
        mosec_process.wait(timeout=10)

cleanup_vram()


# ============================================================
# Comparisons
# ============================================================
print(f"\n{'=' * 60}")
print("CROSS-BACKEND COMPARISONS")
print(f"{'=' * 60}\n")

if "direct" in results["backends"] and results["backends"]["direct"]["status"] == "SUCCESS":
    import numpy as np

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    direct_sample = results["backends"]["direct"].get("sample", [0] * EXPECTED_DIM)

    for backend in ["vllm", "openrouter", "mosec"]:
        if backend in results["backends"] and results["backends"][backend]["status"] == "SUCCESS":
            sim = cosine_sim(
                direct_sample, results["backends"][backend].get("sample", [0] * EXPECTED_DIM)
            )
            print(f"  Direct vs {backend}: {sim * 100:.4f}%")


# ============================================================
# Save
# ============================================================
output_file = "/home/asedov/cmw-rag/docs/analysis/experiments/2026-02-20-qwen3-embedding-validation/data/2026-02-20-4b-comprehensive-test.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print("FINAL SUMMARY (averaged over 10 runs)")
print(f"{'=' * 60}\n")

print(f"| Backend   | Status   | Avg Latency | Min    | Max    | VRAM   |")
print(f"|-----------|----------|-------------|--------|--------|--------|")
for b in ["direct", "vllm", "openrouter", "mosec"]:
    if b in results["backends"]:
        d = results["backends"][b]
        status = "✓" if d["status"] == "SUCCESS" else "✗"
        if d["status"] == "SUCCESS":
            lat_avg = f"{d['latency_avg_ms']:.1f}ms"
            lat_min = f"{d['latency_min_ms']:.1f}ms"
            lat_max = f"{d['latency_max_ms']:.1f}ms"
            vram = f"{d.get('vram_gb', 'N/A')}GB" if d.get("vram_gb") else "N/A"
        else:
            lat_avg = lat_min = lat_max = "N/A"
            vram = "N/A"
        print(
            f"| {b:9} | {status} {d['status']:6} | {lat_avg:11} | {lat_min:6} | {lat_max:6} | {vram:6} |"
        )

print(f"\nResults: {output_file}")
