#!/usr/bin/env python3
"""Qwen3-Embedding-4B VRAM and Performance Test - Isolated"""

import gc
import json
import os
import sys
import time
from datetime import datetime

import torch
import requests
from dotenv import load_dotenv

load_dotenv("/home/asedov/cmw-rag/.env")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
EXPECTED_DIM = 2560

TEST_TEXTS = [
    "What is machine learning?",
    "Natural language processing",
    "Deep learning neural networks",
]


def get_instruction(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"


TASK = "Given a web search query, retrieve relevant passages that answer the query"


def format_vram(label: str = "") -> str:
    if not torch.cuda.is_available():
        return "N/A"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"{label} {allocated:.2f}GB / {reserved:.2f}GB" if label else f"{allocated:.2f}GB"


def cleanup_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(1)


print(f"\n{'=' * 60}")
print(f"QWEN3-4B BACKEND TEST (16GB VRAM LIMIT)")
print(f"{'=' * 60}")
print(f"Time: {datetime.now()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Initial VRAM: {format_vram()}")
print(f"{'=' * 60}\n")

results = {
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "model": MODEL_NAME,
    "backends": {},
}

# ============================================================
# TEST 1: Direct
# ============================================================
print("[1/4] Direct Transformers...")
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
    load_start = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to("cuda")
    model.eval()
    load_time = time.time() - load_start
    vram_after_load = torch.cuda.memory_allocated() / 1024**3

    texts = [get_instruction(TASK, t) for t in TEST_TEXTS]
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=8192)
    encoded = {k: v.cuda() for k, v in encoded.items()}

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model(**encoded)
    torch.cuda.synchronize()
    latency = (time.time() - start) * 1000 / len(TEST_TEXTS)

    emb = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
    emb = F.normalize(emb, p=2, dim=1)

    results["backends"]["direct"] = {
        "status": "SUCCESS",
        "load_time_sec": round(load_time, 1),
        "latency_ms": round(latency, 1),
        "dimensions": emb.shape[1],
        "vram_gb": round(vram_after_load, 2),
        "sample": emb[0, :5].tolist(),
    }
    print(f"  ✓ {latency:.1f}ms, dim={emb.shape[1]}, VRAM={vram_after_load:.2f}GB")

    del model, tokenizer
    cleanup_vram()
    print(f"  After cleanup: {format_vram()}")

except Exception as e:
    results["backends"]["direct"] = {"status": "FAILED", "error": str(e)[:100]}
    print(f"  ✗ FAILED: {str(e)[:80]}")


# ============================================================
# TEST 2: vLLM (16GB = 0.33 utilization)
# ============================================================
print("\n[2/4] vLLM (16GB limit)...")
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
        gpu_memory_utilization=0.33,  # 16GB of 48GB
        enforce_eager=True,
        max_model_len=4096,
        disable_log_stats=True,
    )
    load_time = time.time() - load_start
    vram_after_load = torch.cuda.memory_allocated() / 1024**3

    input_texts = [get_instruction(TASK, t) for t in TEST_TEXTS]

    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.embed(input_texts)
    torch.cuda.synchronize()
    latency = (time.time() - start) * 1000 / len(TEST_TEXTS)

    vllm_embeddings = [o.outputs.embedding for o in outputs]

    results["backends"]["vllm"] = {
        "status": "SUCCESS",
        "load_time_sec": round(load_time, 1),
        "latency_ms": round(latency, 1),
        "dimensions": len(vllm_embeddings[0]),
        "vram_gb": round(vram_after_load, 2),
        "sample": vllm_embeddings[0][:5],
    }
    print(f"  ✓ {latency:.1f}ms, dim={len(vllm_embeddings[0])}, VRAM={vram_after_load:.2f}GB")

    del llm
    cleanup_vram()
    print(f"  After cleanup: {format_vram()}")

except Exception as e:
    results["backends"]["vllm"] = {"status": "FAILED", "error": str(e)[:150]}
    print(f"  ✗ FAILED: {str(e)[:100]}")


# ============================================================
# TEST 3: OpenRouter
# ============================================================
print("\n[3/4] OpenRouter...")
cleanup_vram()

try:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
    )

    input_texts = [get_instruction(TASK, t) for t in TEST_TEXTS]

    start = time.time()
    response = client.embeddings.create(
        model="qwen/qwen3-embedding-4b",
        input=input_texts,
    )
    latency = (time.time() - start) * 1000 / len(TEST_TEXTS)

    or_embeddings = [d.embedding for d in response.data]

    results["backends"]["openrouter"] = {
        "status": "SUCCESS",
        "latency_ms": round(latency, 1),
        "dimensions": len(or_embeddings[0]),
        "sample": or_embeddings[0][:5],
    }
    print(f"  ✓ {latency:.1f}ms, dim={len(or_embeddings[0])}")

except Exception as e:
    results["backends"]["openrouter"] = {"status": "FAILED", "error": str(e)[:100]}
    print(f"  ✗ FAILED: {str(e)[:80]}")


# ============================================================
# TEST 4: Mosec (measure VRAM delta)
# ============================================================
print("\n[4/4] Mosec (measuring VRAM)...")
cleanup_vram()

vram_before_mosec = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
print(f"  VRAM before Mosec: {vram_before_mosec:.2f}GB")

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
    time.sleep(30)  # Wait for server to start

    vram_during_load = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    print(f"  VRAM during Mosec: {vram_during_load:.2f}GB")

    input_texts = [get_instruction(TASK, t) for t in TEST_TEXTS]

    start = time.time()
    resp = requests.post(
        "http://localhost:7998/v1/embeddings",
        json={"input": input_texts, "model": MODEL_NAME},
        timeout=60,
    )
    latency = (time.time() - start) * 1000 / len(TEST_TEXTS)

    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}")

    data = resp.json()
    mosec_embeddings = [d["embedding"] for d in data["data"]]

    vram_during_inference = (
        torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    )

    results["backends"]["mosec"] = {
        "status": "SUCCESS",
        "latency_ms": round(latency, 1),
        "dimensions": len(mosec_embeddings[0]),
        "vram_gb_before": round(vram_before_mosec, 2),
        "vram_gb_during_load": round(vram_during_load, 2),
        "vram_gb_during_inference": round(vram_during_inference, 2),
        "vram_delta": round(vram_during_load - vram_before_mosec, 2),
        "sample": mosec_embeddings[0][:5],
    }
    print(f"  ✓ {latency:.1f}ms, dim={len(mosec_embeddings[0])}")
    print(f"  VRAM delta (load): +{vram_during_load - vram_before_mosec:.2f}GB")

except Exception as e:
    results["backends"]["mosec"] = {"status": "FAILED", "error": str(e)[:100]}
    print(f"  ✗ FAILED: {str(e)[:80]}")

finally:
    if mosec_process:
        print("  Stopping Mosec...")
        mosec_process.terminate()
        mosec_process.wait(timeout=10)

cleanup_vram()
vram_after_cleanup = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
print(f"  VRAM after cleanup: {vram_after_cleanup:.2f}GB")


# ============================================================
# Comparisons
# ============================================================
print(f"\n{'=' * 60}")
print("COMPARISONS")
print(f"{'=' * 60}\n")

if "direct" in results["backends"] and results["backends"]["direct"]["status"] == "SUCCESS":
    direct_sample = results["backends"]["direct"]["sample"]

    import numpy as np

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for backend in ["vllm", "openrouter", "mosec"]:
        if backend in results["backends"] and results["backends"][backend]["status"] == "SUCCESS":
            sim = cosine_sim(direct_sample, results["backends"][backend]["sample"])
            print(f"  Direct vs {backend}: {sim * 100:.4f}%")
            results["backends"][f"direct_vs_{backend}"] = round(sim * 100, 4)


# ============================================================
# Save
# ============================================================
output_file = "/home/asedov/cmw-rag/docs/analysis/experiments/2026-02-20-qwen3-embedding-validation/data/2026-02-20-4b-backend-test.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}\n")

print(f"| Backend   | Status   | Latency | Dim   | VRAM   |")
print(f"|-----------|----------|---------|-------|--------|")
for b, d in results["backends"].items():
    if "vs_" in b:
        continue
    status = "✓" if d["status"] == "SUCCESS" else "✗"
    lat = f"{d.get('latency_ms', 0):.1f}ms" if d["status"] == "SUCCESS" else "N/A"
    dim = d.get("dimensions", "N/A")
    vram = f"{d.get('vram_gb', 'N/A')}GB" if d.get("vram_gb") else "N/A"
    print(f"| {b:9} | {status} {d['status']:6} | {lat:7} | {dim:5} | {vram:6} |")

print(f"\nResults: {output_file}")
