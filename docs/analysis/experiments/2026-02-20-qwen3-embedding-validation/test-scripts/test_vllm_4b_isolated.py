#!/usr/bin/env python3
"""vLLM-only test for Qwen3-Embedding-4B with low VRAM settings."""

import gc
import json
import os
import sys
import time
from datetime import datetime

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
EXPECTED_DIM = 2560


def get_instruction(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"


TASK = "Given a web search query, retrieve relevant passages that answer the query"
TEST_TEXT = "What is machine learning?"


def cleanup_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)


print(f"\n{'=' * 60}")
print(f"vLLM QWEN3-EMBEDDING-4B TEST (LOW VRAM)")
print(f"{'=' * 60}\n")

cleanup_vram()

from vllm import LLM

print("Testing with gpu_memory_utilization=0.10...")
try:
    llm = LLM(
        model=MODEL_NAME,
        runner="pooling",
        convert="embed",
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.10,
        enforce_eager=True,
        max_model_len=4096,
        disable_log_stats=True,
    )

    input_text = get_instruction(TASK, TEST_TEXT)

    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.embed([input_text])
    torch.cuda.synchronize()
    latency = (time.time() - start) * 1000

    emb = outputs[0].outputs.embedding

    print(f"✓ SUCCESS: {latency:.1f}ms, dim={len(emb)}")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    del llm
    cleanup_vram()

except Exception as e:
    print(f"✗ FAILED (0.10): {e}")

print("\n" + "=" * 60)
