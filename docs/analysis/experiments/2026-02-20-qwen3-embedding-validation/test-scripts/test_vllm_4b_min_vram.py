#!/usr/bin/env python3
"""Find minimum vLLM VRAM for Qwen3-Embedding-4B."""

import gc
import os
import time
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"


def get_instruction(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"


TASK = "Given a web search query, retrieve relevant passages that answer the query"
TEST_TEXT = "What is machine learning?"


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)


print(f"\n{'=' * 60}")
print(f"Finding minimum vLLM VRAM for Qwen3-4B")
print(f"{'=' * 60}\n")

cleanup()

print(f"Free VRAM before: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

from vllm import LLM

utilizations = [0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14]

for util in utilizations:
    print(f"\nTesting gpu_memory_utilization={util}...", end=" ")
    cleanup()

    try:
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
        allocated = util * 47.37

        text = get_instruction(TASK, TEST_TEXT)
        torch.cuda.synchronize()
        start = time.time()
        outputs = llm.embed([text])
        torch.cuda.synchronize()
        latency = (time.time() - start) * 1000

        print(f"✓ SUCCESS! Latency: {latency:.1f}ms, VRAM: {vram_used:.2f}GB")

        del llm
        cleanup()

        print(f"\n{'=' * 60}")
        print(f"MINIMUM: gpu_memory_utilization={util} ({allocated:.1f}GB)")
        print(f"Model uses: {vram_used:.2f}GB")
        print(f"{'=' * 60}")
        break

    except Exception as e:
        err_str = str(e)
        if "Free memory" in err_str:
            print(f"✗ Not enough free memory")
        elif "OOM" in err_str or "out of memory" in err_str.lower():
            print(f"✗ OOM")
        else:
            print(f"✗ {err_str[:60]}")

        if "llm" in locals():
            del llm
        cleanup()
        continue

print("\n" + "=" * 60)
