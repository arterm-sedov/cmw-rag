#!/usr/bin/env python3
"""vLLM-only test for Qwen3-Embedding-4B"""

import gc
import os
import time
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"


def get_instruction(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"


TASK = "Given a web search query, retrieve relevant passages that answer the query"
TEST_TEXT = "What is machine learning?"


def cleanup():
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(1)


def main():
    print(f"\n{'=' * 60}")
    print(f"vLLM QWEN3-4B TEST")
    print(f"{'=' * 60}\n")

    cleanup()

    import torch

    print(f"VRAM before: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    from vllm import LLM

    print("Testing with gpu_memory_utilization=0.30 (14GB)...")
    try:
        llm = LLM(
            model=MODEL_NAME,
            runner="pooling",
            convert="embed",
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.30,
            enforce_eager=True,
            max_model_len=4096,
            disable_log_stats=True,
        )

        vram_after = torch.cuda.memory_allocated() / 1024**3
        print(f"  Load VRAM: {vram_after:.2f}GB")

        text = get_instruction(TASK, TEST_TEXT)
        torch.cuda.synchronize()
        start = time.time()
        outputs = llm.embed([text])
        torch.cuda.synchronize()
        latency = (time.time() - start) * 1000

        emb = outputs[0].outputs.embedding
        print(f"  ✓ SUCCESS: {latency:.1f}ms, dim={len(emb)}")
        print(f"  Sample: {emb[:5]}")

        del llm
        cleanup()
        print(f"  VRAM after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
