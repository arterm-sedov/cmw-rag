# Qwen3-Embedding-4B Backend Performance Report

**Date:** 2026-02-20  
**Model:** Qwen/Qwen3-Embedding-4B (2560 dimensions)  
**GPU:** NVIDIA GeForce RTX 4090 (48GB total, ~16GB available)  
**Status:** ✅ ALL BACKENDS VALIDATED

---

## Test Results Summary

| Backend | Status | Avg Latency | Min | Max | VRAM | Notes |
|---------|--------|-------------|-----|-----|------|-------|
| **Mosec** | ✅ | **4.16ms** | 3.88ms | 5.85ms | N/A* | Fastest! |
| **Direct** | ✅ | 5.4ms | 3.5ms | 21.7ms | 7.57GB | |
| **vLLM** | ✅ | 18.6ms | - | - | **~7.5GB** (14GB alloc) | Min: 0.30 util |
| **OpenRouter** | ✅ | 311ms | 263ms | 478ms | N/A | Cloud API |

*Mosec runs in separate process

---

## Cross-Backend Accuracy

| Comparison | Cosine Similarity | Status |
|------------|-------------------|--------|
| Direct vs Mosec | 99.99% | ✅ PASS |
| Direct vs vLLM | 99.97% | ✅ PASS |

---

## vLLM Minimum VRAM Analysis

**Finding:** Minimum `gpu_memory_utilization=0.30` (14.2GB allocated)

| Utilization | Allocated | Model VRAM | KV Cache | Status |
|------------|-----------|------------|----------|--------|
| 0.30 | 14.2GB | ~7.55GB | ~6GB | ✅ WORKS |
| 0.28 | 13.3GB | - | - | ❌ Fork error |
| 0.26 | 12.3GB | - | - | ❌ Fork error |
| 0.20 | 9.5GB | - | - | ❌ Fork error |

**Conclusion:** Model itself requires ~7.5GB. vLLM needs additional VRAM for KV cache and overhead. With 16GB available, 0.30 (14.2GB) is the minimum working configuration.

---

## Latency Breakdown (10 runs)

### Mosec
- Avg: 4.16ms
- Min: 3.88ms  
- Max: 5.85ms
- Very consistent, lowest latency

### Direct Transformers  
- Avg: 5.4ms
- Min: 3.5ms
- Max: 21.7ms
- Higher variance due to Python overhead

### vLLM
- Avg: 18.6ms
- Clean VRAM: 26.2ms
- More overhead from multiprocessing

### OpenRouter
- Avg: 311ms
- Min: 263ms
- Max: 478ms
- Network latency dominates

---

## VRAM Requirements Summary

| Model | Direct | vLLM (min) | Mosec |
|-------|--------|------------|-------|
| 0.6B | ~2.4GB | ~1.1GB | ~2GB |
| **4B** | **~7.5GB** | **~7.5GB** | **~8GB** |
| 8B | ~16GB (offload) | N/A | N/A |

---

## Recommendations

| Use Case | Recommended Backend | Latency | VRAM |
|----------|-------------------|---------|------|
| **Fastest** | Mosec | 4ms | ~8GB |
| **Balanced** | Direct | 5ms | 7.5GB |
| **Production** | vLLM | 19ms | 14GB |
| **Cloud** | OpenRouter | 311ms | N/A |

---

## Test Files

- `test_qwen3_4b_comprehensive.py` - Full benchmark
- `test_vllm_4b_min_vram.py` - VRAM minimum finder
- `2026-02-20-4b-comprehensive-test.json` - Raw results
