## VRAM Requirements для LLM Inference

### Формула расчёта VRAM

**Total VRAM = Model Weights + KV Cache + Activations + Framework Overhead**

**Approximation:**
```
VRAM ≈ (Parameters × Bytes/Weight) / Tensor Parallelism + KV_Cache + Overhead
```

**Правило большого пальца:**
- FP16: **~2 GB VRAM на 1B параметров**
- BF16: Аналогично FP16
- INT8: **~1 GB VRAM на 1B параметров**
- INT4: **~0.5 GB VRAM на 1B параметров**

### Throughput Estimation

**Теоретический максимум:**
```
Max tok/sec ≈ Memory Bandwidth (GB/s) / Model Size (GB)
```

| GPU | Bandwidth | 7B (Q4) | 70B (Q4) |
|-----|-----------|---------|----------|
| RTX 4090 | 1,008 GB/s | ~288 tok/s | ~29 tok/s |
| A100-80GB | 2,039 GB/s | ~583 tok/s | ~58 tok/s |
| H100-80GB | 3,352 GB/s | ~958 tok/s | ~96 tok/s |

### Sizing по классам моделей

| Model | Precision | Weights | KV Cache (8K, batch 8) | Total Recommended |
|-------|-----------|---------|------------------------|-------------------|
| Llama 3.2 3B | Q4 | 1.5GB | 1.2GB | 8GB |
| Mistral 7B | Q4 | 3.5GB | 2.8GB | 12GB |
| Llama 3.1 8B | Q4 | 4GB | 3.2GB | 12-16GB |
| Llama 3.1 8B | FP16 | 16GB | 3.2GB | 24GB |
| Llama 3.1 70B | Q4 | 35GB | 28GB | 80GB (2x40GB) |

---
