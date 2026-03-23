# Executive Summary: AI Sizing, CapEx & OpEx for Clients

**Repositories:** cmw-rag, cmw-mosec, cmw-vllm
**Date:** 2026-03-23
**Prepared for:** Technical Management
**Status:** Final

---

## 1. Executive Overview

This document provides a detailed analysis of hardware requirements (sizing), capital expenditure (CapEx), and operational expenditure (OpEx) for deploying the CMW AI ecosystem. The analysis is based on actual VRAM measurements from `cmw-mosec` and vLLM specifications from `cmw-vllm`, using an NVIDIA RTX 4090 (48GB VRAM) as the reference hardware.

**Key Findings:**
- **Small-Scale Deployment:** ~5GB VRAM (0.6B models) suitable for RTX 4090 or cloud A10G.
- **Medium-Scale Deployment:** ~18GB VRAM (4B models) requires high-end GPU (RTX 6000 Ada, A6000).
- **Large-Scale Deployment:** >48GB VRAM (8B models) requires data-center GPUs (A100, H100).

---

## 2. Hardware Sizing Analysis

### 2.1 VRAM Requirements (Verified Data)

#### Individual Model VRAM Usage (cmw-mosec)

| Model | Type | VRAM Usage | Notes |
|-------|------|------------|-------|
| `ai-forever/FRIDA` | Embedding | +3.6 GB | T5-based, fp32 |
| `Qwen3-Embedding-0.6B` | Embedding | +1.9 GB | fp16 |
| `Qwen3-Embedding-4B` | Embedding | +8.9 GB | fp16 |
| `DiTy/cross-encoder` | Reranker | +2.3 GB | Cross-encoder |
| `Qwen3-Reranker-0.6B` | Reranker | +1.5 GB | Single worker fix |
| `Qwen3-Reranker-4B` | Reranker | +8.1 GB | Single worker fix |
| `Qwen3Guard-Gen-0.6B` | Guard | +1.8 GB | bf16 |
| `Qwen3Guard-Gen-4B` | Guard | +8.8 GB | bf16 |

#### Model Combinations (cmw-mosec)

| Combination | Total VRAM | Free VRAM (48GB) | Status |
|-------------|------------|------------------|--------|
| Embed 0.6B + Rerank 0.6B | ~3.5 GB | ~44 GB | ✅ Safe |
| Embed 4B + Rerank 0.6B | ~10.5 GB | ~37 GB | ✅ Safe |
| FRIDA + DiTy + Guard 0.6B | ~7.7 GB | ~40 GB | ✅ Safe |
| Embed 4B + Rerank 4B | ~17 GB | ~31 GB | ✅ Safe |
| **Any 8B Model** | ~16-18 GB | <30 GB | ⚠️ OOM Risk |

#### vLLM Overhead (cmw-vllm)

vLLM incurs higher overhead due to KV cache and continuous batching:

| Model Type | Example | Total VRAM (vLLM) | vs. MOSEC |
|------------|---------|-------------------|-----------|
| Embedding | Qwen3-0.6B | ~4-6 GB | +2 GB |
| Reranker | Qwen3-0.6B | ~3-5 GB | +1.5 GB |
| Guard | Qwen3Guard-0.6B | ~3-4 GB | +1 GB |
| **Combination (All 0.6B)** | Embed+Rerank+Guard | **~10-15 GB** | +5 GB |

### 2.2 System Requirements

| Component | Minimum | Recommended | High-Performance |
|-----------|---------|-------------|------------------|
| **GPU** | RTX 3060 (12GB) | RTX 4090 (48GB) | A100 (80GB) |
| **System RAM** | 16 GB | 32 GB | 64 GB+ |
| **Storage** | 50 GB SSD | 200 GB NVMe | 1 TB NVMe |
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **Network** | 1 Gbps | 10 Gbps | 25 Gbps+ |

---

## 3. Capital Expenditure (CapEx)

### 3.1 On-Premise Deployment

#### Small-Scale (Departmental)
*   **Hardware:** RTX 4090 Desktop Workstation
*   **Cost:** ~$2,500 (GPU + CPU + RAM + Storage)
*   **Capacity:** 1-3 concurrent users, 0.6B models
*   **Use Case:** Development, testing, small teams

#### Medium-Scale (Departmental/Regional)
*   **Hardware:** RTX 6000 Ada or A6000 Workstation/Server
*   **Cost:** ~$8,000 - $12,000
*   **Capacity:** 5-10 concurrent users, 4B models
*   **Use Case:** Production for medium teams

#### Large-Scale (Enterprise)
*   **Hardware:** NVIDIA A100/H100 Server (4-8 GPUs)
*   **Cost:** ~$50,000 - $200,000+
*   **Capacity:** 50+ concurrent users, 8B+ models
*   **Use Case:** Enterprise-wide deployment

### 3.2 Cloud Deployment (AWS/GCP/Azure)

| Instance Type | GPU | VRAM | Hourly Cost | Monthly Cost (730h) |
|---------------|-----|------|-------------|---------------------|
| **g4dn.xlarge** | T4 | 16 GB | $0.526 | ~$384 |
| **g5.xlarge** | A10G | 24 GB | $1.006 | ~$734 |
| **p3.2xlarge** | V100 | 16 GB | $3.06 | ~$2,234 |
| **p4d.24xlarge** | A100 (8x) | 320 GB | $32.77 | ~$23,922 |

**Recommendation:** Start with **g5.xlarge (A10G)** for small-scale production (~$734/month).

---

## 4. Operational Expenditure (OpEx)

### 4.1 Recurring Costs

#### Cloud Hosting
*   **Small-Scale:** $700 - $1,000/month (g5.xlarge)
*   **Medium-Scale:** $2,000 - $4,000/month (g5.4xlarge or multi-instance)
*   **Large-Scale:** $10,000+/month (multiple A100/H100 instances)

#### LLM API Costs (if using OpenRouter/Gemini)
*   **Gemini Pro:** ~$0.001 per 1k tokens (input), $0.003 per 1k tokens (output)
*   **OpenRouter:** Variable, ~$0.10 - $1.00 per 1M tokens depending on model
*   **Estimate:** $50 - $500/month for moderate usage

#### Electricity (On-Premise)
*   **RTX 4090 Workstation:** ~400W under load → ~$50-100/month (24/7)
*   **A100 Server:** ~2,000W → ~$300-500/month

### 4.2 Maintenance & Support

| Activity | Frequency | Effort | Cost Impact |
|----------|-----------|--------|-------------|
| Model Updates | Monthly | 2-4 hours | Low |
| ChromaDB Maintenance | Quarterly | 4 hours | Low |
| System Monitoring | Continuous | 1 hour/week | Medium |
| Backup & Recovery | Weekly | 2 hours | Low |
| Security Patching | Monthly | 4 hours | Medium |

**Estimated Annual Maintenance:** $5,000 - $15,000 (depending on scale)

---

## 5. Total Cost of Ownership (TCO) Analysis

### 5.1 3-Year TCO Comparison

| Deployment | Initial CapEx | Annual OpEx | 3-Year TCO | Users Supported |
|------------|---------------|-------------|------------|-----------------|
| **On-Premise (Small)** | $2,500 | $2,000 | $8,500 | 1-3 |
| **Cloud (Small)** | $0 | $12,000 | $36,000 | 1-3 |
| **On-Premise (Medium)** | $10,000 | $5,000 | $25,000 | 5-10 |
| **Cloud (Medium)** | $0 | $30,000 | $90,000 | 5-10 |
| **On-Premise (Large)** | $100,000 | $20,000 | $160,000 | 50+ |
| **Cloud (Large)** | $0 | $120,000 | $360,000 | 50+ |

**Key Insight:** On-premise is more cost-effective for sustained workloads (>1 year); cloud is better for variable workloads or rapid scaling.

---

## 6. Client Sizing Recommendations

### 6.1 Small Business / Department
*   **Hardware:** RTX 4090 Workstation
*   **Models:** Qwen3-0.6B (Embedding, Reranker, Guard)
*   **VRAM Usage:** ~5 GB
*   **Users:** 1-3 concurrent
*   **CapEx:** $2,500
*   **OpEx:** $2,000/year (maintenance)
*   **TCO (3yr):** $8,500

### 6.2 Medium Enterprise
*   **Hardware:** RTX 6000 Ada or Cloud g5.4xlarge
*   **Models:** Qwen3-4B (Embedding), Qwen3-0.6B (Reranker/Guard)
*   **VRAM Usage:** ~12 GB
*   **Users:** 5-10 concurrent
*   **CapEx:** $10,000 (on-prem) or $0 (cloud)
*   **OpEx:** $5,000/year (on-prem) or $30,000/year (cloud)
*   **TCO (3yr):** $25,000 (on-prem) or $90,000 (cloud)

### 6.3 Large Enterprise
*   **Hardware:** A100/H100 Server (4-8 GPUs)
*   **Models:** Qwen3-8B (Embedding/Reranker) or Mix of 4B models
*   **VRAM Usage:** 50-100 GB
*   **Users:** 50+ concurrent
*   **CapEx:** $100,000+ (on-prem) or $0 (cloud)
*   **OpEx:** $20,000/year (on-prem) or $120,000/year (cloud)
*   **TCO (3yr):** $160,000 (on-prem) or $360,000 (cloud)

---

## 7. Cost Optimization Strategies

1.  **Model Selection:** Use 0.6B models for cost-sensitive applications; reserve 4B+ for high-accuracy needs.
2.  **Mixed Precision:** Use fp16/bf16 to reduce VRAM by 50% vs fp32.
3.  **Multi-Model Serving:** MOSEC allows single-server deployment of multiple models, reducing hardware needs.
4.  **Cloud Spot Instances:** Use AWS Spot or GCP Preemptible for 60-90% cost savings on non-critical workloads.
5.  **Auto-Scaling:** Scale cloud instances based on demand to minimize idle costs.

---

## 8. Conclusion

The CMW AI ecosystem offers flexible sizing options suitable for various client scales:
- **Small-scale** deployments are feasible on consumer GPUs (RTX 4090).
- **Medium-scale** requires professional GPUs (RTX 6000, A10G).
- **Large-scale** requires data-center GPUs (A100, H100).

**CapEx vs OpEx Trade-off:**
- **On-premise:** Higher upfront cost, lower long-term TCO for sustained workloads.
- **Cloud:** Zero upfront cost, higher long-term TCO, ideal for variable workloads.

**Recommendation:** Start with cloud deployment for proof-of-concept, then migrate to on-premise for production if workload is stable and predictable.
