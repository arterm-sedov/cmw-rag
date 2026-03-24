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

### 3.3 Russian Cloud Providers (for comparison)

**Note:** For Russian market, consider local providers for data sovereignty compliance.

#### Cloud.ru (Evolution Foundation Models) [[source]](https://cloud.ru/documents/tariffs/evolution/foundation-models)

| Model | Input Tokens (per million) | Output Tokens (per million) | Price incl. VAT (₽/million) |
|-------|---------------------------|----------------------------|------------------------------|
| GigaChat3-10B-A1.8B | 10 | 10 | 12.2 |
| Qwen3-235B-A22B-Instruct-2507 | 17 | - | 20.74 |
| GigaChat-2-Max | 466.67 | 466.67 | 569.34 |
| MiniMax-M2 | 33 | 130 | 40.26 / 158.6 |

#### Yandex Cloud (YandexGPT) [[source]](https://www.akm.ru/eng/press/yandex-b2b-tech-has-opened-access-to-the-largest-language-model-on-the-russian-market/)

*   **Price:** ~50 kopecks per 1,000 tokens (~500 ₽/million tokens)
*   **Models:** YandexGPT Pro 5.1, Alice AI LLM

#### SberCloud (GigaChat API) [[source]](https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs)

| Package | Tokens | Price (incl. VAT) |
|---------|--------|------------------|
| GigaChat 2 Lite (1B) | 1,000,000,000 | 65,000 ₽ |
| GigaChat 2 Pro (120M) | 120,000,000 | 60,000 ₽ |

**Equivalent price per million tokens:**
*   GigaChat 2 Lite: ~65 ₽/million
*   GigaChat 2 Pro: ~500 ₽/million

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

### Russian Market Recommendations

For the Russian market, consider local cloud providers for data sovereignty compliance:

1. **Cost Comparison (per 1M tokens):**
   | Provider | Model | Price (₽/million) | USD Equiv. |
   |----------|-------|-------------------|------------|
   | Cloud.ru | GigaChat3-10B | 12.2 | ~$0.13 |
   | Cloud.ru | Qwen3-235B | 20.74 | ~$0.22 |
   | Yandex Cloud | YandexGPT | ~500 | ~$5.50 |
   | SberCloud | GigaChat 2 Lite | ~65 | ~$0.72 |
   | SberCloud | GigaChat 2 Pro | ~500 | ~$5.50 |
   | OpenRouter (US) | Various | 100-1000 | $1-10 |

2. **Best Value:** Cloud.ru's GigaChat3-10B-A1.8B at ~12.2 ₽/million tokens (~10x cheaper than US alternatives)

3. **Local Inference:** Use cmw-mosec/cmw-vllm for fully offline solutions with RTX 4090 (0.6B-4B models) or A100/H100 (8B+ models)

### RTX 4090 (48GB) Benchmarks [[source]](https://t.me/neuraldeep/1476)

Performance on RTX 4090 (data from NeuralDeep community):

| Model | Tokens/sec | Parameters | Context |
|-------|-------------|------------|---------|
| Llama 3.1 8B | ~50-60 | 8B | 8K |
| Qwen 2.5 32B | ~20-30 | 32B | 32K |
| Mistral 7B | ~40-50 | 7B | 8K |
| Qwen3-30B-A3B | ~25-35 | 30B MoE | 32K |

**Budget Cluster Recommendations [[source]](https://t.me/neuraldeep/1627):**
*   Budget cluster: 4x RTX 4090 for parallel processing
*   Power consumption: ~400W per card (~1600W total)
*   Suitable for: RAG bots, transcription, streaming

### Advanced RAG Architectures and Cost Impact [[source]](https://t.me/ai_archnadzor)

From **@ai_archnadzor** channel — key architectures and their economics:

#### Performance Optimization Architectures

| Architecture | Optimization | Cost Reduction | Use Case |
|------------|-------------|---------------|---------|
| **REFRAG** | Chunk compression via RoBERTa + RL policy | 30.85x TTFT speedup | Tier-1 systems, millions of queries |
| **HippoRAG 2** | Dual-Node architecture (entity + passage nodes) | 12x cheaper indexing (9M vs 115M tokens) | Mass indexing |
| **Topo-RAG** | Multi-vector indexes per cell | Hallucinations 45%→8%, Index 12.4GB→4.1GB | Fintech, logistics, tables |
| **Doc-to-LoRA** | Hypernetwork generates LoRA from docs | VRAM: 12GB→50MB (99%), <1s assimilation | Long documents, context optimization |
| **BitNet** | 1-bit weights for Attention/MLP | CPU inference, no GPU monopoly | Edge AI, local solutions, cost reduction |

#### Semantic Quality Architectures

| Architecture | Innovation | Quality Improvement | Use Case |
|-------------|-----------|---------------------|---------|
| **Cog-RAG** | Dual hypergraphs (themes + entities) | Win Rate +84.5% vs Naive RAG | Medicine, science, complex domains |
| **Disco-RAG** | Rhetorical Structure Theory (RST) | Transforms RAG into logical analyst | Jurisprudence, medicine |
| **Semantic Gravity** | Physics-based rejection sampling | 100% Safety Compliance | Enterprise, high-risk queries |
| **GraphOS** | 16-layer architecture with Redis + Neo4j | 47% cost savings on routing | Complex enterprise RAG |

#### Storage & Infrastructure

| Architecture | Innovation | Storage Reduction | Use Case |
|-------------|-----------|-------------------|---------|
| **LEANN** | On-demand embedding computation | 97% reduction (201GB → 6GB for 60M chunks) | Offline-first, privacy |
| **Topo-RAG** | Cell-aware late interaction (CALI) | 18.4% nDCG@10 improvement | Table-heavy documents |

### Benchmark Data from NeuralDeep [[source]](https://t.me/neuraldeep/1476)

RTX 4090 (48GB) performance:

| Model | Tokens/sec | Parameters | Context |
|-------|------------|------------|---------|
| Llama 3.1 8B | ~50-60 | 8B | 8K |
| Qwen 2.5 32B | ~20-30 | 32B | 32K |
| Mistral 7B | ~40-50 | 7B | 8K |
| Qwen3-30B-A3B | ~25-35 | 30B MoE | 32K |
| Qwen3-30B-A3B (Raspberry Pi 5) | ~8-8.5 | 30B MoE | 32K |

**Budget Cluster Recommendations [[source]](https://t.me/neuraldeep/1627):**
*   Budget cluster: 4x RTX 4090 for parallel processing
*   Power consumption: ~400W per card (~1600W total)
*   Suitable for: RAG bots, transcription, streaming

### Local Coding Models for Cost Reduction [[source]](https://t.me/ai_archnadzor/167)

Replace Claude Code ($1,500-2,500/month for 5-10 engineers) with local models:

| Model | VRAM | Context | Use Case |
|-------|------|---------|----------|
| **Qwen3-Coder-Next** (MoE) | ~24 GB (Q4_K_M) | 64K+ | Autonomous agents, full repo analysis |
| **GLM-4.7-Flash** (MoE) | 16-24 GB | 200K | Microservices, legacy monoliths |
| **CodeGemma v1.1** | 8-20 GB | 32K | Clean syntax (Python, TS, Go) |
| **StarCoder2 15B** | 10-12 GB | 16K | Rare languages, legacy (COBOL, Fortran) |
| **Phi-4-mini-instruct** | 3-4 GB | 8K | Instant autocomplete in IDE |
| **IBM Granite-20B-Code** | 12-16 GB | 8K | Enterprise compliance (Java, C#, SQL) |

**Deployment Strategy:** Brain + Edge (heavy model on server for architectural review + micro model local for instant autocomplete)

### 2026 Trends Summary

From **@ai_archnadzor** and **NeuralDeep** channels:

**Architecture Trends:**
*   End of "vanilla RAG" era — composite architectures with specialized indexes
*   Graph-based memory (Knowledge Graphs, Temporal Graphs, Hypergraphs)
*   Multi-modal VLM replacing traditional OCR pipelines
*   DSPy + GEPA for automatic prompt optimization

**Cost Optimization Trends:**
*   BitNet and 1-bit inference for CPU-based production
*   Doc-to-LoRA eliminating KV-Cache overhead
*   LEANN for 97% storage reduction
*   REFRAG for 30x latency improvement

**Infrastructure Trends:**
*   CLI replacing MCP for agent tool access (zero overhead)
*   EffGen framework for SLM efficiency (+11.2% for 1.5B models)
*   Arize Phoenix for full observability
*   Guardrails as mandatory infrastructure layer

---

## 10. Latest AI/ML Trends from @ai_machinelearning_big_data [[source]](https://t.me/ai_machinelearning_big_data)

From **@ai_machinelearning_big_data** channel (323,407 subscribers) — fresh AI/ML news and trends:

### 10.1 Coding Agents

**NousResearch Hermes Agent Hackathon:**
*   187 submissions, $11,750 prize pool
*   Winners: Media Tool (ffmpeg), CAD Builder, Sidecar, World Map, Mars Rover
*   Hermes Agent wrote a 79,456-word novel

**Cursor Composer 2:**
*   Competes with Claude Opus 4.6 and GPT-5.4
*   Price: $0.50/1M input tokens, $2.50/1M output tokens
*   Benchmark: 61.3 points (vs 44.2 for v1.5)

**Claude Code Channels (Anthropic):**
*   Claude Code integration with Telegram and Discord via MCP
*   Async AI agent workflow
*   "OpenClaw killer" per community

**OpenAI Superapp (Codex + ChatGPT + Atlas):**
*   Unifying products into single platform
*   Agents for autonomous computer work

### 10.2 AI Infrastructure

**NVIDIA Nemotron-Cascade 2:**
*   MoE 30B (3B active) — Gold at IMO, IOI, ICPC 2025
*   LiveCodeBench v6: 88.4 points
*   Codeforces rating: 2345 (300B+ model level)
*   License: NVIDIA Open Model License

**Huawei Atlas 350:**
*   Accelerator on Ascend 950PR — 2.87x faster than Nvidia H20
*   FP4 compute, 112 GB HBM
*   Load LLMs up to 70B parameters on single card

**GLM 5.1 Open Source:**
*   Zixuan Li (ZAI) announced plans to open source

**Mamba3:**
*   SSM architecture with inference priority
*   SISO: best total prefill + decode latency
*   MIMO: comparable speed, but noticeably more accurate

### 10.3 Robotics & Hardware

**Unitree As2:**
*   Quadruped robot in 3 versions: AIR, PRO, EDU
*   18 kg, 12 DOF, up to 3.7 m/s
*   EDU: NVIDIA Jetson Orin NX support

**Pokemon Go → Robot Navigation:**
*   30B photos from fans for spatial AI training
*   Niantic Spatial: centimeter-accurate visual navigation
*   Coco Robotics: couriers with 4 cameras

### 10.4 Enterprise AI

**Google AI Studio — Vibe Coding:**
*   Antigravity Agent for automatic Firebase deployment
*   Next.js, React, Angular support
*   Gemini 3.1 Pro for full development cycle

**ElevenLabs Music Marketplace:**
*   AI music from ElevenCreative
*   14M generated songs
*   $11M earned on voice marketplace

**Adobe Firefly:**
*   Custom AI models on user data
*   Project Moonlight: agentic interface for all apps

### 10.5 Russian Market

**Agents Week by Yandex SHD (April 6-10):**
*   Intensive on AI agents from Yandex experts
*   Single-agent and multi-agent architectures
*   Production approaches: evaluation, monitoring, scaling

**National AI Olympiad:**
*   7,000 participants, 111 finalists, 18 winners
*   Winners get internship at Sberbank

**Yandex Prompt Hub:**
*   Prompt for generating prompts (4-D methodology)
*   Deconstruct → Diagnose → Develop → Deliver

**Sber One Day Offer for Data Scientists:**
*   March 28, opportunity to get hired in 1 day
