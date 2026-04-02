# Pricing Reference — March 2026
**Canonical Source for All Pricing Data in This Research Pack**

**Date:** March 29, 2026  
**Status:** Validated & Consolidated  
**Currency:** RUB (internal rate: 85 RUB = 1 USD for comparisons)

---

## 1. Russian LLM API Pricing (RUB per 1M tokens)

### 1.1 GigaChat (Sber)

| Model | Input | Output | Mode | Notes |
|-------|-------|--------|------|-------|
| GigaChat 2 Lite | 65 ₽ | 65 ₽ | Sync | |
| GigaChat 2 Pro | 500 ₽ | 500 ₽ | Sync | |
| GigaChat 2 Max | 650 ₽ | 650 ₽ | Sync | |
| GigaChat 2 Lite | 32.5 ₽ | 32.5 ₽ | Async | 50% discount |
| GigaChat 2 Pro | 250 ₽ | 250 ₽ | Async | 50% discount |
| GigaChat 2 Max | 325 ₽ | 325 ₽ | Async | 50% discount |
| Embeddings | 14 ₽ | — | Sync | |
| Embeddings | 7 ₽ | — | Async | |

**Source:** [Sber Developers — Legal Tariffs](https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs) (Updated Jan 29, 2026, effective Feb 1, 2026)

### 1.2 YandexGPT (Yandex)

| Model | Input | Output | Access | Notes |
|-------|-------|--------|--------|-------|
| YandexGPT 5 Pro | Free | Free | Alice/Browser | Since July 2025 |
| YandexGPT 5 Lite | Free | Free | Alice/Browser | Since July 2025 |
| YandexGPT 5.1 | Contact | Contact | API | Enterprise only |

**Source:** [Yandex Cloud AI Studio](https://aistudio.yandex.ru/docs/en/ai-studio/)

### 1.3 Cloud.ru Evolution

| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| GigaChat-2-Max | 569 ₽ | 569 ₽ | |
| GLM-4.6 | 67 ₽ | 268 ₽ | |
| MiniMax-M2 | 40 ₽ | 159 ₽ | |
| Qwen3-235B | 21 ₽ | 21 ₽ | Open weights |
| GigaChat3-10B | 12 ₽ | 12 ₽ | Open weights |

**Source:** [Cloud.ru Evolution Tariffs](https://cloud.ru/documents/tariffs/evolution/foundation-models.html) (v260316, Mar 26, 2026)

---

## 2. Global LLM API Pricing (RUB per 1M tokens, converted at 85 RUB/USD)

### 2.1 OpenAI GPT-5 Family

| Model | Input (USD) | Output (USD) | RUB Input | RUB Output |
|-------|-------------|--------------|-----------|------------|
| GPT-5.4 Nano | $0.20 | $1.25 | 17 | 106 |
| GPT-5.4 Mini | $0.75 | $4.50 | 64 | 383 |
| GPT-5 | $1.25 | $10.00 | 106 | 850 |
| GPT-5.4 (long ctx) | $5.00 | $22.50 | 425 | 1,913 |
| GPT-5 Pro | $15.00 | $120.00 | 1,275 | 10,200 |
| GPT-5.4 Pro (long) | $60.00 | $270.00 | 5,100 | 22,950 |

### 2.2 Anthropic Claude 4.x

| Model | Input (USD) | Output (USD) | RUB Input | RUB Output |
|-------|-------------|--------------|-----------|------------|
| Claude Haiku 4.5 | $1.00 | $5.00 | 85 | 425 |
| Claude Sonnet 4.6 | $3.00 | $15.00 | 255 | 1,275 |
| Claude Opus 4.6 | $5.00 | $25.00 | 425 | 2,125 |

### 2.3 Google Gemini 3.x

| Model | Input (USD) | Output (USD) | RUB Input | RUB Output |
|-------|-------------|--------------|-----------|------------|
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 | 8.5 | 34 |
| Gemini 3.1 Flash-Lite | $0.25 | $1.00 | 21 | 85 |
| Gemini 3.1 Pro | $2.00 | $12.00 | 170 | 1,020 |

---

## 3. GPU Rental Pricing (RUB per month)

### 3.1 Russian Providers

| Configuration | Monthly | Provider | Notes |
|---------------|---------|----------|-------|
| 1×RTX 4090 (ready) | 16,000–26,000 ₽ | HostKey | Quick deploy |
| 1×RTX 4090 (custom) | ~60,000 ₽ | HostKey | Full config |
| 8×A100 80GB | 671,500 ₽ | HostKey | Enterprise |
| 8×A100 40GB | 416,500 ₽ | HostKey | Enterprise |
| 1×H200 141GB | 253,887 ₽ | HostKey | |

**Sources:** [HostKey](https://hostkey.ru/gpu-dedicated-servers/), [1dedic](https://1dedic.ru/gpu-servers)

### 3.2 Global Comparison (converted at 85 RUB/USD)

| GPU | Global Cloud (est.) | Russian Dedicated |
|-----|---------------------|-------------------|
| RTX 4090 | $150–300/mo | 16,000–60,000 ₽ |
| A100 80GB | $3,500–4,500/mo | ~670,000 ₽ |
| H100 | $3,000–4,200/mo | ~250,000 ₽ |

---

## 4. On-Premise GPU Server Costs (CapEx)

| Configuration | Est. CapEx | Annual OpEx (electricity, colocation) |
|---------------|------------|--------------------------------------|
| 1×RTX 4090 server | 500,000–800,000 ₽ | 150,000–300,000 ₽ |
| 4×RTX 4090 server | 1,400,000–1,800,000 ₽ | 300,000–500,000 ₽ |
| 8×H100 server | 8,000,000–15,000,000 ₽ | 500,000–1,000,000 ₽ |

---

## 5. AI Agent Implementation Cost Bands

| Segment | One-time (CapEx) | Monthly (OpEx) | Description |
|---------|------------------|----------------|-------------|
| Basic | 50,000–200,000 ₽ | 5,000–15,000 ₽ | Chatbot + RAG |
| Medium | 300,000–1,500,000 ₽ | 30,000–100,000 ₽ | Multichannel + CRM |
| Enterprise | 5,000,000+ ₽ | 200,000+ ₽ | Full-scale AI agents |

**Source:** [RBC Education — AI agent costs](https://www.rbc.ru/education/05/02/2026/697162269a794772c9cf9991)

---

## 6. Deployment Mode Comparison

| Mode | Key Advantage | Key Disadvantage | Break-even |
|------|---------------|------------------|------------|
| SaaS (managed) | Fast start, no hardware | Vendor lock-in, data concerns | — |
| Cloud API (Russia) | Data residency, local support | OpEx at scale | — |
| On-prem | Full control, sovereignty | High CapEx, maintenance | >60% GPU utilization |

---

## Quick Reference Summary

| Use Case | Recommended | Est. Monthly Cost |
|----------|-------------|-------------------|
| PoC / Pilot | GigaChat Lite (65 ₽/M) or YandexGPT Free | 5,000–15,000 ₽ |
| Production (small) | GigaChat Pro (500 ₽/M) + Cloud.ru | 30,000–100,000 ₽ |
| Production (medium) | Self-hosted RTX 4090 | 80,000–150,000 ₽ |
| Enterprise | Self-hosted A100/H100 cluster | 250,000–700,000 ₽ |

---

*This document is the canonical source. Reference this file from other documents. All prices validated March 2026. Prices include Russian VAT where applicable. For proposals, verify current pricing with vendors.*
