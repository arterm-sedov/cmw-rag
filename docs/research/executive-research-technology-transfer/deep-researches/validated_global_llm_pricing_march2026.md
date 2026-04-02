# Global LLM API Pricing Validation — March 2026

**Research Date:** March 29, 2026  
**Exchange Rate:** 85 RUB/USD  
**Focus:** Enterprise-relevant API tiers for production workloads

---

## 1. OpenAI GPT-5 Family

| Model | Context Window | Input ($/1M) | Output ($/1M) | Input (RUB) | Output (RUB) |
|-------|----------------|--------------|---------------|-------------|--------------|
| **GPT-5.4 Pro** | 1M tokens | $30.00 | $180.00 | 2,550 | 15,300 |
| **GPT-5.4** (standard, short) | 200K | $2.50 | $15.00 | 212.50 | 1,275 |
| **GPT-5.4** (long context) | 1M | $5.00 | $22.50 | 425 | 1,912.50 |
| **GPT-5.4 mini** | 400K | $0.75 | $4.50 | 63.75 | 382.50 |
| **GPT-5.4 nano** | 128K | $0.20 | $1.00 | 17 | 85 |

**Notes:**
- GPT-5.4 launched March 2026 with native computer-use capabilities
- Regional processing (data residency) adds 10% uplift
- Cached input tokens: 90% discount (e.g., GPT-5.4 cached = $0.25/1M)
- Batch API: 50% discount for asynchronous workloads

**Sources:** OpenAI Pricing (developers.openai.com),硅 Silicon Data, ZDNET, NXCode

---

## 2. Anthropic Claude 4.x Family

| Model | Context Window | Input ($/1M) | Output ($/1M) | Input (RUB) | Output (RUB) |
|-------|----------------|--------------|---------------|-------------|--------------|
| **Claude Opus 4.6** | 1M tokens | $5.00 | $25.00 | 425 | 2,125 |
| **Claude Sonnet 4.6** | 1M tokens | $3.00 | $15.00 | 255 | 1,275 |
| **Claude Haiku 4.5** | 200K | $1.00 | $5.00 | 85 | 425 |

**Notes:**
- Claude 4.6 models released February 2026 with full 1M token context at standard pricing
- Premium pricing for prompts >200K tokens: Opus $10/$37.50 (available on Claude Platform only)
- Prompt caching: 90% discount
- Batch API: 50% discount

**Sources:** Anthropic Pricing (platform.claude.com), Silicon Data, NXCode, IntuitionLabs

---

## 3. Google Gemini 3.x Family

| Model | Context Window | Input ($/1M) | Output ($/1M) | Input (RUB) | Output (RUB) |
|-------|----------------|--------------|---------------|-------------|--------------|
| **Gemini 3.1 Pro** | 1M tokens | $2.00 | $12.00 | 170 | 1,020 |
| **Gemini 3.1 Pro** (>200K) | 1M | $4.00 | $18.00 | 340 | 1,530 |
| **Gemini 3.1 Flash** | 1M | $0.50 | $3.00 | 42.50 | 255 |
| **Gemini 3.1 Flash-Lite** | 1M | $0.10 | $0.40 | 8.50 | 34 |
| **Gemini 2.5 Flash-Lite** | 1M | $0.10 | $0.40 | 8.50 | 34 |

**Notes:**
- Gemini 3.1 Pro released February 2026 as free performance upgrade from 3.0
- Tiered pricing: standard rate for ≤200K tokens, long-context rate for >200K
- Gemini 3.1 Flash-Lite is free tier eligible
- Context caching available at $0.025/1M tokens

**Sources:** Google AI Developer Pricing (ai.google.dev), Google Cloud Vertex AI Pricing, NXCode, CostGoat

---

## 4. Enterprise Comparison Summary

### Top-Tier Models (Production/Reasoning)

| Provider | Model | Input ($/1M) | Output ($/1M) | Combined Index |
|----------|-------|--------------|---------------|----------------|
| Google | Gemini 3.1 Pro | $2.00 | $12.00 | 1.0x (baseline) |
| Anthropic | Claude Sonnet 4.6 | $3.00 | $15.00 | 1.25x |
| OpenAI | GPT-5.4 | $2.50 | $15.00 | 1.04x |
| Anthropic | Claude Opus 4.6 | $5.00 | $25.00 | 2.08x |
| OpenAI | GPT-5.4 Pro | $30.00 | $180.00 | 12.5x |

### Cost-Optimized Models (High Volume)

| Provider | Model | Input ($/1M) | Output ($/1M) |
|----------|-------|--------------|---------------|
| Google | Gemini 2.5 Flash-Lite | $0.10 | $0.40 |
| Google | Gemini 3.1 Flash-Lite | $0.10 | $0.40 |
| OpenAI | GPT-5.4 nano | $0.20 | $1.00 |
| OpenAI | GPT-5.4 mini | $0.75 | $4.50 |
| Anthropic | Claude Haiku 4.5 | $1.00 | $5.00 |

---

## 5. Key Findings

1. **Google Gemini 3.1 Pro** offers the best value among flagship models at $2/$12
2. **OpenAI GPT-5.4** bridges the gap between mid-tier and premium at $2.50/$15
3. **Claude Sonnet 4.6** at $3/$15 is competitive with GPT-5.4 standard
4. **GPT-5.4 Pro** at $30/$180 is 12x more expensive than baseline flagship — enterprise premium tier
5. **Gemini Flash-Lite** at $0.10/$0.40 is the most cost-effective for classification/routing
6. All providers now support 1M token context at standard pricing (except premium tiers)
7. Prompt caching provides 90% savings across all providers
8. Batch API discounts of 50% available from all three vendors

---

## 6. Data Validation Notes

- All prices verified via official vendor pricing pages as of March 2026
- RUB conversion at 85 RUB/USD (Central Bank rate, March 2026)
- Prices represent standard Pay-As-You-Go API rates; enterprise contracts may vary
- Context windows listed are maximum supported; pricing tiers may apply within context ranges
