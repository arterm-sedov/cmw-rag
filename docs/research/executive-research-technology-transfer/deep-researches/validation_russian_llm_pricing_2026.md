# Russian LLM Pricing and Versions - March 2026

**Research Date:** March 29, 2026  
**Research Focus:** Russian LLM pricing, versions, and availability

---

## Executive Summary

This report provides validated information on Russian LLM offerings as of March 2026, focusing on GigaChat (Sber), YandexGPT (Yandex), and Cloud.ru services. Key findings include significant price reductions for GigaChat API (3x reduction in February 2026), the release of GigaChat 3.1 with open weights under MIT license, and YandexGPT 5.1 with expanded context windows.

---

## 1. GigaChat 3.1 - Latest Version Details

### Release Information

| Model | Parameters | Release Date | License |
|-------|------------|--------------|---------|
| GigaChat-3.1-Ultra-702B | 702B (MoE) | March 2026 | MIT |
| GigaChat-3.1-Lightning-10B-A1.8B | 10B total / 1.8B active (MoE) | March 2026 | MIT |

**Source:** Reddit r/LocalLLaMA, March 24, 2026; ai-sage Hugging Face collection

### MIT License and Hugging Face Weights

- **License:** MIT License
- **Repository:** github.com/salute-developers/gigachat3
- **Hugging Face Models:**
  - `ai-sage/GigaChat3.1-702B-A36B` - Full precision weights
  - `ai-sage/GigaChat3.1-702B-A36B-GGUF` - GGUF quantized format
  - `ai-sage/GigaChat3.1-10B-A1.8B` - Lightning model (10B/1.8B)
  - `ai-sage/GigaChat3.1-10B-A1.8B-bf16` - Brain float16 precision

### Key Features

- Optimized for multilingual assistant workloads
- Lightning model designed for fast inference (1.5x faster than Qwen3-4B)
- Supports local deployment on laptops
- Data processing and storage in Russia (152-FZ compliance)

---

## 2. GigaChat API Pricing (Sber)

**Effective Date:** February 1, 2026  
**Note:** Prices reduced 3x in February 2026 (IXBT Pro, February 2, 2026)

### Individual Users (Freemium)

| Service | Tokens | Duration | Model |
|---------|--------|----------|-------|
| Free text generation | 900,000 | 12 months | GigaChat 2 Lite |
| Free text generation | 50,000 | 12 months | GigaChat 2 Pro |
| Free text generation | 50,000 | 12 months | GigaChat 2 Max |

### Paid Packages (Individuals)

| Package | Tokens | Price (RUB, incl. VAT) |
|---------|--------|------------------------|
| GigaChat 2 Lite | 20,000,000 | 1,300 ₽ |
| GigaChat 2 Lite | 100,000,000 | 6,500 ₽ |
| GigaChat 2 Pro | 3,000,000 | 1,500 ₽ |
| GigaChat 2 Pro | 15,000,000 | 7,500 ₽ |
| GigaChat 2 Max | 3,000,000 | 1,950 ₽ |
| GigaChat 2 Max | 15,000,000 | 9,750 ₽ |
| Embeddings | 50,000,000 | 700 ₽ |

### Corporate Pricing (Pay-as-you-go)

**Synchronous Mode:**

| Model | Price per 1,000 tokens (incl. VAT) |
|-------|-----------------------------------|
| GigaChat 2 Lite | 0.065 ₽ |
| GigaChat 2 Pro | 0.50 ₽ |
| GigaChat 2 Max | 0.65 ₽ |

**Asynchronous Mode:**

| Model | Price per 1,000 tokens (incl. VAT) |
|-------|-----------------------------------|
| GigaChat 2 Lite | 0.0325 ₽ |
| GigaChat 2 Pro | 0.25 ₽ |
| GigaChat 2 Max | 0.325 ₽ |

**Embeddings:**

| Mode | Price per 1,000 tokens (incl. VAT) |
|------|-----------------------------------|
| Synchronous | 0.014 ₽ |
| Asynchronous | 0.007 ₽ |

**Minimum Monthly Fee:** 600 RUB (incl. VAT)

**Source:** developers.sber.ru/docs/ru/gigachat/tariffs/ (Updated January 29, 2026)

---

## 3. YandexGPT - Latest Version 2026

### Model Lineup (February 2026)

| Model | Context Window | Release | Key Features |
|-------|---------------|---------|--------------|
| YandexGPT 5.1 Pro | 128,000 tokens | August 2025 | 2.5x faster inference, Chain-of-Reasoning |
| Alice AI LLM | 128,000 tokens | February 2026 | Leader of SLAVA benchmark |
| YandexGPT Lite 5 | TBD | 2025 | Fast, cost-effective |
| YandexGPT Pro RC | 32,000 tokens | 2025 | Release candidate |

**Source:** mysummit.school, March 4, 2026; toolarium.ru, March 3, 2026

### Performance Claims (Yandex)

- YandexGPT 5.1 Pro outperforms previous version in 67% of cases
- Outperforms GPT-4.1 in 56% of cases (Yandex internal tests)
- Second place on SLAVA benchmark after Alice AI LLM

---

## 4. YandexGPT API Pricing

**Effective Date:** March 3, 2026 (AI Studio)

### Pricing (Yandex Cloud / AI Studio)

| Model | Input per 1K tokens | Output per 1K tokens |
|-------|---------------------|----------------------|
| Alice AI LLM | 0.50 ₽ | 1.20 ₽ |
| YandexGPT Pro 5.1 | 0.80 ₽ | 0.80 ₽ |
| YandexGPT Lite | 0.20 ₽ | 0.20 ₽ |

**Source:** mysummit.school, March 4, 2026

### Free Access

- **Alice (via ya.ru):** Free, no registration
- **Yandex Browser:** Free AI assistant
- **Chat with Alice:** Free since July 2025 (including Pro features)
- **Free tier:** Starter grants and promo codes available

### Fine-tuning

- Starting from 200,000 RUB per fine-tuning job
- Available for enterprise clients via Yandex Foundation Models

**Source:** toolarium.ru, March 3, 2026

---

## 5. Cloud.ru (Evolution Foundation Models)

### Overview

Cloud.ru (formerly SberCloud) provides access to various LLM models through Evolution Foundation Models catalog.

### Available Models

| Model | Release | Pricing |
|-------|---------|---------|
| GigaChat Lightning (10B-A1.8B) | November 2025 | Free during testing |
| GigaChat 3.0 | November 2025 | Free during testing |
| GLM-4.6 (200K context) | October 2025 | Free until October 31, 2025 |

**Note:** Cloud.ru offers OpenAI-compatible API for GigaChat models.

**Source:** TAdviser, November 2025

### Key Features

- OpenAI-compatible API
- Data processing in Russia
- Free access during testing phases

---

## 6. Summary Comparison

### API Pricing Comparison (RUB per 1,000 tokens)

| Provider | Model | Sync Price | Notes |
|----------|-------|------------|-------|
| Sber (GigaChat) | Lite | 0.065 ₽ | As low as 0.0325₽ async |
| Sber (GigaChat) | Pro | 0.50 ₽ | As low as 0.25₽ async |
| Sber (GigaChat) | Max | 0.65 ₽ | As low as 0.325₽ async |
| Yandex | Lite | 0.20 ₽ | |
| Yandex | Pro 5.1 | 0.80 ₽ | |
| Yandex | Alice AI LLM | 0.50-1.20 ₽ | Input/Output differentiated |

### Context Windows

| Model | Context Window |
|-------|---------------|
| GigaChat 2 Pro/Max | 128,000 tokens |
| YandexGPT 5.1 Pro | 128,000 tokens |
| GLM-4.6 (Cloud.ru) | 200,000 tokens |

---

## 7. Changes Since Late 2025

### Key Developments

1. **February 2026:** Sber announced 3x price reduction for GigaChat API
2. **March 2026:** GigaChat 3.1 released with open weights (MIT license)
3. **November 2025:** Cloud.ru opened free access to GigaChat 3.0 and Lightning
4. **August 2025:** YandexGPT 5.1 Pro released with 128K context
5. **July 2025:** Yandex made Pro features free in Chat with Alice

---

## 8. API Endpoints

### GigaChat API

- **Base URL:** `https://gigachat.devices.sberbank.ru`
- **Documentation:** developers.sber.ru/docs/ru/gigachat/api/overview

### YandexGPT API

- **Base URL:** `https://llm.api.cloud.yandex.net/foundationModels/v1`
- **Documentation:** yandex.cloud/ru/services/yandexgpt

### Cloud.ru Evolution

- **Base URL:** OpenAI-compatible
- **Documentation:** cloud.ru Evolution Foundation Models

---

## Sources

- developers.sber.ru (Sber documentation)
- yandex.cloud (Yandex Cloud)
- toolarium.ru (YandexGPT review, March 2026)
- mysummit.school (YandexGPT and GigaChat reviews, March 2026)
- TAdviser (Cloud.ru Evolution Foundation Models)
- Reddit r/LocalLLaMA (GigaChat 3.1 release)
- Hugging Face ai-sage collection
- IXBT Pro (Price reduction announcement, February 2026)

---

*Report compiled: March 29, 2026*
