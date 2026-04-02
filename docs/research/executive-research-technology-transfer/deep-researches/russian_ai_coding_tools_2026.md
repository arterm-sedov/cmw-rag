# Russian AI Coding Assistants & Enterprise Solutions

**Research Date:** March 29, 2026  
**Classification:** Executive Research - Technology Transfer  
**Sources:** Russian and English web searches

---

## Executive Summary

This research identifies Russian alternatives to international AI coding assistants like Tabnine and Sourcegraph Cody. The market has evolved significantly with several domestic solutions now available, ranging from free tier offerings to enterprise-grade platforms.

### Key Findings

| Tool | Developer | Pricing (RUB) | Enterprise/Self-Hosted | Status |
|------|-----------|---------------|------------------------|--------|
| **GigaCode 2.0** | SberTech | Free (limited) | Cloud-only | Active |
| **SourceCraft** | Yandex Cloud | 250-700 RUB/user | Cloud-only | Active |
| **Koda** | Koda | 0-500 RUB/mo | Cloud-only | Beta |
| **Qodo** | Israeli/Russian market | ~1,500 RUB/mo | Cloud | Active |

---

## 1. GigaCode (SberTech)

### Overview
- **Developer:** SberTech (СберТех)
- **Product:** AI-ассистент разработчика
- **Version:** GigaCode 2.0 (released March 2026)
- **IDE Support:** Visual Studio Code, JetBrains IDEs, GigaIDE

### Features
- Autocomplete
- Agent mode (autonomous code generation)
- Code review
- Integration with GigaChat models

### Pricing
- **Free Tier:** Limited usage
- **GigaChat API:** Free and paid packages available (as of February 2026)
- **Note:** Exact enterprise pricing requires contacting SberTech directly

### Enterprise Capabilities
- Cloud-based solution
- Integration with Sber ecosystem
- Agent mode for autonomous task completion

---

## 2. SourceCraft (Yandex Cloud)

### Overview
- **Developer:** Yandex Cloud
- **Product:** SourceCraft Code Assistant
- **Release:** General availability June 2025
- **IDE Support:** Visual Studio Code, JetBrains IDEs

### Features
- AI-powered code completion
- Chat mode
- Agent mode (released September 2025)
- Code review (NeuroReview)
- PR description generation
- Code navigation
- CI/CD integration

### Pricing (RUB, incl. VAT)

#### Platform Plans

| Plan | Price | Storage | CI Minutes | Features |
|------|-------|---------|------------|----------|
| **Free** | Free | 2 GB | 1,000/mo | Basic |
| **Pro** | 250 RUB/user/mo | 10 GB | 3,000/mo | Full |

#### Code Assistant Plans (as of April 2026)

| Plan | Neurocredits | Autocomplete | Price |
|------|--------------|--------------|-------|
| **Free** (personal org only) | 500 | 4,000 | Free |
| **Pro** (max 5 seats, until July 2026) | 4,000 | 4,000,000 | **700 RUB/workspace/mo** |
| **Pro** (from July 2026) | 2,000 | Paid extension | **700 RUB/workspace/mo** |

#### Additional Services
- Extra CI time: 0.80 RUB/minute
- Extra storage: 50 RUB/GB
- Security (private repos): 3,000 RUB/active committer/mo
- Extra neurocredits: 1 RUB/credit

### Enterprise Capabilities
- Cloud-only deployment
- Integration with Yandex Cloud
- SSO via SAML (limited, free tier)
- 90-day free trial for Pro

---

## 3. Koda (KodaCode)

### Overview
- **Developer:** Koda (Russian startup)
- **Product:** AI-помощник для разработчиков
- **Status:** Beta
- **IDE Support:** VS Code, JetBrains IDEs, CLI

### Features
- Agent mode
- Multi-model support
- Russian language optimization
- Works without VPN in Russia

### Pricing (RUB)

| Plan | Autocomplete | Daily Base | Monthly Pro | External Models |
|------|--------------|------------|-------------|-----------------|
| **Free** | Unlimited | 300/day | 300/mo | 300/mo |
| **Paid** | Unlimited | 150/day (2nd mo) | 250/mo (2nd mo) | 0/mo (2nd mo) |

### Notes
- Free tier includes Koda Base 300/day
- External models (Codex, GLM): 300 credits/month free
- Pro tier: 500 RUB/month (first month), then 250 RUB/month

---

## 4. Qodo (formerly CodiumAI)

### Overview
- **Company:** Qodo (Israeli with Russian market presence)
- **Product:** AI code quality platform
- **Status:** Active in Russian market

### Features
- PR code review
- Test generation
- IDE assistance
- CLI tooling

### Pricing (USD converted to RUB)

| Plan | Price | PRs Included |
|------|-------|---------------|
| **Developer** | Free | 30 PRs/mo |
| **Teams** | ~1,500 RUB/mo ($19) | Unlimited |
| **Enterprise** | Contact sales | Unlimited |

### Note
- Originally CodiumAI, rebranded to Qodo
- Not a Russian company but actively marketed in Russian market

---

## 5. Benchmark: International Solutions

### Tabnine
- **Price:** $39/user/month (Pro), $89/user/month (Enterprise)
- **Features:** Local + cloud autocomplete, multi-language
- **Self-hosted:** Available (Enterprise)

### Sourcegraph Cody
- **Price:** $59/user/month (Enterprise)
- **Features:** Code search + AI, context-aware
- **Self-hosted:** Available (Enterprise)

### GitHub Copilot
- **Price:** $10/user/month (Individual), $19/user/month (Business)
- **Features:** Codex-based, IDE integration

---

## 6. Comparison Matrix

| Feature | GigaCode | SourceCraft | Koda | Tabnine | Cody |
|---------|----------|-------------|------|---------|------|
| **Autocomplete** | Yes | Yes | Yes | Yes | Yes |
| **Chat/Agent** | Yes | Yes | Yes | Limited | Yes |
| **Code Review** | Yes | Yes | Limited | No | Yes |
| **Self-hosted** | No | No | No | Enterprise | Enterprise |
| **Russian Lang** | Yes | Yes | Yes | Limited | Limited |
| **No VPN (Russia)** | Yes | Yes | Yes | N/A | N/A |
| **Enterprise** | Contact | 250 RUB | Contact | $89/mo | $59/mo |
| **Free Tier** | Limited | Yes | Yes | Limited | Limited |

---

## 7. Enterprise Considerations

### On-Premise/Self-Hosted
- **Russian solutions:** No current on-premise offerings identified
- **Tabnine Enterprise:** Available (local deployment)
- **Sourcegraph Enterprise:** Available (self-hosted)

### Regulatory Compliance
- Russian solutions operate under Russian data localization laws
- Cloud solutions require data transfer considerations

### Integration Ecosystem
- **SourceCraft:** Yandex Cloud ecosystem
- **GigaCode:** Sber/GigaChat ecosystem
- **Koda:** Independent, focused on IDEs

---

## 8. Market Context

### Russian Market Projections
- Up to 30% of Russian developers expected to use domestic AI solutions by 2026 (Izvestia, September 2025)
- Government initiatives supporting import substitution in IT

### Key Trends (2026)
1. Agent mode becoming standard feature
2. Transition from free to freemium models
3. Enterprise features expanding
4. Language model improvements for Russian

---

## 9. Recommendations

### For Russian Enterprises
- **Small teams (<10):** SourceCraft Pro (700 RUB/user) or Koda (250 RUB/mo)
- **Large enterprises:** Contact Sber/GigaCode for custom enterprise terms
- **Security-sensitive:** Consider Tabnine Enterprise ($89/user) for self-hosted option

### For Technology Transfer Scenarios
- SourceCraft offers best Russian market fit with clear pricing
- GigaCode positioned as full-stack development platform
- International alternatives remain superior for self-hosted requirements

---

## Sources

- Yandex Cloud SourceCraft Pricing (sourcecraft.dev)
- GigaCode/SberTech official materials
- Koda official website (kodacode.ru)
- Tproger, VC.ru, Habr articles
- TechBehemoths, PeerSpot comparisons

---

*Research conducted using web search across Russian and English sources. Pricing verified as of March 2026.*
