# OpenCode AI Coding Assistant: Pricing and Capabilities Research 2026

**Research Date:** March 29, 2026  
**Exchange Rate Used:** 85 RUB/USD

---

## Executive Summary

OpenCode is an open-source AI coding assistant that operates as a terminal-based interface, desktop application, or IDE extension. The platform supports 75+ LLM providers and offers flexible deployment options ranging from individual use to enterprise deployments with full data sovereignty.

---

## Pricing Tiers Overview

| Tier | Monthly Price (USD) | Monthly Price (RUB) | Status |
|------|---------------------|-------------------|--------|
| OpenCode Go | $5 first month, then $10 | 850 RUB | Beta |
| OpenCode Black | Paused | Paused | Paused |
| Enterprise | Contact sales | Contact sales | Available |
| Open Source (DIY) | Free | Free | Stable |

---

## 1. OpenCode Go (Individual Plan)

### Pricing
- **First month:** $5 USD (425 RUB)
- **Subsequent months:** $10 USD (850 RUB) per month
- **Currently in beta**

### Included Models
OpenCode Go provides access to curated open-source coding models:

| Model | Type | Notes |
|-------|------|-------|
| GLM-5 | Open model | High capability |
| Kimi K2.5 | Open model | Strong coding performance |
| MiniMax M2.5 | Open model | Cost-effective option |
| MiniMax M2.7 | Open model | Latest version |

### Usage Limits

The Go plan includes tiered usage limits based on dollar value:

| Time Period | Dollar Limit | Est. Requests (GLM-5) | Est. Requests (Kimi K2.5) | Est. Requests (MiniMax M2.5) |
|-------------|--------------|----------------------|--------------------------|----------------------------|
| 5 hours | $12 | ~1,150 | ~1,850 | ~20,000 |
| Weekly | $30 | ~2,880 | ~4,630 | ~50,000 |
| Monthly | $60 | ~5,750 | ~9,250 | ~100,000 |

**Notes:**
- Limits are calculated in dollar value; actual request counts depend on model pricing
- Cheaper models (MiniMax M2.5) allow more requests than premium models (GLM-5)
- Users can track usage in the OpenCode Zen console
- If limits are exceeded, users can enable "Use balance" fallback to their Zen balance

### Model Endpoint Access
Go models are accessible via API endpoints:

| Model | Model ID | Endpoint |
|-------|----------|----------|
| GLM-5 | glm-5 | `https://opencode.ai/zen/go/v1/chat/completions` |
| Kimi K2.5 | kimi-k2.5 | `https://opencode.ai/zen/go/v1/chat/completions` |
| MiniMax M2.7 | minimax-m2.7 | `https://opencode.ai/zen/go/v1/messages` |
| MiniMax M2.5 | minimax-m2.5 | `https://opencode.ai/zen/go/v1/messages` |

### Geographic Availability
Models are hosted in the US, EU, and Singapore for stable global access. Providers follow a zero-retention policy and do not use user data for model training.

---

## 2. OpenCode Black (Premium Individual Plan)

### Status: Enrollment Temporarily Paused

OpenCode Black was designed to provide access to the world's best coding models including:
- Claude (Anthropic)
- GPT (OpenAI)
- Gemini (Google)
- Other premium models

**Note:** Black plan enrollment is currently paused as of March 2026. Interested users should check for updates.

---

## 3. OpenCode Enterprise

### Pricing Model
- **Structure:** Per-seat pricing (contact sales for quote)
- **Token charges:** No charges for tokens if you use your own LLM gateway
- **Contact:** contact@anoma.ly for pricing and implementation

### Enterprise Features

#### 3.1 Data Security
- **No data retention:** OpenCode does not store any code or context data
- **Local processing:** All processing happens locally or through direct API calls to your AI provider
- **Code ownership:** Full ownership of all code produced by OpenCode; no licensing restrictions or ownership claims

#### 3.2 SSO Integration
- Central config integrates with organization's SSO provider
- Credentials for internal AI gateway obtained through existing identity management system

#### 3.3 Internal AI Gateway Support
- Configure OpenCode to use only your internal AI gateway
- Disable all other AI providers to ensure requests go through approved infrastructure

#### 3.4 Centralized Configuration
- Single central config for entire organization
- Ensures all users access only internal AI gateway
- Can be managed at organizational level

#### 3.5 Self-Hosting Options
- Share pages can be self-hosted on your infrastructure (roadmap)
- Recommended: Disable share pages during trial to ensure data never leaves organization

```json
// Recommended config for enterprise trial
{
  "$schema": "https://opencode.ai/config.json",
  "share": "disabled"
}
```

#### 3.6 Private NPM Registry Support
OpenCode supports private npm registries through Bun's native `.npmrc` file support:
- JFrog Artifactory
- Nexus
- Other private registries

---

## 4. Self-Hosted and Local Models Support

### Question: Can OpenCode Enterprise connect to local/self-hosted models?

**Answer: Yes, absolutely.**

OpenCode supports self-hosted model providers that implement an OpenAI-compatible API, enabling:

- Running models on your own hardware
- Using custom inference servers
- Experimenting with fine-tuned or custom models
- Full data and privacy control

### Supported Inference Servers

| Server | Description | Port |
|--------|-------------|------|
| Ollama | Easy local model running | 11434 |
| LM Studio | Desktop application for running models | 1234 |
| vLLM | High-throughput inference server | Configurable |
| Text Generation WebUI | OpenAI-compatible API extension | 5000 |

### Configuration Example

```bash
# Set endpoint
export LOCAL_ENDPOINT="http://localhost:1234/v1"
```

```json
// .opencode.json
{
  "agents": {
    "coder": {
      "model": "local.llama-3.3-70b-instruct",
      "maxTokens": 5000
    }
  }
}
```

### Model Requirements

For optimal experience, self-hosted models should support:
- **Tool calling:** Required for file operations, code execution
- **Streaming:** Server-sent events (SSE) for better UX
- **Context window:** 32K+ tokens recommended for complex coding tasks

### Recommended Models for Self-Hosting
- Llama 3.3 70B Instruct
- Qwen 2.5 Coder
- Granite 3.1 (IBM)
- Mistral Large

---

## 5. Open Source / Free Tier

OpenCode core is open source (Apache 2.0 license) and free to use:

- **No cost** to download and use the application
- **Bring your own API key:** Connect to 75+ LLM providers
- **Full functionality:** Same core features as paid plans
- **Model flexibility:** Use any provider (OpenAI, Anthropic, Google, local models, etc.)

This makes OpenCode accessible for:
- Individual developers
- Teams with existing API budgets
- Organizations wanting to use their own LLM infrastructure

---

## 6. Summary Comparison

| Feature | Go ($10/mo) | Black (Paused) | Enterprise | Free/Open Source |
|---------|-------------|----------------|------------|------------------|
| **Price** | $10/month | Paused | Per-seat contact | Free |
| **Models** | 4 open models | Premium models | Bring your own | Bring your own |
| **SSO** | No | No | Yes | No |
| **Internal Gateway** | No | No | Yes | Yes |
| **Self-Hosted Models** | No | No | Yes | Yes |
| **Data Retention** | Provider-dependent | Provider-dependent | None | None |
| **Central Config** | No | No | Yes | No |
| **Support** | Community | Community | Dedicated | Community |

---

## Key Takeaways

1. **OpenCode Go** at $10/month (850 RUB) offers excellent value for access to curated open-source coding models with generous limits

2. **Enterprise** offers full data sovereignty with SSO integration, internal AI gateway support, and self-hosted model options - pricing is custom per organization

3. **Self-hosted/local models** are fully supported - OpenCode can connect to Ollama, LM Studio, vLLM, or any OpenAI-compatible API, making it ideal for organizations with internal AI infrastructure

4. **Free tier** is genuinely free - the open-source version works with your own API keys from any of 75+ providers

5. **Black plan** enrollment is paused - no timeline for reopening

---

## Sources

- OpenCode Official Documentation: https://opencode.ai/docs/
- OpenCode 
