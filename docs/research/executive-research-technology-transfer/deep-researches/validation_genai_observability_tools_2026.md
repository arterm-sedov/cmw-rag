# GenAI/LLM Observability Tools and Frameworks

**Research Date:** March 2026  
**Classification:** Executive Technology Transfer Research  
**Focus:** Enterprise-grade solutions for Russian market (152-FZ compliance)

---

## Executive Summary

This research evaluates current GenAI/LLM observability tools with focus on enterprise readiness, self-hosted deployment options, and suitability for Russian regulatory context (Federal Law No. 152-FZ on Personal Data). The landscape has matured significantly since early 2025, with OpenTelemetry's GenAI semantic conventions reaching development status and multiple open-source options achieving production maturity.

---

## 1. OpenTelemetry GenAI Semantic Conventions

### Current Status (March 2026)

**Status:** Development (not yet stable)

The OpenTelemetry GenAI semantic conventions are actively evolving with ongoing development across multiple areas:

- **Spans:** Defines standard attributes for LLM inference, embeddings, and retrieval operations
- **Metrics:** Standardized metrics for token usage, latency, and costs
- **Events:** Event-based telemetry for streaming and content capture

### Included Fields

**Core Span Attributes:**
- `gen_ai.operation.name` - Operation type (chat, generate_content, text_completion)
- `gen_ai.provider.name` - Provider identifier (openai, aws.bedrock, gcp.gen_ai)
- `gen_ai.request.model` / `gen_ai.response.model` - Model identifiers
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` - Token counts
- `gen_ai.usage.cache_read.input_tokens` / `gen_ai.usage.cache_creation.input_tokens` - Cache metrics
- `gen_ai.conversation.id` - Session/conversation tracking
- `gen_ai.request.temperature` / `gen_ai.request.top_p` / `gen_ai.request.max_tokens` - Request parameters
- `gen_ai.response.finish_reasons` - Completion reasons
- `server.address` / `server.port` - Connection details

**Opt-In Fields (Sensitive Data):**
- `gen_ai.input.messages` - Full chat history (PII risk)
- `gen_ai.output.messages` - Model responses
- `gen_ai.system_instructions` - System prompts
- `gen_ai.tool.definitions` - Tool definitions

### In Development

- **Agentic Systems Conventions** - GitHub Issue #2664 (August 2025)
- **Task Conventions** (`gen_ai.task.*`) - PR #2713 (September 2025)
- **User-Centric Entry Spans** - Issue #3418 (February 2026)
- **Session ID Attribute** - Issue #2883 (October 2025)
- **Citations Support** - Issue #1952 (February 2025)

### Migration Notes

Existing instrumentations using v1.36.0 or prior should:
1. NOT change convention versions by default
2. Introduce `OTEL_SEMCONV_STABILITY_OPT_IN` environment variable
3. Use `gen_ai_latest_experimental` to opt into latest conventions

**Recommendation:** Monitor stable release (expected Q3-Q4 2026). For production Russian deployments, consider using stable attributes only and avoid opt-in sensitive fields to minimize PII handling under 152-FZ.

---

## 2. LangSmith (LangChain)

### Overview

Enterprise-grade observability platform from the creators of LangChain. Strong integration with LangChain ecosystem but usable standalone.

### Capabilities

| Feature | Support |
|---------|---------|
| Traces | Full |
| Metrics | Yes (latency, token usage, costs) |
| Evaluation | Built-in evaluation framework |
| Cost Tracking | Detailed per-model, per-request |
| Dataset Management | Yes |
| A/B Testing | Yes |
| Runtime Debugging | Excellent (playground, chain visualization) |

### Pricing Model

- **Free Tier:** 5,000 traces/month
- **Pay-as-you-go:** $0.50/1,000 traces (paid plans)
- **Team/Enterprise:** Custom pricing, volume discounts

Pricing is trace-volume based. Enterprise includes:
- SSO/SAML
- Role-based access control
- Audit logs
- Higher retention

### Data Residency

**Key Limitation for Russian Market:**
- **No self-hosted option** - Cloud-only
- **Data centers:** US/EU only
- **No data residency options** for Russian jurisdiction
- **GDPR compliant** but no 152-FZ specific guarantees

**Verdict:** Not suitable for Russian deployments requiring data localization. Best for EU-based operations with no Russian data handling.

---

## 3. Arize Phoenix

### Overview

Open-source LLM observability platform with strong enterprise backing. One of the most popular OSS observability solutions.

### Capabilities

| Feature | Support |
|---------|---------|
| Traces | Full (OpenTelemetry compatible) |
| Metrics | Yes (customizable) |
| Evaluation | Built-in eval framework |
| Cost Tracking | Manual (via token counting) |
| Dataset Management | Yes |
| OTEL Integration | Native |

### Self-Hosted Option

**Fully self-hosted available:**
- Docker Compose deployment
- Kubernetes support
- PostgreSQL backend required
- No cloud dependency

**Resources:**
- 9k+ GitHub stars
- 2.5M+ monthly downloads
- 2.4M+ OTel instrumentations

### Enterprise Features

- **Phoenix Enterprise:** Managed cloud option (SaaS)
- **Self-hosted:** Full control, no data leaves infrastructure
- **Open source:** No license costs

### Data Privacy

- Self-hosted option ensures full data residency control
- Can be deployed within Russian Federation
- Compatible with 152-FZ requirements when self-hosted
- No PII leaves infrastructure in self-hosted mode

**Verdict:** Strong recommendation for Russian market. Open-source + self-hosted = full compliance control.

---

## 4. Langfuse

### Overview

Open-source LLM engineering platform focused on debugging and iteration. Strong community presence, actively developed.

### Capabilities

| Feature | Support |
|---------|---------|
| Traces | Full |
| Metrics | Limited (basic) |
| Evaluation | Via datasets |
| Cost Tracking | Basic (manual calculation) |
| Dataset Management | Yes |
| Annotations | Yes |

### Self-Hosted Deployment

**Deployment Options:**
- Docker Compose (simplest)
- Kubernetes (production)
- Serverless (limited)

**Architecture:**
- PostgreSQL database (required)
- Optional: ClickHouse for analytics
- Object storage for traces/prompts

**Version:** Currently v3 (v2 EOL Q1 2025)

**Requirements:**
- 2+ CPU cores
- 4GB+ RAM
- PostgreSQL instance
- S3-compatible storage (or local)

### Enterprise Features

- **Langfuse Cloud:** Managed SaaS
- **Self-hosted:** Full OSS feature set
- **Add-on features:** Some require license key in self-hosted

### Data Privacy

- Self-hosted option: Full data residency
- Can be deployed in Russia
- No external data transmission in self-hosted mode
- 152-FZ compliant when properly deployed

**Pricing:**
- Cloud: Free tier + paid plans
- Self-hosted: Free (OSS)

**Verdict:** Good option for Russian market. Active development, strong community.

---

## 5. Other Enterprise Options

### Portkey AI

**Type:** AI Gateway + Observability  
**Self-hosted:** Limited (enterprise only)

| Feature | Support |
|---------|---------|
| Traces | Yes |
| Metrics | Yes |
| Cost Tracking | Yes |
| AI Gateway | Yes (1600+ models) |

**Enterprise Features:**
- SOC 2 Type II
- ISO 27001
- Custom deployments

**Data Residency:** Enterprise options available, contact sales

**Pricing:** Custom enterprise pricing

**Verdict:** Good for enterprise, limited Russian data residency info.

### Helicone

**Type:** Open-source observability proxy  
**GitHub:** 5,342 stars

**Features:**
- One-line integration
- Request/response logging
- Caching layer
- Cost optimization

**Self-hosted:** Yes (fully open-source)
- Docker deployment
- Rust-based (high performance)
- AI Gateway also available (open-source)

**Data Residency:** Full control with self-hosted

**Verdict:** Good lightweight option, strong for cost optimization.

### Additional Options

| Tool | Type | Self-Hosted | Russian Market |
|------|------|-------------|-----------------|
| **SigNoz** | OSS (full-stack) | Yes | Good |
| **PostHog** | OSS (with LLM features) | Yes | Good |
| **Weave** | SaaS | No | Limited |
| **Weights & Biases** | SaaS | No | Limited |
| **Datadog** | Enterprise | Via agent | Limited |
| **New Relic** | Enterprise | Via agent | Limited |

---

## 6. Feature Comparison Matrix

| Tool | Traces | Metrics | Eval | Cost Tracking | Self-Hosted | OTEL |
|------|--------|---------|------|---------------|-------------|------|
| **LangSmith** | Full | Full | Built-in | Full | No | Partial |
| **Arize Phoenix** | Full | Custom | Built-in | Manual | Yes | Native |
| **Langfuse** | Full | Basic | Via datasets | Manual | Yes | Export |
| **Helicone** | Full | Basic | No | Yes | Yes | No |
| **Portkey AI** | Full | Full | No | Full | Enterprise | Yes |
| **SigNoz** | Full | Full | No | Manual | Yes | Native |

---

## 7. Russian Market Compliance (152-FZ)

### Requirements

1. **Data Localization:** Personal data must be stored on servers in Russia
2. **Consent:** Explicit consent for data processing
3. **Cross-border Transfer:** Restricted for non-friendly countries
4. **Data Processing:** Must register with Roskomnadzor

### Recommended Solutions

**Tier 1 - Fully Compliant:**
- **Arize Phoenix** (self-hosted)
- **Langfuse** (self-hosted)
- **Helicone** (self-hosted)
- **SigNoz** (self-hosted)

**Tier 2 - Potential (verify):**
- Portkey AI (enterprise)
- Custom OpenTelemetry + Grafana stack

**Tier 3 - Not Recommended:**
- LangSmith (cloud-only, US/EU)
- Weights & Biases
- Weave

### Architecture Recommendation

For Russian enterprise deployments:

```
┌─────────────────────────────────────────────────┐
│                  Application                      │
├─────────────────────────────────────────────────┤
│              LangChain / LlamaIndex              │
├─────────────────────────────────────────────────┤
│         OpenTelemetry SDK (gen_ai semconv)      │
├─────────────────────────────────────────────────┤
│    Self-Hosted Observability Stack               │
│  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Phoenix   │  │   PostgreSQL            │  │
│  │   (traces)  │  │   (data storage)        │  │
│  └─────────────┘  └─────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   SigNoz    │  │   Object Storage        │  │
│  │  (metrics)  │  │   (traces/blobs)         │  │
│  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────┘
              Deploy in Russia (Mojel/Regyard)
```

---

## 8. Recommendations

### For Russian Enterprise (152-FZ Compliance)

1. **Primary:** Arize Phoenix (self-hosted)
   - Best maturity
   - Native OTEL support
   - Strong eval framework
   - Active community

2. **Alternative:** Langfuse (self-hosted)
   - Good debugging features
   - Active development
   - Simpler deployment

3. **Lightweight:** Helicone
   - Minimal resource usage
   - Good for cost optimization
   - Proxy architecture

### For EU Enterprise (GDPR)

1. **Cloud-first:** LangSmith (if acceptable data residency)
2. **Self-hosted:** Arize Phoenix / Langfuse
3. **Full-stack:** SigNoz (traces + metrics + logs)

### Migration Path

Current best practice:
1. Start with OpenTelemetry instrumentation (provider-agnostic)
2. Export to any backend (Phoenix, SigNoz, etc.)
3. Switch backends without code changes

---

## 9. Updates Since March 2026

No major breaking changes observed in observability space. Key trends:

- **OpenTelemetry:** GenAI conventions moving toward stability
- **Open-source:** Increasing adoption, enterprise features maturing
- **AI Gateways:** Growing market (Portkey, Helicone)
- **Evaluation:** becoming standard feature (Phoenix, LangSmith)

---

## Sources

- OpenTelemetry GenAI Semantic Conventions (opentelemetry.io)
- Arize Phoenix documentation (arize.com, phoenix.arize.com)
- Langfuse self-hosting docs (langfuse.com)
- LangSmith pricing (langchain.com, langsmith.com)
- Portkey AI enterprise docs (portkey.ai)
- Helicone GitHub (github.com/Helicone/helicone)
- Industry comparisons (PostHog, TrueFoundry, Maxim AI)
