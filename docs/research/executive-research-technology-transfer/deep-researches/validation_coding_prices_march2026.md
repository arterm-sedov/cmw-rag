# Coding Agent Pricing Validation — March 2026

**Research Date:** March 29, 2026  
**Exchange Rate:** 85 RUB/USD  
**Purpose:** Validate and update Appendix E pricing figures for cloud coding agents and LLM APIs

---

## Executive Summary

This document provides verified March 2026 pricing for five leading AI coding agents and LLM platforms. All prices are converted to Russian rubles (RUB) at 85 RUB/USD.

**Key Findings:**
- **OpenCode Go**: $10/month — VERIFIED (700-850 RUB depending on conversion)
- **Cursor Teams**: $40/user/month — VERIFIED (3,400 RUB/user/month)
- **GitHub Copilot Business**: $19/user/month — VERIFIED (1,615 RUB/user/month)
- **Claude Code**: API-based pricing ($6-12/day avg) — UPDATED
- **Windsurf (Codeium)**: Pricing changed March 2026 — UPDATED

---

## 1. OpenCode Pricing

### 1.1 Current Plans

| Plan | USD/month | RUB/month | Notes |
|------|-----------|-----------|-------|
| Free (core) | $0 | 0 | 75+ LLM providers, BYOK |
| Go (beta) | $5 first month, then $10 | 425→850 | GLM-5, Kimi K2.5, MiniMax M2.5/M2.7 |
| Zen (pay-as-you-go) | ~$20 | ~1,700 | Optimized models, pay-per-use |
| Enterprise | Custom | Custom | Per-seat, SSO, internal gateway |

### 1.2 OpenCode Go Usage Limits

- 5-hour limit: $12 equivalent
- Weekly limit: $30 equivalent
- Monthly limit: $60 equivalent

### 1.3 Enterprise Pricing

OpenCode Enterprise uses per-seat pricing. If organizations use their own LLM gateway, no token charges apply. Contact sales for custom quotes.

**Source:** _«[OpenCode Go](https://opencode.ai/go)»_, _«[OpenCode Enterprise](https://opencode.ai/docs/enterprise/)»_ (verified March 2026)

### 1.4 Validation vs Appendix E

| Figure | Appendix E (old) | March 2026 (verified) | Status |
|--------|------------------|----------------------|--------|
| OpenCode Go | $10/month (850 RUB) | $10/month (850 RUB) | ✅ VERIFIED |
| OpenCode Enterprise | Custom | Custom | ✅ VERIFIED |

---

## 2. Cursor Pricing

### 2.1 Individual Plans

| Plan | USD/month | RUB/month | Key Features |
|------|-----------|-----------|--------------|
| Hobby (Free) | $0 | 0 | 2,000 completions, 50 slow requests |
| Pro | $20 | 1,700 | Unlimited completions, $20 credit pool |
| Pro+ | $60 | 5,100 | 3x usage multiplier |
| Ultra | $200 | 17,000 | 20x usage, priority features |

### 2.2 Team/Business Plans

| Plan | USD/month | RUB/month | Key Features |
|------|-----------|-----------|--------------|
| Teams | $40/user | 3,400/user | Pro-equivalent, centralized billing, SSO |
| Enterprise | Custom | Custom | Pooled usage, invoice billing, SCIM |

**Source:** _«[Cursor Pricing](https://cursor.com/pricing)»_, _«[Cursor Teams Documentation](https://cursor.com/docs/account/teams/pricing)»_ (verified March 2026)

### 2.3 Validation vs Appendix E

| Figure | Appendix E (old) | March 2026 (verified) | Status |
|--------|------------------|----------------------|--------|
| Cursor Teams (5 users) | 17,000 RUB | 17,000 RUB (5×3,400) | ✅ VERIFIED |
| Cursor Teams (10 users) | 34,000 RUB | 34,000 RUB (10×3,400) | ✅ VERIFIED |
| Cursor Pro | 1,700 RUB | 1,700 RUB | ✅ VERIFIED |

---

## 3. GitHub Copilot Pricing

### 3.1 Individual Plans

| Plan | USD/month | RUB/month | Premium Requests |
|------|-----------|-----------|------------------|
| Free | $0 | 0 | 50/month |
| Pro | $10 | 850 | 300/month |
| Pro+ | $39 | 3,315 | 1,500/month |

### 3.2 Organization Plans

| Plan | USD/month | RUB/month | Premium Requests | Notes |
|------|-----------|-----------|------------------|-------|
| Business | $19/user | 1,615/user | 300/user/month | SSO, audit logs, IP indemnity |
| Enterprise | $39/user | 3,315/user | 1,000/user/month | + Knowledge bases, custom models |

**Note:** Enterprise requires GitHub Enterprise Cloud ($21/user/month additional), making true cost ~$60/user/month.

**Source:** _«[GitHub Copilot Plans](https://github.com/features/copilot/plans)»_, _«[GitHub Copilot Billing](https://docs.github.com/en/copilot/concepts/billing/organizations-and-enterprises)»_ (verified March 2026)

### 3.3 Validation vs Appendix E

| Figure | Appendix E (old) | March 2026 (verified) | Status |
|--------|------------------|----------------------|--------|
| Copilot Business (5 users) | 8,075 RUB | 8,075 RUB (5×1,615) | ✅ VERIFIED |
| Copilot Business (10 users) | 16,150 RUB | 16,150 RUB (10×1,615) | ✅ VERIFIED |
| Copilot Pro | 850 RUB | 850 RUB | ✅ VERIFIED |

---

## 4. Claude Code / Anthropic API Pricing

### 4.1 Claude Code Model

Claude Code is not a standalone product — it's a CLI tool that uses Anthropic's API. Pricing depends on subscription plan or direct API usage.

### 4.2 Subscription Plans

| Plan | USD/month | RUB/month | Notes |
|------|-----------|-----------|-------|
| Free | $0 | 0 | Limited access, Sonnet only |
| Pro | $20 | 1,700 | 5x Free, Claude Code included |
| Max 5x | $100 | 8,500 | 5x Pro usage |
| Max 20x | $200 | 17,000 | 20x Pro usage |

### 4.3 Team Plans

| Plan | USD/month | RUB/month | Notes |
|------|-----------|-----------|-------|
| Team Standard | $25/user | 2,125/user | Min 5 seats, billed annually |
| Team Premium | $150/user | 12,750/user | Includes Claude Code |
| Enterprise | Custom | Custom | SSO, audit logs, dedicated support |

### 4.4 API Pay-Per-Use Pricing

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Context |
|-------|----------------------|----------------------|---------|
| Haiku 4.5 | $0.25 | $1.25 | 200K |
| Sonnet 4.6 | $3.00 | $15.00 | 200K |
| Opus 4.6 | $5.00 | $25.00 | 200K |

**Extended Context Pricing (>200K tokens):**
- Input: $6.00/M tokens (vs $3.00 standard)
- Output: $22.50/M tokens (vs $15.00 standard)

### 4.5 Real-World Usage Costs

Anthropic reports average Claude Code usage:
- **Average developer**: ~$6/day
- **90th percentile**: ~$12/day
- **Monthly projection**: $100-200/developer (Sonnet 4.6)

**Source:** _«[Claude Pricing](https://www.anthropic.com/api/pricing)»_, _«[Claude Code Pricing 2026](https://o-mega.ai/articles/claude-code-pricing-2026-costs-plans-and-alternatives)»_ (verified March 2026)

### 4.6 Validation vs Appendix E

| Figure | Appendix E (old) | March 2026 (verified) | Status |
|--------|------------------|----------------------|--------|
| Claude Pro | N/A | 1,700 RUB/month | ✅ NEW |
| Claude Max 20x | N/A | 17,000 RUB/month | ✅ NEW |
| API (avg developer) | N/A | $6-12/day (~5,100-10,200 RUB/month) | ✅ NEW |

---

## 5. Windsurf (Codeium) Pricing

### 5.1 Pricing Update — March 2026

Windsurf announced new pricing effective **March 19, 2026**:

| Plan | USD/month | RUB/month | Key Features |
|------|-----------|-----------|--------------|
| Free | $0 | 0 | 25 credits/month, limited models |
| Pro | $20 | 1,700 | Quota-based (replaces credits), all models |
| Teams | $40/user | 3,400/user | Centralized billing, admin, ZDR |
| Max (NEW) | $200 | 17,000 | Highest quotas, priority support |
| Enterprise | Custom | Custom | SSO, compliance, self-hosted |

**Note:** Previous pricing was Pro $15/month, Teams $30/user — these were grandfathered for existing subscribers.

**Source:** _«[Windsurf Pricing Update](https://windsurf.com/blog/windsurf-pricing-plans)»_ (March 18, 2026)

### 5.2 Validation vs Appendix E

| Figure | Appendix E (old) | March 2026 (verified) | Status |
|--------|------------------|----------------------|--------|
| Windsurf Teams (old) | 17,000 RUB (5 users) | Now 17,000 RUB | ✅ UPDATED |
| Windsurf Teams (old) | 34,000 RUB (10 users) | Now 34,000 RUB | ✅ UPDATED |
| Windsurf Pro (old) | N/A | Now 1,700 RUB (was 1,275) | ✅ UPDATED |
| Windsurf Max | N/A | 17,000 RUB | ✅ NEW TIER |

---

## 6. Summary Comparison Table

### 6.1 Team Pricing (5-10 Users) — Monthly Cost in RUB

| Solution | 5 Users | 10 Users | Per-User |
|----------|---------|----------|----------|
| OpenCode Go* | 4,250 | 8,500 | 850 |
| GitHub Copilot Business | 8,075 | 16,150 | 1,615 |
| Cursor Teams | 17,000 | 34,000 | 3,400 |
| Windsurf Teams | 17,000 | 34,000 | 3,400 |
| GitHub Copilot Enterprise | 25,500+ | 51,000+ | 5,100+ |

*OpenCode Go: only one subscription per workspace; additional users need separate subscriptions or BYOK

### 6.2 Individual Developer Pricing

| Solution | Monthly Cost (RUB) | Best For |
|----------|-------------------|----------|
| OpenCode (BYOK) | 0 + API costs | Flexible, provider-agnostic |
| OpenCode Go | 850 | Low-cost open models |
| GitHub Copilot Pro | 850 | Basic autocomplete + chat |
| Cursor Pro | 1,700 | Agent mode, multi-file editing |
| Windsurf Pro | 1,700 | Cascade agents |
| Claude Pro + API | 1,700 + usage | Best reasoning, terminal-native |

---

## 7. Key Validation Results

### 7.1 Previously Cited Figures — Validation Status

| Key Figure | Old Value (Appendix E) | March 2026 Verified | Status |
|------------|----------------------|---------------------|--------|
| OpenCode Go | 850 RUB/month | 850 RUB/month | ✅ VERIFIED |
| Cursor Teams (5) | 17,000 RUB/month | 17,000 RUB/month | ✅ VERIFIED |
| Cursor Teams (10) | 34,000 RUB/month | 34,000 RUB/month | ✅ VERIFIED |
| Copilot Business (5) | 8,075 RUB/month | 8,075 RUB/month | ✅ VERIFIED |
| Copilot Business (10) | 16,150 RUB/month | 16,150 RUB/month | ✅ VERIFIED |

### 7.2 New Information Added

- Claude Code subscription options (Pro, Max, Team)
- Claude Code API pay-per-use pricing
- Claude Code real-world usage costs ($6-12/day)
- Windsurf March 2026 pricing update (new Max tier)
- Extended context pricing for Claude (>200K tokens)
- OpenCode Enterprise pricing model

---

## 8. Sources

- [OpenCode Go](https://opencode.ai/go)
- [OpenCode Enterprise](https://opencode.ai/docs/enterprise/)
- [Cursor Pricing](https://cursor.com/pricing)
- [Cursor Teams Documentation](https://cursor.com/docs/account/teams/pricing)
- [GitHub Copilot Plans](https://github.com/features/copilot/plans)
- [GitHub Copilot Billing](https://docs.github.com/en/copilot/concepts/billing/organizations-and-enterprises)
- [Claude Pricing](https://www.anthropic.com/api/pricing)
- [Windsurf Pricing](https://windsurf.com/pricing)
- [Windsurf Pricing Update](https://windsurf.com/blog/windsurf-pricing-plans)
- [Cursor Pricing Comparison](https://www.nxcode.io/resources/news/cursor-ai-pricing-plans-guide-2026)
- [Claude Code Pricing 2026](https://o-mega.ai/articles/claude-code-pricing-2026-costs-plans-and-alternatives)
