# Enterprise AI Coding Agents Market Research — March 2026

**Дата исследования:** 29 марта 2026 г.  
**Курс USD → RUB:** 85 RUB/USD  
**Назначение:** Сравнительный анализ pricing и enterprise-функций ведущих AI-ассистентов программирования

---

## Executive Summary

Рынок enterprise AI-ассистентов программирования в 2026 году характеризуется консолидацией трёх основных игроков (Cursor, Windsurf, GitHub Copilot) и появлением нишевых решений для compliance-требовательных сегментов (Tabnine, Sourcegraph Cody). Ключевой тренд — переход от кредитной модели к квотам с ежедневным/недельным обновлением (Windsurf, март 2026). Все основные вендоры предлагают SSO/SAML, но полноценная on-premise поддержка доступна только у Tabnine и частично у Sourcegraph.

---

## 1. Cursor (Anysphere)

### Pricing (2026)

| Plan | Price (USD/month) | Price (RUB/month) | Key Features |
|------|-------------------|-------------------|--------------|
| **Hobby** | Free | Free | Limited agent requests, 2,000 tab completions/month |
| **Pro** | $20 | 1,700 | $20 credit pool, unlimited tab completions, frontier models |
| **Pro+** | $60 | 5,100 | 3x usage credits ($70), unlimited completions |
| **Ultra** | $200 | 17,000 | 20x usage ($400), priority feature access |
| **Teams** | $40/user | 3,400/user | $20 credits/user, centralized billing, SSO, analytics, org-wide privacy mode |
| **Enterprise** | Custom | Custom | Pooled usage, invoice billing, SCIM, audit logs, priority support |

**Notes:**
- Annual billing saves 20% across all paid tiers.
- Teams plan introduced in 2025 at $40/user/mo.
- Bugbot add-on: $40/user/mo (Pro), $40/user/mo (Teams), custom (Enterprise).
- Cursor crossed $1B ARR in 2025; over 1M paying developers.

### Enterprise Features

| Feature | Teams | Enterprise |
|---------|-------|------------|
| SSO (SAML/OIDC) | ✅ | ✅ |
| SCIM | ❌ | ✅ |
| Centralized billing | ✅ | ✅ |
| Usage analytics | ✅ | ✅ |
| Org-wide privacy mode | ✅ | ✅ (enforced) |
| Role-based access control | ✅ | ✅ |
| Invoice/PO billing | ❌ | ✅ |
| AI code tracking API | ❌ | ✅ |
| Audit logs | ❌ | ✅ |
| On-premise | ❌ | ❌ |
| Local models support | ❌ | ❌ |

---

## 2. Windsurf (Cognition, formerly Codeium)

### Pricing (2026)

| Plan | Price (USD/month) | Price (RUB/month) | Key Features |
|------|-------------------|-------------------|--------------|
| **Free** | Free | Free | 25 prompt credits/mo, unlimited tab, limited models |
| **Pro** | $20 (since March 2026; was $15) | 1,700 | 500 prompt credits/mo, all premium models, SWE-1.5, Fast Context |
| **Max** | $200 | 17,000 | Significantly higher quotas, priority support |
| **Teams** | $40/user | 3,400/user | 500 credits/user, centralized billing, admin analytics, priority support |
| **Enterprise** | $60/user | 5,100/user | 1,000 credits/user, RBAC, SSO+SCIM, hybrid deployment |

**Notes:**
- March 2026: Windsurf replaced credit system with quota system (daily/weekly refresh).
- Previous Pro pricing was $15/month; increased to $20 in March 2026.
- SSO available as add-on: +$10/user/month for Teams.
- Enterprise: hybrid deployment option, FedRAMP High support.
- Self-service SSO coming soon to Enterprise.
- Add-on credits: $10 for 250 credits (Pro), $40 for 1,000 pooled credits (Teams/Enterprise).
- ARR ~$82M (July 2025); 350+ enterprise customers; 4,000+ enterprise deployments.

### Enterprise Features

| Feature | Teams | Enterprise |
|---------|-------|------------|
| SSO (SAML/OIDC) | +$10/user | ✅ (included) |
| SCIM | ❌ | ✅ |
| Centralized billing | ✅ | ✅ |
| Usage analytics | ✅ | ✅ |
| Zero data retention (ZDR) | ✅ (auto) | ✅ |
| Role-based access control | ❌ | ✅ |
| Hybrid deployment | ❌ | ✅ |
| On-premise | ❌ | ❌ (hybrid only) |
| Local models support | ❌ | ❌ |
| FedRAMP | ❌ | ✅ (High) |

---

## 3. GitHub Copilot (Microsoft)

### Pricing (2026)

| Plan | Price (USD/month) | Price (RUB/month) | Key Features |
|------|-------------------|-------------------|--------------|
| **Free** | Free | Free | 2,000 completions/mo, 50 chat requests/mo |
| **Pro** | $10 | 850 | Unlimited completions, 300 premium requests/mo, all models |
| **Pro+** | $39 | 3,315 | 1,500 premium requests/mo, priority access to new features |
| **Business** | $19/user | 1,615/user | 300 premium requests/user, org management, IP indemnity, SAML SSO, audit logs |
| **Enterprise** | $39/user | 3,315/user | 1,000 premium requests/user, fine-tuned custom models, knowledge bases, GitHub.com chat |

**Notes:**
- Business requires GitHub Team or higher.
- Enterprise requires GitHub Enterprise Cloud ($21/user/mo extra), making true cost $60/user/mo.
- Overages: $0.04 per additional premium request.
- Students and teachers: free on Pro.
- Popular open source maintainers: free on Pro.
- GitHub Copilot is cloud-only; no on-premise option.

### Enterprise Features

| Feature | Business | Enterprise |
|---------|----------|------------|
| SSO (SAML/OIDC) | ✅ | ✅ |
| SCIM | ✅ | ✅ |
| Centralized billing | ✅ | ✅ |
| Usage analytics | ✅ | ✅ |
| Audit logs | ✅ | ✅ |
| IP indemnification | ✅ | ✅ |
| Content exclusions | ✅ | ✅ |
| Custom fine-tuned models | ❌ | ✅ |
| Knowledge bases | ❌ | ✅ |
| On-premise | ❌ | ❌ |
| Local models support | ❌ | ❌ |

---

## 4. Other Notable Enterprise AI Coding Tools

### 4.1 Tabnine

**Positioning:** Enterprise-grade privacy and compliance. The only major vendor with full on-premise and air-gapped deployment.

| Plan | Price (USD/month) | Price (RUB/month) |
|------|-------------------|-------------------|
| **Basic** | Free | Free |
| **Pro** | $12 | 1,020 |
| **Enterprise** | From $39/user | From 3,315/user |

**Enterprise Features:**
- ✅ SSO/SAML
- ✅ On-premise deployment
- ✅ VPC deployment
- ✅ Air-gapped environments
- ✅ Zero data retention
- ✅ Private model fine-tuning on customer code
- ✅ SOC 2 Type II, ISO 27001
- ✅ 80+ languages, all major IDEs

**Notes:** Tabnine is a Visionary in Gartner Magic Quadrant for AI Code Assistants (September 2025). Unique for fully local AI models — code never leaves customer infrastructure.

---

### 4.2 Sourcegraph Cody (now "Amp")

**Positioning:** Best for large monorepos and code graph understanding.

| Plan | Price (USD/month) | Price (RUB/month) |
|------|-------------------|-------------------|
| **Free** | Free | Free |
| **Pro** | $19 | 1,615 |
| **Enterprise** | From $59/user | From 5,015/user |

**Enterprise Features:**
- ✅ SSO/SAML
- ✅ Self-hosted deployment
- ✅ Private cloud
- ✅ Multi-repo context
- ✅ Code graph integration

**Notes:** Enterprise-only for organizations with 500+ developers. Strong for monorepo navigation and code search.

---

### 4.3 Amazon Q Developer

**Positioning:** AWS ecosystem integration.

| Plan | Price (USD/month) | Price (RUB/month) |
|------|-------------------|-------------------|
| **Free** | Free | 50 agentic requests, 1,000 lines/month |
| **Pro** | $19 | 1,615 |

**Enterprise Features:**
- ✅ SSO via AWS IAM
- ✅ AWS-specific context
- ✅ Security scanning
- ❌ On-premise

---

### 4.4 Claude Code (Anthropic)

**Positioning:** Terminal-first, open-source-agnostic.

| Plan | Price (USD/month) | Price (RUB/month) |
|------|-------------------|-------------------|
| **Pro** | $20 | 1,700 |
| **Team** | $30/user | 2,550/user |

**Enterprise Features:**
- ✅ SSO (via organization)
- ✅ MCP support
- ❌ On-premise
- ❌ Local models (depends on deployment)

---

### 4.5 Cline (Open Source)

**Positioning:** Developer-focused, open source, local-first.

| Plan | Price (USD/month) | Price (RUB/month) |
|------|-------------------|-------------------|
| **Free** | Free | — |

**Notes:** Model usage billed separately (API costs). Fully self-hostable. Growing adoption among developers wanting transparency and control.

---

## 5. Enterprise Features Comparison Summary

| Feature | Cursor | Windsurf | GitHub Copilot | Tabnine | Sourcegraph Cody |
|---------|--------|----------|----------------|---------|------------------|
| **SSO/SAML** | ✅ (Teams+) | ✅ (+$10/user on Teams) | ✅ | ✅ | ✅ |
| **SCIM** | ❌ (Enterprise only) | ✅ (Enterprise) | ✅ | ✅ | ✅ |
| **On-premise** | ❌ | ❌ | ❌ | ✅ | ✅ (self-hosted) |
| **Air-gapped** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Local models** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Zero data retention** | Org-wide (Enterprise) | ✅ (auto on Teams) | ✅ | ✅ | ✅ |
| **IP indemnification** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Audit logs** | ❌ (Enterprise only) | ✅ (Enterprise) | ✅ | ✅ | ✅ |
| **Custom fine-tuned models** | ❌ | ❌ | ✅ (Enterprise) | ✅ | ❌ |
| **Compliance certifications** | SOC 2 | SOC 2, FedRAMP High | SOC 2 | SOC 2, ISO 27001 | SOC 2 |

---

## 6. Key Findings

1. **Price Leadership (Individual):** GitHub Copilot Pro at $10/month remains the best value for solo developers.

2. **Price Leadership (Teams):** Windsurf Teams at $30/user (or $40 with SSO) vs. Cursor Teams at $40/user. GitHub Copilot Business at $19/user is cheapest but lacks deep IDE integration.

3. **Enterprise Compliance:** Only Tabnine offers full on-premise and air-gapped deployment. Sourcegraph Cody offers self-hosted. All others are cloud-only.

4. **Local Models:** Tabnine is the only vendor offering fully local AI models with zero data leaving customer infrastructure.

5. **Pricing Trend (2026):** Windsurf shifted from credit system to quota system (March 2026), aligning with industry shift toward consumption-based billing tied to API costs.

6. **True Enterprise Cost:**
   - Cursor Enterprise: Custom (typically $40+/user)
   - Windsurf Enterprise: $60/user (+ SSO add-on)
   - GitHub Copilot Enterprise: $39/user + GitHub Enterprise Cloud $21/user = **$60/user total**

---

## 7. Sources

- cursor.com/pricing
- windsurf.com/pricing
- github.com/features/copilot/plans
- tabnine.com/enterprise
- Sourcegraph official documentation
- Various industry reviews: Vendr, SaaS Price Pulse, Verdent AI, UI Bakery (March 2026)
- TechCrunch, Reuters (ARR figures)
- Gartner Magic Quadrant (September 2025)

---

*Document prepared for internal research. Prices and features subject to change. Exchange rate: 1 USD = 85 RUB.*
