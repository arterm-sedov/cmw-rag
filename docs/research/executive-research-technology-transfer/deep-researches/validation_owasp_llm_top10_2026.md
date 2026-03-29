# Validation: OWASP LLM Top 10 2025 and Agentic Top 10 2026

**Research Date:** March 29, 2026  
**Status:** Validated and Updated

---

## Executive Summary

Both OWASP LLM Top 10 2025 and OWASP Agentic Top 10 2026 have been officially released. The current appendix D correctly references both documents with proper links. However, the appendix lacks detailed enumeration of all 10 threat categories and does not include prevalence statistics. This validation confirms the lists are accurate and provides additional data for appendix enrichment.

---

## 1. OWASP LLM Top 10 2025 — Validated List

**Official Source:** _«[OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/)»_

| # | Code | Category | Brief Description |
|---|------|----------|------------------|
| 1 | LLM01:2025 | Prompt Injection | Malicious input manipulates LLM behavior; includes direct and indirect (stored) injection |
| 2 | LLM02:2025 | Sensitive Information Disclosure | Model reveals confidential data, PII, credentials, or IP through outputs |
| 3 | LLM03:2025 | Supply Chain Vulnerabilities | Compromised models, datasets, dependencies, or third-party services |
| 4 | LLM04:2025 | Data and Model Poisoning | Training data or RAG corpora manipulated to introduce backdoors or biases |
| 5 | LLM05:2025 | Improper Output Handling | Unsanitized LLM output passed directly to downstream systems, APIs, or code execution |
| 6 | LLM06:2025 | Excessive Agency | LLM granted too much functionality/permissions, enabling unintended harmful actions |
| 7 | LLM07:2025 | System Prompt Leakage | Unauthorized exposure of foundational system instructions and operational logic |
| 8 | LLM08:2025 | Vector and Embedding Weaknesses | Security flaws in retrieval mechanisms, embedding spaces, or vector databases |
| 9 | LLM09:2025 | Misinformation | Generation of false or misleading content compounded by user overreliance |
| 10 | LLM10:2025 | Unbounded Consumption | Excessive resource utilization leading to DoS or "denial of wallet" cost attacks |

### Key Changes from 2023 Version

- **LLM02 rose from #6 to #2** — sensitive information disclosure now ranked as second-highest risk
- **LLM07 (System Prompt Leakage)** — new category, previously not explicitly listed
- **LLM08 (Vector/Embedding Weaknesses)** — new category addressing RAG-specific attack vectors
- **LLM10 expanded** — renamed from "Model Denial of Service" to "Unbounded Consumption" reflecting both compute and cost impacts

### Prevalence Data

Direct prevalence statistics for LLM-specific vulnerabilities are limited in public reporting. However:

- The 2025 list is based on real-world incidents, community feedback, and contributions from 100+ industry experts (_«[OWASP announcement](https://www.prnewswire.com/news-releases/owasp-reveals-updated-2025-top-10-risks-for-llms-announces-new-llm-project-sponsorship-program-and-inaugural-sponsors-302309429.html)»_)
- HackerOne's 2025 report notes a **surge in AI vulnerability reports** and sharp rise in prompt-injection findings (_«[Cycode analysis](https://cycode.com/blog/the-2025-owasp-top-10-addressing-software-supply-chain-and-llm-risks-with-cycode/)»_)
- Traditional OWASP Web Top 10 2025 data: ~318,000 instances of Broken Access Control (94% of apps tested); combined Injection vulnerabilities (XSS + SQLi) account for ~29% of all findings (_«[Cobalt pentest data](https://www.cobalt.io/blog/comparing-the-owasp-top-10-2025-with-real-world-pentest-data)»_)

---

## 2. OWASP Agentic Top 10 2026 — Validated List

**Official Source:** _«[OWASP Top 10 for Agentic Applications for 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)»_  
**Release Date:** December 9, 2025 (published for 2026)

| # | Code | Category | Brief Description |
|---|------|----------|------------------|
| 1 | ASI01:2026 | Agent Goal Hijack | Attackers manipulate agent's objectives or instructions to cause harmful outcomes |
| 2 | ASI02:2026 | Tool Misuse & Exploitation | Legitimate tools used unsafely; ambiguous prompts cause agents to call tools with destructive parameters |
| 3 | ASI03:2026 | Identity & Privilege Abuse | Compromise of agent credentials, tokens, or session identity; privilege escalation across agents |
| 4 | ASI04:2026 | Agentic Supply Chain Vulnerabilities | Compromised skills, MCP servers, models, or dependencies loaded at runtime |
| 5 | ASI05:2026 | Unexpected Code Execution (RCE) | Agents execute untrusted code, commands, or scripts leading to system compromise |
| 6 | ASI06:2026 | Memory & Context Poisoning | Adversarial manipulation of agent memory, context window, or shared state |
| 7 | ASI07:2026 | Insecure Inter-Agent Communication | Attacks on communication channels between agents; data interception or manipulation |
| 8 | ASI08:2026 | Cascading Failures | Failure in one agent triggers chain reactions across multi-agent systems |
| 9 | ASI09:2026 | Human-Agent Trust Exploitation | Users manipulated into trusting malicious agent actions; social engineering at AI interface |
| 10 | ASI10:2026 | Rogue Agents | Compromised or misaligned agents acting harmfully while appearing legitimate |

### Key Distinctions from LLM Top 10

The Agentic Top 10 addresses **autonomy, delegation, and multi-agent coordination** — risks that extend beyond single-response LLM applications. Each ASI maps to one or more LLM categories but introduces new attack vectors due to:
- Persistent memory across sessions
- Tool chaining and execution
- Inter-agent communication protocols
- Elevated privileges for autonomous actions

### Prevalence Statistics

| Metric | Figure | Source |
|--------|--------|--------|
| Skills scanned (Snyk ToxicSkills) | 3,984 | Snyk (Feb 2026) |
| Skills with security flaws | 1,467 (36.82%) | Snyk ToxicSkills (Feb 2026) |
| Skills with critical issues | 534 (13.4%) | Snyk ToxicSkills (Feb 2026) |
| Confirmed malicious payloads | 76+ | Snyk ToxicSkills (Feb 2026) |
| Malicious skills (ClawHavoc) | 1,184 | Antiy CERT (Feb 2026) |
| OpenClaw instances exposed | 135,000+ | SecurityScorecard (Feb 2026) |
| Correlated with prior breaches | 53,000+ | SecurityScorecard (Feb 2026) |
| CVEs in OpenClaw | 9 (3 with public exploits) | Endor Labs (Feb 2026) |
| MCP servers SSRF-vulnerable | 36.7% | BlueRock Security (2026) |
| Skills with vulnerabilities (industry) | >25% | National CIO Review / Cisco (2026) |

Sources: _«[OWASP Agentic Skills Top 10](https://owasp.org/www-project-agentic-skills-top-10/)»_, _«[Astrix Security](https://astrix.security/learn/blog/the-owasp-agentic-top-10-just-dropped-heres-what-you-need-to-know/)»_, _«[Aikido Security](https://www.aikido.dev/blog/owasp-top-10-agentic-applications)»_

---

## 3. Comparison with Current Appendix D

### What Appendix D Currently States

The security observability appendix (paragraphs 32, 62, 79, 107, 190) correctly references:

- **OWASP Top 10 for LLM Applications (2025)** — with correct link
- **OWASP Top 10 for Agentic Applications (2026)** — with correct link

### What's Missing

1. **Enumeration of all 10 categories** — appendix references the frameworks but does not list them
2. **Prevalence statistics** — no incident data or industry metrics
3. **Mapping between LLM and Agentic categories** — no cross-reference showing how ASI risks extend LLM risks

### Recommendation

Appendix D should be enhanced with:
- A table summarizing all 10 LLM categories (paragraph 32 or new subsection)
- A table summarizing all 10 Agentic categories (paragraph 62 or new subsection)
- Prevalence statistics section citing Snyk ToxicSkills and OpenClaw exposure data

---

## 4. 2025-2026 Updates Summary

| Document | Release Date | Status |
|----------|--------------|--------|
| OWASP LLM Top 10 2023/24 | 2023 | Superseded |
| OWASP LLM Top 10 2025 | Late 2024 (effective 2025) | **Current** |
| OWASP Agentic Top 10 2026 | December 9, 2025 | **Current** |

No additional updates or modifications to either list have been published as of March 2026.

---

## 5. Sources

- [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/)
- [OWASP Top 10 for Agentic Applications for 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
- [OWASP Agentic Skills Top 10](https://owasp.org/www-project-agentic-skills-top-10/)
- [OWASP LLM Top 10 2025 — Navigating the Risks](https://digital.nemko.com/news/navigating-the-owasp-top-10-for-llm-applications-2025)
- [Invicti — OWASP Top 10 for LLMs 2025](https://www.invicti.com/blog/web-security/owasp-top-10-risks-llm-security-2025/)
- [Qualys — OWASP Top 10 LLM Risks 2025](https://blog.qualys.com/vulnerabilities-threat-research/2024/11/25/ai-under-the-microscope-whats-changed-in-the-owasp-top-10-for-llms-2025)
- [Astrix Security — OWASP Agentic Top 10 Released](https://astrix.security/learn/blog/the-owasp-agentic-top-10-just-dropped-heres-what-you-need-to-know/)
- [Aikido Security — OWASP Top 10 for Agentic Applications](https://www.aikido.dev/blog/owasp-top-10-agentic-applications)
- [Teleport — OWASP Top 10 for Agentic Applications 2026](https://goteleport.com/blog/owasp-top-10-agentic-applications/)
- [Snyk ToxicSkills Report (Feb 2026)](https://securityscorecard.com/research/openclaw-enterprise-security-advisory/)
- [SecurityScorecard — OpenClaw Exposure (Feb 2026)](https://securityscorecard.com/)
- [Cycode — OWASP Top 10 2025 Analysis](https://cycode.com/blog/the-2025-owasp-top-10-addressing-software-supply-chain-and-llm-risks-with-cycode/)
- [Cobalt — OWASP Top 10 2025 vs Real-World Pentest Data](https://www.cobalt.io/blog/comparing-the-owasp-top-10-2025-with-real-world-pentest-data)
