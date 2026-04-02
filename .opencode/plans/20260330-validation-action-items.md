# Validation Action Items — Executive Research Pack

**Created:** March 30, 2026  
**Status:** Pending Execution

---

## Critical Priority (Must Fix)

### 1. Add Sources Section to Main Methodology

**File:** `docs/research/executive-research-technology-transfer/report-pack/20260325-research-report-methodology-main-ru.md`

**Issue:** Missing required `## Источники` section per AGENTS.md requirements

**Action:**
- Review all inline citations in the document
- Add `## Источники` section at end of document
- Format sources as plain Markdown links (no guillemets, no italic)

**Template:**
```markdown
## Источники

- [Source Title](https://example.com)
- [Another Source](https://example2.com)
```

---

## High Priority (Should Fix)

### 2. Standardize Cross-References

**Files:** All three target documents

**Issue:** Some cross-references not using AGENTS.md standard format `_«[Title](path)»_`

**Action:**
- Audit all cross-references in each document
- Update to standard format:
  - Internal: `_«[Document Title](./relative/path.md#anchor)»_`
  - External: `_«[Source Title](https://...)»_`

### 3. Add Admonition Blocks

**Files:** All three target documents

**Issue:** Admonitions (callout blocks) severely underutilized

**Action:** Add appropriate admonitions for:
- `!!! tip "Рекомендация"` — for best practices and recommendations
- `!!! warning "Важно"` — for critical warnings and pitfalls
- `!!! note "Примечание"` — for additional context

**Example placement:**
- After KPI sections (utilization, efficiency, quality)
- Before action plans (30/60/90 days)
- Around regulatory compliance requirements

---

## Medium Priority (Consider)

### 4. Consolidate Market Data

**Current state:** Market statistics duplicated between Executive Methodology and referenced in Main Methodology

**Recommended approach:**
- Keep full market data in Main Methodology (as primary analytical document)
- Keep only key figures (2-3 bullets) in Executive Summary with reference to Main Methodology
- Format: "Подробнее см. _«[Методология внедрения](./20260325-research-report-methodology-main-ru.md#method_market_benchmarks)»_"

### 5. Unified Delivery Models Section

**Current state:** Delivery models (BOT, create-transfer) fragmented across documents

**Recommended approach:**
- Create consolidated delivery models overview in Main Methodology
- Reference Appendix B for detailed procedures
- Cross-link at both document and section level

### 6. Enhanced Regulatory Cross-Links

**Current state:** Regulatory information in multiple documents without explicit links

**Recommended approach:**
- Add explicit "См. также" references between regulatory sections
- Ensure 152-FZ, Presidential Decree №124, and NIST AI RMF references are consistent

---

## Validation Summary

| Priority | Issue | Effort | Impact |
|-----------|-------|--------|--------|
| Critical | Missing Sources section | Low | High |
| High | Cross-reference standardization | Medium | Medium |
| High | Admonition blocks | Medium | Medium |
| Medium | Market data consolidation | Medium | Low |
| Medium | Delivery models unification | High | Low |
| Medium | Regulatory cross-links | Low | Low |

---

## Execution Notes

- Do NOT commit changes unless explicitly requested by user
- All edits should maintain existing content (additive only)
- Preserve Russian language formatting and terminology
- Use subagents if parallel execution needed

---

*Action items generated from master_validation_report.md findings*