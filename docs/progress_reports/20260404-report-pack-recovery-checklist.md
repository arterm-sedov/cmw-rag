# Report-Pack Recovery Checklist

## Goal

Recover the best final state from the last few hours of parallel edits by:

- preserving earlier reviewed smart-sync decisions,
- integrating useful additions from side branches,
- fixing real regressions,
- and keeping the final report-pack business-oriented, executive-legible, grounded, and free of editorial drift.

## File-by-File Plan

### 1. `20260325-research-report-methodology-main-ru.md`

**Restore integrity**
- Fix malformed markdown links for `Yandex Cloud`, `SberCloud`, and `MWS GPT`.

**Preserve good additions**
- Keep Gartner / BCG / McKinsey grounding where it strengthens executive argumentation.
- Keep stronger wording around data readiness and organizational transformation.

**Guardrails**
- Do not roll back the useful new evidence just to recover link syntax.

### 2. `20260325-research-report-sizing-economics-main-ru.md`

**Restore integrity**
- Fix the `Приложение E` trend cross-link back to `#app_e_trends_2026_summary`.

**Restore commercial precision**
- Re-check the `Российские модели (март 2026)` table.
- Remove over-broad equivalence wording like “Цены едины для Cloud.ru и SberCloud.”
- Reintroduce provider / SKU / price-date precision where it was stronger and more defensible.

**Preserve good additions**
- Keep stronger FinOps / observability / unit-economics framing where it improves the TCO narrative.

### 3. `20260325-research-appendix-a-index-ru.md`

**Restore integrity**
- Re-sync the two hardware-sizing links to the anchors that actually exist in the sizing report.

**Preserve good additions**
- Keep neutral wording improvements that do not break navigation.

### 4. `20260325-research-appendix-d-security-observability-ru.md`

**Restore useful legal navigation**
- Reintroduce a concise but reusable `EU AI Act` comparative context section and anchor.
- Restore enough structure for:
  - timeline,
  - scope,
  - penalties,
  - practical implication for EU-facing deals.

**Preserve good additions**
- Keep stronger OWASP / TRiSM / Agentic framing that remains grounded and business-relevant.
- Keep newer wording where it improves readability without losing legal utility.

**Guardrails**
- Avoid turning the section into legal drafting notes.
- Keep it as decision support, not legal memo prose.

### 5. `20260325-research-appendix-e-market-technical-signals-ru.md`

**Preserve aligned additions**
- Keep `MWS Octapi`, `Yandex Agent Atelier`, and voice-layer additions if sourceable and business-relevant.
- Keep useful enterprise-product framing that supports RF market positioning and corporate-agent architecture.

**Guardrails**
- Do not let it drift into product-marketing or unsupported hype.

### 6. `20260331-research-executive-unified-ru.md`

**Preserve aligned additions**
- Keep stronger business evidence and enterprise-adoption/value-gap framing.

**Reinstate concise compliance pointers if useful**
- Re-add brief references to draft-law / EU context only if they improve executive readiness without clutter.

**Guardrails**
- Stay concise and executive-facing.
- No editorial or author-facing wording.

## Verification Checklist

- Search for malformed markdown link patterns like `]](` and `[[`.
- Search for dead cross-links we already know about.
- Search for expected restored anchors in `Appendix D`.
- Check the edited wording for business orientation, source precision, and absence of task/workflow phrasing.

## Acceptance Criteria

- No malformed markdown links remain in edited sections.
- `Appendix A`, `Sizing`, and `Appendix E` cross-links are coherent.
- `Sizing` pricing language is defensible in a business / offer context.
- `Appendix D` again contains reusable EU comparative navigation.
- Useful net-new additions from parallel agents remain where they improve the pack.
- Final wording reads like an executive research pack, not an editing workspace.
