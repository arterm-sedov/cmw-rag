# Pricing & Sizing Consolidation — FINAL v2.2
**Date:** March 29, 2026  
**Status:** ✅ COMPLETE

---

## Summary of Changes

### What Was Done

1. **Added global pricing** to main report (lines 283-290)
   - GPT-5.4, Claude Sonnet 4.6, Claude Opus 4.6, Gemini 3.1 Pro

2. **Removed duplicate pricing** from Appendix E
   - Changed lines 446-467 to reference main report only
   - Added note: "Цены на модели — единый источник"

3. **Enhanced Russian pricing section** (lines 241-250)
   - Added GLM-4.6 pricing
   - Organized into sections: Российские / Китайские (доступны в РФ)

---

## Final Architecture: ONE Source of Truth

| Document | Role |
|----------|------|
| `20260325-research-report-sizing-economics-main-ru.md` | **SINGLE SOURCE** for all pricing/sizing |
| `20260325-research-executive-sizing-ru.md` | Summary — references main report |
| `20260325-research-appendix-e-market-technical-signals-ru.md` | Technical specs only — references main report |

---

## What Now Lives Where

**Main Report (SINGLE SOURCE):**
- Russian LLM pricing (GigaChat, YandexGPT, Cloud.ru) ✓
- Chinese models in Cloud.ru (GLM, Qwen, MiniMax) ✓
- Global comparison (GPT-5, Claude, Gemini) ✓
- Cost segments (Basic/Medium/Enterprise) ✓
- Token calculations ✓
- GPU pricing ✓

**Other Reports:**
- Reference main report for prices
- Keep technical content (specs, VRAM, capabilities)
- No duplicate pricing tables

---

## NOT Committed — Working Files Only

- `.opencode/plans/20260329-pricing-sizing-consolidation-plan.md`
- `deep-researches/pricing_reference_march2026.md` (quick ref, not canonical)
