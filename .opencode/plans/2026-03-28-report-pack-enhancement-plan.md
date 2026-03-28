# Plan: Report-Pack Enhancement — C-Level Sales Enablement

**Date:** 2026-03-28  
**Status:** Ready for execution  
**Scope:** 10 files in `docs/research/executive-research-technology-transfer/report-pack/`  
**Business Goal:** Make this pack the **definitive C-Level reference** for Comindware's AI implementation and expertise transfer sales motion.

---

## Business Purpose Alignment

### The Pack's Job

Comindware sells **AI implementation + expertise transfer** — not "research PDFs". The pack enables C-Level executives to:

1. **Quickly brief** themselves before client meetings
2. **Build credible proposals** with evidence-backed numbers
3. **Handle objections** confidently (ROI, security, residency, lock-in)
4. **Navigate** between methodology and economics seamlessly
5. **Compose sales kits** for specific deals without opening the repo

### Pack Architecture

```
COMMERCIAL OFFER          ← C-Level pitch: "What we sell"
├── Main Methodology       ← Deep: TOM, phases, KT, compliance
├── Main Sizing           ← Deep: CapEx/OpEx/TCO, scenarios, tariffs
├── Appendix A            ← Navigation + sources registry
├── Appendix B            ← KT/IP/legal framework
├── Appendix C            ← Comindware stack boundaries
├── Appendix D            ← Security, compliance, observability
├── Appendix E            ← Market signals (supplementary)
├── Exec Methodology      ← 1-2 pager: methodology essence
└── Exec Sizing           ← 1-2 pager: economics essence
```

---

## Critical Finding

Deep research identified that the three "43%" CMO figures are from **different sources** and correctly disambiguated in Appendix E (lines 179, 182) but need verification across other files.

---

## Execution Phases

### Phase 1: Cosmetic & Structural (PRISTINE presentation)
**Why first:** C-Level will notice typos, missing `---`, broken formatting.

| # | Action | File | Issue |
|---|--------|------|-------|
| 1.1 | Fix YAML closing `---` | Commercial Offer | Missing delimiter |
| 1.2 | Fix YAML closing `---` | Appendix E | Missing delimiter |
| 1.3 | Fix duplicate YAML tags | Main:M | `методология` appears twice |
| 1.4 | Fix duplicate YAML tags | Main:S | `сайзинг` appears twice |
| 1.5 | Fix duplicate YAML tags | Appendix A | `методология` appears twice |
| 1.6 | Fix duplicate YAML tags | Appendix D | `безопасность`, `комплаенс`, `наблюдаемость` each appear twice |
| 1.7 | Fix tag typo | Commercial Offer | `продажипродажи` → `продажи` |

**Verification:** All YAML blocks properly closed; each tag unique.

---

### Phase 2: Cross-Reference Integrity
**Why:** Executives navigate via links; broken links destroy credibility.

| # | Action | File | Issue |
|---|--------|------|-------|
| 2.1 | Add to task manifest §1б | `20260324-research-task.md` | Append E + Commercial Offer missing |
| 2.2 | Add Appendix E to "Связанные документы" | Appendix A | Cross-ref missing |
| 2.3 | Add Appendix E to "Связанные документы" | Appendix B | Cross-ref missing |
| 2.4 | Add Appendix E to "Связанные документы" | Appendix D | Cross-ref missing |
| 2.5 | Add Appendix E to "Связанные документы" | Main:M | Cross-ref missing |
| 2.6 | Add Appendix E to "Связанные документы" | Main:S | Cross-ref missing |
| 2.7 | Normalize ~44 anchors in Appendix E | Appendix E | `sizing_`/`method_` → `app_e_` |
| 2.8 | Update ~8 incoming anchor references | Multiple files | Broken links to Appendix E |

**Note:** Commercial Offer uses "Навигация в комплекте" — verify this matches AGENTS.md.

---

### Phase 3: Sources & Citations (Evidence credibility)
**Why:** C-Level needs traceable claims for proposal defense.

| # | Action | File | Issue |
|---|--------|------|-------|
| 3.1 | Add `## Источники` section | Commercial Offer | Section missing |
| 3.2 | Fix citation format | Commercial Offer | Plain `[brackets]` → `_«[...]»_` |
| 3.3 | Verify `## Источники` | Exec Methodology | Section exists, verify completeness |
| 3.4 | Verify `## Источники` | Exec Sizing | Section exists, verify completeness |

---

### Phase 4: Currency Consistency (Financial credibility)
**Why:** Executives cite numbers in proposals; inconsistent FX destroys trust.

| # | Action | File | Issue |
|---|--------|------|-------|
| 4.1 | Add FX policy reference | Commercial Offer | No `#app_a_fx_policy` link |
| 4.2 | Verify FX policy reference | Appendix B | Check if present, add if missing |
| 4.3 | Fix USD figure without RUB | Main:S ~line 1009 | `~$0,001–0,005/токен` needs conversion |

---

### Phase 5: Statistics Validation (Claims credibility)
**Why:** Three 43% figures from different sources — must never be conflated.

| # | Action | File | Issue |
|---|--------|------|-------|
| 5.1 | Audit "43%" references | Main:S | Verify correct disambiguation |
| 5.2 | Audit "43%" references | Exec Sizing | Verify correct disambiguation |
| 5.3 | Audit "43%" references | Appendix D | Verify correct disambiguation |
| 5.4 | Audit "43%" references | Appendix E | Verify lines 179, 182 accurate |

**Deep research verdict:** Appendix E correctly disambiguates all three 43% figures. Verify other files reference correctly.

---

### Phase 6: Objection Handling (Sales enablement)
**Why:** C-Level needs ready answers for client pushback.

| # | Action | File | Issue |
|---|--------|------|-------|
| 6.1 | Add objection handling section | Commercial Offer | After "Типовые компромиссы" |

**Draft content:** Address ROI proof, vendor lock-in, security/residency, integration complexity.

---

### Phase 7: Deep Content Validation (Quality assurance)
**Why:** Ensure pack tells coherent, non-contradictory story.

| # | Action | Scope |
|---|--------|-------|
| 7.1 | Cross-validate key figures | 212,500₽ context, 4-6mo break-even, 40-60% threshold |
| 7.2 | Validate external links | NIST, OWASP, CBR, Cloud.ru, McKinsey/BCG/Bain |
| 7.3 | Verify GigaChat-3.1 MIT attribution | Sizing, Appendix B, Appendix E |

---

## Rollback Plan

```bash
git status
git checkout -- <file>  # Restore individual file
```

---

## Verification Checklist

After completion:

- [ ] All YAML blocks properly closed with `---`
- [ ] No duplicate YAML tags
- [ ] No `продажипродажи` typo
- [ ] Appendix E in all "Связанные документы" sections
- [ ] All anchor references resolve
- [ ] `## Источники` in Commercial Offer
- [ ] Citation format `_«[...]»_` in Commercial Offer
- [ ] FX policy reference in Commercial Offer
- [ ] USD figures have RUB conversions
- [ ] Three 43% figures never conflated
- [ ] Objection handling in Commercial Offer
- [ ] External links work
- [ ] Key figures consistent across files

---

## File Reference

| # | Short | Key Issues |
|---|-------|------------|
| 1 | Comm. Offer | YAML `---`, typo, missing Источники, no FX ref, citation format, objection handling |
| 2 | App A | Cross-refs, duplicate tags |
| 3 | App B | Cross-refs, FX ref |
| 4 | App C | Cross-refs |
| 5 | App D | Cross-refs, duplicate tags, 43% check |
| 6 | App E | YAML `---`, anchor normalization |
| 7 | Exec:M | Источники verification |
| 8 | Exec:S | 43% check |
| 9 | Main:M | Cross-refs, duplicate tags |
| 10 | Main:S | Cross-refs, duplicate tags, USD figure |

**Task file:** `tasks/20260324-research-task.md`
