# Research Documents Cross-Validation Plan

> **For agentic workers:** This plan focuses on document validation and alignment tasks, not code implementation. Tasks involve reading, comparing, and editing markdown files.

**Goal:** Cross-validate the 20260325 research pack documents against business goals of selling and transferring AI expertise, ensuring internal consistency and C-level readiness.

**Architecture:** Review 4 key documents (2 executive summaries + 2 main reports), verify cross-references, align terminology and metrics, strengthen messaging for selling/transferring AI knowledge.

**Scope:** 4 primary documents in `docs/research/executive-research-technology-transfer/report-pack/`

---

## Task 1: Verify Business Goal Alignment

**Files:**
- Review: `20260325-research-executive-methodology-ru.md`
- Review: `20260325-research-executive-sizing-ru.md`
- Review: `20260325-research-report-methodology-main-ru.md`
- Review: `20260325-research-report-sizing-economics-main-ru.md`

- [ ] **Step 1: Read executive methodology (lines 1-120)**
  
  Focus on: Value proposition, KT/BOT model, 30/60/90 plan

- [ ] **Step 2: Read executive sizing (lines 1-66)**
  
  Focus on: CapEx/OpEx ranges, currency section, role matrix

- [ ] **Step 3: Verify selling message alignment**
  
  Check: Does each document emphasize "суверенность" and "передача знаний (KT), а не подписка"?

- [ ] **Step 4: Verify transferring message alignment**
  
  Check: Does each document describe BOT model and handover timeline?

- [ ] **Step 5: Document alignment findings**
  
  Create notes on: Gaps, inconsistencies, strength of business goal messaging

---

## Task 2: Verify Currency/FX Consistency

**Files:**
- Modify: `20260325-research-executive-methodology-ru.md` (line 40-42)
- Modify: `20260325-research-executive-sizing-ru.md` (line 29-31)

- [ ] **Step 1: Check currency section in executive methodology**
  
  Current: `## Валюта {: #exec_method_currency }` pointing to Appendix A

- [ ] **Step 2: Check currency section in executive sizing**
  
  Current: `## Валюта {: #exec_sizing_currency }` pointing to Appendix A

- [ ] **Step 3: Decide on anchor consistency**
  
  Option A: Use same anchor `#exec_currency` in both
  Option B: Keep separate anchors (documented differently)
  Recommendation: Keep separate (different documents may evolve independently)

- [ ] **Step 4: Verify both point to correct appendix anchor**
  
  Target: `#app_a_fx_policy` in `20260325-research-appendix-a-index-ru.md`

- [ ] **Step 5: Commit changes if any**

---

## Task 3: Verify Terminology Consistency

**Files:**
- All 4 documents

- [ ] **Step 1: Check "отчуждение" usage**
  
  Verify: All documents use consistent terminology for knowledge transfer

- [ ] **Step 2: Check "BOT" model description**
  
  Verify: Build/Operate/Transfer phases described consistently

- [ ] **Step 3: Check KPI terminology**
  
  Verify: "Utilization", "ROI", "TCO" used consistently
  Check: Target metrics (>60% utilization, -30-40% ticket time)

- [ ] **Step 4: Check 152-FZ/compliance references**
  
  Verify: Consistent framing of Russian regulatory requirements

- [ ] **Step 5: Document terminology findings**

---

## Task 4: Verify Quantitative Claims

**Files:**
- Executive methodology (lines 66-72): Russian market figures
- Executive sizing: CapEx/OpEx ranges
- Main methodology: Implementation timelines
- Main sizing: Detailed economic tables

- [ ] **Step 1: Check Russian market figures**
  
  From executive methodology:
  - 13 трлн руб. к 2030 (Yakov Partners) ✓
  - 46% companies testing autonomous agents
  - 86% using open models
  - +62% job growth

- [ ] **Step 2: Check CapEx/OpEx ranges**
  
  Verify: Executive sizing matches main sizing report
  - SaaS: ~200-250 тыс. руб./мес
  - On-prem CapEx: ~7-11 млн руб.
  - Hybrid: ~0,3-1,5 млн руб.

- [ ] **Step 3: Check duplicate 43% warning**
  
  Verify: Executive sizing line 43 correctly warns about two different 43% figures (hallucinations vs data leak)

- [ ] **Step 4: Verify global benchmark caveats**
  
  Check: Both executive summaries include "выборка enterprise-клиентов" / "не типовой резидентный продакшн" caveats

- [ ] **Step 5: Document quantitative findings**

---

## Task 5: Verify Cross-References

**Files:**
- All 4 documents

- [ ] **Step 1: Check internal cross-references**
  
  Verify: Executive summaries reference main reports correctly
  - Executive methodology line 20: References commercial offer
  - Executive sizing line 20: References commercial offer

- [ ] **Step 2: Check appendix references**
  
  Verify: Both executive summaries reference Appendix A for currency

- [ ] **Step 3: Check source citations**
  
  Verify: External sources cited with URLs (not internal paths)

- [ ] **Step 4: Fix broken cross-references**

- [ ] **Step 5: Commit cross-reference fixes**

---

## Task 6: Strengthen Selling/Transferring Messaging

**Files:**
- Modify: `20260325-research-executive-methodology-ru.md`
- Modify: `20260325-research-executive-sizing-ru.md`

- [ ] **Step 1: Review current value proposition**
  
  From executive methodology lines 29-38:
  - Суверенность как конкурентное преимущество ✓
  - Передача знаний (KT), а не подписка ✓
  - Измеримый ROI ✓
  - Предсказуемое TCO ✓

- [ ] **Step 2: Enhance "transfer" messaging if needed**
  
  Consider: Explicit section on what client receives (code, documentation, training, runbook)

- [ ] **Step 3: Enhance "selling" messaging if needed**
  
  Consider: Clearer differentiation from subscription/SaaS competitors

- [ ] **Step 4: Verify 30/60/90 plan supports transfer**
  
  Check: Phase 3 (90 days) includes knowledge transfer activities

- [ ] **Step 5: Commit messaging enhancements**

---

## Task 7: Final Review and Checkpoints

**Files:**
- All 4 documents

- [ ] **Step 1: Run final consistency check**
  
  Verify:
  - [ ] SCQA structure in both executive summaries
  - [ ] Role decision tables present and aligned
  - [ ] Currency sections point to same appendix
  - [ ] No contradictory figures between documents
  - [ ] External sources use public URLs only
  - [ ] Business goal messaging (selling/transferring) is clear

- [ ] **Step 2: Verify document status**

  Check: All documents have correct YAML front matter (date, status, tags)

- [ ] **Step 3: Final cross-validation report**

  Summary: Document readiness for C-level use

---

## Checkpoint Summary

| Checkpoint | Status | Notes |
|------------|--------|-------|
| Business goal alignment | ✅ Done | Commit 0cc1917 - aligned with selling/transferring |
| Currency/FX consistency | ✅ Done | Commit 61abcc0 - currency terminology updated |
| Terminology consistency | ✅ Done | Utilization targets aligned (>60%) |
| Quantitative claims | ✅ Done | Commit 46f6d76 - 43% disclaimers added |
| Cross-references | ✅ Done | Anchors fixed, links formatted |
| Selling/transferring messaging | ✅ Done | Commit 0cc1917 |
| Final review | ✅ Done | Role tables aligned, YAML dates unified |
| Appendix A table trim | ✅ Done | Commit d2c8779 - 70+ → 22 rows |

## Execution Summary

**Completed:** March 29, 2026
**Commits:**
- `0cc1917` - exec: align methodology docs with business goals
- `61abcc0` - fix(report-pack): update currency terminology
- `46f6d76` - fix(report-pack): resolve D7/D8/D11, add 43% disclaimers
- `6b052f1` - docs(research): update section headings
- `1af926c` - docs(research): add missing heading anchors
- `255255b` - docs(research): update document titles and references
- `d2c8779` - docs(appendix-a): trim topic tables

**Result:** Research pack is C-level ready with consistent terminology, cross-references, and business goal messaging.
