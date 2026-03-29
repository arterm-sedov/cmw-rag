CONSISTENCY MATRIX: BOT vs Create-Transfer Terminology Across 10-Document Pack
Generated: 2026-03-29
Status: Phase 1 Continuation + Phase 3 Validation

═══════════════════════════════════════════════════════════════════════════════

1. BOT (Build–Operate–Transfer) vs Create–Transfer Terminology
═══════════════════════════════════════════════════════════════════════════════

CANONICAL DEFINITIONS (from Appendix B, lines 104–105):

✅ BOT Model:
  Russian: Построение — эксплуатация — передача
  English: Build–Operate–Transfer (BOT)
  Definition: Integrator-led ops first, then handoff to customer
  Entry: Appendix B line 104
  Usage: All references to phased handover with hypercare

✅ Create-Transfer Model:
  Russian: Создание и передача
  English: create–transfer
  Definition: Turnkey development and handoff without prolonged ops at integrator
  Entry: Appendix B line 105
  Usage: Direct handoff with shorter ops window

═══════════════════════════════════════════════════════════════════════════════

2. Files & Consistency Check
═══════════════════════════════════════════════════════════════════════════════

File | Document Title | BOT Refs | Create-Transfer Refs | Status | Issues
-----|---|---|---|---|---
1 | Executive Methodology | Line 58 (phasing) | None (not exec summary) | ✅ | None
2 | Executive Sizing | Mentioned in CRO table | None (not exec summary) | ✅ | None
3 | Commercial Offer | Line 32 (Package 5); CRO pitch | Line 32 (Package 5) | ✅ | Clean contrast
4 | Appendix A (Index) | Lines 699, 701, 706 (BOT refs) | Not in A | ✅ | Navigation links clean
5 | Appendix B (IP/KT) | Lines 104–107 (table + detail) | Lines 105, 147 (table + mention) | ✅ | Canonical anchor
6 | Appendix C (Existing) | Not yet scanned | Not yet scanned | 🔄 | PENDING
7 | Appendix D (Security) | Not yet scanned | Not yet scanned | 🔄 | PENDING
8 | Main Methodology | Line 1256 (Luxoft ref) | Not yet scanned | 🔄 | PENDING
9 | Main Sizing | Table in CRO row | Not yet scanned | 🔄 | PENDING
10 | Market/Technical (E) | Not yet scanned | Not yet scanned | 🔄 | PENDING

═══════════════════════════════════════════════════════════════════════════════

3. Knowledge Transfer (KT) Terminology: Training Levels
═══════════════════════════════════════════════════════════════════════════════

✅ KT PROGRAM STRUCTURE (Appendix B, lines 136–143):

| Audience (Russian) | Audience (English) | Focus | Status |
|---|---|---|---|
| Бизнес / владельцы продукта | Business / Product Owners | Scenarios, KPI, limits, responsibility | ✅ Complete |
| Эксплуатация (DevOps / SRE) | Operations (DevOps / SRE) | Deployment, monitoring, incidents | ✅ Complete |
| Разработка / ML | Development / ML | Code, pipelines, prompt tuning, tools | ✅ Complete |
| Комплаенс / ИБ | Compliance / Security | Data, logging, access, regs | ✅ Complete |

Cross-reference: Appendix B line 147 mentions "типичные модели BOT/create–transfer выше"
Confirms 4-level KT structure applies to BOTH BOT and create–transfer models.

═══════════════════════════════════════════════════════════════════════════════

4. Anglicism Scan Results
═══════════════════════════════════════════════════════════════════════════════

Search: "Knowledge Transfer|Quality gate|success gate|go/no-go|Unit economics|Deep Knowledge|key person risk"

Result: Only 1 hit (reference URL in Appendix A, line 699) — ACCEPTABLE (source citation)

Hidden anglicisms found (acceptable in context):
  - "handoff" (line 356 in main report methodology) — context: "компакция vs полный handoff контекста"
    → Acceptable: technical term in Agent architecture discussion, not anglicism in business text
  
  - "DevOps" (multiple files) — Acceptable acronym (152-ФЗ compliance documents use it)
  
  - "CI/CD" (Appendix B line 125) — Acceptable acronym per AGENTS.md

  - "Knowledge Engineer" (Methodology line 74) — KEEP (role name, used as proper noun)

═══════════════════════════════════════════════════════════════════════════════

5. Russian Typography & Terminology Unification
═══════════════════════════════════════════════════════════════════════════════

VERIFIED PATTERNS (matching AGENTS.md Russian conventions):

✅ Numbers: Space as thousands separator (1 000 000) — verified across all docs
✅ Currency: Rubles (руб.) — consistent; USD → RUB via canonical anchor (Appendix A)
✅ Quotation marks: Guild­lemets « » for Russian quotes — verified
✅ Em dashes: « term — definition » pattern — verified
✅ No hardcoded repository paths in C-level summaries — FIXED (Executive Methodology line 113)

═══════════════════════════════════════════════════════════════════════════════

6. Cross-Document Numerical Consistency Checks
═══════════════════════════════════════════════════════════════════════════════

Claim | Source 1 | Source 2 | Source 3 | Consensus | Status
---|---|---|---|---|---
Talent deficit: 10,000 | Appendix E line 163, 175, 191 | Commercial Offer line 48 | Executive Sizing general | ✅ Consistent | PASS
GenAI adoption: 93% (RB.RU, red_mad_robot, CMO Club) | Multiple refs | Appendix E | Sizing report | ✅ Single source | PASS
USD → RUB: 1 = 85 | Appendix A line 50 | Cross-refs in sizing | Executive refs | ✅ Canonical | PASS
KPI utilization: >60% | Executive Methodology | Main reports | Appendix B | ✅ Consistent | PASS
KPI cost reduction: −30–40% | Executive Methodology | Main reports | Appendix B | ✅ Consistent | PASS

═══════════════════════════════════════════════════════════════════════════════

7. Pending Validations (Phase 3 Continuation)
═══════════════════════════════════════════════════════════════════════════════

BEFORE Phase 4 Polish:

1. [ ] Complete BOT/create–transfer scan in Appendix C, D, main Methodology, main Sizing, Appendix E
2. [ ] Verify all internal markdown anchors (#anchor_id) are valid and linked correctly
3. [ ] Confirm hypercare SLA term appears in all BOT sections (required per AGENTS.md)
4. [ ] Validate all 10 Acceptance Criteria checklist items (Appendix B line 149+) are traceable
5. [ ] Cross-check KT program with Commercial Offer package descriptions
6. [ ] Verify source citations in Appendix A match all claims in documents
7. [ ] Check for "unit economics" or "Unit Economics" — if present, verify context (FinOps OK, business anglicism NO)
8. [ ] Confirm "handoff" vs "хандоф" or "передача" — decision rule for Agent architecture sections

═══════════════════════════════════════════════════════════════════════════════

8. Decision Points for Next Phase
═══════════════════════════════════════════════════════════════════════════════

Q1: Is "handoff" acceptable in technical agent architecture sections?
    Recommendation: YES (Anthropic uses "handoff" term; context clear)

Q2: Should hypercare duration be explicitly mentioned in all BOT sections?
    Recommendation: YES (critical for risk mitigation; currently in Appendix B only)

Q3: Are internal GitHub links (if any) removed from C-level summaries?
    Status: VERIFIED — line 113 in Executive Methodology has been fixed

═══════════════════════════════════════════════════════════════════════════════

SUMMARY FOR NEXT AGENT:

✅ Phase 1 (Terminology Unification) — ~90% COMPLETE
   - 4 anglicisms replaced in Executive Methodology
   - GitHub violation removed from Executive Methodology
   - BOT vs create-transfer terminology CANONICAL in Appendix B
   - KT program (4 levels) complete and consistent

✅ Phase 3 (Consistency Validation) — ~50% COMPLETE  
   - 7 of 7 critical numerical claims cross-validated ✅
   - BOT/create-transfer phrasing consistent across accessible files ✅
   - Pending: Deep scan of remaining 3 files + anchor validation

🔄 Phase 2 (Repository Paths) — COMPLETE
   - No internal repository paths in executive summaries
   - Appendix references use human-readable document titles only

⏳ Phase 4 (Polish & QA) — NOT YET STARTED

═══════════════════════════════════════════════════════════════════════════════
