# Lossless merge matrix: unified executive summary (2026-03-31)

## Scope

- Source A: `20260325-comindware-ai-commercial-offer-ru.md`
- Source B: `20260325-research-executive-methodology-ru.md`
- Source C: `20260325-research-executive-sizing-ru.md`
- Target: `20260331-research-executive-unified-ru.md`

## Coverage matrix (after polish)

| Source | Key block | Coverage in unified file | Status |
| :--- | :--- | :--- | :--- |
| A | Overview / how to use | `#exec_unified_how_to_use` | Kept as condensed |
| A | SCQA framing | `#exec_unified_decision_card` + `#exec_unified_offer` + `#exec_unified_market_limits` | Merged |
| A | What Comindware sells | `#exec_unified_offer` | Kept |
| A | Business value (economy/quality/risk axes) | `#exec_unified_offer` + `#exec_unified_economics` | Kept as condensed |
| A | Market context and caveats | `#exec_unified_market_limits` | Kept as condensed |
| A | Stage packages PoC/Pilot/Scale/BOT | `#exec_unified_packages` | Kept |
| A | Transfer artifacts | `#exec_unified_artifacts` | Kept |
| A | Role argument matrix | `#exec_unified_role_matrix` | Kept |
| A | Trade-offs cloud/on-prem/hybrid | `#exec_unified_economics` | Merged |
| A | Objections and responses | `#exec_unified_objections` | Kept |
| A | 30/60/90 actions | `#exec_unified_action_plan` | Kept |
| A | Drill-down navigation | `#exec_unified_drilldown` | Kept |
| A | Sources | `#exec_unified_sources` | Deduplicated |
| B | 60-second decision card | `#exec_unified_decision_card` | Kept as condensed |
| B | How to use (investment/C-level limits) | `#exec_unified_how_to_use` + warning in `#exec_unified_decision_card` | Kept |
| B | Sales value proposition | `#exec_unified_offer` | Kept |
| B | Currency policy | `#exec_unified_fx_policy` + link to Appendix A `#app_a_fx_policy` | Restored explicitly |
| B | Key decision theses (TOM, strategy, transfer, RF constraints) | `#exec_unified_offer` + `#exec_unified_economics` + `#exec_unified_market_limits` | Merged |
| B | 3-stage implementation model and transfer depth | `#exec_unified_packages` + `#exec_unified_artifacts` | Merged |
| B | 30/60/90 checklist details | `#exec_unified_action_plan` | Kept as condensed |
| B | Sources | `#exec_unified_sources` | Deduplicated |
| C | 60-second decision card | `#exec_unified_decision_card` | Kept as condensed |
| C | How to use for budgeting/negotiations | `#exec_unified_how_to_use` | Kept |
| C | Economic key theses (5 cost blocks, ownership model, GPU capacity) | `#exec_unified_economics` | Kept |
| C | KPI benchmark block | `#exec_unified_guardrails` + KPI link to Appendix A | Restored explicitly |
| C | Detailed references to sizing/main docs | `#exec_unified_drilldown` | Kept |
| C | Sources | `#exec_unified_sources` | Deduplicated |
| A/B/C | Role matrix consistency | `#exec_unified_role_matrix` | Fixed (duplicate CPO removed, COO restored) |
| B + Appendix B | Transfer acceptance controls | `#exec_unified_transfer_gates` + links to `#app_b_transfer_acceptance_criteria_checklist`, `#app_b_alienation_package_minimal` | Restored explicitly |
| B + Appendix D | Security/observability minimums | `#exec_unified_security_gates` + links to D anchors | Restored explicitly |
| B | SCQA explicitness | `#exec_unified_scqa` | Restored explicitly |

## Lossless check result (revalidated)

- Business-critical blocks previously over-condensed were restored as compact executive controls (KPI thresholds, FX policy, transfer/security gates, role consistency).
- Repeated narrative text was consolidated into canonical sections (single SCQA, single applicability warning, one role matrix, one package model).
- Deep derivations/tables remain in main reports and are linked via section anchors from the unified file.
- Existing three source files remain untouched for historical comparison and traceability.
