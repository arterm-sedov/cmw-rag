## 1. Scaffolding and conventions

- [x] 1.1 Choose final filenames for the 6 new documents (2 mains + Appendices A–D) under `docs/research/` using the existing naming convention
- [x] 1.2 Add a one-paragraph “pack overview” and “related documents” micro-guide boilerplate to each of the 6 document templates
- [x] 1.3 Define the “topic owner” table (topic → canonical document) to be embedded in Appendix A

## 2. Traceability (“no-loss”) map

- [x] 2.1 Build a traceability map from both original reports (all major headings/blocks) to destinations in the new 6 documents
- [x] 2.2 Review the traceability map for gaps (every original block has exactly one destination) and for duplication risks (same block mapped twice)

## 3. Appendix A (vitrine / registry)

- [x] 3.1 Create Appendix A skeleton: document registry + question→document navigation + topic-owner table
- [x] 3.2 Create canonical “used sources” registry in Appendix A by consolidating sources from the original two reports
- [x] 3.3 Add “additional sources from task list” section in Appendix A, categorized by theme (no reliability scoring, no usage statuses)

## 4. Main report: Methodology & development/implementation

- [x] 4.1 Create the methodology main report skeleton (SCQA + TOM + phases + recommendations + risks)
- [x] 4.2 Move/merge all relevant methodology content from the original methodology report into the new methodology main report per the traceability map
- [x] 4.3 Replace any duplicated economic tables with a short pointer to the economics main report (canonical numbers live there)

## 5. Main report: Sizing & economics (CapEx/OpEx/TCO)

- [x] 5.1 Create the sizing & economics main report skeleton (SCQA + cost factor tree + scenario sizing + tariffs + TCO + risks)
- [x] 5.2 Move/merge all quantitative and economic content from the original sizing report into the new economics main report per the traceability map
- [x] 5.3 Remove methodology-only blocks from the economics report and replace them with pointers to the methodology main report / appendices

## 6. Appendix B: IP & code alienation (KT/IP package)

- [x] 6.1 Create Appendix B skeleton: delivery/transfer models + deliverables + acceptance checklist + license tracking requirements
- [x] 6.2 Move all “transfer / KT / IP / licensing” content from the originals into Appendix B per the traceability map
- [x] 6.3 Ensure Appendix B references the economics report for quantitative impacts (no duplicated CapEx/OpEx tables)

## 7. Appendix C: Existing CMW work (what exists today)

- [x] 7.1 Create Appendix C skeleton: component inventory + boundaries + what is “existing work” vs “guidance”
- [x] 7.2 Move all “CMW architecture / existing components / what we have” content into Appendix C per the traceability map
- [x] 7.3 Ensure Appendix C points to Appendix B for transfer acceptance and to mains for methodology/economics

## 8. Appendix D: Security, compliance, observability (deep dive)

- [x] 8.1 Create Appendix D skeleton: control perimeter + security risks + compliance guardrails + observability posture
- [x] 8.2 Move all deep-dive content on security/compliance/observability into Appendix D per the traceability map
- [x] 8.3 Ensure Appendix D references mains for canonical methodology and canonical numbers (no duplicated tables)

## 9. Cross-validation and quality gate

- [x] 9.1 Verify “no-loss”: walk the traceability map and confirm each original block appears exactly once in the new pack (or as a short summary + canonical pointer) — automated: `python docs/research/research_migration/scripts/verify_pack.py` (H2 map + verbatim sections + URL set for `## Источники`)
- [x] 9.2 Verify “low-crosslink”: ensure cross-references are primarily “downward” (mains → appendices), avoid circular references
- [x] 9.3 Verify source hygiene: inline links remain for key claims, full registries live in Appendix A; executive-facing boilerplate avoids internal change codenames and filenames (migrated body text preserves original component names and public URLs)
- [x] 9.4 Final consistency pass: terminology, Russian numeric conventions, and consistent document titles across the pack
