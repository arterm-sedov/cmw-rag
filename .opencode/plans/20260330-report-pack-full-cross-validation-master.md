# Master plan: full report-pack cross-validation and harmonization

**Created:** 2026-03-30  
**Language:** English (internal); Russian deliverables in `report-pack/`  
**Policy:** No git commits for this engagement (user instruction).

## Goal

Deliver a coherent, evidence-aligned **10-file** research pack for C-level decision support: methodology, sizing, KT/IP, security, market signals, commercial summary—**better coherence, not larger volume**, without dropping validated data.

## Artifacts

| Artifact | Path |
|----------|------|
| Full-pack matrix | `docs/research/executive-research-technology-transfer/deep-researches/20260330-full-pack-validation-matrix.md` |
| Track memos T1–T5 | `docs/research/executive-research-technology-transfer/deep-researches/20260330-track-t*.md` |
| This master plan | `.opencode/plans/20260330-report-pack-full-cross-validation-master.md` |

## Checkpoints

### Checkpoint A — Inventory

**Status:** Complete (2026-03-30)

- Matrix covers all 10 `report-pack` files against the 8-dimension rubric (business, cross-doc, metrics, non-contradiction, DRY, structure, C-level rules, other).
- Prior trio-only validations merged (`master_validation_report`, `topic_coverage_analysis`, `internal_consistency_validation`, etc.).

**Delta A:** Identified GAPs: duplicate Russian market block in executive methodology; Yakov “13 трлн” without band in main methodology; hh.ru metrics split across files; Appendix C market duplicate of E.

### Checkpoint B — Evidence

**Status:** Complete (2026-03-30)

- High-impact Russian market figures for this sprint rely on existing `real_world_metrics_validation.md` (March 2026) and in-pack tables; no new live scrape required for acceptance.
- **Fencing:** Decree №124 **GDP influence** (~11+ трлн) vs Yakov **economic effect** band (~8–13 трлн) documented in matrix—distinct indicators.

**Delta B:** None conflicting; labels must stay explicit in prose when both appear.

### Checkpoint C — Patch list

**Status:** Complete (2026-03-30)

1. `20260325-research-executive-methodology-ru.md` — stub Russian market stats → Appendix E.
2. `20260325-research-report-methodology-main-ru.md` — Yakov band (two sections); hh line + E cross-link.
3. `20260325-research-appendix-b-ip-code-alienation-ru.md` — hh line + ~2.7× gloss + E link.
4. `20260325-research-appendix-c-cmw-existing-work-ru.md` — compress market context to pointers.
5. `20260325-research-executive-sizing-ru.md` — AGENTS inline link style for internal docs.

### Checkpoint D — Consistency pass

**Status:** Complete (2026-03-30)

- Verified: one `## Источники` per pack file; no duplicate headers.
- C-level bodies: no `rag_engine` / repo paths (grep on report-pack).
- Источники list: `20260325-research-executive-methodology-ru.md` Yakov/App E entries use plain `- [title](url)` bullets per AGENTS.

### Checkpoint E — Red-team read-through

**Status:** Complete (2026-03-30)

- Pack remains decision-support: harmonization **removed** duplicated market bullets from exec methodology in favor of Appendix E canon; macro bands explicit; decree vs analyst effect disambiguated in Appendix E.

## Execution note

Harmonization applied only where matrix flagged GAP or clear DRY violation; no drive-by refactors of Appendix D/E bodies.

## Completion

**Status:** **Closed** (2026-03-30). No remaining undocumented contradictions: GDP-influence target (Указ / стратегия) is labeled separately from Yakov «экономический эффект» band; hh metrics reconciled (~2,7× vs +170%).

---
**Delta (2026-03-30):** Checkpoints A–E complete. Artifacts: matrix, T1–T5 memos, report-pack harmonization (exec methodology, main methodology, appendix B/C/E, exec sizing). **No git commit.**
