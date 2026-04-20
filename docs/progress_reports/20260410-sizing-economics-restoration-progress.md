# Progress Report: Sizing Economics Document Restructuring

**Date:** 2026-04-10
**Status:** In Progress

## Completed Steps

### Step 1: Token Sizing Tables ✅

- Renamed H2 `## Экономический каркас и токен-экономика` → `## Модель затрат`
- Added H3 `### Калькуляция по классам задач`
- Added table "Класс агента (ориентир длины системного промпта)"
- Added table "Класс данных по длине пользовательского текста"

### Step 2: Risk Mitigation Table ✅

- Added H3 `### Риски внедрения ИИ-проектов` (renamed from "Риски бюджета")
- 12-row table with risks and mitigations
- Updated content:
  - "Дефицит GPU": Added practical alternatives (Russian clouds, consumer GPUs, Mac Studio 256GB, Chinese accelerators)
  - "Галлюцинации": Updated impact to "крах агента, репутация" + specific mitigations

### Step 3: Edge/Sovereign Section ✅

- Created new H2 `## Локальный и edge-инференс` (renamed from "Автономный инференс")
- Added H3 `### Потребительское железо для суверенности` (Qwen on Mac)
- Added H3 `### Edge-агенты на минимальном железе` (Picoclaw on Raspberry Pi)
- Added H3 `### Протоколы для корпоративных систем` (CLI vs MCP)

### Step 4: Technical Optimization Admonition ✅

- Added `!!! note "Ключевые публикации по оптимизации"` with links to:
  - Accenture Memex(RL)
  - Databricks KARL

## In Progress

### Step 5: Research Enhancements 🔄

**Hallucinations Impact Research:**
- Task launched to research hallucination impact on AI agents
- Focus: business impact, real incidents, mitigation approaches
- Will update risk table with specific data

**Chinese GPU Alternatives Research:**
- Task launched to research Huawei Ascend, Moore Threads, Cambricon, MetaX
- Focus: specifications, performance benchmarks, Russia/CIS availability
- Will add practical alternatives section if data is significant

### Step 6: Final Verification ⏳

Pending completion of research tasks.

## Statistics

| Metric | Original | Restored | Change |
|--------|----------|----------|--------|
| Lines | 1,275 | ~1,007+ | +~50 (research pending) |
| H2 Sections | 12 | 13 | +1 (Локальный и edge-инференс) |
| Tables | 23 | 25 | +2 (klassy, risk table) |

## Key Decisions

1. **Structure**: Moved from "Экономический каркас" to "Модель затрат" for business clarity
2. **Risks**: Expanded from budget-focused to implementation risks
3. **Edge/Sovereign**: Separated edge-AI from sovereign AI conceptually
4. **Technical**: Added links to research papers for credibility

## Files Modified

- `docs/research/executive-research-technology-transfer/report-pack/20260325-research-report-sizing-economics-main-ru_RESTRUCTURED.md`

## Files Created

- `.opencode/plans/20260410-sizing-economics-restoration-plan.md`
- `.opencode/plans/20260410-sizing-economics-restoration-step5.md`
- `docs/progress_reports/20260410-sizing-economics-restoration-progress.md`