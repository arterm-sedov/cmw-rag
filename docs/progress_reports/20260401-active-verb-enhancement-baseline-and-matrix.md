# Active verb enhancement — baseline, taxonomy, matrix

**Date:** 2026-04-01  
**Scope:** [docs/research/executive-research-technology-transfer/report-pack/](../research/executive-research-technology-transfer/report-pack/)  
**Policy:** Balanced — imperative active voice for directives; neutral voice for evidence/facts.

## Taxonomy

| Bucket | When to use | Examples |
|--------|-------------|----------|
| **Must-convert** | Decision rules, policies, «как использовать», КП/пресейл, чек-листы, риск-директивы | `рекомендуется X` → `сделайте X` / `X — рекомендуемый шаг` |
| **Optional-convert** | Mixed fact + implied action | Split: fact sentence + imperative sentence |
| **Keep-neutral** | Market data, citations, legal/regulatory description, third-party claims | `оценивается рынок`, `публикуется отчёт` |

## Section-level rules

- **Executive / «Как использовать» / политики:** prefer imperative or explicit «Comindware рекомендует…» + verb.
- **SCQA / выводы с цифрами:** keep neutral for the fact; add separate «Действие: …» if needed.
- **Архитектурные пайплайны (RAG шаги):** keep passive or use «модуль … выполняет» if clarity improves; avoid fake imperatives to the reader.
- **Приложения (IP, security):** convert handover/checklist language to active; keep statutory passive where accuracy requires.

## Baseline counts (heuristic grep)

**High-signal impersonal lexemes** (`рекомендуется`, `используется`, `определяется`, …):

| File | Count |
|------|------:|
| appendix-d-security-observability-ru.md | 17 |
| report-sizing-economics-main-ru.md | 14 |
| report-methodology-main-ru.md | 13 |
| appendix-b-ip-code-alienation-ru.md | 4 |
| appendix-c-cmw-existing-work-ru.md | 2 |
| executive-unified-ru.md | 2 |
| appendix-e-market-technical-signals-ru.md | 1 |
| appendix-a-index-ru.md | 0 |

**Broader `-ется/-ются` forms** (includes many neutral technical passives):

| File | Count |
|------|------:|
| appendix-d-security-observability-ru.md | 57 |
| report-methodology-main-ru.md | 70 |
| report-sizing-economics-main-ru.md | 35 |
| appendix-e-market-technical-signals-ru.md | 17 |
| appendix-b-ip-code-alienation-ru.md | 16 |
| executive-unified-ru.md | 13 |
| appendix-a-index-ru.md | 3 |
| appendix-c-cmw-existing-work-ru.md | 4 |

**Classification coverage:** 100% of high-signal lexeme hits addressed in editorial pass; broader passive forms reviewed in context (not line-by-line enumerated — would duplicate the source files).

## Candidate → decision matrix (summary)

| Pattern / area | Decision |
|----------------|----------|
| `рекомендуется` in guidance blocks | Convert to imperative or «закрепите / выберите / перепроверьте» |
| `принимаются решения` in policy | Convert to «принимайте решения» |
| `требуется` for reader obligation | Convert to «учтите / получите / включите» |
| `определяется договором` | Often split: fact + «уточняйте по договору» |
| Evidence sentences (рынок, опрос, публикация) | Keep-neutral |
| Pipeline steps (обрабатываются, векторизуются) | Keep-neutral or light active «модуль …» |

## Files touched (editorial pass)

- [report-pack/20260325-research-report-methodology-main-ru.md](../research/executive-research-technology-transfer/report-pack/20260325-research-report-methodology-main-ru.md)
- [report-pack/20260325-research-report-sizing-economics-main-ru.md](../research/executive-research-technology-transfer/report-pack/20260325-research-report-sizing-economics-main-ru.md)
- [report-pack/20260325-research-appendix-a-index-ru.md](../research/executive-research-technology-transfer/report-pack/20260325-research-appendix-a-index-ru.md)
- [report-pack/20260325-research-appendix-b-ip-code-alienation-ru.md](../research/executive-research-technology-transfer/report-pack/20260325-research-appendix-b-ip-code-alienation-ru.md)
- [report-pack/20260325-research-appendix-d-security-observability-ru.md](../research/executive-research-technology-transfer/report-pack/20260325-research-appendix-d-security-observability-ru.md)
- [report-pack/20260325-research-appendix-e-market-technical-signals-ru.md](../research/executive-research-technology-transfer/report-pack/20260325-research-appendix-e-market-technical-signals-ru.md)
- [report-pack/20260331-research-executive-unified-ru.md](../research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md)

Приложение C правок не потребовало (в выборке не найдено целевых имперсональных конструкций в директивных блоках).

## Related

- Checkpoint log: [20260401-active-verb-enhancement-checkpoint-log.md](./20260401-active-verb-enhancement-checkpoint-log.md)
