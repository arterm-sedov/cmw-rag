# Plan: Restructure Sizing Economics Document

## Objective
Consolidate `20260325-research-report-sizing-economics-main-ru.md` (1275 lines, 12,346 words) into a coherent, non-duplicating document (~10,000 words) with single authoritative sections for each idea.

## Constraints
- Preserve ALL unique content
- Merge overlapping sections into ONE place
- Keep cross-references working
- Match section naming with adjacent `20260325-research-report-methodology-main-ru.md`

## Target Structure (Single Document)

```
1. Обзор                                          # Keep existing
2. Концепция финансовой модели (SCQA)           # Consolidate SCQA + decisions + metrics
3. Рыночный контекст                             # Keep, minor cleanup
4. Тарифы API и провайдеры                        # MERGE: Russian + Chinese + Global + Aggregators + Infra providers
5. Токен-экономика и юнит-экономика              # MERGE: Token tables + FinOps + Cost model
6. Инфраструктура GPU                             # MERGE: Hardware pricing + VRAM + Throughput + Selection guide
7. Облачные провайдеры РФ                         # MERGE: Cloud.ru + Yandex + Selectel detailed tables
8. TCO и сценарии развёртывания                   # MERGE: Break-even + 3-year TCO + Business sizing
9. Риски и оптимизация                            # MERGE: Risk table + Cost optimization + Technical trends
10. Аппаратные требования и кейсы                # MERGE: Local inference + Edge cases + Case studies
11. Заключение                                    # Keep existing
```

## Execution Strategy: Step-by-Step

### Step 1: Create restructured foundation
- Write cleaned Обзор and Концепция sections
- Decision matrix + metrics table consolidated

### Step 2: Market context
- Review and clean, no major duplication there

### Step 3: Тарифы API (Tariffs)
- MERGE from:
  - Lines 239-310 (Russian models)
  - Lines 256-271 (Chinese models)
  - Lines 285-291 (Global models)
  - Lines 296-310 (API aggregators)
  - Lines 316-324 (Infrastructure providers)
- INTO single authoritative section with sub-tables

### Step 4: Токен-экономика
- MERGE from:
  - Lines 419-430 (FinOps)
  - Lines 437-460 (Cost model)
  - Lines 461-616 (Token consumption estimates + reasoning tokens)
- INTO single section

### Step 5: GPU Infrastructure
- MERGE from:
  - Lines 665-720 (Hardware profile + pricing matrix)
  - Lines 721-750 (Depreciation + VRAM)
  - Lines 751-800 (Throughput + Local models)
  - Lines 801-836 (TCO adjustment + Recommendations)
  - Lines 845-892 (VRAM requirements + vLLM/MOSEC)
- INTO single authoritative hardware section

### Step 6: Cloud Providers
- MERGE from:
  - Lines 1001-1049 (Detailed Cloud.ru/Yandex/Selectel tables)
  - Lines 1093-1101 (Cloud hosting summary)
- INTO single cloud section

### Step 7: TCO Section
- MERGE from:
  - Lines 1050-1082 (Break-even analysis)
  - Lines 1089-1151 (TCO comparison tables + Sizing)
  - Lines 976-1000 (Local deployment scenarios)
- INTO single TCO section

### Step 8: Risks
- MERGE from:
  - Lines 631-663 (Risk mitigation table)
  - Lines 1201-1250 (Optimization + Technical trends)
- INTO single section

### Step 9: Hardware cases
- MERGE case studies (Apple M3, Picoclaw)

### Step 10: Final polish
- Cross-reference validation
- Anchor consistency

## Duplication Sources Identified

| Topic | Locations | Lines |
|-------|-----------|-------|
| GPU prices | Quick ref (673-685), Matrix (700-719), VRAM (782-796), Adjustment (808-819), Selection (969-974) | ~120 lines overlap |
| Cloud tariffs | Russian (239-310), Providers (316-324), Quick (681-685), Detailed (1001-1049), Summary (1093-1101) | ~200 lines overlap |
| TCO | SCQA (44-56), Framework (339-369), Segments (441-446), Break-even (1050-1082), Comparison (1138-1151), Sizing (1157-1180) | ~150 lines overlap |

## Progress Tracking
- [ ] Step 1: Foundation
- [ ] Step 2: Market context
- [ ] Step 3: Tariffs
- [ ] Step 4: Token economics
- [ ] Step 5: GPU Infrastructure
- [ ] Step 6: Cloud Providers
- [ ] Step 7: TCO
- [ ] Step 8: Risks
- [ ] Step 9: Hardware cases
- [ ] Step 10: Final polish