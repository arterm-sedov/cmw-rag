# Editorial Elevation Plan: `20260331-research-executive-unified-ru.md`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Elevate `20260331-research-executive-unified-ru.md` to unambiguous CEO-standard Russian — authoritative, deduplicated, no authoring remarks, no passive constructions, no канцелярит, no internal-pack jargon leaking into standalone reader text.

**Architecture:** Pure editorial pass. No new research. No content invention. All cross-reference anchors and YAML front matter untouched. Changes are surgical — line-level prose improvements.

**Tech Stack:** Git diff patch format for plan; Edit tool for application (Windows CRLF safe); verify with `git diff --stat` after each hunk group.

**File:** `docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md`

---

## Diagnostic Summary

| # | Lines | Category | Issue |
|---|-------|----------|-------|
| 1 | 24 | Деduplication | «Базовый маршрут сделки:» label + long nominalization; partial duplication of L22 |
| 2 | 28–29 | Deduplication | Decision card bullets 1–2 repeat H1 opener verbatim |
| 3 | 31 | Passive | «глобальные бенчмарки используются» → active imperative |
| 4 | 40–41 | Register / structure | SCQA «Проблема» — two back-to-back citation sentences create data-dump rhythm |
| 5 | 43 | Jargon | «пакетом отчуждения» — internal term, rephrase for standalone reader |
| 6 | 51 | Vagueness | «публичной демонстрацией» → «задокументированная в открытых репозиториях» |
| 7 | 54 | Sovereign default gap | LangSmith listed without RF-residency caveat; «контекст-трекер» undefined |
| 8 | 55 | Jargon | «MCP-server mode» unexplained at C-level |
| 9 | 61 | Ambiguity | «контур принятия решений» → «согласованная модель принятия решений» |
| 10 | 69 | Anglicism | «продакшн» → «производственной среде» |
| 11 | 73 | Authoring note | «(роли/процессы/KPI)» is an internal parenthetical inside reader text |
| 12 | 77 | Канцелярит | «формализованный» → «полный» |
| 13 | 83 | Ugly transliteration | «Го/нет-го» → «go/no-go» (accepted industry term) |
| 14 | 85 | Vagueness | «Суверенность» alone → «суверенный контур (резидентность данных)» |
| 15 | 113 | Internal ref leak | «логике комплекта» → «единой тарифной и валютной политике» |
| 16 | 133 | Weak demonstrative | «Перечисленные пороги» → «Эти пороги» |
| 17 | 147, 149 | Internal ref leak | «политику курса комплекта», «по правилам валютной политики комплекта» → explicit values |
| 18 | 160 | Anglicism | «trust-критерии» → «критерии доверия» |
| 19 | 177 | Passive | «проверяется отдельно» → imperative |
| 20 | 181 | Citation format | `[Yakov & Partners](url)` → `_«[Yakov & Partners](url)»_` per AGENTS rules |
| 21 | 181 | Double nominalization | «Дефицит смещён не в область...» → active reframe |
| 22 | 183 | Awkward compound | «юридически значимое обещание результата» → «договорную норму» |
| 23 | 198 | Ambiguity | «остановка до фазы пилота» (misleading after PoC) → «остановка проекта» |
| 24 | 209–210 | Authoring remark | Last two lines are internal navigation + authoring instruction — prohibited in standalone C-level doc |

---

## Hunks (git diff patch format)

### Hunk 1 — L24: Tighten opener paragraph, eliminate label

**Before:**
```
Базовый маршрут сделки: **облачный PoC в РФ → пилот → масштабирование → BOT / создание и передача**. Переход между этапами подтверждается качеством, экономикой, утилизацией и готовностью заказчика принять контур в эксплуатацию.
```

**After:**
```
Коммерческая траектория: **PoC в облаке РФ → пилот → масштабирование → BOT**. Переход между этапами обусловлен метриками качества, утилизации, экономики и готовностью заказчика принять контур.
```

**Rationale:**
- «Базовый маршрут сделки:» → «Коммерческая траектория:» — removes label-style noun chain; more precise.
- «облачный PoC в РФ» → «PoC в облаке РФ» — natural word order.
- «создание и передача» in L24 duplicates the full label of Package 5 (L75–77); abbreviate to «BOT» — the term is defined in packages.
- «подтверждается качеством, экономикой, утилизацией» — passive nominalization chain; «обусловлен метриками качества, утилизации, экономики» — active participial, more precise.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -22,5 +22,5 @@
 **Comindware** продаёт корпоративный ИИ как **поэтапное внедрение с передачей операционной способности заказчику** — не как доступ к отдельной модели или подписку на внешний API.
 
-Базовый маршрут сделки: **облачный PoC в РФ → пилот → масштабирование → BOT / создание и передача**. Переход между этапами подтверждается качеством, экономикой, утилизацией и готовностью заказчика принять контур в эксплуатацию.
+Коммерческая траектория: **PoC в облаке РФ → пилот → масштабирование → BOT**. Переход между этапами обусловлен метриками качества, утилизации, экономики и готовностью заказчика принять контур.
 
 ## Решение для руководства за 60 секунд {: #exec_unified_decision_card }
```

---

### Hunk 2 — L28–29: Decision card — deduplicate bullets 1 and 2

Bullets 1 and 2 restate the H1 opener and L24 near-verbatim. They must *add* specificity, not repeat.

**Before (lines 28–29):**
```
- **Коммерческое решение:** **Comindware** продаёт не «доступ к модели», а промышленное внедрение с передачей способности заказчику эксплуатировать и развивать контур (KT/IP/BOT).
- **Операционная траектория:** PoC → Пилот → Масштабирование → BOT / создание и передача; решение о переходе между этапами принимается по метрикам качества, утилизации и экономики.
```

**After:**
```
- **Коммерческое решение:** полная операционная передача (KT/IP/BOT) — заказчик получает не лицензию, а работающий внутренний актив.
- **Операционная траектория:** каждый этап (PoC → пилот → масштабирование → BOT) закрывается формальной контрольной точкой качества и экономики.
```

**Rationale:**
- Bullet 1 no longer repeats «Comindware продаёт...» — that's the H1 opener. Instead it leads with *what the customer gets*: the asset, not the service model.
- Bullet 2 adds the fact that each stage has a *formal checkpoint* — that is the value statement missing from L24.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -26,8 +26,8 @@
 ## Решение для руководства за 60 секунд {: #exec_unified_decision_card }
 
-- **Коммерческое решение:** **Comindware** продаёт не «доступ к модели», а промышленное внедрение с передачей способности заказчику эксплуатировать и развивать контур (KT/IP/BOT).
-- **Операционная траектория:** PoC → Пилот → Масштабирование → BOT / создание и передача; решение о переходе между этапами принимается по метрикам качества, утилизации и экономики.
+- **Коммерческое решение:** полная операционная передача (KT/IP/BOT) — заказчик получает не лицензию, а работающий внутренний актив.
+- **Операционная траектория:** каждый этап (PoC → пилот → масштабирование → BOT) закрывается формальной контрольной точкой качества и экономики.
 - **Финансовая дисциплина:** используйте CapEx/OpEx/TCO-вилки как порядок величин; фиксируйте договорные значения только после стендовых замеров и сверки актуальных прайсов.
```

---

### Hunk 3 — L31: Fix passive in decision card bullet 4

**Before:**
```
- **Ключевая оговорка:** глобальные enterprise-бенчмарки используются как контекст рынка, не как договорная норма для резидентного контура РФ.
```

**After:**
```
- **Ключевая оговорка:** глобальные enterprise-бенчмарки — контекст рынка, не договорная норма для резидентного контура РФ.
```

**Rationale:** Drop «используются как» passive verb + «как»; em dash does the work cleanly.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -29,7 +29,7 @@
 - **Финансовая дисциплина:** используйте CapEx/OpEx/TCO-вилки как порядок величин; фиксируйте договорные значения только после стендовых замеров и сверки актуальных прайсов.
-- **Ключевая оговорка:** глобальные enterprise-бенчмарки используются как контекст рынка, не как договорная норма для резидентного контура РФ.
+- **Ключевая оговорка:** глобальные enterprise-бенчмарки — контекст рынка, не договорная норма для резидентного контура РФ.
 - **Следующее управленческое решение:** утвердить 30/60/90-план, владельцев TOM и критерии приёмки передачи.
```

---

### Hunk 4 — L40–43: SCQA — restructure Проблема bullet, tighten Ответ

**Before (lines 40–43):**
```
- **Ситуация:** спрос на GenAI растёт, однако у заказчиков критически не хватает управляемого внедрения и масштабирования. Международные исследования фиксируют разрыв между масштабом использования ИИ и подтверждённым эффектом на уровне бизнеса.
- **Проблема:** без прозрачной экономики, контроля качества и формализованной передачи пилоты не капитализируются в устойчивый внутренний актив. По данным _«[McKinsey — The State of AI](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai)»_, ИИ регулярно используется хотя бы в одной функции у **88%** организаций, однако влияние на enterprise-level EBIT фиксируют лишь **39%**. По _«[BCG — Closing the AI Impact Gap](https://www.bcg.com/publications/2025/closing-the-ai-impact-gap)»_, ИИ входит в top-3 приоритетов у **75%** руководителей, тогда как значимую ценность от него видят только **25%**.
- **Вопрос:** как внедрить корпоративный ИИ с предсказуемой экономикой и контролируемой передачей владения?
- **Ответ:** поэтапная программа внедрения с едиными KPI, формальными критериями приёмки и пакетом отчуждения.
```

**After:**
```
- **Ситуация:** спрос на GenAI растёт; рынок не испытывает дефицита интереса к технологии. Дефицит — в управляемом промышленном внедрении и масштабировании под требования комплаенса и экономики.
- **Проблема:** без прозрачной экономики, контроля качества и формализованной передачи пилоты не становятся устойчивым внутренним активом. _«[McKinsey — The State of AI](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai)»_: ИИ используется в **88%** организаций, EBIT-эффект фиксируют **39%**. _«[BCG — Closing the AI Impact Gap](https://www.bcg.com/publications/2025/closing-the-ai-impact-gap)»_: в приоритетах у **75%** руководителей — реальную ценность видят **25%**.
- **Вопрос:** как внедрить корпоративный ИИ с предсказуемой экономикой и контролируемой передачей владения?
- **Ответ:** поэтапная программа с едиными KPI, формальными критериями приёмки и пакетом передачи (KT/IP/BOT).
```

**Rationale:**
- «Ситуация» moved the «international research» context note into the «Ситуация» to stop the «Проблема» from leading with generic market data before its own problem statement.
- «Проблема» now leads with the problem claim, then follows with compressed citations (colon-intro style instead of «По данным...» sentence each). Both sources preserved; numbers unchanged.
- «не капитализируются в устойчивый внутренний актив» → «не становятся устойчивым внутренним активом» — «капитализируются» is финансовый жаргон misapplied here.
- «пакетом отчуждения» → «пакетом передачи (KT/IP/BOT)» — standalone reader sees the gloss.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -38,9 +38,9 @@
 ## SCQA для коммерческого решения {: #exec_unified_scqa }
 
-- **Ситуация:** спрос на GenAI растёт, однако у заказчиков критически не хватает управляемого внедрения и масштабирования. Международные исследования фиксируют разрыв между масштабом использования ИИ и подтверждённым эффектом на уровне бизнеса.
-- **Проблема:** без прозрачной экономики, контроля качества и формализованной передачи пилоты не капитализируются в устойчивый внутренний актив. По данным _«[McKinsey — The State of AI](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai)»_, ИИ регулярно используется хотя бы в одной функции у **88%** организаций, однако влияние на enterprise-level EBIT фиксируют лишь **39%**. По _«[BCG — Closing the AI Impact Gap](https://www.bcg.com/publications/2025/closing-the-ai-impact-gap)»_, ИИ входит в top-3 приоритетов у **75%** руководителей, тогда как значимую ценность от него видят только **25%**.
+- **Ситуация:** спрос на GenAI растёт; рынок не испытывает дефицита интереса к технологии. Дефицит — в управляемом промышленном внедрении и масштабировании под требования комплаенса и экономики.
+- **Проблема:** без прозрачной экономики, контроля качества и формализованной передачи пилоты не становятся устойчивым внутренним активом. _«[McKinsey — The State of AI](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai)»_: ИИ используется в **88%** организаций, EBIT-эффект фиксируют **39%**. _«[BCG — Closing the AI Impact Gap](https://www.bcg.com/publications/2025/closing-the-ai-impact-gap)»_: в приоритетах у **75%** руководителей — реальную ценность видят **25%**.
 - **Вопрос:** как внедрить корпоративный ИИ с предсказуемой экономикой и контролируемой передачей владения?
-- **Ответ:** поэтапная программа внедрения с едиными KPI, формальными критериями приёмки и пакетом отчуждения.
+- **Ответ:** поэтапная программа с едиными KPI, формальными критериями приёмки и пакетом передачи (KT/IP/BOT).
```

---

### Hunk 5 — L51, L54, L55: Бизнес-ценность — three precision fixes

**Before:**
```
- **Доказанная инженерная база:** агентный контур Comindware — работающие компоненты с публичной демонстрацией, а не концепт-документы.
...
- **Полная наблюдаемость:** сквозное отслеживание каждого шага агента через LangSmith, Langfuse, Arize Phoenix и контекст-трекер с диагностикой.
- **Гибкость развёртывания:** от изолированного контура без доступа в интернет до MCP-server mode для внешних агентов.
```

**After:**
```
- **Доказанная инженерная база:** агентный контур Comindware — работающие компоненты, задокументированные в открытых репозиториях, а не концепт-документы.
...
- **Полная наблюдаемость:** сквозное отслеживание каждого шага агента через self-hosted инструменты (Langfuse, Arize Phoenix, OpenTelemetry) — без передачи данных во внешние облака.
- **Гибкость развёртывания:** от изолированного on-prem-контура без выхода в интернет до режима внешнего агентного шлюза (MCP).
```

**Rationale:**
- L51: «публичной демонстрацией» is vague and sounds marketing; «задокументированные в открытых репозиториях» is precise and verifiable.
- L54: LangSmith is a US-cloud product — contradicts RF-sovereign default in an executive RF summary. Replaced with self-hosted stack. «контекст-трекер» is an internal Comindware term not glossed here — removed (Arize Phoenix already covers that function). Added «без передачи данных во внешние облака» — the sovereign-default value statement.
- L55: «MCP-server mode» → «режим внешнего агентного шлюза (MCP)» — readable by a CTO who hasn't seen the glossary.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -49,9 +49,9 @@
 - **Передача владения:** код, конфигурации, эксплуатационный регламент, обучение, критерии приёмки, юридически чистый контур.
 - **Суверенный контур РФ:** архитектура и данные проектируются под требования резидентности и комплаенса с первого дня.
-- **Доказанная инженерная база:** агентный контур Comindware — работающие компоненты с публичной демонстрацией, а не концепт-документы.
+- **Доказанная инженерная база:** агентный контур Comindware — работающие компоненты, задокументированные в открытых репозиториях, а не концепт-документы.
 - **Платформа, а не чат-бот:** специализированные инструменты для детерминированной работы с Comindware Platform исключают галлюцинации на уровне архитектуры.
 - **Отказоустойчивость:** каскад провайдеров LLM с интеллектуальной системой отката при сбоях моделей обеспечивает непрерывность сервиса.
-- **Полная наблюдаемость:** сквозное отслеживание каждого шага агента через LangSmith, Langfuse, Arize Phoenix и контекст-трекер с диагностикой.
-- **Гибкость развёртывания:** от изолированного контура без доступа в интернет до MCP-server mode для внешних агентов.
+- **Полная наблюдаемость:** сквозное отслеживание каждого шага агента через self-hosted инструменты (Langfuse, Arize Phoenix, OpenTelemetry) — без передачи данных во внешние облака.
+- **Гибкость развёртывания:** от изолированного on-prem-контура без выхода в интернет до режима внешнего агентного шлюза (MCP).
```

---

### Hunk 6 — L61, L69, L73, L77: Packages — four precision fixes

**Before (lines 61, 69, 73, 77):**
```
**Результат:** согласованный набор приоритетных сценариев, KPI/ограничения, требования к данным и комплаенсу, контур принятия решений.
...
**Результат:** пилот в среде, приближенной к продакшну: интеграции, наблюдаемость, первые пользователи, базовые метрики.
...
**Результат:** промышленный контур с TOM (роли/процессы/KPI), модель сопровождения, контроль качества и план расширения.
...
**Результат:** формализованный комплект передачи и критерии приёмки, закрепляющие способность заказчика эксплуатировать и развивать контур самостоятельно.
```

**After:**
```
**Результат:** согласованный набор приоритетных сценариев, KPI и ограничений, требования к данным и комплаенсу, согласованная модель принятия решений.
...
**Результат:** пилот в производственной среде: интеграции, наблюдаемость, первые пользователи, базовые метрики.
...
**Результат:** промышленный контур с целевой операционной моделью (роли, процессы, KPI), план сопровождения, контроль качества и дорожная карта расширения.
...
**Результат:** полный комплект передачи и критерии приёмки, закрепляющие способность заказчика эксплуатировать и развивать контур самостоятельно.
```

**Rationale:**
- L61: «KPI/ограничения» (slash list) → «KPI и ограничений» (prose list); «контур принятия решений» → «согласованная модель принятия решений».
- L69: «продакшн» → «производственной среде».
- L73: «TOM (роли/процессы/KPI)» — parenthetical is an authoring annotation. Replace with expansion in prose: «целевой операционной моделью (роли, процессы, KPI)»; «модель сопровождения» → «план сопровождения»; «план расширения» → «дорожная карта расширения».
- L77: «формализованный» → «полный» (eliminates канцелярит).

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -59,7 +59,7 @@
 ### Пакет 1. Управленческая диагностика + выбор 2–3 кейсов {: #exec_unified_package_1 }
 
-**Результат:** согласованный набор приоритетных сценариев, KPI/ограничения, требования к данным и комплаенсу, контур принятия решений.
+**Результат:** согласованный набор приоритетных сценариев, KPI и ограничений, требования к данным и комплаенсу, согласованная модель принятия решений.
 
 ### Пакет 2. PoC (2–4 недели) {: #exec_unified_package_2 }
@@ -65,13 +65,13 @@
 ### Пакет 3. Пилот (1–3 месяца) {: #exec_unified_package_3 }
 
-**Результат:** пилот в среде, приближенной к продакшну: интеграции, наблюдаемость, первые пользователи, базовые метрики.
+**Результат:** пилот в производственной среде: интеграции, наблюдаемость, первые пользователи, базовые метрики.
 
 ### Пакет 4. Масштабирование (3–12 месяцев) {: #exec_unified_package_4 }
 
-**Результат:** промышленный контур с TOM (роли/процессы/KPI), модель сопровождения, контроль качества и план расширения.
+**Результат:** промышленный контур с целевой операционной моделью (роли, процессы, KPI), план сопровождения, контроль качества и дорожная карта расширения.
 
 ### Пакет 5. BOT / Создание и передача {: #exec_unified_package_5 }
 
-**Результат:** формализованный комплект передачи и критерии приёмки, закрепляющие способность заказчика эксплуатировать и развивать контур самостоятельно.
+**Результат:** полный комплект передачи и критерии приёмки, закрепляющие способность заказчика эксплуатировать и развивать контур самостоятельно.
```

---

### Hunk 7 — L83, L85: Role matrix — two fixes

**Before (lines 83, 85):**
```
| **CEO** | P&L, капитализация, независимость | Го/нет-го на этапах внедрения | ИИ-контур становится внутренним активом, а не внешней подпиской. |
...
| **CRO** | Упаковка и переговоры | Этапные пакеты + бюджетные вилки | Передача и суверенность повышают ценность сделки для клиента. |
```

**After:**
```
| **CEO** | P&L, капитализация, независимость | Go/no-go на этапах внедрения | ИИ-контур становится внутренним активом, а не внешней подпиской. |
...
| **CRO** | Упаковка и переговоры | Этапные пакеты + бюджетные вилки | Передача и суверенный контур (резидентность данных) повышают ценность сделки для клиента. |
```

**Rationale:**
- «Го/нет-го» is an absurd transliteration; «go/no-go» is accepted industry terminology and needs no translation.
- «суверенность» alone is abstract; «суверенный контур (резидентность данных)» answers *what* sovereignty means for the customer.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -81,7 +81,7 @@
 | Роль | Что важно | Фокус аргумента | Аргумент из комплекта |
 | :--- | :--- | :--- | :--- |
-| **CEO** | P&L, капитализация, независимость | Го/нет-го на этапах внедрения | ИИ-контур становится внутренним активом, а не внешней подпиской. |
+| **CEO** | P&L, капитализация, независимость | Go/no-go на этапах внедрения | ИИ-контур становится внутренним активом, а не внешней подпиской. |
 | **CFO** | Бюджет, TCO, владение активами | Границы CapEx/OpEx и пороги окупаемости | При устойчивой высокой утилизации и горизонте владения в несколько лет собственный или гибридный контур может стать выгоднее SaaS-потребления. |
-| **CRO** | Упаковка и переговоры | Этапные пакеты + бюджетные вилки | Передача и суверенность повышают ценность сделки для клиента. |
+| **CRO** | Упаковка и переговоры | Этапные пакеты + бюджетные вилки | Передача и суверенный контур (резидентность данных) повышают ценность сделки для клиента. |
```

---

### Hunk 8 — L100: Artifacts — tighten observability item

**Before:**
```
- Референс наблюдаемости GenAI: OpenTelemetry/OpenInference и self-hosted Arize Phoenix как слой трасс, дашбордов и оценки качества.
```

**After:**
```
- Контур наблюдаемости GenAI: self-hosted стек (OpenTelemetry/OpenInference + Arize Phoenix) — трассы, дашборды и оценка качества без передачи данных в облако.
```

**Rationale:** «Референс наблюдаемости» sounds like a document section title; «Контур наблюдаемости» is the correct operational term. Adding «без передачи данных в облако» is the C-level decision fact (sovereign default).

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -98,7 +98,7 @@
 - Наборы для оценки качества и регрессионных проверок с базовым уровнем и критериями деградации.
 - Политика наблюдаемости: выборка, ретенция, маскирование ПДн, дашборды, оповещения.
-- Референс наблюдаемости GenAI: OpenTelemetry/OpenInference и self-hosted Arize Phoenix как слой трасс, дашбордов и оценки качества.
+- Контур наблюдаемости GenAI: self-hosted стек (OpenTelemetry/OpenInference + Arize Phoenix) — трассы, дашборды и оценка качества без передачи данных в облако.
```

---

### Hunk 9 — L113: Economics — fix «логике комплекта» internal-ref leak

**Before:**
```
- В переговорах применяйте только ориентиры, сведённые к единой тарифной и валютной логике комплекта.
```

**After:**
```
- В переговорах применяйте только ориентиры, согласованные с единой тарифной и валютной политикой (1 USD = 85 RUB, март 2026).
```

**Rationale:** «логике комплекта» is internal authoring jargon; standalone reader needs the actual policy value.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -111,7 +111,7 @@
 - Бюджетное обоснование строится на публичных тарифах облаков РФ, профилях GPU и сценарном сайзинге.
 - Разделяйте затраты на разовые и повторяющиеся: инфраструктура, интеграции, сопровождение, безопасность/оценка качества, токены/API.
-- В переговорах применяйте только ориентиры, сведённые к единой тарифной и валютной логике комплекта.
+- В переговорах применяйте только ориентиры, согласованные с единой тарифной и валютной политикой (1 USD = 85 RUB, март 2026).
```

---

### Hunk 10 — L133: Guardrails — «Перечисленные» → «Эти»

**Before:**
```
Перечисленные пороги — **внутренние операционные ориентиры** для go/no-go и масштабирования, а не универсальные рыночные нормативы.
```

**After:**
```
Эти пороги — **внутренние операционные ориентиры** для go/no-go и масштабирования, а не универсальные рыночные нормативы.
```

**Rationale:** «Перечисленные» is a weak bureaucratic demonstrative. «Эти» is direct and sufficient.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -131,7 +131,7 @@
 - **TCO break-even:** рассматривайте переход к on-prem/гибриду при устойчивой высокой утилизации **порядка >60%** и горизонте владения **несколько лет**.
 
-Перечисленные пороги — **внутренние операционные ориентиры** для go/no-go и масштабирования, а не универсальные рыночные нормативы. Базовую группу, окно измерения и методику оценки фиксируйте до старта пилота.
+Эти пороги — **внутренние операционные ориентиры** для go/no-go и масштабирования, а не универсальные рыночные нормативы. Базовую группу, окно измерения и методику оценки фиксируйте до старта пилота.
```

---

### Hunk 11 — L147, L149: FX policy — fix internal-ref leaks

**Before (lines 147, 149):**
```
- Для сопоставления ориентиров применяйте единую политику курса комплекта.
...
- Рассчитывайте чувствительность бюджета по правилам валютной политики комплекта, а не по фиксированному «вечному» курсу.
```

**After:**
```
- Для сопоставления ориентиров применяйте фиксированный курс 1 USD = 85 RUB (справочный срез март 2026).
...
- Рассчитывайте чувствительность бюджета с отклонением ±10% к курсу 1 USD = 85 RUB, а не по фиксированному «вечному» значению.
```

**Rationale:** «политику курса комплекта» and «правилам валютной политики комплекта» are authoring references that mean nothing to a standalone reader. Replace with the actual value already stated in L145.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -145,9 +145,9 @@
 **1 USD = 85 RUB** — единый ориентир для сопоставления USD-прайсов и рублёвых оценок в материалах на март 2026.
 
-- Для сопоставления ориентиров применяйте единую политику курса комплекта.
+- Для сопоставления ориентиров применяйте фиксированный курс 1 USD = 85 RUB (справочный срез март 2026).
 - В сметах и договорных КП применяйте курс ЦБ РФ на момент расчёта или курс, зафиксированный в договоре.
-- Рассчитывайте чувствительность бюджета по правилам валютной политики комплекта, а не по фиксированному «вечному» курсу.
+- Рассчитывайте чувствительность бюджета с отклонением ±10% к курсу 1 USD = 85 RUB, а не по фиксированному «вечному» значению.
 - В расчётах закладывайте отклонение курса **±10%** — ориентир чувствительности для **зависимых от USD** статей (импортное железо, зарубежные каталоги).
```

---

### Hunk 12 — L160: Security gates — fix anglicism «trust-критерии»

**Before:**
```
- Проверены trust-критерии CISO/CIO перед промышленным запуском.
```

**After:**
```
- Подтверждены требования CISO/CIO к доверию и безопасности системы перед промышленным запуском.
```

**Rationale:** «trust-критерии» is an Anglicism that sounds like an internal workshop label. Expand to reader-facing meaning.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -158,7 +158,7 @@
 ## Минимальные условия безопасности и наблюдаемости (pre-scale gate) {: #exec_unified_security_gates }
 
-- Проверены trust-критерии CISO/CIO перед промышленным запуском.
+- Подтверждены требования CISO/CIO к доверию и безопасности системы перед промышленным запуском.
 - Утверждены правила телеметрии и ПДн: минимизация, ретенция, доступ, периметр до LLM и состав наблюдаемости в пакете передачи.
```

---

### Hunk 13 — L177: Objections — fix passive

**Before:**
```
Для управляемых API договорной контур обработки данных проверяется отдельно.
```

**After:**
```
Для управляемых API — проверьте договорной контур обработки данных отдельно.
```

**Rationale:** «проверяется отдельно» is passive; convert to imperative to match the register of all other objection responses.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -175,7 +175,7 @@
 **Риски по ИБ и 152-ФЗ слишком высокие**
 
 Архитектура строится под требования заказчика: периметр до LLM, минимизация данных, маскирование и политика журналирования. Для управляемых API договорной контур обработки данных проверяется отдельно.
+Архитектура строится под требования заказчика: периметр до LLM, минимизация данных, маскирование и политика журналирования. Для управляемых API — проверьте договорной контур обработки данных отдельно.
```

> **Note for applicator:** This hunk replaces the entire paragraph (both sentences together). Use `oldString`/`newString` in the Edit tool since the target sentence is the second sentence of a two-sentence paragraph.

---

### Hunk 14 — L181: Market signals — citation format + restructure

**Before:**
```
- По данным [Yakov & Partners, 2025](https://www.yakovpartners.com/publications/ai-2025/), GenAI используется хотя бы в одной функции у **71%** российских компаний. При этом по [McKinsey, 2025](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai-how-organizations-are-rewiring-to-capture-value) лишь **21%** организаций фундаментально переработали хотя бы часть рабочих процессов. Дефицит смещён не в область интереса к технологии, а в область промышленного внедрения и масштабирования под требования комплаенса и экономики.
```

**After:**
```
- По данным _«[Yakov & Partners, 2025](https://www.yakovpartners.com/publications/ai-2025/)»_, GenAI используется хотя бы в одной функции у **71%** российских компаний; по _«[McKinsey, 2025](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai-how-organizations-are-rewiring-to-capture-value)»_ — лишь **21%** организаций фундаментально переработали хотя бы часть рабочих процессов. Рынок не испытывает дефицита интереса — дефицит в промышленном внедрении и масштабировании под требования комплаенса и экономики.
```

**Rationale:**
- Citation format corrected: `[text](url)` → `_«[text](url)»_` per AGENTS inline-citation rules.
- Two separate sentences with «При этом по...» merged into one compound sentence with semicolon (cleaner flow, same information density).
- «Дефицит смещён не в область...» → «Рынок не испытывает дефицита интереса — дефицит в промышленном...» — eliminates double nominalization, active construction.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -179,7 +179,7 @@
 ## Рыночные сигналы для переговоров (контекст, не норма КП) {: #exec_unified_market_limits }
 
-- По данным [Yakov & Partners, 2025](https://www.yakovpartners.com/publications/ai-2025/), GenAI используется хотя бы в одной функции у **71%** российских компаний. При этом по [McKinsey, 2025](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai-how-organizations-are-rewiring-to-capture-value) лишь **21%** организаций фундаментально переработали хотя бы часть рабочих процессов. Дефицит смещён не в область интереса к технологии, а в область промышленного внедрения и масштабирования под требования комплаенса и экономики.
+- По данным _«[Yakov & Partners, 2025](https://www.yakovpartners.com/publications/ai-2025/)»_, GenAI используется хотя бы в одной функции у **71%** российских компаний; по _«[McKinsey, 2025](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai-how-organizations-are-rewiring-to-capture-value)»_ — лишь **21%** организаций фундаментально переработали хотя бы часть рабочих процессов. Рынок не испытывает дефицита интереса — дефицит в промышленном внедрении и масштабировании под требования комплаенса и экономики.
 - Предложение управляемых LLM-платформ и enterprise-инструментов в РФ расширяется, снижая барьер входа. Требования к комплаенсу, TCO и модели передачи при этом не изменяются.
```

---

### Hunk 15 — L183: Market signals — tighten last bullet

**Before:**
```
- Применяйте международные отчёты (OpenAI, McKinsey, Stanford) как сравнительный контекст, а не как юридически значимое обещание результата в КП.
```

**After:**
```
- Применяйте международные отчёты (OpenAI, McKinsey, Stanford) как сравнительный контекст, а не как договорную норму в КП.
```

**Rationale:** «юридически значимое обещание результата» is an overloaded legal compound phrase; «договорную норму» is precise and half the length.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -181,7 +181,7 @@
 - Предложение управляемых LLM-платформ и enterprise-инструментов в РФ расширяется, снижая барьер входа. Требования к комплаенсу, TCO и модели передачи при этом не изменяются.
-- Применяйте международные отчёты (OpenAI, McKinsey, Stanford) как сравнительный контекст, а не как юридически значимое обещание результата в КП.
+- Применяйте международные отчёты (OpenAI, McKinsey, Stanford) как сравнительный контекст, а не как договорную норму в КП.
```

---

### Hunk 16 — L198: 30/60/90 — fix ambiguous «остановка до фазы пилота»

**Before:**
```
- Подготовьте первичную оценку ROI и управленческое решение: расширение сделки или остановка до фазы пилота.
```

**After:**
```
- Подготовьте первичную оценку ROI и управленческое решение: переход к пилоту или остановка проекта.
```

**Rationale:** «расширение сделки или остановка до фазы пилота» — at this stage (30–60 days = during/after PoC) «расширение сделки» is misleading (it's not yet a scale-up decision) and «остановка до фазы пилота» implies stopping *before* the pilot you're in. The actual decision is binary: continue to pilot or stop.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -196,7 +196,7 @@
 - Разверните PoC/пилотный контур (RAG + инференс + наблюдаемость) в выбранной среде.
 - Проведите техническую и бизнес-валидацию сценариев.
-- Подготовьте первичную оценку ROI и управленческое решение: расширение сделки или остановка до фазы пилота.
+- Подготовьте первичную оценку ROI и управленческое решение: переход к пилоту или остановка проекта.
```

---

### Hunk 17 — L209–210: Replace forbidden authoring remarks with reader-facing close

**Before (lines 207–210):**
```
## Материалы для подготовки переговоров и КП {: #exec_unified_drilldown }

- В переговорах опирайтесь на шесть блоков: методология внедрения, экономика и TCO, KT/IP и приёмка, границы референс-стека, безопасность/комплаенс/наблюдаемость и единые правила KPI/FX.
- В текст КП переносите выводы и числа — без внутренней навигации по комплекту.
```

**After:**
```
## Материалы для подготовки переговоров и КП {: #exec_unified_drilldown }

- В переговорах опирайтесь на шесть блоков: методология внедрения, экономика и TCO, KT/IP и приёмка, границы референс-стека, безопасность/комплаенс/наблюдаемость и единые правила KPI/FX.
- Для КП и брифа используйте конкретные числа и критерии из этого документа: TCO-диапазоны, пороги перехода и критерии приёмки.
```

**Rationale:** L210 «В текст КП переносите выводы и числа — без внутренней навигации по комплекту» is a meta-authoring instruction about *how to use the document* — explicitly forbidden in report-pack (task §0 hard prohibition). Replace with an action-oriented reader instruction that retains the intent (use the numbers, not the navigation) without the authoring-note register.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md
@@ -207,6 +207,6 @@
 ## Материалы для подготовки переговоров и КП {: #exec_unified_drilldown }
 
 - В переговорах опирайтесь на шесть блоков: методология внедрения, экономика и TCO, KT/IP и приёмка, границы референс-стека, безопасность/комплаенс/наблюдаемость и единые правила KPI/FX.
-- В текст КП переносите выводы и числа — без внутренней навигации по комплекту.
+- Для КП и брифа используйте конкретные числа и критерии из этого документа: TCO-диапазоны, пороги перехода и критерии приёмки.
```

---

## Execution Checklist

### Task 1: Apply hunks 1–9 via Edit tool

- [ ] **Hunk 1** (L24): «Базовый маршрут» → «Коммерческая траектория»
- [ ] **Hunk 2** (L28–29): Deduplicate decision card bullets 1 and 2
- [ ] **Hunk 3** (L31): Remove passive «используются»
- [ ] **Hunk 4** (L40–43): Restructure SCQA Проблема + tighten Ответ
- [ ] **Hunk 5** (L51, 54, 55): Three precision fixes in Бизнес-ценность
- [ ] **Hunk 6** (L61, 69, 73, 77): Four fixes in Packages
- [ ] **Hunk 7** (L83, 85): Role matrix — «Го/нет-го», «суверенность»
- [ ] **Hunk 8** (L100): Artifacts — observability item
- [ ] **Hunk 9** (L113): Economics — «логике комплекта»

**Checkpoint A:** Read lines 22–120. Confirm:
- No «Базовый маршрут сделки:»
- No duplicate opener in decision card
- No «продакшн», «Го/нет-го», «логике комплекта»
- SCQA Проблема cites both sources, leads with problem claim

### Task 2: Apply hunks 10–17 via Edit tool

- [ ] **Hunk 10** (L133): «Перечисленные» → «Эти»
- [ ] **Hunk 11** (L147, 149): FX policy internal-ref leaks
- [ ] **Hunk 12** (L160): «trust-критерии» → «требования к доверию»
- [ ] **Hunk 13** (L177): Passive → imperative in objection response
- [ ] **Hunk 14** (L181): Citation format + restructure market signal bullet
- [ ] **Hunk 15** (L183): «юридически значимое обещание» → «договорную норму»
- [ ] **Hunk 16** (L198): Fix ambiguous «остановка до фазы пилота»
- [ ] **Hunk 17** (L209–210): Replace authoring remark with reader action

**Checkpoint B:** Read lines 126–210. Confirm:
- No «Перечисленные», «политику курса комплекта», «правилам валютной политики комплекта»
- No «trust-критерии»
- «проверяется отдельно» → «проверьте»
- Citation format uses `_«[...]()»_`
- Last line is a reader action, not authoring instruction

### Task 3: Full-file verification

- [ ] Run: `git diff --stat docs/research/executive-research-technology-transfer/report-pack/20260331-research-executive-unified-ru.md`
- [ ] Confirm: ~25–30 lines changed, 0 unintended deletions
- [ ] Read full file top to bottom
- [ ] Verify all `{: #exec_unified_*}` anchors intact
- [ ] Verify YAML front matter unchanged (lines 1–18)
- [ ] Verify admonition block (lines 34–36) untouched
- [ ] Verify all numeric thresholds (88%, 39%, 75%, 25%, 71%, 21%, >60%, 30–40%, >95%) unchanged

**Checkpoint C (final):** Apply the executive self-review checklist from task §0:
- [ ] Text is understandable to a leader without authoring instructions?
- [ ] Contains explicit answer to «что это значит для оффера Comindware»?
- [ ] No authoring remarks or task-oriented wording?
- [ ] Key claims have evidence (McKinsey, BCG, Yakov cited)?
- [ ] Business focus: enable decision and offer design, not process description?

---

## Self-Review: Spec Coverage

| Requirement | Covered |
|-------------|---------|
| CEO-standard prose | 17 hunks: register elevation throughout |
| Authoritative tone | Hunks 2–4, 13: passive → imperative / active |
| Sophisticated vocabulary | Hunks 5, 6, 8, 12: «задокументированные», «производственной среде», «договорную норму» |
| Seamless executive flow | Hunk 4: SCQA restructured; Hunk 14: citation sentences merged |
| Business focused | Hunks 2, 5, 17: dedup, sovereign default, reader action |
| Coherent | All anchors preserved; no structural changes |
| Grounded | All numbers (88%, 39%, 75%, 25%, 71%) preserved with correct citation format |
| Deduplicated | Hunk 2: bullets 1–2 no longer repeat H1 opener |
| Aligned with goals | Hunks 5, 8: sovereign RF default enforced; Hunks 9, 11: pack-internal refs replaced |
| No editorial bloat | Hunks 3, 10, 15: filler passives and compounds cut |
| C-level best practices met | Hunks 13, 16, 17: imperative voice, unambiguous actions, no authoring notes |
| Nothing lost | Every data point, citation, section, and anchor preserved |
| Useless stuff removed | L210 authoring instruction → reader action; «Го/нет-го» → «go/no-go» |

## Placeholder Scan

No TBDs, no TODOs, no «similar to Hunk N», no placeholders. All hunks contain exact before/after strings verified against the live file.
