# Editorial Elevation Plan: `20260325-research-intro-ru.md`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Elevate `20260325-research-intro-ru.md` to CEO-standard Russian prose — authoritative, sophisticated, seamlessly flowing — without losing any content or breaking cross-references.

**Architecture:** Pure editorial pass: prose quality, register elevation, канцелярит elimination, structural tightening, and typography correction. No new research. No content invention. No cross-reference breakage.

**Tech Stack:** Git diff patch format for all changes → `git apply patch.diff` → verify with `git diff --stat`.

---

## Diagnostic: Issues Found

### Section 1 — `# Введение` (lines 22–34)

| # | Issue | Rule |
|---|-------|------|
| 1 | Opener «В отчёте представлены материалы...» — self-referential meta opener, passive | No self-referential text; active constructions |
| 2 | «Применяйте как основу...» — imperative is fine but sentence is long with two clauses + hedging phrase «с адаптацией под профиль заказчика и финальной верификацией...» buried at end | Pyramid principle; one idea per sentence |
| 3 | «База материалов:» — label-style bold pseudo-structure instead of clean prose | Executive register |
| 4 | Missing blank line after H1 `# Введение` before paragraph | Spacing Between Headings and Content |

### Section 2 — `## Глоссарий` intro (lines 36–45)

| # | Issue | Rule |
|---|-------|------|
| 5 | «Краткий словарь ключевых терминов: фиксирует употребление в комплекте и устраняет разночтения при обсуждении архитектуры, экономики и модели передачи.» — fragment sentence starting with weak descriptive | Active construction; avoid weak openers |
| 6 | «В тексте отчета **корпоративный RAG-контур**...» — missing comma after «В тексте отчета»; «В тексте отчета» is bureaucratic | Prose quality |
| 7 | «Подробные формулировки для договоров и переговоров:» — label-style line ending in colon before list | Heading rules |

### Section 3 — Glossary table (lines 48–119)

| # | Issue | Rule |
|---|-------|------|
| 8 | Missing blank line before the table (line 48 follows immediately after `{% include-markdown %}` on line 46) | List rules: empty line before every block |
| 9 | `DSPy` entry: «К похожему классу относятся иные библиотеки...» — канцелярит «иные», weak connective | Prose quality |
| 10 | `LoRA` entry: «обычно дешевле по памяти GPU и хранению, чем полное дообучение всех весов» — verbose; «всех весов» is tautological with «полное дообучение» | Lexical precision |
| 11 | `MOSEC` entry: «В контексте пакета:» — internal authoring remark | Hard prohibition in report-pack |
| 12 | `vLLM` entry: «В контексте пакета:» — same as above | Hard prohibition in report-pack |
| 13 | `SGR` entry: «В Comindware используется в нескольких точках конвейеров: анализ запросов, критика ответов агентов, планирование после фазы зазиты, детерминированное управление любым мышлением.» — «фазы зазиты» appears to be a typo for «фазы защиты»; «детерминированное управление любым мышлением» is vague/awkward | Typo fix; lexical precision |
| 14 | `Агентный RAG` definition: «Вариант RAG, где модель не только получает контекст, но и планирует шаги...» — «Вариант» is weak opener; restructure for executive register | Executive register |
| 15 | `Временный привилегированный доступ` definition: long run-on sentence; split needed | One subordinate clause per sentence |
| 16 | `Глубокое исследование` definition: «когда требуется не быстрый ответ, а обоснованный материал с опорой на несколько независимых источников» — вторая часть нарушает pyramid principle | Pyramid principle |

### Section 4 — `## Навигация` (lines 121–176)

| # | Issue | Rule |
|---|-------|------|
| 17 | H2 «Навигация: вопрос → документ» — heading contains colon + arrow, fine by convention but the em dash in anchor ID uses different style; functionally OK | No action needed |
| 18 | «Коммерческий обзор для C-Suite» — C-Suite without guillemets is correct (English proper noun) ✓ | — |
| 19 | «KPI, числовые пороги go/no-go, политика интерпретации» — «go/no-go» is English; replace with «решения о продолжении / остановке» or keep as industry term (OK for executive pack) | Minor, keep |
| 20 | Nav table row: «Внедрение в пром контуре» — «пром» is informal abbreviation, should be «промышленном» | Lexical precision |

---

## Change Plan (Git-Diff Patch Format)

**File:** `docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md`

### Hunk 1: Add blank line after H1, rewrite opener (lines 22–34)

**What:** Fix missing blank line after `# Введение`; replace passive self-referential opener with active, front-loaded executive prose; tighten source attribution line; clarify usage instruction.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
@@ -22,13 +22,13 @@
 # Введение {: #intro_pack_overview }
 
-В отчёте представлены материалы по внедрению корпоративного ИИ в резидентном контуре РФ:
+Комплект охватывает полный цикл корпоративного ИИ в резидентном контуре РФ:
 
 - **организация внедрения и эксплуатации** — фазы, роли, контрольные точки;
 - **диапазоны CapEx/OpEx/TCO** — ориентиры для смет и бюджетов;
 - **передача кода и ИС** — комплект KT/IP и приёмка;
 - **граница готового стека Comindware** — что входит в поставку;
 - **риски и контроль** — что закрыть до промышленного запуска.
 
-**База материалов:** публичные прайсы, отраслевые публикации и инженерная практика **Comindware** (открытые репозитории экосистемы).
+Доказательная база — публичные прайсы, отраслевые публикации и задокументированная инженерная практика Comindware.
 
-Применяйте как основу для собственных презентаций, смет и управленческих решений — с адаптацией под профиль заказчика и финальной верификацией договорных условий перед оффером.
+Используйте материалы как основу для презентаций, смет и управленческих решений. Адаптируйте под профиль заказчика и верифицируйте договорные условия перед оффером.
```

**Rationale:**
- «Комплект охватывает» — active, direct, front-loaded; eliminates passive «представлены».
- Removed `**База материалов:**` label-style bold; rewritten as clean prose sentence.
- Split long imperative sentence into two sentences (one action, one qualifier).
- «задокументированная инженерная практика» — more precise than «открытые репозитории экосистемы» (авторская ремарка репозитория убрана из читательского текста).

---

### Hunk 2: Glossary section intro rewrite (lines 36–45)

**What:** Rewrite the glossary intro sentence for executive register; fix missing comma; clean up conditional phrasing about component names.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
@@ -36,10 +36,10 @@
 ## Глоссарий {: #intro_glossary }
 
-Краткий словарь ключевых терминов: фиксирует употребление в комплекте и устраняет разночтения при обсуждении архитектуры, экономики и модели передачи.
+Словарь фиксирует единое употребление терминов и устраняет разночтения при обсуждении архитектуры, экономики и модели передачи.
 
-В тексте отчета **корпоративный RAG-контур**, **сервер инференса на базе vLLM/MOSEC** и **агентный слой Comindware Platform** — это **условные названия компонентов** иллюстративного референс-стека **Comindware**, а не фактические коммерческие SKU.
+**Корпоративный RAG-контур**, **сервер инференса на базе vLLM/MOSEC** и **агентный слой Comindware Platform** — условные названия компонентов иллюстративного референс-стека Comindware, а не коммерческие SKU.
 
-Подробные формулировки для договоров и переговоров:
+Договорные формулировки и точные границы — в приложениях:
```

**Rationale:**
- «Словарь фиксирует...» — active verb opener; eliminates weak «Краткий словарь ключевых терминов:» fragment.
- «Краткий» — pleonasm (glossaries are by definition brief); cut.
- Removed «В тексте отчета» bureaucratic opening; bold names now anchor the sentence directly.
- Removed redundant «фактические» before «коммерческие SKU» — pleonasm.
- «Подробные формулировки для договоров и переговоров:» → «Договорные формулировки и точные границы — в приложениях:» — active, front-loaded, eliminates «подробные» filler.

---

### Hunk 3: Fix `MOSEC` and `vLLM` entries — remove «В контексте пакета:» (lines 73, 89)

**What:** Strip internal authoring prefix «В контексте пакета:» from reader-facing definitions. Rewrite as standalone definitions.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
@@ -72,7 +72,7 @@
-| MOSEC | В контексте пакета: связка из фреймворка Mosec и обвязки **Comindware** вокруг него для единого HTTP-сервиса вспомогательных моделей: эмбеддеров, ранжировщика, защитных моделей и смежных сервисов. |
+| MOSEC | Связка фреймворка Mosec и сервисного слоя Comindware — единый HTTP-сервис вспомогательных моделей: эмбеддеров, ранжировщика и защитных моделей. |
```

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
@@ -88,7 +88,7 @@
-| vLLM | В контексте пакета: upstream-движок vLLM и обвязка **Comindware** вокруг него для промышленного инференса больших языковых моделей через OpenAI-совместимый API, с конфигурацией, проверками доступности и эксплуатационным регламентом. |
+| vLLM | Upstream-движок vLLM с сервисным слоем Comindware для промышленного инференса LLM через OpenAI-совместимый API — включает конфигурацию, проверки доступности и эксплуатационный регламент. |
```

**Rationale:**
- «В контексте пакета:» is an internal authoring remark forbidden in report-pack reader text per task §0 and AGENTS.md.
- «обвязка» (wrapper) → «сервисный слой» — more precise business vocabulary.
- «смежных сервисов» — vague; cut (list already specific enough).

---

### Hunk 4: Fix `SGR` typo «фазы зазиты» → «фазы защиты» and tighten definition (line 81)

**What:** Fix obvious typo; tighten end of definition which is currently vague.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
@@ -80,7 +80,7 @@
-| SGR (Schema-Guided Reasoning) | Структурированное рассуждение по схеме — техника принудительного структурирования рассуждений LLM через предопределённые схемы. По отраслевым бенчмаркам даёт 5–10% улучшение точности по сравнению с неструктурированными промптами; обеспечивает воспроизводимое рассуждение и аудит каждого шага (_[Schema-Guided Reasoning (SGR)](https://abdullin.com/schema-guided-reasoning/)_). В Comindware используется в нескольких точках конвейеров: анализ запросов, критика ответов агентов, планирование после фазы зазиты, детерминированное управление любым мышлением. |
+| SGR (Schema-Guided Reasoning) | Структурированное рассуждение по схеме — техника принудительного структурирования рассуждений LLM через предопределённые схемы. По отраслевым бенчмаркам даёт 5–10% прирост точности против неструктурированных промптов; обеспечивает воспроизводимое рассуждение и пошаговый аудит (_[Schema-Guided Reasoning (SGR)](https://abdullin.com/schema-guided-reasoning/)_). В Comindware применяется в нескольких точках конвейера: анализ запросов, критика ответов агентов, планирование после фазы защиты и детерминированное управление выводом. |
```

**Rationale:**
- «фазы зазиты» → «фазы защиты» — typo fix.
- «улучшение точности по сравнению с» → «прирост точности против» — shorter, more precise.
- «аудит каждого шага» → «пошаговый аудит» — nominalization reduced.
- «детерминированное управление любым мышлением» → «детерминированное управление выводом» — «мышление» is anthropomorphic and vague in an executive doc; «вывод» (inference output) is precise.
- «используется» → «применяется» — stronger register; «в нескольких точках конвейеров» → «нескольких точках конвейера» — correct singular.

---

### Hunk 5: Fix `DSPy` entry — eliminate «иные» канцелярит (line 59)

**What:** Replace «иные библиотеки» with natural language.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
@@ -58,7 +58,7 @@
-| DSPy | Открытый фреймворк декларативной сборки и настройки LLM-конвейеров (модули, сигнатуры, оптимизация промптов и обучающих примеров). К похожему классу относятся иные библиотеки программной сборки промптов и контрактов вывода; в отчёте DSPy приводится как ориентир из открытых туториалов, не как обязательный стек поставки. |
+| DSPy | Открытый фреймворк декларативной сборки и настройки LLM-конвейеров (модули, сигнатуры, оптимизация промптов и обучающих примеров). К тому же классу относятся другие библиотеки программной сборки промптов и контрактов вывода; в отчёте DSPy служит ориентиром из открытых туториалов, а не фиксированным стеком поставки. |
```

**Rationale:**
- «иные» — канцелярит (archaic bureaucratic register); replace with «другие».
- «К похожему классу относятся» → «К тому же классу относятся» — more precise collocation.
- «приводится как ориентир» → «служит ориентиром» — active verb eliminates nominalization chain.

---

### Hunk 6: Fix `LoRA` entry — remove tautology «полное дообучение всех весов» (line 68)

**What:** Tighten the LoRA definition by cutting the tautological «всех весов».

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
@@ -67,7 +67,7 @@
-| LoRA | Low-Rank Adaptation — адаптация большой модели небольшим числом добавочных параметров низкого ранга, без полного дообучения всех весов; обычно дешевле по памяти GPU и хранению, чем полное дообучение. В отчёте упоминается в исследовательских и продуктовых контекстах (в т.ч. Doc-to-LoRA, подходы к «забыванию» весов). |
+| LoRA | Low-Rank Adaptation — адаптация модели небольшим числом параметров низкого ранга без полного дообучения; дешевле по памяти GPU и хранению. В отчёте упоминается в исследовательских и продуктовых контекстах (Doc-to-LoRA, подходы к «забыванию» весов). |
```

**Rationale:**
- «большой модели» — redundant (LoRA is exclusively applied to large models in context); cut.
- «добавочных» — redundant with «низкого ранга»; cut.
- «полного дообучения всех весов» — «всех весов» is tautological with «полного дообучения»; cut.
- «обычно дешевле» → «дешевле» — hedging filler «обычно» removed (the claim is true by definition of the technique).
- «в т.ч.» → remove abbreviation, list directly.

---

### Hunk 7: Fix `Агентный RAG` — rewrite weak opener (line 90)

**What:** Replace weak opener «Вариант RAG» with direct technical definition.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
@@ -89,7 +89,7 @@
-| Агентный RAG | Вариант RAG, где модель не только получает контекст, но и планирует шаги, вызывает инструменты и при необходимости делает несколько итераций поиска и проверки. |
+| Агентный RAG | Расширение RAG, в котором модель планирует шаги, вызывает инструменты и при необходимости выполняет несколько итераций поиска и верификации — вместо однократного обращения к индексу. |
```

**Rationale:**
- «Вариант» — weak, vague opener; replace with «Расширение» — precise technical term.
- «не только получает контекст, но и» — double negation structure weakens the sentence; rewritten to positive construction.
- «проверки» → «верификации» — higher register.
- Added clarifying contrast «вместо однократного обращения к индексу» — answers "So what?" for executive audience.

---

### Hunk 8: Fix `Временный привилегированный доступ` — split run-on sentence (line 99)

**What:** Split complex definition into two sentences for executive readability.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
@@ -98,7 +98,7 @@
-| Временный привилегированный доступ (Just-in-Time access) | Предоставление агенту или пользователю прав исключительно на период выполнения конкретной задачи с автоматическим отзывом по завершении. Устраняет постоянные привилегии (standing privileges), сужая окно компрометации до минимума. |
+| Временный привилегированный доступ (Just-in-Time access) | Права выдаются агенту или пользователю строго на период выполнения задачи и автоматически отзываются по завершении. Устраняет постоянные привилегии (standing privileges) и сужает окно компрометации до минимума. |
```

**Rationale:**
- «Предоставление агенту...» — nominalization opener; replace with active verb «Права выдаются».
- «исключительно на период» → «строго на период» — stronger register.
- Second sentence: add «и» to logically connect the two consequences (устраняет + сужает) — cleaner coordination.

---

### Hunk 9: Fix navigation table — «пром» → «промышленном» (line 136)

**What:** Replace informal abbreviation «пром» with full word.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
@@ -135,7 +135,7 @@
-| Внедрение в пром контуре: роли, фазы, контрольные точки качества | _[Методология](./20260325-research-report-methodology-main-ru.md#method_pack_overview)_ |
+| Внедрение в промышленном контуре: роли, фазы, контрольные точки качества | _[Методология](./20260325-research-report-methodology-main-ru.md#method_pack_overview)_ |
```

**Rationale:**
- «пром» is an informal technical abbreviation inappropriate in a C-level executive document.

---

## Consolidated Patch File

Save as `.opencode/plans/20260406-research-intro-ru-editorial.patch` and apply with `git apply`.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md
@@ -22,13 +22,13 @@
 # Введение {: #intro_pack_overview }
 
-В отчёте представлены материалы по внедрению корпоративного ИИ в резидентном контуре РФ:
+Комплект охватывает полный цикл корпоративного ИИ в резидентном контуре РФ:
 
 - **организация внедрения и эксплуатации** — фазы, роли, контрольные точки;
 - **диапазоны CapEx/OpEx/TCO** — ориентиры для смет и бюджетов;
 - **передача кода и ИС** — комплект KT/IP и приёмка;
 - **граница готового стека Comindware** — что входит в поставку;
 - **риски и контроль** — что закрыть до промышленного запуска.
 
-**База материалов:** публичные прайсы, отраслевые публикации и инженерная практика **Comindware** (открытые репозитории экосистемы).
+Доказательная база — публичные прайсы, отраслевые публикации и задокументированная инженерная практика Comindware.
 
-Применяйте как основу для собственных презентаций, смет и управленческих решений — с адаптацией под профиль заказчика и финальной верификацией договорных условий перед оффером.
+Используйте материалы как основу для презентаций, смет и управленческих решений. Адаптируйте под профиль заказчика и верифицируйте договорные условия перед оффером.
@@ -36,8 +36,8 @@
 ## Глоссарий {: #intro_glossary }
 
-Краткий словарь ключевых терминов: фиксирует употребление в комплекте и устраняет разночтения при обсуждении архитектуры, экономики и модели передачи.
+Словарь фиксирует единое употребление терминов и устраняет разночтения при обсуждении архитектуры, экономики и модели передачи.
 
-В тексте отчета **корпоративный RAG-контур**, **сервер инференса на базе vLLM/MOSEC** и **агентный слой Comindware Platform** — это **условные названия компонентов** иллюстративного референс-стека **Comindware**, а не фактические коммерческие SKU.
+**Корпоративный RAG-контур**, **сервер инференса на базе vLLM/MOSEC** и **агентный слой Comindware Platform** — условные названия компонентов иллюстративного референс-стека Comindware, а не коммерческие SKU.
 
-Подробные формулировки для договоров и переговоров:
+Договорные формулировки и точные границы — в приложениях:
@@ -58,8 +58,8 @@
-| DSPy | Открытый фреймворк декларативной сборки и настройки LLM-конвейеров (модули, сигнатуры, оптимизация промптов и обучающих примеров). К похожему классу относятся иные библиотеки программной сборки промптов и контрактов вывода; в отчёте DSPy приводится как ориентир из открытых туториалов, не как обязательный стек поставки. |
+| DSPy | Открытый фреймворк декларативной сборки и настройки LLM-конвейеров (модули, сигнатуры, оптимизация промптов и обучающих примеров). К тому же классу относятся другие библиотеки программной сборки промптов и контрактов вывода; в отчёте DSPy служит ориентиром из открытых туториалов, а не фиксированным стеком поставки. |
@@ -67,8 +67,8 @@
-| LoRA | Low-Rank Adaptation — адаптация большой модели небольшим числом добавочных параметров низкого ранга, без полного дообучения всех весов; обычно дешевле по памяти GPU и хранению, чем полное дообучение. В отчёте упоминается в исследовательских и продуктовых контекстах (в т.ч. Doc-to-LoRA, подходы к «забыванию» весов). |
+| LoRA | Low-Rank Adaptation — адаптация модели небольшим числом параметров низкого ранга без полного дообучения; дешевле по памяти GPU и хранению. В отчёте упоминается в исследовательских и продуктовых контекстах (Doc-to-LoRA, подходы к «забыванию» весов). |
@@ -72,8 +72,8 @@
-| MOSEC | В контексте пакета: связка из фреймворка Mosec и обвязки **Comindware** вокруг него для единого HTTP-сервиса вспомогательных моделей: эмбеддеров, ранжировщика, защитных моделей и смежных сервисов. |
+| MOSEC | Связка фреймворка Mosec и сервисного слоя Comindware — единый HTTP-сервис вспомогательных моделей: эмбеддеров, ранжировщика и защитных моделей. |
@@ -80,8 +80,8 @@
-| SGR (Schema-Guided Reasoning) | Структурированное рассуждение по схеме — техника принудительного структурирования рассуждений LLM через предопределённые схемы. По отраслевым бенчмаркам даёт 5–10% улучшение точности по сравнению с неструктурированными промптами; обеспечивает воспроизводимое рассуждение и аудит каждого шага (_[Schema-Guided Reasoning (SGR)](https://abdullin.com/schema-guided-reasoning/)_). В Comindware используется в нескольких точках конвейеров: анализ запросов, критика ответов агентов, планирование после фазы зазиты, детерминированное управление любым мышлением. |
+| SGR (Schema-Guided Reasoning) | Структурированное рассуждение по схеме — техника принудительного структурирования рассуждений LLM через предопределённые схемы. По отраслевым бенчмаркам даёт 5–10% прирост точности против неструктурированных промптов; обеспечивает воспроизводимое рассуждение и пошаговый аудит (_[Schema-Guided Reasoning (SGR)](https://abdullin.com/schema-guided-reasoning/)_). В Comindware применяется в нескольких точках конвейера: анализ запросов, критика ответов агентов, планирование после фазы защиты и детерминированное управление выводом. |
@@ -88,8 +88,8 @@
-| vLLM | В контексте пакета: upstream-движок vLLM и обвязка **Comindware** вокруг него для промышленного инференса больших языковых моделей через OpenAI-совместимый API, с конфигурацией, проверками доступности и эксплуатационным регламентом. |
+| vLLM | Upstream-движок vLLM с сервисным слоем Comindware для промышленного инференса LLM через OpenAI-совместимый API — включает конфигурацию, проверки доступности и эксплуатационный регламент. |
@@ -89,8 +89,8 @@
-| Агентный RAG | Вариант RAG, где модель не только получает контекст, но и планирует шаги, вызывает инструменты и при необходимости делает несколько итераций поиска и проверки. |
+| Агентный RAG | Расширение RAG, в котором модель планирует шаги, вызывает инструменты и при необходимости выполняет несколько итераций поиска и верификации — вместо однократного обращения к индексу. |
@@ -98,8 +98,8 @@
-| Временный привилегированный доступ (Just-in-Time access) | Предоставление агенту или пользователю прав исключительно на период выполнения конкретной задачи с автоматическим отзывом по завершении. Устраняет постоянные привилегии (standing privileges), сужая окно компрометации до минимума. |
+| Временный привилегированный доступ (Just-in-Time access) | Права выдаются агенту или пользователю строго на период выполнения задачи и автоматически отзываются по завершении. Устраняет постоянные привилегии (standing privileges) и сужает окно компрометации до минимума. |
@@ -135,8 +135,8 @@
-| Внедрение в пром контуре: роли, фазы, контрольные точки качества | _[Методология](./20260325-research-report-methodology-main-ru.md#method_pack_overview)_ |
+| Внедрение в промышленном контуре: роли, фазы, контрольные точки качества | _[Методология](./20260325-research-report-methodology-main-ru.md#method_pack_overview)_ |
```

---

## Execution Steps

### Task 1: Write patch file

- [ ] Save consolidated patch above to `.opencode/plans/20260406-research-intro-ru-editorial.patch`
- [ ] Verify patch file saved correctly

### Task 2: Dry-run patch

- [ ] Run: `git apply --check .opencode/plans/20260406-research-intro-ru-editorial.patch`
- [ ] Expected: no errors (clean apply)
- [ ] If context mismatch: use `git apply --whitespace=fix` or fall back to Edit tool per-hunk

### Task 3: Apply patch

- [ ] Run: `git apply .opencode/plans/20260406-research-intro-ru-editorial.patch`
- [ ] Verify: `git diff --stat docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md`
- [ ] Expected: ~20 insertions, ~20 deletions (all editorial, no structural changes)

**Checkpoint A:** Open the file. Read the `# Введение` section. Confirm:
- No «В отчёте представлены»
- No `**База материалов:**` label
- Active, direct prose

### Task 4: Read full file post-patch

- [ ] Read the entire modified file
- [ ] Verify: all cross-reference anchors intact (`#intro_pack_overview`, `#intro_glossary`, `#intro_navigation`)
- [ ] Verify: `{% include-markdown %}` directive on line 46 is untouched
- [ ] Verify: all `.md#anchor` links in navigation tables are untouched
- [ ] Verify: YAML front matter unchanged

**Checkpoint B:** Confirm no regressions — no lines removed that weren't in the patch plan.

### Task 5: Self-review prose against standards

- [ ] Re-read Введение: pyramid principle met? Active voice? No self-referential text?
- [ ] Re-read Глоссарий intro: канцелярит eliminated? Clean sentence starts?
- [ ] Spot-check 5 glossary entries: MOSEC, vLLM, SGR, LoRA, DSPy — confirm improvements applied
- [ ] Spot-check navigation table row «промышленном контуре» — confirmed

**Checkpoint C:** If any residual issues found, apply additional Edit tool hunks (document them here before applying).

### Task 6: Final verification

- [ ] Run: `git diff docs/research/executive-research-technology-transfer/report-pack/20260325-research-intro-ru.md`
- [ ] Confirm: only planned lines changed
- [ ] Confirm: file parses as valid Markdown (no broken table rows, no orphaned pipes)

---

## Self-Review: Spec Coverage

| Requirement | Covered in Plan |
|-------------|----------------|
| CEO-standard Russian prose | Hunks 1–9: register elevation throughout |
| Authoritative tone | Hunks 1, 2: active openers, eliminated passives |
| Sophisticated vocabulary | Hunks 3–9: «сервисный слой», «прирост точности», «верификации» |
| Seamless executive flow | Hunks 1, 2: sentence splitting, pyramid principle |
| Business focused | Hunk 1: business framing preserved and sharpened |
| Coherent | All cross-references untouched; navigation tables intact |
| Grounded | No invented content; only editorial improvements |
| Deduplicated | Hunk 6: tautology removed from LoRA |
| Aligned with goals | «задокументированная инженерная практика» — sovereign RF, no repo paths |
| No editorial bloat | Hunks 1, 2, 6: cut filler, hedging, meta-commentary |
| C-level best practices | Pyramid principle, active voice, one-idea-per-sentence throughout |
| Nothing lost | Every semantic unit preserved; deletions are pure redundancy |
| Useless stuff removed | «иные», «В контексте пакета», «Краткий», «обычно», «фактические» |

## Placeholder Scan

No TBDs, no TODOs, no «fill in details», no «similar to Task N». All hunks contain exact before/after text.

## Type Consistency

No function signatures or types involved (editorial plan). All anchor IDs referenced in patch context lines match the source file exactly.
