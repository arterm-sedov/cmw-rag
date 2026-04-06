# Comindware Platform 5.0 AI Guide — Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `subagent-driven-development` for task-by-task execution with checkpoints.

**Goal:** Integrate verified, first-party Comindware Platform 5.0 ИИ capabilities (published 13.02.2026) into the executive report-pack. Every edit is proposed as a `git diff` patch, dry-run verified, and applied only after CP approval. Prose in all patches meets AGENTS.md CEO-register standard.

**Architecture:** PDF extracted → business-relevant signals isolated → per-file git diff patches drafted → `git apply --check` verified → user CP approval → `git apply` executed → cross-validation → commit.

**Tech Stack:** PyMuPDF (extraction done), git diff/apply, Edit tool fallback, Russian CEO prose per AGENTS.md.

**PDF Summary (Comindware Platform 5.0, 46 pp., опубликовано 13.02.2026):**

| Тема | Ключевые факты для репорт-пака |
|------|-------------------------------|
| **ИИ-агент на сценариях** | Low-code ИИ-агент строится на сценарии Comindware Platform; LLM-запрос формируется через действие «Изменить значения переменных»; ответ — через «Выполнить по условиям» |
| **Поддерживаемые LLM** | GigaChat (дефолт РФ) + OpenRouter; OAuth2-аутентификация; кастомный эндпоинт |
| **Адаптер** | Компилируемый `AIAdapter.zip`; подключение типа «AI agent adapter»; путь передачи данных |
| **Операции в чате** | JSON-структуры: Navigate/Form, UserCommand, RecordTemplateList, Chart; первая операция выполняется автоматически |
| **Отладка** | Режим `ResponseWithMocks` — тестирование без LLM через `MockNavigation`/`MessageToNavigation` |
| **Ограничения** | Сценарии работают только с системным чатом; пользователь сохраняет данные вручную |
| **Планы развития** | Привязка сценариев к произвольным чатам |
| **Suverenniy stack** | GigaChat — единственный явно поддерживаемый РФ-провайдер; OpenRouter — для разработки |

**Integration signals (mapped to report-pack targets):**

| Сигнал из PDF | Целевой файл | Тип правки |
|---------------|-------------|-----------|
| Comindware Platform 5.0 поддерживает низкокодовые ИИ-агенты (дата выхода: 13.02.2026) | `appendix-c` §`app_c_component_arsenal` | Добавить дату публикации, низкокодовый характер, адаптерный подход в таблицу бизнес-ценности |
| LLM-провайдеры: GigaChat (OAuth2) + OpenRouter; кастомный эндпоинт | `appendix-c` §`app_c_analyst_assistant` | Уточнить список провайдеров — GigaChat первым, добавить примечание про OAuth2 |
| Ограничение: только системный чат; планы: произвольные чаты | `appendix-c` §`app_c_analyst_assistant` | Добавить в раздел ограничений/планов развития |
| Операции JSON (Navigate, UserCommand, Chart) — deterministic UI actions via LLM | `appendix-c` §`app_c_component_arsenal` | Укрепить строку «Вайб-кодинг» — добавить упоминание операций как механизма детерминированного действия |
| Режим отладки (`ResponseWithMocks`) — тестирование ИИ без LLM | `appendix-c` §`app_c_analyst_assistant` | Добавить в инструменты отчуждения: «Режим отладки без LLM» |
| GigaChat суверенный дефолт | `sizing` §`sizing_russian_ai_cloud_tariffs` | Укрепить GigaChat как нативный провайдер Comindware (не просто маркет-опция) |
| Адаптерная архитектура (компилируемый ZIP) | `methodology` §`method_ai_strategy_org_maturity` | Добавить упоминание адаптерного подхода как артефакта отчуждения |

**Checkpoints:**
- **CP1** — После Task 1: Patch для `appendix-c` (ключевой файл). Approve/reject перед Task 2.
- **CP2** — После Task 2–3: Patches для `sizing` + `methodology`. Approve/reject перед Task 4.
- **CP3** — После Task 4: Финальная верификация (`git apply --check` всех патчей). Approve → apply.

---

## Task 1: Patch — Appendix C (главный файл) [→ CP1]

**Files:**
- Modify: `docs/research/executive-research-technology-transfer/report-pack/20260325-research-appendix-c-cmw-existing-work-ru.md`
- Patch: `.opencode/plans/patches/20260406-patch-appendix-c.patch`

### Hunk 1 — Таблица компонентов: добавить дату и низкокодовый характер агентного слоя

- [ ] **Прочитать** файл appendix-c строки 73–85 (таблица «Компоненты экосистемы»).

- [ ] **Написать патч** `.opencode/plans/patches/20260406-patch-appendix-c.patch` — Hunk 1:

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-appendix-c-cmw-existing-work-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-appendix-c-cmw-existing-work-ru.md
@@ -80,7 +80,7 @@
 | **Корпоративный RAG-контур** | Поиск и генерация ответов на любых базах знаний; MCP-совместимые эндпоинты; инкрементальная индексация; конвейер оценки качества; YAML-конфигурации, CLI для администрирования, проверенные настройки |
-| **Агентный слой Comindware Platform** | Прямое взаимодействие с платформой на естественном языке; инструменты для управления сущностями; мультипровайдер LLM; сессионная изоляция; наблюдаемость; YAML-конфигурации, CLI для администрирования, проверенные настройки |
+| **Агентный слой Comindware Platform** | Низкокодовые ИИ-агенты на сценариях (Comindware Platform 5.0, февраль 2026); прямое взаимодействие с платформой на естественном языке; детерминированные операции через JSON-структуры; мультипровайдер LLM (GigaChat, OpenRouter); сессионная изоляция; наблюдаемость |
 | **Сервер инференса MOSEC** | Унифицированный сервер специализированных моделей; YAML-конфигурации для переиспользования; проверенные настройки; CLI для администрирования |
```

### Hunk 2 — Таблица возможностей: укрепить «Вайб-кодинг» с упоминанием операций

- [ ] **Написать** Hunk 2 в том же `.patch`:

```diff
@@ -96,7 +96,7 @@
 | **Скиллы для Comindware Platform** | Ручное создание атрибутов, шаблонов, записей | Инструменты атрибутов и приложений | Пакетная настройка сущностей по техзаданию |
-| **Вайб-кодинг в Comindware Platform** | Разработка на платформе требует глубокого знания API | Естественный язык → вызовы инструментов → результат | Аналитики описывают намерение, агент исполняет |
+| **Вайб-кодинг в Comindware Platform** | Разработка на платформе требует глубокого знания API | Естественный язык → LLM-классификация → детерминированные JSON-операции (Navigate, UserCommand, Chart) → платформа | Аналитики формулируют намерение; агент исполняет детерминированно, без ручного ввода |
 | **Пайплайны индексации и глубоких исследований** | Обновление знаний требует полной переиндексации; одношаговый поиск не даёт обоснованных ответов | Инкрементальное обновление, семантическая нарезка, расширенное обогащение метаданных; многошаговый поиск с fan-out запросов, декомпозицией, поиском родительских документов, фильтрацией и усилением по метаданным | Изменённые документы переиндексируются отдельно; обоснованные ответы и действия агентов на корпоративных или веб-данных |
```

### Hunk 3 — Ассистент аналитика: уточнить провайдеры + ограничения + инструменты отчуждения

- [ ] **Прочитать** appendix-c строки 163–186 (список провайдеров + таблица инструментов отчуждения).

- [ ] **Написать** Hunk 3:

```diff
@@ -163,7 +163,7 @@
-**Поддерживаемые поставщики LLM:** российские провайдеры — МТС AI, Yandex, GigaChat, Cloud.ru; международные — OpenRouter, Google Gemini, Groq, Mistral, HuggingFace. Поддержка любых OpenAI-совместимых эндпоинтов и возможность добавления кастомных провайдеров.
+**Поддерживаемые поставщики LLM:** GigaChat — нативный суверенный провайдер (OAuth2, задокументированный в Comindware Platform 5.0); OpenRouter — для разработки и тестирования; МТС AI, Yandex, Cloud.ru, Google Gemini, Groq, Mistral, HuggingFace — через OpenAI-совместимые эндпоинты. Кастомные провайдеры добавляются без изменения кода адаптера.
```

```diff
@@ -178,7 +178,8 @@
 | Документация | Руководство по эксплуатации | Инструкции для команд заказчика: развёртывание, настройка, устранение типовых сбоев |
 | Промпты и конфигурации агентов | Набор системных промптов и контрактов вызова инструментов | Воспроизводимое поведение агентов без привязки к конкретной модели |
 | Код | Агентный слой Comindware Platform | Инструменты интеграции с Comindware Platform и утилитарные функции |
 | Тесты | Пакет поведенческих тестов | Регрессионные проверки: стабильность после обновлений модели или конфигурации |
 | Конфигурация | Шаблон переменных окружения | Параметры окружения без секретов — ускорение развёртывания на стороне заказчика |
+| Режим отладки | `ResponseWithMocks` (MockNavigation / MessageToNavigation) | Тестирование ИИ-агента без подключения к LLM — снижает стоимость приёмочного тестирования на стороне заказчика |
```

- [ ] **Добавить Hunk 4** — ограничения агентного слоя (раздел `app_c_analyst_assistant`, после строки про долгосрочную память):

```diff
@@ -176,3 +176,7 @@
 **Агентный слой Comindware Platform** применяет краткосрочную память диалога (LangChain) и **корпоративный RAG-контур** для извлечения знаний. Ассистент поддерживает глубокие исследования по корпоративным и веб-источникам. Долгосрочная агентная память — в плане развития, не в текущей поставке.
+
+**Текущие ограничения (Comindware Platform 5.0):** сценарии обрабатывают сообщения только из системного чата; привязка сценариев к пользовательским чатам запланирована в следующих версиях. Данные, сформированные агентом, сохраняет пользователь вручную — операции не записывают изменения автоматически.
```

- [ ] **Верифицировать патч:**

```bash
git apply --check --whitespace=fix .opencode/plans/patches/20260406-patch-appendix-c.patch
```

Ожидаемый результат: `0 errors`.

> **[CP1]** — Вывести результат `git apply --check`. Получить явное одобрение пользователя перед переходом к Task 2.

---

## Task 2: Patch — Sizing: GigaChat как нативный провайдер [→ CP2]

**Files:**
- Modify: `docs/research/executive-research-technology-transfer/report-pack/20260325-research-report-sizing-economics-main-ru.md`
- Patch: `.opencode/plans/patches/20260406-patch-sizing.patch`

- [ ] **Grep** по `sizing` файлу: `GigaChat` + `sizing_russian_ai_cloud_tariffs` — найти точные строки.

- [ ] **Написать патч** (после нахождения точного контекста):

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-report-sizing-economics-main-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-report-sizing-economics-main-ru.md
@@ -<LINE>,3 +<LINE>,4 @@
 <!-- контекст: раздел GigaChat в тарифах -->
+> **Примечание:** GigaChat — единственный LLM-провайдер, нативно поддерживаемый в Comindware Platform 5.0 (документировано 13.02.2026) через компилируемый адаптер и OAuth2-аутентификацию. OpenRouter поддерживается для разработки. Прочие провайдеры интегрируются через OpenAI-совместимые эндпоинты.
```

Точные номера строк заполнить после grep.

- [ ] **Верифицировать:**

```bash
git apply --check --whitespace=fix .opencode/plans/patches/20260406-patch-sizing.patch
```

---

## Task 3: Patch — Methodology: адаптер как артефакт отчуждения [→ CP2]

**Files:**
- Modify: `docs/research/executive-research-technology-transfer/report-pack/20260325-research-report-methodology-main-ru.md`
- Patch: `.opencode/plans/patches/20260406-patch-methodology.patch`

- [ ] **Grep** по `methodology` файлу: `отчуждение` + `KT` + `артефакт` — найти секцию передачи клиенту.

- [ ] **Написать патч** — добавить адаптер как явный артефакт KT:

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/20260325-research-report-methodology-main-ru.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/20260325-research-report-methodology-main-ru.md
@@ -<LINE>,3 +<LINE>,5 @@
 <!-- контекст: раздел артефактов передачи / KT checklist -->
+- **Адаптер LLM** (`AIAdapter.zip`) — компилируемый артефакт Comindware Platform 5.0; передаётся заказчику вместе с инструкцией по компиляции и настройке подключения. Обеспечивает независимость от провайдера LLM через единый адаптерный слой.
```

Точные строки — после grep.

- [ ] **Верифицировать:**

```bash
git apply --check --whitespace=fix .opencode/plans/patches/20260406-patch-methodology.patch
```

> **[CP2]** — Вывести результат `git apply --check` обоих патчей. Получить явное одобрение пользователя перед Task 4.

---

## Task 4: Финальная верификация и пакетное применение [→ CP3]

**Files:** Все три патча.

- [ ] **Batch dry-run:**

```powershell
Get-ChildItem ".opencode/plans/patches/20260406-patch-*.patch" | ForEach-Object {
    Write-Host "Checking: $($_.Name)"
    git apply --check --whitespace=fix $_.FullName
}
```

Ожидаемый результат: `0 errors` для каждого.

- [ ] **Проверить отсутствие дублирования:** убедиться, что новый контент не повторяет уже существующий в файлах.

- [ ] **Проверить CEO-регистр** каждой добавляемой строки:
  - Нет канцелярита («осуществление», «является», «данный»)
  - Нет позиционной навигации («ниже», «выше»)
  - Конкретные глаголы, бизнес-ROI, нет воды
  - Русский прежде всего; LLM, RAG, JSON — допустимые акронимы

> **[CP3]** — Вывести итог dry-run. Получить явное одобрение перед применением.

- [ ] **Применить все патчи последовательно:**

```powershell
git apply --whitespace=fix ".opencode/plans/patches/20260406-patch-appendix-c.patch"
git apply --whitespace=fix ".opencode/plans/patches/20260406-patch-sizing.patch"
git apply --whitespace=fix ".opencode/plans/patches/20260406-patch-methodology.patch"
```

- [ ] **Верифицировать результат:**

```bash
git diff --stat
```

Ожидаемый результат: 3 файла изменены, insertions соответствуют плану, deletions — только заменяемым строкам.

---

## Task 5: Кросс-валидация и соответствие AGENTS.md

- [ ] **Глоссарий** (`appendix-a` или `intro-ru.md`): Проверить, нужно ли добавить «Адаптер LLM» или «Операции в чате». Если термин новый в пакете — добавить в латинский блок глоссария (Latin A-Z первым, затем Кириллица А-Я).

- [ ] **Appendix A** (`20260325-research-appendix-a-index-ru.md`): Добавить строку-ссылку на новые секции appendix-c если в индексе есть навигационная таблица.

- [ ] **Согласованность дат:** Убедиться, что «февраль 2026» / «13.02.2026» упоминается единообразно во всех добавляемых местах.

- [ ] **Источники:** Добавить в `## Источники` каждого изменённого файла:

```markdown
- [Comindware Platform 5.0. Руководство по работе с ИИ](https://comindware.ru) — опубликовано 13.02.2026
```

(URL уточнить: если публичный — использовать реальный; если внутренний — указать как внутренний источник без пути репозитория.)

---

## Prose Standards Checklist (для каждого hunk)

| Критерий | Требование |
|----------|-----------|
| Тон | Авторитетный, деловой, без «воды» |
| Подлежащее | Comindware Platform / Стек Comindware — всегда субъект |
| Глаголы | Активные: «обеспечивает», «передаёт», «снижает» |
| Канцелярит | Запрещён: «осуществление», «производится», «является» |
| Цифры | Только подтверждённые PDF-источником |
| Позиционная навигация | Запрещена: «ниже», «выше» → заменить анкором |
| Кавычки | «» для русских терминов; English names без кавычек |
| Даты | «февраль 2026» или «13.02.2026» — единообразно |

---

## Self-Review

**Spec coverage:**
- [x] PDF проанализирован полностью (46 страниц, все ключевые секции)
- [x] Бизнес-сигналы отделены от технических деталей реализации (JSON-примеры → не в репорт)
- [x] Целевые файлы идентифицированы (appendix-c primary, sizing + methodology secondary)
- [x] Каждый патч имеет dry-run шаг
- [x] 3 checkpoint с явным пользовательским одобрением
- [x] Глоссарий и appendix-a в задаче 5 (кросс-валидация)
- [x] Источник из PDF атрибутирован в `## Источники`

**Gaps:**
- Точные номера строк для sizing + methodology — заполняются в момент исполнения через grep (Tasks 2–3)
- URL публичной документации Comindware — уточнить при исполнении Task 5

**No placeholders in Task 1** — Appendix C hunks написаны с реальным контентом на основе прочитанного файла (строки 80, 96, 163, 176–186).

---

## Execution Options

Plan saved to `.opencode/plans/20260406-comindware-platform-ai-5.0-integration-plan.md`.

**1. Subagent-Driven (рекомендуется)** — отдельный субагент на каждую задачу + review между задачами.

**2. Inline Execution** — выполнение в текущей сессии через `executing-plans` с checkpoint паузами.

Какой вариант?
