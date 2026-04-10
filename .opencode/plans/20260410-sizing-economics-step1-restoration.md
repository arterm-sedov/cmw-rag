# Step 1: Restore "Слой перед LLM и режимы нагрузки" Section

## Status: COMPLETED ✅

## Decision
- **Placement:** New section after "Около-LLM инфраструктура" (line 343)
- **Engineering benchmark:** Included full paragraph
- **Structure:** Standalone H3 section with clear business focus

## Changes Made
- Added anchor `{: #sizing_pre_llm_layer_load_modes }`
- Restructured content for business clarity:
  - Opening sentence connects to cost model
  - "Что закладывать в смету" bullet list for actionable items
  - Engineering benchmark preserved as technical context
- Russian punctuation: properNBSP, dashes, formatting

## Original Content Location
- OLD file: lines 380-393
- Section: "Слой перед LLM и режимы нагрузки (ориентиры для модели затрат)"

---

# Step 2: Restore "Наблюдаемость LLM/RAG: сценарии размещения и бюджет" (PENDING)

## Original Content Location
- OLD file: lines 396-404
- Section: "Наблюдаемость LLM/RAG: сценарии размещения и бюджет"

## Content to Restore
```markdown
| Сценарий | Типичные статьи затрат | Применимость в РФ (ориентир) |
| :--- | :--- | :--- |
| **Зарубежный SaaS** наблюдаемости | Абонентская плата по объёму трасс/сидов, мало CapEx | Допустим при **явной** правовой оценке, ДПО и допустимости размещения копий данных; для **ПДн** в проде часто **не** дефолт |
| **Self-hosted** (on-prem или IaaS заказчика) | CapEx/аренда ВМ и СХД, занятость специалистов на сопровождение стека | Соответствует типовым ожиданиям **локализации** и контроля журналов при работе с чувствительными данными |
| **Облако РФ** (управляемые ВМ/К8s/SaaS у РФ-провайдера) | OpEx по потреблению + сетевой трафик телеметрии | Компромисс: меньше CapEx, чем свой ЦОД, при сохранении договорного контура РФ |
| **Гибрид** (метаданные в корпорации, контент трасс в зашифрованном хранилище) | Инжиниринг конвейера + отдельное хранилище | Соотносится с рекомендациями **OpenTelemetry** не хранить полный текст промптов в атрибутах спанов по умолчанию |
```

## Questions
- Should this be restored as a table or reformatted?
- Where in the NEW structure should it go?

---

# Step 3: Restore Detailed Token Calculation Formulas (PENDING)

## Original Content Location
- OLD file: lines 511-556
- Section: "Средние длины по корпусу заявок (заявка + ответ)"

## Content to Restore
- Calculation formula block
- Source attribution
- RAG component explanation

---

# Step 4: Restore"Анализ чувствительности" Table (PENDING)

## Original Content Location
- OLD file: lines 618-629

## Content to Restore
```markdown
| Параметр | Консервативный (Small) | Базовый (Medium) | Агрессивный (Enterprise) |
| :--- | :--- | :--- | :--- |
| **Нагрузка (DAU)** | 10–50 пользователей | 50–500 пользователей | 500+ пользователей |
| **Запросов/день** | ~200 | ~2 500 | ~10 000+ |
| **Средний контекст** | 4K токенов | 16K токенов | 32K–128K токенов |
| **Норматив задержки** | < 5 с | < 2 с | < 1 с (реальное время) |
| **Рекомендуемое железо** | 1× RTX 4090 / аналог | 2×–4× RTX 4090 или A100 | GPU-кластер |
```

---

# Step 5: Restore Picoclaw Section Details (PENDING)

## Original Content Location
- OLD file: lines 919-954

## Content to Restore
- Separate "Характеристики" and "Функционал" subsections
- "Self-modification (перезапуск без смертей)" detail