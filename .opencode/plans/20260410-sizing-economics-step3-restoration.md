# Step 3: Restore "Наблюдаемость LLM/RAG: сценарии размещения и бюджет"

## Status: ANALYZING

## Original Content (OLD lines 395-416)

### Table 1: Observability Placement Scenarios
```markdown
| Сценарий | Типичные статьи затрат | Применимость в РФ (ориентир) |
| **Зарубежный SaaS** наблюдаемости | Абонентская плата по объёму трасс/сидов, мало CapEx | Допустим при явной правовой оценке, ДПО и допустимости размещения копий данных; для ПДн в проде часто не дефолт |
| **Self-hosted** (on-prem или IaaS заказчика) | CapEx/аренда ВМ и СХД, занятость специалистов на сопровождение стека | Соответствует типовым ожиданиям локализации и контроля журналов при работе с чувствительными данными |
| **Облако РФ** (управляемые ВМ/К8s/SaaS у РФ-провайдера) | OpEx по потреблению + сетевой трафик телеметрии | Компромисс: меньше CapEx, чем свой ЦОД, при сохранении договорного контура РФ |
| **Гибрид** (метаданные в корпорации, контент трасс в зашифрованном хранилище) | Инжиниринг конвейера + отдельное хранилище | Соотносится с рекомендациями OpenTelemetry не хранить полный текст промптов в атрибутах спанов по умолчанию |
```

### OpenTelemetry Reference
```markdown
Метрики в духе `gen_ai.client.token.usage` и `gen_ai.client.operation.duration` из конвенций OpenTelemetry GenAI дают общий язык с биллингом API и упрощают аллокацию FinOps по продуктам; методологическая связка — см. «[Промышленная наблюдаемость LLM, RAG и агентов](./20260325-research-appendix-d-security-observability-ru.md#app_d_llm_rag_agent_observability)».
```

### Table 2: CapEx/OpEx Split
```markdown
| Вид затрат | Часто у интегратора | Часто у заказчика |
| **CapEx** | Лицензии на инструменты разработки, стенды пилота | Серверы, GPU, СХД, сетевое оборудование |
| **OpEx (проект)** | Аналитика, разработка, настройка RAG, интеграции | Внутренний PM, приёмка, обучение пользователей |
| **OpEx (эксплуатация)** | Поддержка по SLA, доработки по бэклогу | Облачные API, электроэнергия/ЦОД, мониторинг, ИБ |
| **Передача (KT / IP)** | Документация, сессии передачи, hypercare | Владение репозиторием, дальнейшее развитие |
```

### MTS AI Example (Marginal - marked as "not verified by third party")
- Claims ~1B RUB savings from moving training/inference to cloud
- Should only be used as hypothesis for financial modeling

---

## NEW File Context (current structure)

```
### Около-LLM инфраструктура: журналы, события, нагрузка [lines 337-343]
```
(Infrastructure tools: seq-db, file.d, framer)

```
### Слой перед LLM и режимы нагрузки [lines 345-362]
```
(Pre-LLM processing: fast path vs cascade)

```
### FinOps и юнит-экономика нагрузки [lines 364-371]
```
(Budget metrics and allocation)

---

## Value Analysis

| Content | Value | Recommendation |
|---------|-------|----------------|
| Observability table (4 scenarios) | HIGH - helps decide SaaS vs self-hosted vs hybrid | Include |
| OpenTelemetry reference | MEDIUM - links to standards and Comindware reference | Include, keep concise |
| CapEx/OpEx split table | MEDIUM - clarifies who pays for what | Include if space permits |
| MTS AI example | LOW - illustrative but unverifiable | Skip (marked as unreliable) |

---

## Questions for User

1. Should the CapEx/OpEx split table be included?
2. Where should this content go - after "Около-LLM инфраструктура" or before "FinOps"?