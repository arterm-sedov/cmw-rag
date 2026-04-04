---
title: 'Обзор и ведомость документов: навигация, реестр и источники'
date: 2026-03-28
status: 'Черновой комплект материалов для руководства (v1, март 2026)'
tags:
  - AI
  - CapEx
  - комплаенс
  - корпоративный
  - методология
  - наблюдаемость
  - OpEx
  - RAG
  - коммерциализация
  - TCO
  - комплект
  - сайзинг
  - экономика
---

# Обзор и ведомость документов: навигация, реестр и источники {: #app_a }

## Обзор {: #app_a_pack_overview }

Настоящее приложение помогает **быстро найти ответ** в комплекте отчётов _«Внедрение ИИ: методология и сайзинг»_ по ключевым вопросам внедрения GenAI в резидентном контуре:

- **Организация внедрения и эксплуатации** — фазы, роли, контрольные точки
- **Диапазоны CapEx/OpEx/TCO** — ориентиры для смет и бюджетов
- **Передача кода и ИС** — комплект KT/IP и приёмка
- **Граница готового стека Comindware** — что входит в поставку
- **Риски и контроли** — что закрыть до промышленного запуска

**Состав комплекта:** коммерческое резюме для руководства, два основных отчёта (методология; сайзинг и экономика), приложения (A–E), а также два **кратких резюме** для быстрого ознакомления.

**В этом документе:** навигация по вопросам → документам, ведомость комплекта и **единый реестр источников** (дополняется перекрёстными ссылками внутри отчётов).

**База материалов:** публичные прайсы, отраслевые публикации и инженерная практика **Comindware** (открытые репозитории экосистемы). Используйте как основу для собственных презентаций, смет и управленческих решений — с перепроверкой цифр на дату и адаптацией под профиль заказчика.

## Как пользоваться комплектом {: #app_a_reading_guide_executives }

По [таблице «вопрос → документ»](#app_a_question_document_navigation) перейдите к требуемому разделу.

### Быстрый маршрут {: #app_a_reading_guide_quick_route }

1. **Обосновать бюджет** — _[Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md)_.
2. **Оценить риски и комплаенс** — _[Безопасность и наблюдаемость](./20260325-research-appendix-d-security-observability-ru.md)_.
3. **Понять как передать контур** — _[Отчуждение ИС и кода](./20260325-research-appendix-b-ip-code-alienation-ru.md)_.
4. **Принять решение для C-level (быстро)** — _[Резюме для руководства: коммерческое обоснование, методология и экономика внедрения ИИ](./20260331-research-executive-unified-ru.md)_.

### Полный маршрут {: #app_a_reading_guide_complete_route }

1. **Решения по модели внедрения, ролям, фазам и качеству** — _[Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md)_.
2. **Сметы, тарифы, TCO и сценарии сайзинга** — _[Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md)_.
3. **Передача ИС и кода, KT/IP** — _[Отчуждение ИС и кода](./20260325-research-appendix-b-ip-code-alienation-ru.md)_.
4. **Фактический состав стека Comindware** — _[Наработки Comindware](./20260325-research-appendix-c-cmw-existing-work-ru.md)_.
5. **ИБ, комплаенс и промышленная наблюдаемость** — _[Безопасность и наблюдаемость](./20260325-research-appendix-d-security-observability-ru.md)_.
6. Цифры и ссылки в тексте **проверяйте по сводному реестру источников** — _[Сводный реестр источников (часть III)](#app_a_sources_registry)_.

## Словарь терминов, акронимов и условных обозначений {: #app_a_conventions_and_scope }

Ниже — не полный ИТ-словарь, а перечень обозначений, которые в этом комплекте имеют **специальный**, **неочевидный** или **внутренний для пакета** смысл.

В тексте комплекта **корпоративный RAG-контур**, **vLLM**, **MOSEC** и агентный слой **Comindware Platform** — это **условные названия ролей компонентов** иллюстративного референс-стека **Comindware**, а не обязательный коммерческий продукт или фиксированные SKU.

Подробные формулировки для договоров и переговоров — в _[Приложении B «Отчуждение ИС и кода (KT, IP)»](20260325-research-appendix-b-ip-code-alienation-ru.md)_.

| Термин / акроним | Что означает в этом комплекте |
| --- | --- |
| Корпоративный RAG-контур | Внутреннее обозначение референсного контура ассистента **Comindware** на базе поиска по корпоративным данным и генерации ответа с опорой на найденный контекст. |
| Агентный слой платформы (Comindware Platform) | Внутреннее обозначение сценариев, где LLM не только отвечает, но и инициирует действия в платформе через разрешённые инструменты и API. |
| GenAI | Generative AI, то есть генеративный ИИ: модели, которые создают текст, код, структуры данных и иные артефакты. |
| RAG | Retrieval-Augmented Generation — генерация ответа с предварительным поиском по корпоративным данным, документам или базе знаний. |
| Agentic RAG | Вариант RAG, где модель не только получает контекст, но и планирует шаги, вызывает инструменты и при необходимости делает несколько итераций поиска и проверки. |
| LLM | Large Language Model — большая языковая модель для генерации и анализа текста, кода и инструкций. |
| SLM | Small Language Model — более компактная модель, которую используют для дешёвых или быстрых сценариев, где не нужен максимальный уровень рассуждения. |
| vLLM | Высокопроизводительный движок инференса LLM с OpenAI-совместимым API; в пакете это один из базовых вариантов промышленной подачи больших моделей. |
| MOSEC | Фреймворк и сервисный шаблон для подачи ML-моделей по HTTP; в пакете обычно означает единый контур для эмбеддингов, реранка, guardrails и смежных вспомогательных моделей. |
| KT | Knowledge Transfer — передача знаний, регламентов, runbook-ов, практики эксплуатации и обучения команды заказчика. |
| IP | Intellectual Property — интеллектуальная собственность: код, артефакты, модели, документация, права использования и условия передачи. |
| PoC | Proof of Concept — короткий этап проверки гипотезы: работает ли сценарий технически и есть ли шанс на бизнес-эффект. |
| BOT | Build–Operate–Transfer — модель «создать, эксплуатировать, передать», при которой поставщик сначала собирает и ведёт контур, а затем передаёт его заказчику. |
| TOM | Target Operating Model — целевая операционная модель: роли, процессы, метрики, контуры ответственности и правила эксплуатации после внедрения. |
| CapEx | Capital Expenditures — капитальные затраты: вложения в оборудование, лицензии, базовую инфраструктуру и ввод в эксплуатацию. |
| OpEx | Operating Expenditures — операционные затраты: облачные счета, поддержка, сопровождение, мониторинг, ИБ и команда эксплуатации. |
| TCO | Total Cost of Ownership — совокупная стоимость владения решением на горизонте нескольких лет, а не только стартовый бюджет. |
| KPI | Key Performance Indicator — измеримый показатель результата; в пакете используется для контроля внедрения, качества ответов и эффекта для бизнеса. |
| SLA / SLO | SLA — обещанный уровень сервиса для заказчика; SLO — внутренняя целевая метрика качества сервиса, например по задержке, доступности или точности. |
| Observability / наблюдаемость | Практика, при которой контур можно разбирать по трассам, метрикам, логам и событиям, а не по косвенным симптомам. |
| FinOps | Подход к управлению облачными и ИИ-затратами через прозрачные метрики потребления, аллокацию по продуктам и контроль unit economics. |
| LLMOps / ModelOps / AgentOps | Управленческие и инженерные практики эксплуатации моделей и агентов: релизы, оценка качества, мониторинг, версии, инциденты и аудит. |
| Guardrails | Набор ограничений и проверок вокруг модели: политики ответа, фильтры, валидация формата, защитные сценарии и правила безопасного вызова инструментов. |
| MCP | Model Context Protocol — протокол подключения инструментов и внешних ресурсов к агенту через явные серверы, права и контракты вызова. |
| AI TRiSM | AI Trust, Risk and Security Management — рамка доверия, рисков и безопасности ИИ: объяснимость, защита данных и моделей, устойчивость и контроль соответствия. |
| NIST AI RMF | AI Risk Management Framework от NIST — методологическая рамка управления рисками ИИ; в пакете используется как ориентир, а не как замена нормам РФ. |
| Open weights | Открыто опубликованные веса модели, которые можно разворачивать в своём контуре; это не то же самое, что «бесплатно» или «без лицензионных ограничений». |
| On-prem | Размещение в собственном или выделенном контуре заказчика, а не в публичном управляемом API провайдера. |
| VRAM | Память видеоускорителя; один из главных ограничителей при локальном инференсе, реранке и запуске эмбеддинговых моделей. |
| ADR | Architecture Decision Record — зафиксированное архитектурное решение: что выбрано, почему и при каких ограничениях. |
| RAGAS / DeepEval / MERA | Примеры фреймворков и контуров оценки качества RAG/LLM: помогают измерять полезность ответа, релевантность контекста и регрессии после изменений. |
| LLM-as-a-judge | Подход, при котором отдельная модель используется как «судья» для оценки качества ответов по заранее заданной рубрике. |
| TOON | Token-Oriented Object Notation — компактный формат структурированных данных, который в ряде сценариев уменьшает токеновые затраты по сравнению с JSON. |

## Единые KPI и их интерпретация {: #app_a_kpi_semantics }

Используйте следующие KPI как **операционные ориентиры**, они не заменяют юридические критерии соответствия и комплаенса:

- `>60 %` — доля регулярного использования ИИ в целевой группе пользователей.
- `30–40 %` — целевой диапазон сокращения времени выполнения типового сценария.
- `>95 %` — доля ответов, прошедших внутреннюю рубрику качества.

## Курс USD для смет {: #app_a_fx_policy }

**1 USD = 85 RUB** — единый ориентир для сопоставления USD-прайсов и рублёвых оценок в материалах на март 2026.

В **сметах и в договорных КП** ориентируйтесь на **курс ЦБ РФ на текущую дату** или на **курс, зафиксированный в договоре**.

Закладывайте отклонение курса **±10 %** — ориентир чувствительности для **зависимых от USD** статей (импортное железо, зарубежные каталоги); но это не прогноз.

## Связанные документы {: #app_a_related_documents }

- [Резюме для руководства: коммерческое обоснование, методология и экономика внедрения ИИ](./20260331-research-executive-unified-ru.md)
- [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_pack_overview)
- [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_pack_overview)
- [Приложение B: отчуждение ИС и кода (KT, IP)](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_pack_overview)
- [Приложение C. Имеющиеся наработки Comindware](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_pack_overview)
- [Приложение D: безопасность, комплаенс и наблюдаемость (observability)](./20260325-research-appendix-d-security-observability-ru.md#app_d__pack_overview)
- [Приложение E. Рыночные и технические сигналы (справочно)](./20260325-research-appendix-e-market-technical-signals-ru.md#app_e_root)

## Базовые документы по теме {: #app_a_topic_owners_canonical_docs }

| Тема | Документ |
| --- | --- |
| Коммерческое резюме предложения: что продаём и как передаём способность | [Резюме для руководства: коммерческое обоснование, методология и экономика внедрения ИИ](./20260331-research-executive-unified-ru.md) |
| Экономика: цифры, тарифы, сценарии сайзинга, CapEx / OpEx / TCO | [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_pack_overview) |
| Методология: TOM, фазы внедрения, производственная модель (таблицы затрат — в отчёте по сайзингу) | [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_pack_overview) |
| Отчуждение ИС и кода: KT / IP, лицензии, комплект передачи, приёмка | [Приложение B: отчуждение ИС и кода (KT, IP)](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_pack_overview) |
| Наработки **Comindware**: состав стека, границы «что есть сегодня» | [Приложение C. Имеющиеся наработки Comindware](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_pack_overview) |
| Безопасность, комплаенс, наблюдаемость (углубление) | [Приложение D: безопасность, комплаенс и наблюдаемость (observability)](./20260325-research-appendix-d-security-observability-ru.md#app_d__pack_overview) |
| Рыночный и инженерный контекст (сигналы, дайджесты, публичные кейсы; не для смет) | [Приложение E. Рыночные и технические сигналы (справочно)](./20260325-research-appendix-e-market-technical-signals-ru.md#app_e_root) |
| Навигация по комплекту, реестр документов и источников | [Обзор и ведомость документов](#app_a_pack_overview) (этот документ) |
| Краткое резюме: коммерческое обоснование, методология и экономика | [Резюме для руководства: коммерческое обоснование, методология и экономика внедрения ИИ](./20260331-research-executive-unified-ru.md) |

## Навигация «вопрос → документ» {: #app_a_question_document_navigation }

- Нужен коммерческий C-level обзор: что именно предлагаем (типовые проектные пакеты по этапам PoC → Пилот → Масштабирование → BOT), что остаётся у заказчика после передачи и как использовать матрицу аргументов по ролям ЛПР? → [Резюме для руководства: коммерческое обоснование, методология и экономика внедрения ИИ](./20260331-research-executive-unified-ru.md)
- Как внедрять и разрабатывать в пром контуре (роли, фазы, контрольные точки качества)? → [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_pack_overview)
- Где в корпоративном ИИ формируется преимущество (данные, семантика, агенты)? → [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_pack_overview)
- Где глобальные бенчмарки корпоративного внедрения по публичному отчёту OpenAI (2025) и оговорки по выборке (не норма для КП в РФ)? → [Отчёт. Методология разработки и внедрения ИИ — эмпирика корпоративного внедрения](./20260325-research-report-methodology-main-ru.md#method_openai_implementation_report)
- Стратегия внедрения, организационная зрелость, барьеры, пилот vs scale, обучение руководителей (СКОЛКОВО)? → [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_ai_strategy_org_maturity); поведенческие риски — [Приложение D](./20260325-research-appendix-d-security-observability-ru.md#app_d__org_behavioral_risk_factors); риск бюджета организационная зрелость и пилот — [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_budget_risks_mitigation)
- Какие цифры/диапазоны CapEx/OpEx/TCO заложить клиенту и как обосновать? → [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_pack_overview)
- Где примерные расчёты расхода токенов по данным портала поддержки и допущениям? → параграф _«[Примерные расчёты расхода токенов по данным корпуса заявок (портал поддержки)](./20260325-research-report-sizing-economics-main-ru.md#sizing_token_consumption_estimates)», отчёта _«Сайзинг и экономика»_
- Как устроен комплект отчуждения ИС/кода и что именно передаём клиенту (KT/IP)? → [Приложение B: отчуждение ИС и кода (KT, IP)](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_pack_overview)
- Как оформлять бизнес-процессы для KT (BPMN 2.0, генерация LLM, проверка)? → [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_bpmn_process_formalization_llm) и [Приложение B: комплект отчуждения](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_alienation_package_minimal)
- Что есть в **Comindware** сегодня (состав стека, границы ‘что есть’ vs ‘методология’)? → [Приложение C: имеющиеся наработки Comindware](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_pack_overview)
- Как обеспечить security, комплаенс и промышленную observability (контур контроля, data minimization posture)? → [Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__pack_overview)
- Как проектировать изоляцию и сеть для агентского исполнения (граница доверия, egress, краткоживущие учётные данные)? → [Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__trust_boundary_agent_environment)
- Какие паттерны среды для агента в PR и долгоживущей dev, модель риска по сценарию и минимальный состав платформы задач? → [Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__risk_model_platform_patterns); для KT/IP и PR — [Приложение B: отчуждение ИС и кода (KT, IP)](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_reference_agent_pr_artifacts)
- Как сравнивать E2B / Modal / Daytona и бенчмаркать песочницы (сеть, сессии, метрики прода)? → [Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__managed_sandboxes_benchmarks) и [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_sandbox_evaluation_benchmarks)
- Как за ~30 дней вывести безопасный MVP контура исполнения агента, какие враждебные сценарии и критерии готовности? → [Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__secure_mvp_execution_environment) и [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_agent_execution_mvp)
- Где цифры и барьеры зрелости GenAI в маркетинге крупных брендов РФ (опрос CMO, red_mad_robot × CMO Club, 2025)? → [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_russian_market); [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_genai_marketing_teams); [Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__org_barriers_risk_survey_2025); для концентрации SaaS, каталога моделей и ИС в договоре — [Приложение B: отчуждение ИС и кода (KT, IP)](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_shadow_genai_marketing_model_routing)
- Где ландшафт российского рынка GenAI (онтология сегментов, сценарный контур до 2030, публичные материалы red_mad_robot 2025) и согласование с цифрами IMARC/сегментами? → [Отчёт. Методология](./20260325-research-report-methodology-main-ru.md#method_russian_genai_market_map); [Основной отчёт: сайзинг — ИИ-рынок России](./20260325-research-report-sizing-economics-main-ru.md#sizing_russia_ai_market_stats_forecasts); [AI TRiSM — Приложение D](./20260325-research-appendix-d-security-observability-ru.md#app_d__ai_trism_trust_management)
- Нужен единый реестр источников и расширенные списки по темам? → [Сводный реестр источников (часть III)](#app_a_sources_registry) (этот документ)
- Нужен **сжатый** C-level обзор по методологии, TCO, CapEx/OpEx, валюте, передаче и аргументации по ролям ЛПР? → [Резюме для руководства: коммерческое обоснование, методология и экономика внедрения ИИ](./20260331-research-executive-unified-ru.md)

## Краткие резюме (март 2026) {: #app_a_executive_level_summaries }

| Назначение | Документ |
| --- | --- |
| SCQA по методологии внедрения, TOM, суверенитету и происхождению практики | [«Отчёт. Методология разработки и внедрения ИИ»](./20260325-research-report-methodology-main-ru.md#method_scqa) |
| SCQA по TCO, CapEx/OpEx, сценариям РФ, валюте и границам применимости глобальных бенчмарков | [«Отчёт. Сайзинг и экономика»](./20260325-research-report-sizing-economics-main-ru.md#sizing_scqa) |

## Часть II. Соответствие тем документам комплекта {: #app_a_source_to_pack_mapping }

Ниже — **навигация по темам**: в каком документе комплекта раскрыт тот или иной раздел. **Полный реестр ссылок** — в **части III** («Сводный реестр источников»).

### Темы из отчета «Методология внедрения и отчуждения ИИ» (март 2026) {: #app_a_ai_implementation_methodology_source }

| Тема | Документ |
| --- | --- |
| **Назначение документа и границы применения** | Отчёт. Методология разработки и внедрения ИИ |
| **Резюме для руководства** | Отчёт. Методология разработки и внедрения ИИ |
| **Источник преимущества в корпоративном ИИ (2026): внутренний контекст и рабочий слой данных** | [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_corporate_ai_advantage_source) |
| **Стратегия внедрения ИИ и организационная зрелость** | [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_ai_strategy_org_maturity) |
| **Целевая операционная модель (Target Operating Model)** | Отчёт. Методология разработки и внедрения ИИ |
| **Публичные ориентиры рынка (@Redmadnews, 2026)** | [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_market_benchmarks_2026) |
| **Обзор текущей архитектуры Comindware** | [Приложение C: имеющиеся наработки Comindware](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_pack_overview) |
| **Карта российского рынка GenAI (обзор red_mad_robot, публичные материалы 2025)** | [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_russian_genai_market_map) |
| **Журнал метрик карты GenAI (ноябрь 2025; ограниченное извлечение)** | [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_genai_map_nov2025_metrics_extract) |
| **Эмпирика корпоративного внедрения (отчёт OpenAI, 2025; оговорки по выборке)** | [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_openai_implementation_report) |
| **Управление рисками и соответствие (Compliance)** | [Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__pack_overview) |

### Темы из отчета «Сайзинг, CapEx и OpEx для клиентов» (март 2026) {: #app_a_sizing_capex_opex_source }

| Тема | Документ |
| --- | --- |
| **Аренда GPU (IaaS РФ): дополнительные поставщики и ориентиры** | [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_gpu_rental_iaas_providers) |
| **Профиль on-prem-GPU в проектах Comindware** | [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_onprem_gpu_profile_cmw) |
| **Топология ёмкости GPU и типы источников цифр** | [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_gpu_capacity_topology_bench_classes) |
| **Ориентиры для углублённого аппаратного сайзинга (официальные бенчмарки и документация)** | [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_hardware_deep_research_pointers) |
| **Бенчмарки RTX 4090 (реф. 24 ГБ GeForce; 48 ГБ — коммерческая аренда)** | [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_rtx_4090_benchmarks) |
| **Ориентиры сообщества: Qwen3.5-35B-A3B и потребительское железо (март 2026)** | [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_community_qwen_consumer_hardware) |
| **Требования к VRAM при инференсе LLM** | [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_vram_requirements_llm_inference) |
| **Сегментные ориентиры РФ (GPU-облако, B2B LLM)** | [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_russia_segment_benchmarks) |
| **Практические рекомендации по сайзингу (дерево решений)** | Отчёт. Сайзинг и экономика |
| **Новые тренды 2026 (Дополнительно)** | Отчёт. Сайзинг и экономика |
| **Планирование мощности ИИ-инфраструктуры (2025-2030)** | Отчёт. Сайзинг и экономика |

## Часть III. Сводный реестр источников {: #app_a_sources_registry }

Ниже — **единый перечень** внешних ссылок комплекта, сгруппированный по темам. Каждый URL приведён **один раз**.

#### Инженерия обвязки и мультиагентная разработка {: #app_a_wrapper_engineering_multiagent }

- [Хабр — Инженер будущего строит обвязку для агентов](https://habr.com/ru/articles/1005032/)
- [Martin Fowler — Harness Engineering (Thoughtworks)](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)
- [OpenAI — Harness engineering](https://openai.com/ru-RU/index/harness-engineering/)
- [Anthropic — Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Anthropic — Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)

#### OWASP GenAI Security, тестирование и адаптации на русском {: #app_a_owasp_genai_security_ru }

- [OWASP Gen AI Security Project — Introduction](https://genai.owasp.org/introduction-genai-security-project/)
- [GenAI Security — OWASP Top 10 for Agentic Applications for 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
- [GenAI Security — OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/)
- [GitHub — OWASP Application Security Verification Standard 5.0.0 (PDF, RU)](https://github.com/OWASP/ASVS/blob/master/5.0/OWASP_Application_Security_Verification_Standard_5.0.0_ru.pdf)
- [GitHub — OWASP www-project-ai-testing-guide](https://github.com/OWASP/www-project-ai-testing-guide)
- [Habr — OWASP (вводные по тестированию и материалам сообщества)](https://habr.com/ru/companies/owasp/articles/817241/)
- [Habr — OWASP: LLM TOP 10 2025 (адаптация)](https://habr.com/ru/companies/owasp/articles/893712/)
- [Habr — OWASP (смежные публикации сообщества)](https://habr.com/ru/companies/owasp/articles/896328/)
- [Habr — OWASP (смежные публикации сообщества)](https://habr.com/ru/companies/owasp/articles/900276/)
- [OWASP — проект Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OWASP — Web Security Testing Guide (WSTG), stable](https://owasp.org/www-project-web-security-testing-guide/stable/)

#### Безопасность GenAI, OWASP и сигналы рынка (TCO / риски) {: #app_a_genai_security_owasp_tco }

- [CodeWall — разбор red team: McKinsey AI platform](https://codewall.ai/blog/how-we-hacked-mckinseys-ai-platform)
- [GitHub — NVIDIA Garak (сканер для LLM, только изолированные стенды)](https://github.com/NVIDIA/garak)
- [OpenAI — приобретение PromptFoo (контекст рынка тестирования)](https://openai.com/index/openai-to-acquire-promptfoo/)
- [Kaspersky — пресс-релиз: угрозы под видом популярных ИИ-сервисов (бенчмарк тренда)](https://www.kaspersky.com/about/press-releases/kaspersky-chatgpt-mimicking-cyberthreats-surge-115-in-early-2025-smbs-increasingly-targeted)
- [The Hacker News — TeamPCP: LiteLLM и Telnyx скомпрометированы через PyPI (март 2026)](https://thehackernews.com/2026/03/teampcp-pushes-malicious-telnyx.html)
- [Datadog Security Labs — LiteLLM and Telnyx compromised on PyPI: Tracing the TeamPCP supply chain campaign](https://securitylabs.datadoghq.com/articles/litellm-compromised-pypi-teampcp-supply-chain-campaign/)
- [Коммерсантъ — рынок и атаки на ИИ-системы (журналистский контекст)](https://www.kommersant.ru/doc/8363105)

#### Угрозы GenAI и иллюстративные материалы третьих лиц (не реклама) {: #app_a_genai_threats_third_party_materials }

- [Securelist — webinar: AI agents vs. prompt injections](https://securelist.com/webinars/ai-agents-vs-prompt-injections/)
- [Kaspersky — press release: training Large Language Models Security (описание программы)](https://www.kaspersky.com/about/press-releases/kaspersky-introduces-a-new-training-large-language-models-security)
- [Kaspersky Blog — How LLMs can be compromised in 2025](https://www.kaspersky.com/blog/new-llm-attack-vectors-2025/54323/)
- [Kaspersky Blog — Agentic AI security measures and OWASP ASI Top 10](https://www.kaspersky.com/blog/top-agentic-ai-risks-2026/29988/)
- [Kaspersky Resource Center — What Is Prompt Injection?](https://www.kaspersky.com/resource-center/threats/prompt-injection)

#### Нормативные и стратегические материалы {: #app_a_regulatory_strategic_materials }

- [Официальное опубликование — Приказ Роскомнадзора от 19.06.2025 № 140 (обезличивание ПДн)](http://publication.pravo.gov.ru/document/0001202508010002)
- [ACSOUR — обязанность операторов передавать анонимизированные ПДн в ГИС (152-ФЗ)](https://acsour.com/en/news-and-articles/tpost/2g13ahnab1-mandatory-anonymized-personal-data-shari)
- [NIST AIRC — Roadmap for the AI Risk Management Framework](https://airc.nist.gov/airmf-resources/roadmap)
- [NIST — AI RMF to ISO/IEC 42001 Crosswalk (PDF)](https://airc.nist.gov/docs/NIST_AI_RMF_to_ISO_IEC_42001_Crosswalk.pdf)
- [Известия (EN) — создание офисов внедрения ИИ](https://en.iz.ru/en/node/1985740)
- [DataGuidance — поправки к национальной стратегии развития ИИ РФ](https://www.dataguidance.com/news/russia-president-issues-amendments-national-ai)
- [Фонтанка — проект закона о госрегулировании ИИ (Минцифры, 18.03.2026)](https://www.fontanka.ru/2026/03/18/76318717/)
- [ISO/IEC 42001:2023 — Artificial intelligence management system](https://www.iso.org/standard/81230.html)
- [NIST — AI RMF: Generative AI Profile (NIST.AI.600-1, 2024)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)

#### Данные и стратегические сигналы {: #app_a_data_strategic_signals }

- [Gartner — пресс-релиз: нехватка AI-ready data подрывает ИИ-проекты (26.02.2025)](https://www.gartner.com/en/newsroom/press-releases/2025-02-26-lack-of-ai-ready-data-puts-ai-projects-at-risk)

#### Российский рынок GenAI, сегменты и AI TRiSM (публичные ссылки) {: #app_a_russian_genai_market_segments_trism }

- [Хабр — red_mad_robot: анонс тренд-репорта и события в Сколково](https://habr.com/ru/companies/redmadrobot/articles/879750/)
- [red_mad_robot — раздел «Исследования»](https://redmadrobot.ru/issledovaniya-1/)
- [red_mad_robot — мероприятие: тренд-репорт рынка GenAI (2025)](https://redmadrobot.ru/meropriyatiya/trend-report-rynok-gen-ai-v-2025-godu/)
- [Gartner — AI TRiSM (глоссарий)](https://www.gartner.com/en/information-technology/glossary/ai-trism)
- [McKinsey — The state of AI](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai)
- [РБК — объём рынка B2B LLM в России (MTS AI)](https://www.rbc.ru/technology_and_media/26/11/2024/67449d909a79478a2052d490)
- [Сколково — событие «Состояние рынка GenAI в России и в мире» (12.02.2025)](https://www.skolkovo.ru/events/120225-sostoyanie-rynka-genai-v-rossii-i-v-mire/)
- [Ведомости — рынок облачных сервисов с GPU (МНИАП)](https://www.vedomosti.ru/technology/articles/2024/12/11/1080600-rinok-oblachnih-servisov-s-gpu-virastet)

#### Подкасты (первичная запись): AI-First, red_mad_robot {: #app_a_podcasts_ai_first_red_mad_robot }

- [YouTube — «Ноосфера» #129: Илья Самофеев (red_mad_robot), AI-First / AI-Native](https://www.youtube.com/watch?v=jTKhg1jqF_M)

#### Стек инференса (MOSEC, vLLM) и открытая документация {: #app_a_inference_stack_mosec_vllm_docs }

- [vLLM — OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)
- [сервер инференса MOSEC — README проекта (пример публичного зеркала)](https://github.com/arterm-sedov/cmw-mosec)
- [mosecorg/mosec (GitHub)](https://github.com/mosecorg/mosec)
- [MOSEC — документация](https://mosecorg.github.io/mosec/index.html)

#### Экономика, рынок, enterprise AI {: #app_a_economics_market_enterprise_ai }

- [a16z — Top 100 Gen AI Apps (6)](https://a16z.com/100-gen-ai-apps-6/)
- [OpenAI — The state of enterprise AI 2025 (PDF)](https://cdn.openai.com/pdf/7ef17d82-96bf-4dd1-9df2-228f7f377a29/the-state-of-enterprise-ai_2025-report.pdf)
- [Dataoorts — GPU cloud providers in Russia](https://dataoorts.com/top-5-plus-gpu-cloud-providers-in-russia/)
- [Хабр — Релиз Claude Opus 4.6](https://habr.com/ru/news/993322/)
- [ITNext — GPU infrastructure as foundational to enterprise AI strategy](https://itnext.io/why-gpu-infrastructure-is-foundational-to-an-enterprise-ai-strategy-5b574ef1eebc)
- [Larridin — State of Enterprise AI in 2025 (независимый обзор)](https://larridin.com/blog/state-of-enterprise-ai-in-2025)
- [OpenAI — The state of enterprise AI (обзор, декабрь 2025)](https://openai.com/index/the-state-of-enterprise-ai-2025-report)
- [Claude Docs — Models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Claude Docs — What's new in Claude 4.6](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-6)
- [Anthropic — Introducing Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)
- [Anthropic — Introducing Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)
- [FinOps Foundation — Framework: Unit Economics (Capability)](https://www.finops.org/framework/capabilities/unit-economics/)
- [FinOps Foundation — Generative AI / Unit Economics](https://www.finops.org/wg/generative-ai/)
- [IMARC — Russia Artificial Intelligence Market](https://www.imarcgroup.com/russia-artificial-intelligence-market)
- [MarketsandMarkets — Russia AI Inference PaaS Market](https://www.marketsandmarkets.com/ResearchInsight/russia-ai-inference-platform-as-a-service-paas-market.asp)
- [Microsoft Research — Fara-7B: An Efficient Agentic Model for Computer Use (PDF)](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/11/Fara-7B-An-Efficient-Agentic-Model-for-Computer-Use.pdf)
- [РБК Education — во сколько обойдётся ИИ-агент: подсчёты экспертов (2026)](https://www.rbc.ru/education/05/02/2026/697162269a794772c9cf9991)
- [Yakov & Partners — публикация AI 2025](https://yakovpartners.com/publications/ai-2025/)

#### Оценка качества и мониторинг (LangSmith) {: #app_a_quality_monitoring_langsmith }

- [LangChain Docs — Evaluation concepts (LangSmith)](https://docs.langchain.com/langsmith/evaluation-concepts)
- [LangSmith — Online evaluations (how-to)](https://docs.smith.langchain.com/observability/how_to_guides/online_evaluations)

#### Исследования (edge–cloud routing, агентная память и обучение; ориентиры НИОКР) {: #app_a_research_edge_cloud_routing_memory }

- [arXiv — PRISM: Privacy-Aware Routing for Cloud-Edge LLM Inference](https://arxiv.org/html/2511.22788v1)
- [arXiv — HybridFlow: Resource-Adaptive Subtask Routing for Edge-Cloud LLM Inference](https://arxiv.org/html/2512.22137v4)
- [arXiv — Moonshot AI: ускорение синхронного RL](https://arxiv.org/pdf/2511.14617)
- [arXiv — Agent0: co-evolving curriculum and executor agents](https://arxiv.org/pdf/2511.16043)
- [arXiv — MoE на стеке AMD (IBM, Zyphra и др.)](https://arxiv.org/pdf/2511.17127)
- [arXiv — General Agentic Memory (GAM)](https://arxiv.org/pdf/2511.18423)

#### Edge-инференс и оптимизации памяти (Apple Silicon, локальные модели) {: #app_a_edge_inference_memory_optimizations }

- [Apple ML Research — LLM in a Flash: Efficient Large Language Model Inference with Limited Memory (ACL 2024)](https://machinelearning.apple.com/research/efficient-large-language)
- [GitHub — matt-k-wong/mlx-flash (реализация для MLX, март 2026)](https://github.com/matt-k-wong/mlx-flash)
- [arXiv — LLM in a Flash: Efficient Large Language Model Inference with Limited Memory (оригинальная статья, 2312.11514)](https://arxiv.org/html/2312.11514v3)
- [Apple Developer — WWDC 2025: Explore large language models on Apple silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/)

#### Агентная память и модели (ориентиры НИОКР и прайсинга, не строка КП) {: #app_a_agent_memory_benchmarks }

- [Anthropic — Pricing](https://www.anthropic.com/pricing)

#### Облачные провайдеры и тарифы (РФ) {: #app_a_cloud_providers_tariffs_russia }

- [1dedic — GPU-серверы](https://1dedic.ru/gpu-servers)
- [Google — условия использования Gemma](https://ai.google.dev/gemma/terms)
- [Yandex AI Studio — доступные генеративные модели](https://aistudio.yandex.ru/docs/ru/ai-studio/concepts/generation/models.html)
- [Yandex AI Studio — правила тарификации](https://aistudio.yandex.ru/docs/ru/ai-studio/pricing.html)
- [CIO — MTS AI перенесла обучение моделей в облако](https://cio.osp.ru/articles/140525-MTS-AI-perenesla-obuchenie-modeley-v-oblako)
- [МТС Cloud — виртуальная инфраструктура с GPU](https://cloud.mts.ru/services/virtual-infrastructure-gpu/)
- [Cloud.ru — Evolution Foundation Models, тарифы (2026)](https://cloud.ru/documents/tariffs/evolution/foundation-models)
- [Cloud.ru — Evolution Foundation Models (продукт, перечень моделей)](https://cloud.ru/products/evolution-foundation-models)
- [VK Cloud — машинное обучение в облаке (документация)](https://cloud.vk.com/docs/ru/ml)
- [Сбер — GigaChat API: юридические тарифы](https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs)
- [Сбер — портал GigaChat API](https://developers.sber.ru/portal/products/gigachat-api)
- [HOSTKEY — выделенные серверы с GPU](https://hostkey.ru/gpu-dedicated-servers/)
- [Hugging Face — MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)
- [Hugging Face — организация Qwen](https://huggingface.co/Qwen)
- [Hugging Face — deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
- [Hugging Face — deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- [Hugging Face — организация moonshotai](https://huggingface.co/moonshotai)
- [Hugging Face — moonshotai/Kimi-K2-Base](https://huggingface.co/moonshotai/Kimi-K2-Base)
- [Hugging Face — nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)
- [Hugging Face — openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
- [Hugging Face — openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- [Hugging Face — zai-org/GLM-4.6](https://huggingface.co/zai-org/GLM-4.6)
- [Hugging Face — zai-org/GLM-4.7](https://huggingface.co/zai-org/GLM-4.7)
- [Hugging Face — zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
- [Hugging Face — zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5)
- [Immers Cloud — GPU](https://immers.cloud/gpu/)
- [Intelion Cloud](https://intelion.cloud/)
- [MWS — тарифы MWS GPT](https://mws.ru/docs/docum/cloud_terms_mwsgpt_pricing.html)
- [MWS — MWS GPT (продукт)](https://mws.ru/mws-gpt/)
- [MWS — GPU On‑premises](https://mws.ru/services/mws-gpu-on-prem/)
- [NVIDIA — Nemotron 3 (обзор семейства)](https://research.nvidia.com/labs/nemotron/Nemotron-3/)
- [Selectel — Foundation Models Catalog](https://selectel.ru/services/cloud/foundation-models-catalog)
- [Selectel — облако GPU (калькулятор)](https://selectel.ru/services/gpu/)
- [AKM.ru — доступ к крупнейшей языковой модели на рынке РФ (Yandex B2B)](https://www.akm.ru/eng/press/yandex-b2b-tech-has-opened-access-to-the-largest-language-model-on-the-russian-market/)
- [Cloud4Y — облачный GPU-хостинг](https://www.cloud4y.ru/cloud-hosting/gpu/)
- [CNews — кейс: MTS AI и экономия инвестиций за счёт облака MWS (обзор)](https://www.cnews.ru/reviews/provajdery_gpu_cloud_2025/cases/kak_mts_ai_sekonomila_bolee_milliarda)
- [NVIDIA — GeForce Software License](https://www.nvidia.com/en-us/drivers/geforce-license/)

#### Российские облачные GPU: актуальные тарифы и аналитика (2026) {: #app_a_russian_cloud_gpu_pricing_2026 }

- [Cloud.ru — Тарифы «Evolution Compute GPU», Приложение №7G.EVO.1 (январь 2026)](https://cloud.ru/documents/tariffs/evolution/evolution-compute-gpu)
- [Elish Tech — Почасовая аренда GPU A100 vs H100: что выгоднее в 2026 году](https://www.elishtech.com/arenda-gpu-a100-vs-h100-2026/)
- [Elish Tech — Где арендовать GPU-серверы дешевле и выгоднее: сравнение рынка в России и за рубежом](https://www.elishtech.com/gpu-server-rent-market-comparison/)
- [Yandex Cloud — GPU (графические ускорители), документация](https://yandex.cloud/ru/docs/compute/concepts/gpus)
- [Yandex Cloud — Прайс-лист (текущие тарифы)](https://yandex.cloud/ru/prices)
- [Selectel — Cloud GPU (облачные серверы с GPU)](https://selectel.ru/services/cloud/servers/gpu)
- [Selectel — Новости: новые конфигурации GPU-серверов от 50 руб./час](https://myseldon.com/ru/news/index/262426281)

#### Публичные веса с нестандартной лицензией {: #app_a_public_weights_licensing }

- [arXiv — Cache Me If You Must (KV-quantization), 2501.19392](https://arxiv.org/abs/2501.19392)
- [Hugging Face — карточка модели YandexGPT-5-Lite-8B-instruct](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct)
- [Hugging Face — LICENSE (YandexGPT-5-Lite-8B), сырой текст соглашения](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct/raw/main/LICENSE)
- [Yandex Research — принятые к ICML 2025 (список, в т.ч. KV-cache)](https://research.yandex.com/blog/papers-accepted-to-icml-2025)
- [Yandex Research — обзор направлений работ (2025)](https://research.yandex.com/blog/yandex-research-in-2025)

#### Открытые модели ai-sage (GigaChat и спутники) {: #app_a_open_models_ai_sage_gigachat }

- [GitHub — sgl-project/sglang, PR #18802](https://github.com/sgl-project/sglang/pull/18802)
- [GitVerse — GigaTeam/gigachat3.1](https://gitverse.ru/GigaTeam/gigachat3.1)
- [Хабр — GigaChat-3.1: большое обновление больших моделей (блог Сбера)](https://habr.com/ru/companies/sberbank/articles/1014146/)
- [Hugging Face — организация ai-sage](https://huggingface.co/ai-sage)
- [Hugging Face — ai-sage/GigaChat3.1-10B-A1.8B (Lightning 3.0)](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B)
- [Hugging Face — ai-sage/GigaChat3.1-10B-A1.8B (Lightning 3.1)](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B)
- [Hugging Face — ai-sage/GigaChat3.1-702B-A36B (Ultra)](https://huggingface.co/ai-sage/GigaChat3.1-702B-A36B)
- [Hugging Face — коллекция GigaAM](https://huggingface.co/collections/ai-sage/gigaam)
- [Hugging Face — коллекция GigaChat 3.1](https://huggingface.co/collections/ai-sage/gigachat-31)
- [Hugging Face — коллекция GigaChat Lite](https://huggingface.co/collections/ai-sage/gigachat-lite)
- [Hugging Face — коллекция GigaEmbeddings](https://huggingface.co/collections/ai-sage/gigaembeddings)

#### Примерные расчёты токенов (портал поддержки, агрегаторы и обзоры прайсов) {: #app_a_token_estimates_pricing }

- [Хабр — обзор цен на токены](https://habr.com/ru/articles/1000058/)
- [Хабр — гид по топ-20 нейросетям для текстов (в т.ч. цены)](https://habr.com/ru/articles/948672/)
- [LLMoney — калькулятор цен токенов LLM](https://llmoney.ru)
- [Портал поддержки Comindware](https://support.comindware.com/)
- [VC.ru — гайд по тарифам Claude и доступу из России](https://vc.ru/ai/2757771-tarify-claude-2026-gayd-po-planam-i-dostupu-iz-rossii)

#### Формат TOON и оптимизация токенов {: #app_a_toon_format_token_optimization }

- [Спецификация TOON](https://toonformat.dev/)
- [Tensorlake: бенчмарки](https://www.tensorlake.ai/blog/toon-vs-json)
- [Systenics: экономия токенов](https://systenics.ai/blog/2026-01-24-toon-vs-json-how-token-oriented-object-notation-reduces-llm-token-costs)

#### Иллюстративные ориентиры нагрузки (публичные интервью, финсектор) {: #app_a_load_benchmarks_financial }

- [CIO — интервью: чат-бот, масштаб обращений и сценарии](https://cio.osp.ru/articles/5455)
- [«Открытые системы» — RAG и LLM для поддержки операционистов](https://www.osp.ru/articles/2025/0324/13059305)

#### Инструменты разработки с ИИ (ориентиры) {: #app_a_ai_dev_tools_benchmarks }

- [OpenWork (different-ai/openwork)](https://github.com/different-ai/openwork)
- [OpenCode](https://opencode.ai/)
- [OpenCode — документация (Intro)](https://opencode.ai/docs)
- [OpenCode — Ecosystem](https://opencode.ai/docs/ecosystem/)
- [OpenCode Zen](https://opencode.ai/docs/zen)
- [OpenRouter](https://openrouter.ai/)
- [OpenRouter — Logging и политики провайдеров](https://openrouter.ai/docs/guides/privacy/logging)

#### Инференс и VRAM: бенчмарки, движок и калькуляторы {: #app_a_inference_vram_tools_sizing_nav }

- [apxml.com — VRAM calculator](https://apxml.com/tools/vram-calculator)
- [vLLM — документация](https://docs.vllm.ai/)
- [MLCommons — Inference Datacenter](https://mlcommons.org/benchmarks/inference-datacenter/)

#### Финансовая и инфраструктурная база (FinOps/TCO/железо) {: #app_a_finops_tco_infrastructure }

- [Medium — Qwen 3.5 35B A3B (AgentNativeDev)](https://agentnativedev.medium.com/qwen-3-5-35b-a3b-why-your-800-gpu-just-became-a-frontier-class-ai-workstation-63cc4d4ebac1)
- [Hugging Face — Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Introl — планирование мощностей ИИ-инфраструктуры (прогнозы, McKinsey в обзоре)](https://introl.com/blog/ai-infrastructure-capacity-planning-forecasting-gpu-2025-2030)
- [Introl — финансирование CapEx/OpEx и инвестиции в GPU](https://introl.com/blog/ai-infrastructure-financing-capex-opex-gpu-investment-guide-2025)
- [PitchGrade — AI Infrastructure Primer](https://pitchgrade.com/research/ai-infrastructure-primer)
- [OpenAI — Prompt caching (снижение стоимости повторяющегося контекста)](https://platform.openai.com/docs/guides/prompt-caching)
- [Slyd — калькулятор TCO (on-prem и облако)](https://slyd.com/resources/tco-calculator)
- [Runpod — LLM inference optimization playbook (throughput)](https://www.runpod.io/articles/guides/llm-inference-optimization-playbook)
- [SWFTE — экономика частного AI / on-prem](https://www.swfte.com/blog/private-ai-enterprises-onprem-economics)

#### Исследования рынка (зрелость GenAI, не технический сайзинг) {: #app_a_market_research_genai_maturity }

- [RB.RU — 93% команд в маркетинге используют ИИ (обзор исследования CMO Club × red_mad_robot, 2025)](https://rb.ru/news/93-komand-v-marketinge-ispolzuyut-ii-nejroseti-pomogayut-s-kontentom-no-im-pochti-ne-doveryayut-strategiyu-i-byudzhety/)
- [Telegram — CMO Club Russia: анонс исследования GenAI в маркетинге (red_mad_robot × CMO Club, 2025)](https://t.me/cmoclub/197)

#### Наблюдаемость и телеметрия {: #app_a_observability_telemetry }

- [OpenInference — инструментирование ИИ для OpenTelemetry](https://arize-ai.github.io/openinference/)
- [Arize Phoenix — документация](https://docs.arize.com/phoenix)
- [LangSmith — документация](https://docs.smith.langchain.com/)
- [Langfuse — документация observability / tracing](https://langfuse.com/docs/observability/get-started)
- [OpenTelemetry — OpenTelemetry for Generative AI (блог)](https://opentelemetry.io/blog/2024/otel-generative-ai)
- [OpenTelemetry — Semantic conventions for generative AI metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)
- [OpenTelemetry — Semantic conventions for generative client AI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)

#### Публичные материалы Ozon Tech {: #app_a_ozon_tech_materials }

- [GitHub — организация ozontech (открытые репозитории)](https://github.com/ozontech)
- [Хабр — Ozon Tech: анонс ML&DS Meetup (MLOps, программа докладов)](https://habr.com/ru/companies/ozontech/articles/768734/)
- [Хабр — Ozon Tech: пересборка конструктора чат-ботов (Bots Factory, no-code, масштаб)](https://habr.com/ru/companies/ozontech/articles/834812/)
- [Хабр — Ozon Tech: Query Prediction, ANN и обратный индекс](https://habr.com/ru/companies/ozontech/articles/990180/)

#### Методологии внедрения и отраслевые практики {: #app_a_implementation_methodologies_industry_practices }

- [GitHub — redmadrobot-rnd/mcp-registry](https://github.com/redmadrobot-rnd/mcp-registry)
- [Habr — red_mad_robot: кейс RAG для ФСК](https://habr.com/ru/companies/redmadrobot/articles/892882/)
- [Habr — red_mad_robot: MCP Tool Registry и автоматизация RAG](https://habr.com/ru/companies/redmadrobot/articles/982004/)
- [InOrg — бесшовная передача (seamless handover) в модели BOT](https://inorg.com/blog/from-build-to-transfer-key-success-factors-a-seamless-bot-model-transition)
- [Just AI — корпоративный GenAI (упоминается как практикующий вендор)](https://just-ai.com/ru/)
- [Luxoft — модель Build–Operate–Transfer (BOT)](https://www.luxoft.com/blog/master-the-build-operate-transfer-bot-model-with-luxoft)
- [Ведомости — CTO AI red_mad_robot (Влад Шевченко)](https://www.vedomosti.ru/technologies/trendsrub/articles/2026/03/11/1181757-ii-uskoril-kod)

#### Публичные материалы MWS / MTS AI {: #app_a_public_materials_mws_mts_ai }

- [Альянс в сфере искусственного интеллекта](https://a-ai.ru/)
- [МТС Cloud — IaaS 152-ФЗ УЗ-1](https://cloud.mts.ru/services/iaas-152-fz/)
- [Хабр — MTS AI: взаимная оценка LLM при улучшении Cotype](https://habr.com/ru/companies/mts_ai/articles/892176/)
- [Хабр — MTS AI: граф в RAG](https://habr.com/ru/companies/mts_ai/articles/915276/)
- [Хабр — МТС: MWS Octapi и AI-агенты](https://habr.com/ru/companies/ru_mts/articles/932382/)
- [Хабр — МТС: архитектура LLM-платформы MWS GPT](https://habr.com/ru/companies/ru_mts/articles/967478/)
- [Хабр — МТС: RAG-помощник для саппорта (смежная публикация)](https://habr.com/ru/companies/ru_mts/articles/970392/)
- [Хабр — МТС: RAG для поддержки (Confluence, Jira, гибридный поиск)](https://habr.com/ru/companies/ru_mts/articles/970476/)
- [MERA — бенчмарк русскоязычных моделей](https://mera.a-ai.ru/)
- [MWS AI — MWS AI Agents Platform (описание модулей)](https://mts.ai/product/ai-agents-platform/)
- [MWS — MWS Octapi (продукт)](https://mws.ru/dev-tools/octapi/)
- [MWS Docs — условия облачного сегмента 152-ФЗ](https://mws.ru/docs/docum/cloud_terms_152fz.html)
- [MWS Docs — лицензионные условия ПО «MWS GPT»](https://mws.ru/docs/docum/lic_terms_mwsgpt.html)
- [MWS — новость: Octapi и создание ИИ-агентов](https://mws.ru/news/mts-web-services-na-30-uskorila-sozdanie-ii-agentov-pri-pomoshhi-platformy-mws-octapi/)
- [MWS — новость: хранение персональных данных в облаке](https://mws.ru/news/mts-web-services-zapustila-servis-dlya-hraneniya-personalnyh-dannyh-v-oblake/)

#### Публичные материалы финсектора (паттерны внедрения) {: #app_a_financial_sector_public_materials_patterns }

- [Хабр — MLOps и каскады моделей](https://habr.com/ru/companies/alfa/articles/801893/)
- [Хабр — автоматизация обучения и обновления моделей](https://habr.com/ru/companies/alfa/articles/852790/)
- [Хабр — классификация текстов диалогов на большом числе классов](https://habr.com/ru/companies/alfa/articles/900538/)
- [Хабр — обновление LLM: instruction following и tool calling](https://habr.com/ru/companies/tbank/articles/979650/)

#### Telegram-каналы и посты {: #app_a_telegram_channels }

- [AGORA — Industrial AI](https://t.me/AGORA)
- [@Redmadnews (red_mad_robot)](https://t.me/Redmadnews)
- [Redmadnews — MCP Tool Registry / RAG](https://t.me/Redmadnews/5132)
- [Redmadnews — СП с «ВымпелКом», фабрика агентов](https://t.me/Redmadnews/5145)
- [Redmadnews — R&D в AI в 2026](https://t.me/Redmadnews/5146)
- [Redmadnews — AI + Economy, Китай](https://t.me/Redmadnews/5159)
- [Redmadnews — бизнес-завтрак КРОК](https://t.me/Redmadnews/5167)
- [Redmadnews — AI-first подкаст](https://t.me/Redmadnews/5170)
- [ai_archnadzor — RAG и архитектуры](https://t.me/ai_archnadzor)
- [ai_archnadzor — локальные модели для кодинга и снижения затрат](https://t.me/ai_archnadzor/167)
- [ai_archnadzor — CLI вместо MCP](https://t.me/ai_archnadzor/190)
- [Канал @ai_machinelearning_big_data](https://t.me/ai_machinelearning_big_data)
- [CMO Club Russia](https://t.me/cmoclub)
- [@llm_under_hood](https://t.me/llm_under_hood)
- [NeuralDeep](https://t.me/neuraldeep)
- [NeuralDeep — экономика LLM-решений](https://t.me/neuraldeep/1366)
- [NeuralDeep — бенчмарки vLLM / RTX 4090](https://t.me/neuraldeep/1476)
- [NeuralDeep — рекомендации по кластерам](https://t.me/neuraldeep/1627)
- [@rmr_rnd — R&D red_mad_robot](https://t.me/rmr_rnd)
- [«ITипичные аспекты Артёма» (Артём Лысенко)](https://t.me/virrius_tech_chat)

#### Посты NeuralDeep {: #app_a_neuraldeep_posts }

- [Agentic RAG / SGR](https://t.me/neuraldeep/1605)
- [ETL, эмбеддинги, реранкеры, фреймворки RAG, eval, безопасность](https://t.me/neuraldeep/1758)

#### Посты @ai_archnadzor {: #app_a_ai_archnadzor_posts }

- [GraphOS для RAG](https://t.me/ai_archnadzor/151)
- [Semantic Gravity Framework](https://t.me/ai_archnadzor/155)
- [Nested Learning](https://t.me/ai_archnadzor/157)
- [LEANN](https://t.me/ai_archnadzor/161)
- [OpenClaw (ex-Moltbot)](https://t.me/ai_archnadzor/165)
- [Perplexica](https://t.me/ai_archnadzor/166)
- [Guardrails как архитектурный паттерн](https://t.me/ai_archnadzor/168)
- [EffGen / agentic SLM](https://t.me/ai_archnadzor/171)
- [Типы AI-агентов](https://t.me/ai_archnadzor/173)
- [GenAI в продакшне: технологический манифест](https://t.me/ai_archnadzor/175)
- [Локальный стек обсервабильности](https://t.me/ai_archnadzor/177)
- [REFRAG](https://t.me/ai_archnadzor/178)
- [Cog-RAG](https://t.me/ai_archnadzor/179)
- [HippoRAG 2](https://t.me/ai_archnadzor/180)
- [Topo-RAG](https://t.me/ai_archnadzor/182)
- [Disco-RAG](https://t.me/ai_archnadzor/183)
- [DSPy 3 и GEPA](https://t.me/ai_archnadzor/184)
- [OCR: NEMOTRON-PARSE, Chandra, DOTS.OCR](https://t.me/ai_archnadzor/185)
- [BitNet](https://t.me/ai_archnadzor/189)
- [Doc-to-LoRA; память агентов (пост /191)](https://t.me/ai_archnadzor/191)
- [Multimodal LLM](https://t.me/ai_archnadzor/192)

#### Habr и статьи по инженерии RAG {: #app_a_habr_rag_engineering_articles }

- [Raft на Habr — чанкование](https://habr.com/ru/companies/raft/articles/954158/)

#### Препринты (arXiv) {: #app_a_arxiv_preprints }

- [Google — Deep-Thinking Ratio (DTR), 2602.13517](https://arxiv.org/pdf/2602.13517)
- [Oppo AI — Search More, Think Less (SMTL), 2602.22675](https://arxiv.org/pdf/2602.22675)
- [Meta (Экстремистская организация, запрещена в РФ), OpenAI, xAI — непрерывное улучшение моделей (чаты), 2603.01973](https://arxiv.org/pdf/2603.01973)
- [Microsoft Research — безопасность агентов с внешними инструментами, 2603.03205](https://arxiv.org/pdf/2603.03205)
- [Accenture — Memex(RL), 2603.04257](https://arxiv.org/pdf/2603.04257)
- [SkillNet, 2603.04448](https://arxiv.org/pdf/2603.04448)
- [Databricks — KARL, 2603.05218](https://arxiv.org/pdf/2603.05218)
- [OpenAI — контроль рассуждения со скрытыми шагами, 2603.05706](https://arxiv.org/pdf/2603.05706)
- [Princeton — непрерывное обучение из взаимодействия с агентом, 2603.10165](https://arxiv.org/pdf/2603.10165)

#### Продукты и блоги (эмбеддинги, M365; справочно) {: #app_a_products_blogs_embeddings_m365_reference }

- [Google — Gemini Embedding 2 (блог)](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
- [Microsoft — Copilot Cowork (блог Microsoft 365)](https://www.microsoft.com/en-us/microsoft-365/blog/2026/03/09/copilot-cowork-a-new-way-of-getting-work-done/)

#### Открытые проекты {: #app_a_open_third_party_projects }

- [RAGAS — документация](https://docs.ragas.io/en/stable/)
- [EvilFreelancer/openapi-to-cli](https://github.com/EvilFreelancer/openapi-to-cli)
- [Marker-Inc-Korea/AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG)
- [NVIDIA-NeMo/Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)
- [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)
- [chonkie-inc/chonkie](https://github.com/chonkie-inc/chonkie)
- [datalab-to/marker](https://github.com/datalab-to/marker)
- [docling-project/docling](https://github.com/docling-project/docling)
- [langchain-ai/langchain — text-splitters](https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters)
- [langgenius/dify](https://github.com/langgenius/dify/)
- [mastra-ai/mastra](https://github.com/mastra-ai/mastra)
- [microsoft/markitdown](https://github.com/microsoft/markitdown)
- [GitHub — ozontech/file.d](https://github.com/ozontech/file.d)
- [GitHub — ozontech/framer](https://github.com/ozontech/framer)
- [GitHub — ozontech/seq-db](https://github.com/ozontech/seq-db)
- [protectai/rebuff](https://github.com/protectai/rebuff)
- [run-llama/llama_index](https://github.com/run-llama/llama_index)
- [stanford-futuredata/ARES](https://github.com/stanford-futuredata/ARES)
- [vamplabAI/sgr-agent-core](https://github.com/vamplabAI/sgr-agent-core) (ветка tool-confluence)
- [vamplabAI/sgr-agent-core — ветка tool-confluence](https://github.com/vamplabAI/sgr-agent-core/tree/tool-confluence)
- [Lakera — платформа](https://platform.lakera.ai/)
- [Neuraldeep.ru - база навыков для российских сервисов](https://neuraldeep.ru/)

#### Регулирование (проектный контур 2026) {: #app_a_regulation_project_context_2026 }

- [Портал НПА — проект федерального закона (ID 166424)](https://regulation.gov.ru/projects#npa=166424)

## Часть IV. Дополнительные источники (backlog из ТЗ) {: #app_a_additional_sources_backlog }

### 16.1 Международные стандарты и регулирование (Приоритет 1) {: #app_a_international_standards_regulation }

- [ISO/IEC 42001:2023 - PDF Sample and Core Requirements](https://cdn.standards.iteh.ai/samples/81230/4c1911ebc9a641fcb6ee21aa09c28ad3/ISO-IEC-42001-2023.pdf)
- [NIST AI Risk Management Framework 1.0 (Full Portal)](https://nist.gov/itl/ai-risk-management-framework)
- [NIST AI 600-1 Direct PDF Download](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=958388)
- [NIST AI RMF Implementation Guide 2026 (GLACIS)](https://www.glacis.io/guide-nist-ai-rmf)
- [NIST Roadmap for AI RMF 1.0 (Updated 2025)](https://www.nist.gov/itl/ai-risk-management-framework/roadmap-nist-artificial-intelligence-risk-management-framework-ai)
- [EU AI Act: Official Obligations for GPAI Providers](https://digital-strategy.ec.europa.eu/en/factpages/general-purpose-ai-obligations-under-ai-act)
- [EU AI Act: Article 16 (High-Risk AI Systems)](https://artificialintelligenceact.eu/article/16)
- [EU AI Act: Article 53 (GPAI Model Obligations)](https://artificialintelligenceact.eu/article/53)
- [EU Commission: GPAI Code of Practice (Final Draft July 2025)](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai)
- [EU AI Act Compliance Guide 2026 (Unorma)](https://unorma.com/eu-ai-act-compliance-guide-2026-edition/)
- [OECD AI Principles and Governance Framework](https://oecd.ai/en/ai-principles)
- [OECD Catalogue of Tools for Trustworthy AI](https://oecd.ai/en/catalogue/tools)
- [IEEE P7000 Series: Process Model for Ethical AI Design](https://standards.ieee.org/project/7000.html)
- [UNESCO Recommendation on the Ethics of AI (Global Implementation)](https://www.unesco.org/en/artificial-intelligence/recommendation-ethics)
- [G7 Hiroshima AI Process: International Guiding Principles](https://www.moff.go.jp/files/100573473.pdf)
- [UK AI Safety Institute: Systemic Safety Framework (2025)](https://www.gov.uk/government/organisations/ai-safety-institute)

### 16.2 Управленческие методологии внедрения (Big Three & Big Four) {: #app_a_executive_implementation_methodologies }

- [McKinsey: Rewiring the Enterprise for GenAI (2025)](https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/rewiring-for-the-era-of-gen-ai)
- [McKinsey: The GenAI Operating Model Leader's Guide (2025)](https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/a-data-leaders-operating-guide-to-scaling-gen-ai)
- [McKinsey: Seizing the Agentic AI Advantage (June 2025 Report)](https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/seizing%20the%20agentic%20ai%20advantage/seizing-the-agentic-ai-advantage-june-2025.pdf)
- [McKinsey: The State of AI in 2025 - Value Capture](https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/the%20state%20of%20ai/2025/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf)
- [BCG: From Potential to Profit with GenAI (2025 Framework)](https://www.bcg.com/publications/2024/from-potential-to-profit-with-genai)
- [BCG: Closing the AI Impact Gap (2025 Deep Dive)](https://www.bcg.com/publications/2025/closing-the-ai-impact-gap)
- [BCG: The Stairway to GenAI Impact (Maturity Model)](https://www.bcg.com/publications/2024/stairway-to-gen-ai-impact)
- [BCG: How AI Is Paying Off in the Tech Function (2026 Report)](https://www.bcg.com/publications/2026/how-ai-is-paying-off-in-the-tech-function)
- [BCG: Turbocharging Automotive Operations with GenAI (2026 Case)](https://www.bcg.com/publications/2026/turbocharging-automotive-operations-with-genai)
- [Bain: State of the Art Agentic AI Transformation (2025)](https://www.bain.com/insights/state-of-the-art-of-agentic-ai-transformation-technology-report-2025/)
- [Bain: From Pilots to Payoff in Software Development (2025)](https://www.bain.com/insights/from-pilots-to-payoff-generative-ai-in-software-development-technology-report-2025/)
- [Bain: Executive Survey - AI Moves to Production](https://www.bain.com/insights/executive-survey-ai-moves-from-pilots-to-production/)
- [Bain: Why Agentic AI Demands a New Architecture (2026)](https://www.bain.com/de/insights/why-agentic-ai-demands-a-new-architecture/)
- [Bain: Nvidia GTC 2026 - AI Becomes the Operating Layer](https://www.bain.com/el/insights/nvidia-gtc-2026-ai-becomes-the-operating-layer/)
- [Accenture: Making Reinvention Real with GenAI (2025 Blueprint)](https://www.accenture.com/content/dam/accenture/final/industry/cross-industry/document/Making-Reinvention-Real-With-GenAI-TL.pdf)
- [Accenture: Front Runner's Guide to Scaling AI (2025 POV)](https://www.accenture.com/content/dam/accenture/final/accenture-com/document-3/Accenture-Front-Runners-Guide-Scaling-AI-2025-POV.pdf)
- [Accenture: Tech Vision 2025 - Agentic Ecosystems](https://www.accenture.com/content/dam/accenture/final/accenture-com/document-3/Accenture-Tech-Vision-2025.pdf)
- [Deloitte: State of GenAI in the Enterprise (Q3 2025)](https://www2.deloitte.com/us/en/pages/consulting/articles/state-of-generative-ai-in-the-enterprise.html)
- [Deloitte: State of AI in the Enterprise 2026 (Early Preview)](https://deloitte.com/us/state-of-generative-ai)
- [Deloitte: From Ambition to Activation - Press Release 2026](https://www.deloitte.com/us/en/about/press-room/state-of-ai-report-2026.html)
- [KPMG: Trusted AI Framework - Governance & Control (2025 PDF)](https://assets.kpmg.com/content/dam/kpmg/ng/pdf/2025/09/AI%20Governance%20and%20Control.pdf)
- [KPMG: AI Governance for the Agentic Era (TACO Framework 2025)](https://kpmg.com/us/en/articles/2025/ai-governance-for-the-agentic-ai-era.html)
- [KPMG: Quantifying the GenAI Opportunity (2025 Report)](https://kpmg.com/kpmg-us/content/dam/kpmg/pdf/2025/quantifying-genai-opportunity.pdf)
- [KPMG: Agentic AI Advantage - Strategy for Success (2025)](https://assets.kpmg.com/content/dam/kpmgsites/xx/pdf/2025/10/agentic-ai-advantage-report.pdf.coredownload.inline.pdf)
- [PwC: Global AI Study 2025 - The Path to Value](https://www.pwc.com/gx/en/issues/data-and-ai/publications/global-ai-study.html)
- [Gartner: Top Strategic Technology Trends for 2025 - AI focus](https://www.gartner.com/en/articles/gartner-top-10-strategic-technology-trends-for-2025)
- [Gartner — пресс-релиз: нехватка AI-ready data подрывает ИИ-проекты (26.02.2025)](https://www.gartner.com/en/newsroom/press-releases/2025-02-26-lack-of-ai-ready-data-puts-ai-projects-at-risk)
- [Everest Group: Enterprise Generative AI Adoption 2025 Playbook](https://www.everestgrp.com/report/generative-ai-playbook)
- [HFS Research: The Generative AI 2025 Horizon Report](https://www.hfsresearch.com/research/genai-horizon-2025/)

### 16.3 Технические паттерны и инженерные блоги (Промышленный ИИ / Production AI) {: #app_a_technical_patterns_engineering_blogs }

- [DoorDash: How We Built an Internal AI Platform That Works (2025)](https://www.getdot.ai/blog/doordash-ai-platform-agents)
- [DoorDash: Building a Collaborative Multi-Agent AI Ecosystem (2026)](https://www.zenml.io/llmops-database/building-a-collaborative-multi-agent-ai-ecosystem-for-enterprise-knowledge-access)
- [DoorDash: Building an Enterprise LLMOps Stack - Lessons (2026)](https://www.zenml.io/llmops-database/building-an-enterprise-llmops-stack-lessons-from-doordash)
- [Uber Engineering: Genie - GenAI On-Call Copilot Architecture (2025)](https://www.uber.com/en-CO/blog/genie-ubers-gen-ai-on-call-copilot/)
- [Uber Engineering: Raising the Bar on ML Model Deployment Safety (2025)](https://www.uber.com/en-CA/blog/raising-the-bar-on-ml-model-deployment-safety/)
- [Netflix Tech Blog: Foundation Models for Personalized Recommendations (2025)](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39)
- [Netflix Tech Blog: Scaling Generative Recommenders (2025)](https://netflixtechblog.medium.com/integrating-netflixs-foundation-model-into-personalization-applications-cf176b5860eb)
- [Airbnb Tech Blog: Reshaping Customer Support with GenAI (2025)](https://medium.com/airbnb-engineering/how-ai-text-generation-models-are-reshaping-customer-support-at-airbnb-a851db0b4fa3)
- [Airbnb Tech Blog: Agent-in-the-Loop (AITL) Framework Paper (2025)](https://aclanthology.org/2025.emnlp-industry.135.pdf)
- [Meta (Экстремистская организация, запрещена в РФ) AI: Production Pipelines for Llama Deployments (Official 2025)](https://llama.meta.com/docs/deployment/production-deployment-pipelines)
- [vLLM: Performance Optimization and Tuning Guide (2025)](https://docs.vllm.ai/en/latest/performance/optimization.html)
- [vLLM Performance Tuning: The Ultimate Guide (2026)](https://cloud.google.com/blog/topics/developers-practitioners/vllm-performance-tuning-the-ultimate-guide-to-xpu-inference-configuration)
- [vLLM Production Stack Roadmap for 2025 Q2 (GitHub)](https://github.com/vllm-project/production-stack/issues/300)
- [vLLM Production Stack 2026 Roadmap (GitHub)](https://github.com/vllm-project/production-stack/issues/855)
- [vLLM: Practical strategies for performance tuning (Red Hat 2026)](https://developers.redhat.com/articles/2026/03/03/practical-strategies-vllm-performance-tuning)
- [LangGraph: Enterprise Multi-Agent Orchestration Patterns (2025)](https://sumanta9090.medium.com/langgraph-patterns-best-practices-guide-2025-38cc2abb8763)
- [LangGraph: Building Stateful, Multi-Agent AI Workflows (Checklist 2025)](https://bix-tech.com/ai-agents-orchestration-langgraph/)
- [Anthropic: Contextual Retrieval - Improving RAG Accuracy (2025)](https://www.anthropic.com/news/contextual-retrieval)
- [arXiv — General Agentic Memory (GAM), 2025](https://arxiv.org/pdf/2511.18423)
- [arXiv — Agent0: co-evolving curriculum and executor agents, 2025](https://arxiv.org/pdf/2511.16043)
- [OpenAI: Production RAG Best Practices & Evaluation (Cookbook)](https://github.com/openai/openai-cookbook/blob/main/examples/RAG_with_Evaluation.ipynb)
- [Microsoft: RAG Architecture on Azure AI Search (2025 Update)](https://learn.microsoft.com/en-us/azure/architecture/guide/multimodal-rag/multimodal-rag-architecture)
- [RAG - A Deep Dive (System Design Newsletter 2025)](https://newsletter.systemdesign.one/p/how-rag-works)
- [RAG Evaluation with RAGAS and MLflow (Practical Guide 2026)](https://www.safjan.com/ragas-mlflow-rag-evaluation-tutorial/)
- [Giskard: Open-Source Evaluation for LLM Agents (2026 Docs)](https://github.com/Giskard-AI/giskard-oss)
- [Monitaur: AI Governance Software Platform Features (2026)](https://www.monitaur.ai/platform)
- [LiteLLM Docs: Production Gateway for 100+ Models (2026)](https://docs.litellm.ai/)
- [Portkey AI Gateway: Cost Management and Observability (2026)](https://docs.portkey.ai/)
- [Helicone: Provider Routing and AI Gateway Docs (2026)](https://docs.helicone.ai/)

### 16.4 Экономика ИИ, FinOps и сайзинг {: #app_a_ai_economics_finops_sizing }

- [FinOps Foundation: Cost Estimation of AI Workloads (2026 Resource)](https://www.finops.org/wg/cost-estimation-of-ai-workloads)
- [FinOps Framework 2025: Cloud Cost Allocation PDF](https://www.finops.org/wp-content/uploads/2025/05/English-FinOps-Framework-2025.pdf)
- [FinOps in the AI Era: 2026 Survey Report (CloudZero)](https://www.cloudzero.com/guide/finops-in-the-ai-era-2026-survey-report/)
- [CloudZero: FinOps for AI - Why AI Alters Cloud Cost Management (2026)](https://www.cloudzero.com/blog/finops-for-ai/)
- [CloudZero: Cloud Unit Economics In 2026 Guide](https://www.cloudzero.com/guide/cloud-unit-economics-2026/)
- [CloudZero: FinOps Cost-Per-Unit Glossary (Feb 2026)](https://www.cloudzero.com/blog/finops-cost-per-unit-glossary/)
- [OpenAI: Real-time Cost and Token Monitoring (2025)](https://developers.openai.com/api/docs/guides/realtime-costs)
- [Azure AI Search: Semantic Ranker Pricing and Scaling (2025)](https://azure.microsoft.com/en-us/pricing/details/search/)
- [Mavik Labs: LLM Cost Optimization (Routing/Caching/Batching 2026)](https://www.maviklabs.com/blog/llm-cost-optimization-2026)
- [Enrico Piovano: LLM Cost Engineering & Token Budgeting (2026)](https://enricopiovano.com/blog/llm-cost-optimization-caching-strategies)
- [EngineersOfAI: Inference Cost Optimization Curriculum (2025)](https://engineersofai.com/docs/ai-systems/cost-and-finops/Inference-Cost-Optimization)
- [LLM API Pricing Guide 2026: Every Major Model Compared](https://www.decodesfuture.com/articles/llm-api-pricing-guide-2026-every-major-model-compared)
- [LLM API Pricing (March 2026) — GPT-5.4, Claude 4.6, Gemini 3.1](https://www.tldl.io/resources/llm-api-pricing-2026)
- [LLM Pricing in February 2026: What Every Model Actually Costs](https://kaelresearch.com/blog/llm-pricing-comparison-feb-2026)
- [AI Model Pricing 2026: GPT-5 vs Claude 4.6 vs Gemini 3.1 (ClawPort)](https://clawport.io/blog/best-ai-models-cost-comparison-2026)
- [AI Cost Optimization Case Study v2.1 (Feb 2026)](https://medium.com/@alexlewe/ai-cost-optimization-case-study-v2-1-englishfootballhistory-com-a35e29d8a1dc)
- [How AI-First Teams Cut AWS Bills by 60% in 2026 (Groovy Web)](https://www.groovyweb.co/blog/cloud-cost-optimization-ai-first-2026)
- [OptyxStack Case Study: Reducing Inference Cost by 60% (2026)](https://optyxstack.com/case-studies/llm-inference-cost-reduction)
- [AI Agent Cost Optimization: Token Economics in Production (Zylos 2026)](https://zylos.ai/research/2026-02-19-ai-agent-cost-optimization-token-economics)

### 16.5 Российские регуляторные и правовые источники (Приоритет 1) {: #app_a_russian_regulatory_legal_sources }

- [Указ Президента РФ №490: Национальная стратегия развития ИИ до 2030 (Ред. 2024)](https://www.consultant.ru/document/cons_doc_LAW_470015/)
- [Указ Президента РФ от 15.02.2024: Изменения в стратегию ИИ (Актуальная версия)](https://ai.gov.ru/national-strategy/)
- [Минцифры РФ: Пояснительная записка к законопроекту об ИИ (Март 2026)](https://www.m24.ru/news/politika/18032026/883742)
- [Минцифры РФ: Правила маркировки ИИ-контента (Законопроект 2026)](https://www.infox.ru/news/299/375381-mincifry-rf-predstavilo-novye-pravila-dla-regulirovania-ii-s-markirovkoj-kontenta)
- [Банк России: Кодекс этики ИИ на финансовом рынке (Официальный PDF 2025)](https://www.cbr.ru/content/document/file/178667/code_09072025.pdf)
- [Банк России: Пять принципов ответственного использования ИИ (Июль 2025)](https://www.cbr.ru/press/event/?id=25755)
- [Банк России: Информационное письмо №ИН-016-13/91 (Июль 2025)](https://www.consultant.ru/document/cons_doc_LAW_509514/)
- [Банк России: Доклад о применении ИИ на финансовом рынке (Consultation Paper)](http://www.cbr.ru/analytics/d_ok/Consultation_Paper_03112023/)
- [Роскомнадзор: Приказ №140 «Об утверждении требований к обезличиванию ПДн» (2025)](https://normativ.kontur.ru/document?documentId=500957&moduleId=1)
- [Постановление Правительства РФ №1154: Требования и методы обезличивания ПДн (2025)](https://klerk.ru/doc/657888)
- [ФЗ-152 «О персональных данных»: Ст. 18.1 (Меры защиты)](https://legalacts.ru/doc/152_FZ-o-personalnyh-dannyh/glava-4/statja-18.1/)
- [ФЗ-152: Обзор поправок о локализации и сборе с 30 мая 2025 года](https://riverstart.ru/blog/novyie-trebovaniya-kpersonalnyim-dannyim-v2025-pravila-rabotyi-dlya-biznesa-s152-fz)
- [ФЗ-572 «О биометрических данных»: Регулирование в контуре ИИ (2025)](https://www.consultant.ru/document/cons_doc_LAW_435801/)
- [Минцифры РФ: Методические рекомендации по внедрению ИИ в госсекторе](https://digital.gov.ru/ru/documents/9245/)
- [Национальный кодекс этики в сфере ИИ (Альянс в сфере ИИ)](https://a-ai.ru/code-of-ethics/)
- [ГОСТ Р 59277-2020: Системы ИИ. Классификация систем ИИ](https://allgosts.ru/35/240/gost_r_59277-2020)
- [ГОСТ Р 59276-2020: Системы ИИ. Способы обеспечения доверия](https://allgosts.ru/35/240/gost_r_59276-2020)
- [Dentons: Регулирование ИИ в России - обзор 2025-2026](https://www.dentons.com/ru/insights/alerts/)
- [ALRUD: ИИ и персональные данные - новые вызовы 2026](https://alrud.ru/news/legal-alerts/)
- [BGP Litigation: Законопроект об ИИ - что нужно знать бизнесу (2026)](https://bgplaw.com/news/)
- [Melling Voitishkin: Legal Alert - Маркировка ИИ контента в РФ](https://melling.com/ru/insights/)

### 16.6 Российские прикладные исследования и бенчмарки {: #app_a_russian_applied_research_benchmarks }

- [MERA Benchmark: GigaChat 2 MAX Ranking (Top-1 RU 2026)](https://setka.ru/posts/019592e7-54d7-4f94-af58-0b74d6968357)
- [ruMMLU: Benchmarking Russian LLM Intelligence (HSE/Sber)](https://github.com/ai-forever/ru-mmlu)
- [НИУ ВШЭ: Исследование точности RAG-систем на русском языке (2025)](https://www.hse.ru/edu/vkr/1053304649)
- [НИУ ВШЭ: Мультиагентная платформа для отраслевых задач (2026)](https://techpro.hse.ru/ai-solutions/description)
- [НИУ ВШЭ: План исследований мультиагентного ИИ 2025-2026](https://www.hse.ru/news/development/1053986394.html)
- [ИТМО: Мультиагентная система ProAGI для разработки ПО (2026)](https://iai.itmo.ru/news/v-itmo-sozdali-multiagentnuyu-ii-sistemu-proagi,-kotoraya-uskoryaet-sozdanie-promyishlennного-po-ot-2-do-10-raz)
- [ИТМО: Лаборатория композитного ИИ - Фреймворк FEDOT (2025)](https://itmo.ru/ru/viewdepartment/507/laboratoriya_kompozitnogo_iskusstvennogo_intellekta.htm)
- [Сколково: Потенциал GenAI для инженерных задач (Июль 2025)](https://sk.ru/news/skolkovo-i-ano-ce-predstavili-obzor-potencial-primeneniya-generativnogo-ii-dlya-resheniya-inzhenernyh-zadach/)
- [АНО ЦЭ: Аналитический отчет «Будущее искусственного интеллекта» (2025)](https://d-economy.ru/news/ano-cje-vypustila-analiticheskij-otchet-budushhee-iskusstvennogo-intellekta/)
- [Иннополис: Применение ИИ в промышленности и строительстве (2025)](https://innopolis.university/news/)
- [Sber AI: ru-Gemma и open-source инициативы 2025-2026](https://developers.sber.ru/docs/ru/gigachat/models/updates)
- [Yandex Research: Оптимизация инференса LLM для русского языка (2025)](https://yandex.ru/company/research/)

### 16.7 Модели отчуждения и передачи (BOT и передача) {: #app_a_transfer_models_bot_handover }

- [Build-Operate-Transfer (BOT) Model: Full Guide 2025](https://build-operate-transfer.com/post/build-operate-transfer-bot-model-complete-guide-for-software-development-2025)
- [Tech4lyf — чек‑лист передачи ПО (Software handover checklist, 2026)](https://www.tech4lyf.com/blog/software-handover-documentation-checklist-2026/)
- [InCommon: Why BOT Wins for AI Infrastructure](https://www.incommon.ai/blog/build-operate-transfer/)
- [Innowise: BOT Outsourcing Contract and IP Transfer Guide](https://innowise.com/blog/build-operate-transfer-bot-model-guide/)
- [Devico: Checklist for a seamless BOT transition (2025)](https://devico.io/blog/checklist-for-a-seamless-bot-transition)
- [Knowledge Transfer Framework for Enterprise Software Handover](https://www.knowledge-management-tools.net/knowledge-transfer-framework.html)

### 16.8 Кураторские списки и репозитории (Awesome Lists) {: #app_a_curated_lists_repositories }

- [GitHub: Awesome AI Agents 2026 (300+ resources)](https://github.com/caramaschiHG/awesome-ai-agents-2026)
- [GitHub: Awesome Production GenAI (Updated March 2026)](https://ethicalml.github.io/awesome-production-genai/)
- [GitHub: Awesome RAG Production Tools (Curated Feb 2026)](https://github.com/Yigtwxx/Awesome-RAG-Production)
- [GitHub: Awesome AI Apps - Practical Agents (2026 Repo)](https://github.com/rohitg00/awesome-ai-apps)
- [Arxiv: HiChunk - Hierarchical Chunking for Advanced RAG (2025)](http://arxiv.org/abs/2509.11552v3)
- [Arxiv: SmartChunk Retrieval - Query-Aware Compression (2026 Paper)](https://www.arxiv.org/abs/2602.22225)
- [Arxiv: Agentic RAG Taxonomy, Architecture and Research (March 2026)](https://arxiv.org/abs/2603.07379v1)
- [Arxiv: JADE - Strategic-Operational Gap in Agentic RAG (Jan 2026)](https://arxiv.org/abs/2601.21916)
- [Arxiv: OrchMAS - Orchestrated Reasoning with Multi-Agents (March 2026)](https://arxiv.org/abs/2603.03005v1)
- [Arxiv: TreePS-RAG - Tree-based Process Supervision (Jan 2026)](https://arxiv.org/abs/2601.06922)

### 16.9 Кейсы внедрения в российском бизнесе (2025-2026) {: #app_a_russian_business_implementation_cases }

- [Сбер: Эффект от внедрения ИИ в 2026 году (Прогноз 550 млрд руб)](https://www.sostav.ru/publication/sber-ozhidaet-chto-effekt-ot-vnedreniya-ii-v-2026-godu-dostignet-550-mlrd-rublej-80507.html)
- [Сбер: Первый в России ИИ-агент для Process Mining (Янв 2026)](https://pwa.lenta.ru/news/2026/01/22/sber-predstavil-pervogo-v-rossii-ii-agenta-dlya-analiza-biznes-protsessov/)
- [Сбер: Кейс автономного кредитования без участия человека (2025)](https://abnews.ru/news/2026/3/3/sber-97-krupnyh-kompanij-v-rossii-gotovy-rabotat-s-ii-sistemami)
- [ВТБ: Как банк превратит 15 млрд в 50 млрд руб. экономии через ИИ](https://www.comnews.ru/content/242366/2025-11-17/2025-w47/1008/ii-alkhimiya-vtb-kak-bank-15-mlrd-prevratit-50-mlrd-rub-ekonomii)
- [ВТБ Мои Инвестиции: Алгоритм работы ИИ-стратегии «Интеллект» (2026)](https://banks.cnews.ru/news/line/2026-02-20_vtb_moi_investitsii_rasskazali)
- [СИБУР: Экономический эффект от ИИ на «Сибур-Нефтехиме» (200 млн руб)](https://www.sibur.ru/SiburNeftekhim/press-center/ekonomicheskiy-effekt-ot-vnedreniya-tsifrovykh-instrumentov-na-sibur-neftekhime-prevysil-200-mln-rub/)
- [Газпром Нефть: Использование цифровых двойников и ИИ в сейсморазведке](https://neftegaz.ru/analisis/digitalization/908282-ii-v-neftegaze-ot-otdelnykh-algoritmov-k-kompleksnym-resheniyam/)
- [Газпром ЦПС: Внедрение ИИ-помощника в систему «АФИДА» (RAG-кейс)](https://habr.com/ru/companies/gazpromcps/articles/975596/)
- [Магнит: Как ИИ превратил 150 000 отзывов в день в рост NPS (2025)](https://generation-ai.ru/cases/magnit)
- [Ozon: ИИ как инструмент для 60% малых предпринимателей (2025)](https://www.retail.ru/news/ozon-ii-stal-rabochim-instrumentom-dlya-bolee-60-malykh-predprinimateley-v-rossi-15-dekabrya-2025-272572/)
- [Альфа-Банк: ИИ-модерация контента на платформе «Альфа-Инвестор»](https://innovanews.ru/info/news/economics/ii-na-troikh-vedushhie-banki-razlozhili-tekhnologicheskijj-pasjans/)
- [Т-Банк: Эмоциональный ИИ и 500 000 звонков ИИ-Деду Морозу (2026)](https://innovanews.ru/info/news/economics/ii-na-troikh-vedushhie-banki-razlozhili-tekhnologicheskijj-pasjans/)
- [Яндекс: Корпоративный DeepResearch по кодовой базе (Кейс 2025)](https://habr.com/ru/companies/yandex/articles/987388/)
- [Северсталь: ИИ для контроля качества проката и оптимизации плавки](https://www.severstal.com/rus/media/news/)
- [Росатом: ИИ-системы для проектирования АЭС и анализа безопасности](https://rosatom.ru/journalist/news/)
- [Самолет: Кейс «Цифровой рабочий» и ИИ в управлении стройкой (2025)](https://samolet.ru/news/)

### 16.10 Технические статьи и инженерные блоги (Россия) {: #app_a_russian_technical_articles_blogs }

- [Хабр: Оркестрация ИИ-агентов в 2026 - Кейс ритейл-компании](https://habr.com/ru/articles/1008598/)
- [Хабр: Продвинутые техники RAG в действии (Сбербанк 2025)](https://habr.com/ru/companies/sberbank/articles/937242/)
- [Хабр: ИИ-агент внутри 1С - архитектура и DSL-управление (2026)](https://habr.com/ru/articles/1006230/)
- [Хабр: Как превратить сценарного чат-бота в умного ИИ-агента (2025)](https://habr.com/ru/articles/976782/)
- [Хабр: Кейс решения тестового задания 1С-аналитика ИИ-агентом (2025)](https://habr.com/ru/companies/1yes/articles/1001112/)
- [Хабр: Строим корпоративную GenAI-платформу - RAG и ROI (МФТИ 2025)](https://habr.com/ru/companies/mipt_digital/articles/932962/)
- [Хабр: RAG на CPU без GPU - опыт Газпром ЦПС (2025)](https://habr.com/ru/companies/gazpromcps/articles/975596/)
- [VC.ru: ИИ-трансформация 2026 - пошаговый план от хайпа к P&L](https://vc.ru/ai/2734338-ii-transformaciya-biznesa-2026-poshagovyj-plan-vnedreniya-ii)
- [VC.ru: Сколько на самом деле стоит ИИ-агент для бизнеса (2026)](https://vc.ru/ai/2791769-stoimost-ii-agenta-dlya-biznesa)
- [VC.ru: Маркировка ИИ-контента - разбор законопроекта Минцифры](https://vc.ru/ai/2802187-zakonoproekt-mincifry-ob-ii-markirovka-kontenta)
- [CNews: Технологии искусственного интеллекта 2025 - обзор](https://adobe.cnews.ru/reviews/tehnologii_iskusstvennogo_intellekta/)
- [RB.ru: Топ-100 ИИ-стартапов России 2025 - карта рынка](https://rb.ru/list/ai-100-2025/)
- [ComNews: Экономика автоматизации ИИ и точки экономии для бизнеса (2026)](https://www.comnews.ru/digital-economy/content/244350/2026-03-23/2026-w13/1016/ekonomika-avtomatizacii-ii-i-realnye-tochki-ekonomii-dlya-biznesa)

### 16.11 Российская экономика ИИ и отчеты консалтинга {: #app_a_russian_ai_economy_consulting_reports }

- [Яков и Партнёры: Rewiring the Enterprise for GenAI - Russian Context (2025)](https://yakovpartners.ru/publications/ai-2025/)
- [Kept (ex-KPMG): ИИ-агенты KeptStore для корпоративного сектора (2026)](https://www.vedomosti.ru/press_releases/2026/01/14/kept-zapuskaet-platformu-s-ii-agentami-keptstore-dlya-avtomatizatsii-zadach-korporativnogo-segmenta-erid-2VfnxxbJAiD)
- [B1 (ex-EY): Использование ИИ в российских компаниях - опрос 2025](https://www.b1.ru/ru/insights/ai-survey-2025/)
- [Технологии Доверия (ex-PwC): ИИ как драйвер изменений экономики (2025)](https://ict.moscow/research/iskusstvennyi-intellekt-draiver-izmenenii-ekonomiki-i-finansov/)
- [Деловые Решения и Технологии (ex-Deloitte): ИИ в России 2026](https://delret.ru/insights/ai-russia-2026)
- [AIРассвет: Метрики ROI и стратегии внедрения ИИ в РФ 2025-2026](https://airassvet.ru/articles/effektivnost-iskusstvennogo-intellekta-v-rossiyskom-biznese-2025-2026-analiticheskiy-otchet-o-7-klyuchevyh-stsenariyah-metrikah-roi-i-strategiyah-vнедрения)
- [CNews: Прогноз внедрения ИИ в BI-решения 2026](https://www.cnews.ru/news/line/2026-03-16_navikon_v_2026_gbolee_80)
- [TAdviser: ИТ-приоритеты 2026 - ИИ на первом месте](http://www.tadviser.ru/index.php/Статья:TAdviser%3A_%D0%98%D0%A2-%D0%BF%D1%80%D0%B8%D0%BE%D1%80%D0%B8%D1%82%D0%B5%D1%82%D1%8B_2026)
- [Yandex Cloud: Стоимость YandexGPT 4 и кейсы интеграции (2026)](https://cloud.yandex.ru/services/yandexgpt)
- [Минцифры РФ: Национальный прогноз вклада ИИ в ВВП до 2030 года](https://digital.gov.ru/ru/activity/directions/1056/)
- [РБК: Тренды ИИ в медицине и госсекторе 2025-2026](https://www.rbc.ru/trends/innovation/)
- [Коммерсант: Сравнение цен на генерацию YandexGPT и GigaChat (2026)](https://ya-r.ru/2023/12/12/kommersant-sravnil-tseny-na-generatsiyu-yandexgpt-i-gigachat/)
