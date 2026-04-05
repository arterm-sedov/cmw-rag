---
title: 'Ведомость документов, глоссарий, реестр источников и курс валют'
date: '2026-04-05'
status: 'Комплект материалов для руководства (v2, апрель 2026)'
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
hide:
  - tags
---

# Ведомость документов, глоссарий, реестр источников и курс валют {: #app_a_pack_overview .pageBreakBefore }

## Обзор {: #app_a_overview_section }

Приложение даёт **единый вход** в комплект отчётов по внедрению корпоративного ИИ в резидентном контуре РФ: где искать решение по методологии, экономике, передаче, текущему стеку, комплаенсу и расширенному чтению.

- **Организация внедрения и эксплуатации** — фазы, роли, контрольные точки
- **Диапазоны CapEx/OpEx/TCO** — ориентиры для смет и бюджетов
- **Передача кода и ИС** — комплект KT/IP и приёмка
- **Граница готового стека Comindware** — что входит в поставку
- **Риски и контроли** — что закрыть до промышленного запуска

**Состав комплекта:** единое резюме для руководства, два основных отчёта (методология; сайзинг и экономика), приложения A–F.

**В приложении:** один навигатор `вопрос → документ`, глоссарий, единая политика курса валют и реестр источников.

**База материалов:** публичные прайсы, отраслевые публикации и инженерная практика **Comindware** (открытые репозитории экосистемы). Используйте как основу для собственных презентаций, смет и управленческих решений — с адаптацией под профиль заказчика и финальной сверкой договорных условий перед оффером.

## Единая дата-опора для цифр и метрик {: #app_a_reference_date_policy .pageBreakBefore }

Если в тексте не оговорено иное, **цены, тарифы, рыночные метрики и количественные ориентиры** в комплекте приведены как единый справочный срез **на март 2026 года**.

Для **управленческих решений** комплект используйте как основу для сравнения сценариев, архитектур и диапазонов TCO. Для **сметы, КП и договора** финальные значения перепроверяются по актуальным первичным источникам и условиям сделки на момент расчёта.

## Термины и условные обозначения {: #app_a_conventions_and_scope }

Ниже — краткий словарь ключевых терминов, который фиксирует их употребление в этом комплекте и снижает риск разночтений при обсуждении архитектуры, экономики и модели передачи.

В тексте комплекта **корпоративный RAG-контур**, **сервер инференса на базе vLLM/MOSEC** и **агентный слой Comindware Platform** — это **условные названия ролей компонентов** иллюстративного референс-стека **Comindware**, а не обязательные коммерческие SKU.

Подробные формулировки для договоров и переговоров — см. _Приложение B «[Отчуждение ИС и кода: KT, IP, лицензии, критерии приёмки передачи](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_pack_overview)_ и _Приложение C «[Корпоративный ИИ Comindware: состав стека, границы, артефакты](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_pack_overview)»_.

| Термин | Определение в контексте комплекта |
| --- | --- |
| Агентный RAG | Вариант RAG, где модель не только получает контекст, но и планирует шаги, вызывает инструменты и при необходимости делает несколько итераций поиска и проверки. |
| Агентный слой Comindware Platform | Условное обозначение сценариев, где модель не только отвечает, но и инициирует действия в платформе через разрешённые инструменты и API. |
| Агенты для программирования (coding agents) | ИИ-агенты и связанные среды (IDE, CLI, песочницы, внешние каналы), автоматизирующие **цикл разработки**: правки кода, тесты, ревью, интеграция с PR/CI. Англ. **coding agents** — устоявшийся продуктовый ярлык; в комплекте основной термин — **агенты для программирования**. |
| Батч (batch) | Группа запросов, обрабатываемая за один цикл **пакетной обработки**; **размер батча** (англ. **batch size**) влияет на VRAM и задержку. |
| CLI (Command Line Interface) | Интерфейс командной строки для администрирования серверов инференса: запуск, остановка, проверка статуса, тестирование моделей. |
| Вызов инструментов (Tool calling / Function calling) | Способ работы модели, при котором она по явному контракту инициирует вызов внешнего инструмента, API или функции и использует результат в ответе или следующем шаге. |
| Выборка (Sampling) | Отбор только части журналов, трасс или событий для хранения и анализа, чтобы ограничивать объём телеметрии и стоимость наблюдаемости без потери значимых сигналов. |
| Глубокое исследование (Deep research) | Многошаговый поиск и сверка источников с последующей аналитической сборкой выводов, когда требуется не быстрый ответ, а обоснованный материал с опорой на несколько независимых источников. |
| Защитные механизмы (Guardrails) | Набор политик, фильтров, валидаций и ограничений вокруг модели и инструментов, который снижает риск небезопасных, нерелевантных или неразрешённых действий и ответов. |
| Извлечение контекста (Retrieval) | Этап RAG-контура, в котором система находит и отбирает релевантные фрагменты документов, записей или иных данных по запросу пользователя для последующей генерации ответа. |
| Исходный уровень (Baseline) | Начальное значение метрики, процесса или стоимости, от которого затем измеряют изменение, эффект и достижение целевых показателей. |
| Комплаенс (compliance) | Соответствие применимым нормам, договорам и внутренним политикам (регуляторика, ИБ, персональные данные). В комплекте термин обозначает **управленческий и технический** контур согласования, а не замену юридического заключения. |
| Контур оценки качества | Наборы тестов, критерии, метрики и регрессионные проверки, которые позволяют отслеживать деградацию после изменений модели, индекса или промпта. Включает офлайн- и онлайн-оценку качества (см. отдельные строки ниже). |
| Корпоративный RAG-контур | Условное обозначение референсного контура ассистента **Comindware** с поиском по корпоративным данным и генерацией ответа по найденному контексту. |
| Наблюдаемость (observability) | Возможность разбирать контур по трассам, метрикам, журналам и событиям. |
| Онлайн-оценка качества | Оценка ответов или траекторий агента на живом трафике часто без эталонного ответа; требует явной политики телеметрии, выборки и ПДн. В инженерной среде иногда используют термин online evaluation; в отчётах комплекта предпочтительна русская формулировка. |
| Офлайн-оценка качества | Проверки на фиксированных наборах, рубриках и сценариях до выката (регрессия после смены модели, индекса или промпта). В инженерной среде иногда используют термины eval, evals, offline evaluation; в отчётах комплекта предпочтительны эта строка и термин «контур оценки качества». |
| Открытые веса модели (Open weights) | Опубликованные веса модели, которые можно развернуть в своём контуре; это не тождественно отсутствию лицензионных ограничений. |
| Пакетная обработка (Batching) | Совместная подача нескольких запросов в инференс (очередь, групповой проход) для загрузки GPU и пропускной способности. |
| Ретенция данных (Retention) | Установленный срок и правила хранения журналов, трасс, метрик и других артефактов наблюдаемости, после которого данные удаляются, архивируются или переводятся в более дешёвое хранилище. |
| SGR (Schema-Guided Reasoning) | Структурированное рассуждение по схеме — техника принудительного структурирования рассуждений LLM через предопределённые схемы. По отраслевым бенчмаркам даёт 5–10% улучшение точности по сравнению с неструктурированными промптами; обеспечивает воспроизводимое рассуждение и аудит каждого шага (_[Schema-Guided Reasoning (SGR)](https://abdullin.com/schema-guided-reasoning/)_). В Comindware используется в нескольких точках конвейеров: анализ запросов, критика ответов агентов, планирование после фазы гарда, детерминированное управление любым мышлением. |
| Тензорный параллелизм (Tensor Parallelism, TP) | Распределение вычислений и фрагментов весов модели по нескольким GPU (англ. **tensor parallelism**); снижает объём памяти **на одно устройство** ценой обмена между картами. В грубых формулах сайзинга **TP** — число GPU по этой оси (**1**, если шардирования нет). |
| Фактор автобуса (Bus factor) | Степень зависимости проекта или контура от ограниченного числа носителей критичных знаний; чем он ниже, тем выше риск остановки или деградации после выбытия ключевых сотрудников. |
| Человек в контуре (Human-in-the-loop, HITL) | Подход, при котором критические решения, спорные ответы или рискованные действия модели переходят на проверку, подтверждение или коррекцию человеком. |
| Эксплуатационный регламент (Runbook) | Описание штатной эксплуатации, инцидентов, типовых проверок и действий сопровождения. |
| ADR | Architecture Decision Record — зафиксированное архитектурное решение с его основаниями и ограничениями. |
| AI TRiSM | AI Trust, Risk and Security Management — рамка доверия, рисков и безопасности ИИ. |
| ASR | Automatic Speech Recognition — распознавание речи: преобразование речевого сигнала в текст (голосовой ввод, колл-центры, мультимодальные сценарии). |
| Arize Phoenix | Открытый продукт Arize AI для наблюдаемости и экспериментов в контурах LLM/RAG: трассы, дашборды, связка с оценкой качества при self-hosted размещении. В референс-стеке Comindware — слой рядом с OpenTelemetry/OpenInference и инференсом; не заменяет инфраструктурный мониторинг (Prometheus/Grafana/Tempo). Подробно — см. _«[Рынок РФ, наблюдаемость LLM и референс-стек Comindware](./20260325-research-appendix-d-security-observability-ru.md#app_d_russia_llm_observability_phoenix_reference)_ в Приложении D. |
| BOT | Build–Operate–Transfer — модель «создать, эксплуатировать, передать». |
| CapEx | Капитальные затраты: оборудование, лицензии и ввод в эксплуатацию. |
| DPA | Data Processing Agreement — договор об обработке данных между сторонами, где одна передаёт данные другой для обработки (типичный ориентир в практике GDPR). В РФ переносится на роли **оператора**, **поручения обработки ПДн** и цепочку субобработчиков по 152-ФЗ и договору; в комплекте термин не заменяет юридическое заключение. |
| DSPy | Открытый фреймворк декларативной сборки и настройки LLM-конвейеров (модули, сигнатуры, оптимизация промптов и обучающих примеров). К похожему классу относятся иные библиотеки программной сборки промптов и контрактов вывода; в комплекте DSPy приводится как ориентир из открытых туториалов, не как обязательный стек поставки. |
| FinOps | Подход к управлению облачными и ИИ-затратами через прозрачные метрики потребления и аллокацию затрат. |
| GenAI | Генеративный ИИ: модели, которые создают текст, код и иные артефакты. |
| IP | Intellectual Property — интеллектуальная собственность: код, артефакты, модели, документация, права использования и условия передачи. |
| KT | Knowledge Transfer — передача знаний, эксплуатационных регламентов и обучения команде заказчика. |
| KV-кэш | Кэш пар «ключ–значение» для промежуточных состояний внимания при автогрессивной генерации (англ. **KV cache**); вместе с весами модели и **батчем** задаёт основную нагрузку на VRAM при длинном контексте. |
| LLM | Большая языковая модель. |
| LLM-as-a-judge | Подход, при котором отдельная модель используется как судья по заранее заданной рубрике. |
| LLMOps | Практики эксплуатации LLM-контуров: релизы, мониторинг, стоимость, качество и инциденты. |
| LoRA | Low-Rank Adaptation — адаптация большой модели небольшим числом добавочных параметров **низкого ранга**, без полного дообучения всех весов; обычно дешевле по памяти GPU и хранению, чем полное дообучение. В комплекте упоминается в исследовательских и продуктовых контекстах (в т.ч. Doc-to-LoRA, подходы к «забыванию» весов). |
| MCP | Model Context Protocol — протокол подключения инструментов и внешних ресурсов к агенту через явные серверы и контракты вызова. |
| MERA / RAGAS / DeepEval | Контуры и фреймворки оценки качества RAG/LLM. |
| ML | Классическое машинное обучение: модели классификации, прогнозирования и ранжирования без обязательной генерации текста. В комплекте термин нужен, чтобы отделять традиционный ML от GenAI. |
| ModelOps | Практики управления жизненным циклом моделей как производственных компонентов: версии, выкаты, контроль качества и сопровождение. |
| MOSEC | В контексте пакета: связка из фреймворка **Mosec** и обвязки **Comindware** вокруг него для единого HTTP-сервиса вспомогательных моделей: эмбеддингов, реранка, защитных механизмов и смежных сервисов. |
| NIST AI RMF | AI Risk Management Framework от NIST; в пакете используется как методологический ориентир, а не как замена нормам РФ. |
| On-prem | Размещение в собственном или выделенном контуре заказчика, а не в публичном управляемом API. |
| OpEx | Операционные затраты: эксплуатация, сопровождение, мониторинг и сервисы. |
| OpenInference | Открытый набор соглашений и инструментов для OpenTelemetry-совместимого инструментирования GenAI-приложений (спаны, атрибуты, экспорт телеметрии); поддерживается в том числе в Arize Phoenix. |
| PoC | Proof of Concept — короткий этап проверки гипотезы до пилота или масштабирования. |
| RAG | Генерация ответа с опорой на предварительный поиск по документам, данным или базе знаний. |
| SGLang | Открытый фреймворк высокопроизводительного инференса LLM и структурированной генерации; в комплекте — ориентир при выборе движка рядом с **vLLM**, не фиксированный референс поставки без отдельного архитектурного решения. |
| SLA | Обещанный уровень сервиса для заказчика. |
| SLM | Компактная языковая модель для более дешёвых или быстрых сценариев. |
| SLO | Внутренняя целевая метрика качества сервиса. |
| TCO | Совокупная стоимость владения решением на горизонте нескольких лет. |
| TOON | Компактный формат структурированных данных, применяемый для снижения токеновых затрат относительно JSON. |
| TOM | Target Operating Model — целевая операционная модель: роли, процессы, метрики и контуры ответственности. |
| TTS | Text-to-Speech — синтез речи: преобразование текста в речь (голосовой ответ ассистента, озвучивание, IVR). |
| vLLM | В контексте пакета: upstream-движок **vLLM** и обвязка **Comindware** вокруг него для промышленного инференса больших языковых моделей через OpenAI-совместимый API, с конфигурацией, проверками доступности и эксплуатационным регламентом. |

## Курс USD для смет {: #app_a_fx_policy }

**1 USD = 85 RUB** — единый ориентир для сопоставления USD-прайсов и рублёвых оценок в материалах на март 2026.

В **сметах и в договорных КП** ориентируйтесь на **курс ЦБ РФ на текущую дату** или на **курс, зафиксированный в договоре**.

Закладывайте отклонение курса **±10%** — ориентир чувствительности для **зависимых от USD** статей (импортное железо, зарубежные каталоги); но это не прогноз.

## Навигация «вопрос → документ» {: #app_a_question_document_navigation }

### Коммерция и экономика

| Вопрос | Документ |
| --- | --- |
| Коммерческий C-level-обзор: типовые пакеты, что остаётся у заказчика, матрица аргументов по ЛПР | _[Резюме: коммерческое обоснование](./20260331-research-executive-unified-ru.md)_ |
| KPI, числовые пороги go/no-go, политика интерпретации | _[Резюме: числовые пороги](./20260331-research-executive-unified-ru.md#exec_unified_guardrails)_; _[Методология: процессы и KPI](./20260325-research-report-methodology-main-ru.md#method_processes_kpis)_ |
| CapEx/OpEx/TCO — цифры и диапазоны для клиента | _[Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_pack_overview)_ |
| Расчёт расхода токенов (портал поддержки) | _[Сайзинг: токены](./20260325-research-report-sizing-economics-main-ru.md#sizing_token_consumption_estimates)_ |

### Методология внедрения

| Вопрос | Документ |
| --- | --- |
| Внедрение в пром контуре: роли, фазы, контрольные точки качества | _[Методология](./20260325-research-report-methodology-main-ru.md#method_pack_overview)_ |
| Где формируется преимущество в корпоративном ИИ (данные, семантика, агенты) | _[Методология](./20260325-research-report-methodology-main-ru.md#method_pack_overview)_ |
| Глобальные бенчмарки OpenAI 2025 и оговорки по выборке | _[Методология: эмпирика](./20260325-research-report-methodology-main-ru.md#method_openai_implementation_report)_ |
| Стратегия внедрения, организационная зрелость, пилот vs scale, обучение (СКОЛКОВО) | _[Методология: стратегия](./20260325-research-report-methodology-main-ru.md#method_ai_strategy_org_maturity)_ |
| Бизнес-процессы для KT (BPMN 2.0, LLM-генерация) | _[Методология: BPMN](./20260325-research-report-methodology-main-ru.md#method_bpmn_process_formalization_llm)_ |
| SGR в практике Comindware | _[Методология: SGR](./20260325-research-report-methodology-main-ru.md#method_sgr_practice_cmw)_ |
| GenAI в маркетинге крупных брендов РФ (опрос CMO, red_mad_robot, 2025) | _[Сайзинг: рынок](./20260325-research-report-sizing-economics-main-ru.md#sizing_russian_market)_; _[Методология: GenAI в маркетинге](./20260325-research-report-methodology-main-ru.md#method_genai_marketing_teams)_ |
| Российский рынок GenAI: сегменты, прогноз до 2030 | _[Методология: карта рынка](./20260325-research-report-methodology-main-ru.md#method_russian_genai_market_map)_; _[Сайзинг: статистика](./20260325-research-report-sizing-economics-main-ru.md#sizing_russia_ai_market_stats_forecasts)_ |

### Передача и текущий стек

| Вопрос | Документ |
| --- | --- |
| Комплект отчуждения ИС/кода (KT/IP) | _[Приложение B](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_pack_overview)_ |
| Бизнес-процессы для KT (минимальный комплект) | _[Приложение B: отчуждение](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_alienation_package_minimal)_ |
| Состав стека Comindware («что есть» vs «методология») | _[Приложение C](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_pack_overview)_ |
| Возможности агентов (RAG, MCP, SGR, индексация) | _[Приложение C: арсенал](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_component_arsenal)_ |
| Ассистент аналитика (49 инструментов, 6 провайдеров) | _[Приложение C: аналитик](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_analyst_assistant)_ |
| Фреймворки инференса (MOSEC, vLLM, Infinity) | _[Приложение C: инференс](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_inference_frameworks)_ |

### Безопасность и наблюдаемость

| Вопрос | Документ |
| --- | --- |
| Безопасность, комплаенс, наблюдаемость | _[Приложение D](./20260325-research-appendix-d-security-observability-ru.md)_ |
| Наблюдаемость GenAI в РФ: локализация, self-hosted телеметрия, отличие от инфраструктурного мониторинга | _[Приложение D: LLM-observability](./20260325-research-appendix-d-security-observability-ru.md#app_d_russia_llm_observability_phoenix_reference)_ |
| Изоляция и сеть для агентского исполнения (граница доверия, egress) | _[Приложение D: граница доверия](./20260325-research-appendix-d-security-observability-ru.md#app_d_trust_boundary_agent_environment)_ |
| Паттерны среды для агента, модель риска, минимальный состав платформы | _[Приложение D: модель риска](./20260325-research-appendix-d-security-observability-ru.md#app_d_risk_model_platform_patterns)_ |
| Сравнение песочниц E2B / Modal / Daytona | _[Приложение D: песочницы](./20260325-research-appendix-d-security-observability-ru.md#app_d_managed_sandboxes_benchmarks)_; _[Методология: бенчмарки](./20260325-research-report-methodology-main-ru.md#method_sandbox_evaluation_benchmarks)_ |
| Безопасный MVP контура агента за ~30 дней | _[Приложение D: MVP](./20260325-research-appendix-d-security-observability-ru.md#app_d_secure_mvp_execution_environment)_; _[Методология: MVP](./20260325-research-report-methodology-main-ru.md#method_agent_execution_mvp)_ |
| Поведенческие риски | _[Приложение D: риски](./20260325-research-appendix-d-security-observability-ru.md#app_d_org_behavioral_risk_factors)_ |
| AI TRiSM: управление доверием и рисками | _[Приложение D: AI TRiSM](./20260325-research-appendix-d-security-observability-ru.md#app_d_ai_trism_trust_management)_ |

### Справочные документы

| Вопрос | Документ |
| --- | --- |
| Единый реестр источников | [Сводный реестр](#app_a_sources_registry) (этот документ) |
| Сжатый C-level обзор | [Резюме](./20260331-research-executive-unified-ru.md) |
| Бюджетный риск и организационная зрелость | _[Сайзинг: риски](./20260325-research-report-sizing-economics-main-ru.md#sizing_budget_risks_mitigation)_ |
| Shadow GenAI и маршрутизация моделей в маркетинге | _[Приложение B: Shadow GenAI](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_shadow_genai_marketing_model_routing)_ |
| Артефакты PR-веток для агентного контура | _[Приложение B: PR-артефакты](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_reference_agent_pr_artifacts)_ |

## Сводный реестр источников {: #app_a_sources_registry }

Ниже — **канонический перечень** внешних ссылок комплекта, сгруппированный по темам. Каждый URL приведён **один раз** и может использоваться как базовая опора для ссылок и перепроверки тезисов внутри пакета.

### Инженерия обвязки и мультиагентная разработка {: #app_a_wrapper_engineering_multiagent }

- [Хабр — Инженер будущего строит обвязку для агентов](https://habr.com/ru/articles/1005032/)
- [Martin Fowler — Harness Engineering (Thoughtworks)](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)
- [OpenAI — Harness engineering](https://openai.com/ru-RU/index/harness-engineering/)
- [Anthropic — Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Anthropic — Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)

### OWASP GenAI Security, тестирование и адаптации на русском {: #app_a_owasp_genai_security_ru }

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

### Безопасность GenAI, OWASP и сигналы рынка (TCO / риски) {: #app_a_genai_security_owasp_tco }

- [CodeWall — разбор red team: McKinsey AI platform](https://codewall.ai/blog/how-we-hacked-mckinseys-ai-platform)
- [GitHub — NVIDIA Garak (сканер для LLM, только изолированные стенды)](https://github.com/NVIDIA/garak)
- [OpenAI — приобретение PromptFoo (контекст рынка тестирования)](https://openai.com/index/openai-to-acquire-promptfoo/)
- [Kaspersky — пресс-релиз: угрозы под видом популярных ИИ-сервисов (бенчмарк тренда)](https://www.kaspersky.com/about/press-releases/kaspersky-chatgpt-mimicking-cyberthreats-surge-115-in-early-2025-smbs-increasingly-targeted)
- [The Hacker News — TeamPCP: LiteLLM и Telnyx скомпрометированы через PyPI (март 2026)](https://thehackernews.com/2026/03/teampcp-pushes-malicious-telnyx.html)
- [Datadog Security Labs — LiteLLM and Telnyx compromised on PyPI: Tracing the TeamPCP supply chain campaign](https://securitylabs.datadoghq.com/articles/litellm-compromised-pypi-teampcp-supply-chain-campaign/)
- [Коммерсантъ — рынок и атаки на ИИ-системы (журналистский контекст)](https://www.kommersant.ru/doc/8363105)

### Угрозы GenAI и иллюстративные материалы третьих лиц (не реклама) {: #app_a_genai_threats_third_party_materials }

- [Securelist — webinar: AI agents vs. prompt injections](https://securelist.com/webinars/ai-agents-vs-prompt-injections/)
- [Kaspersky — press release: training Large Language Models Security (описание программы)](https://www.kaspersky.com/about/press-releases/kaspersky-introduces-a-new-training-large-language-models-security)
- [Kaspersky Blog — How LLMs can be compromised in 2025](https://www.kaspersky.com/blog/new-llm-attack-vectors-2025/54323/)
- [Kaspersky Blog — Agentic AI security measures and OWASP ASI Top 10](https://www.kaspersky.com/blog/top-agentic-ai-risks-2026/29988/)
- [Kaspersky Resource Center — What Is Prompt Injection?](https://www.kaspersky.com/resource-center/threats/prompt-injection)

### Нормативные и стратегические материалы {: #app_a_regulatory_strategic_materials }

- [Официальное опубликование — Приказ Роскомнадзора от 19.06.2025 № 140 (обезличивание ПДн)](http://publication.pravo.gov.ru/document/0001202508010002)
- [ACSOUR — обязанность операторов передавать анонимизированные ПДн в ГИС (152-ФЗ)](https://acsour.com/en/news-and-articles/tpost/2g13ahnab1-mandatory-anonymized-personal-data-shari)
- [NIST AIRC — Roadmap for the AI Risk Management Framework](https://airc.nist.gov/airmf-resources/roadmap)
- [NIST — AI RMF to ISO/IEC 42001 Crosswalk (PDF)](https://airc.nist.gov/docs/NIST_AI_RMF_to_ISO_IEC_42001_Crosswalk.pdf)
- [Известия (EN) — создание офисов внедрения ИИ](https://en.iz.ru/en/node/1985740)
- [DataGuidance — поправки к национальной стратегии развития ИИ РФ](https://www.dataguidance.com/news/russia-president-issues-amendments-national-ai)
- [Фонтанка — проект закона о госрегулировании ИИ (Минцифры, 18.03.2026)](https://www.fontanka.ru/2026/03/18/76318717/)
- [ISO/IEC 42001:2023 — Artificial intelligence management system](https://www.iso.org/standard/81230.html)
- [NIST — AI RMF: Generative AI Profile (NIST.AI.600-1, 2024)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)

### Данные и стратегические сигналы {: #app_a_data_strategic_signals }

- [Gartner — пресс-релиз: нехватка AI-ready data подрывает ИИ-проекты (26.02.2025)](https://www.gartner.com/en/newsroom/press-releases/2025-02-26-lack-of-ai-ready-data-puts-ai-projects-at-risk)

### Российский рынок GenAI, сегменты и AI TRiSM (публичные ссылки) {: #app_a_russian_genai_market_segments_trism }

- [Хабр — red_mad_robot: анонс тренд-репорта и события в Сколково](https://habr.com/ru/companies/redmadrobot/articles/879750/)
- [red_mad_robot — раздел «Исследования»](https://redmadrobot.ru/issledovaniya-1/)
- [red_mad_robot — мероприятие: тренд-репорт рынка GenAI (2025)](https://redmadrobot.ru/meropriyatiya/trend-report-rynok-gen-ai-v-2025-godu/)
- [Gartner — AI TRiSM (глоссарий)](https://www.gartner.com/en/information-technology/glossary/ai-trism)
- [McKinsey — The state of AI](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai)
- [РБК — объём рынка B2B LLM в России (MTS AI)](https://www.rbc.ru/technology_and_media/26/11/2024/67449d909a79478a2052d490)
- [Сколково — событие «Состояние рынка GenAI в России и в мире» (12.02.2025)](https://www.skolkovo.ru/events/120225-sostoyanie-rynka-genai-v-rossii-i-v-mire/)
- [Ведомости — рынок облачных сервисов с GPU (МНИАП)](https://www.vedomosti.ru/technology/articles/2024/12/11/1080600-rinok-oblachnih-servisov-s-gpu-virastet)

### Подкасты (первичная запись): AI-First, red_mad_robot {: #app_a_podcasts_ai_first_red_mad_robot }

- [YouTube — «Ноосфера» #129: Илья Самофеев (red_mad_robot), AI-First / AI-Native](https://www.youtube.com/watch?v=jTKhg1jqF_M)

### Стек инференса (MOSEC, vLLM) и открытая документация {: #app_a_inference_stack_mosec_vllm_docs }

- [MOSEC — документация](https://mosecorg.github.io/mosec/index.html)
- [mosecorg/mosec (GitHub)](https://github.com/mosecorg/mosec)
- [vLLM — OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)
- [infinity-emb — документация](https://github.com/michaelfeil/infinity)

### Отраслевые бенчмарки и исследования {: #app_a_industry_benchmarks }

- [Bain & Company — The Three Layers of an Agentic AI Platform (апрель 2026)](https://www.bain.com/insights/the-three-layers-of-an-agentic-ai-platform/)
- [Schema-Guided Reasoning (SGR)](https://abdullin.com/schema-guided-reasoning/)
- [Model Context Protocol — официальный сайт](https://modelcontextprotocol.io/)
- [LangGraph — документация](https://docs.langchain.com/)

### Экономика, рынок, enterprise AI {: #app_a_economics_market_enterprise_ai }

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

### Оценка качества и мониторинг (LangSmith) {: #app_a_quality_monitoring_langsmith }

- [LangChain Docs — Evaluation concepts (LangSmith)](https://docs.langchain.com/langsmith/evaluation-concepts)
- [LangSmith — Online evaluations (how-to)](https://docs.smith.langchain.com/observability/how_to_guides/online_evaluations)

### Исследования (edge–cloud routing, агентная память и обучение; ориентиры НИОКР) {: #app_a_research_edge_cloud_routing_memory }

- [arXiv — PRISM: Privacy-Aware Routing for Cloud-Edge LLM Inference](https://arxiv.org/html/2511.22788v1)
- [arXiv — HybridFlow: Resource-Adaptive Subtask Routing for Edge-Cloud LLM Inference](https://arxiv.org/html/2512.22137v4)
- [arXiv — Moonshot AI: ускорение синхронного RL](https://arxiv.org/pdf/2511.14617)
- [arXiv — Agent0: co-evolving curriculum and executor agents](https://arxiv.org/pdf/2511.16043)
- [arXiv — MoE на стеке AMD (IBM, Zyphra и др.)](https://arxiv.org/pdf/2511.17127)
- [arXiv — General Agentic Memory (GAM)](https://arxiv.org/pdf/2511.18423)

### Edge-инференс и оптимизации памяти (Apple Silicon, локальные модели) {: #app_a_edge_inference_memory_optimizations }

- [Apple ML Research — LLM in a Flash: Efficient Large Language Model Inference with Limited Memory (ACL 2024)](https://machinelearning.apple.com/research/efficient-large-language)
- [GitHub — matt-k-wong/mlx-flash (реализация для MLX, март 2026)](https://github.com/matt-k-wong/mlx-flash)
- [arXiv — LLM in a Flash: Efficient Large Language Model Inference with Limited Memory (оригинальная статья, 2312.11514)](https://arxiv.org/html/2312.11514v3)
- [Apple Developer — WWDC 2025: Explore large language models on Apple silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/)

### Агентная память и модели (ориентиры НИОКР и прайсинга, не строка КП) {: #app_a_agent_memory_benchmarks }

- [Anthropic — Pricing](https://www.anthropic.com/pricing)

### Облачные провайдеры и тарифы (РФ) {: #app_a_cloud_providers_tariffs_russia }

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

### Российские облачные GPU: актуальные тарифы и аналитика (2026) {: #app_a_russian_cloud_gpu_pricing_2026 }

- [Cloud.ru — Тарифы «Evolution Compute GPU», Приложение №7G.EVO.1 (январь 2026)](https://cloud.ru/documents/tariffs/evolution/evolution-compute-gpu)
- [Elish Tech — Почасовая аренда GPU A100 vs H100: что выгоднее в 2026 году](https://www.elishtech.com/arenda-gpu-a100-vs-h100-2026/)
- [Elish Tech — Где арендовать GPU-серверы дешевле и выгоднее: сравнение рынка в России и за рубежом](https://www.elishtech.com/gpu-server-rent-market-comparison/)
- [Yandex Cloud — GPU (графические ускорители), документация](https://yandex.cloud/ru/docs/compute/concepts/gpus)
- [Yandex Cloud — Прайс-лист (текущие тарифы)](https://yandex.cloud/ru/prices)
- [Selectel — Cloud GPU (облачные серверы с GPU)](https://selectel.ru/services/cloud/servers/gpu)
- [Selectel — Новости: новые конфигурации GPU-серверов от 50 руб./час](https://myseldon.com/ru/news/index/262426281)

### Публичные веса с нестандартной лицензией {: #app_a_public_weights_licensing }

- [arXiv — Cache Me If You Must (KV-quantization), 2501.19392](https://arxiv.org/abs/2501.19392)
- [Hugging Face — карточка модели YandexGPT-5-Lite-8B-instruct](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct)
- [Hugging Face — LICENSE (YandexGPT-5-Lite-8B), сырой текст соглашения](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct/raw/main/LICENSE)
- [Yandex Research — принятые к ICML 2025 (список, в т.ч. KV-кэш)](https://research.yandex.com/blog/papers-accepted-to-icml-2025)
- [Yandex Research — обзор направлений работ (2025)](https://research.yandex.com/blog/yandex-research-in-2025)

### Открытые модели ai-sage (GigaChat и спутники) {: #app_a_open_models_ai_sage_gigachat }

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

### Примерные расчёты токенов (портал поддержки, агрегаторы и обзоры прайсов) {: #app_a_token_estimates_pricing }

- [Хабр — обзор цен на токены](https://habr.com/ru/articles/1000058/)
- [Хабр — гид по топ-20 нейросетям для текстов (в т.ч. цены)](https://habr.com/ru/articles/948672/)
- [LLMoney — калькулятор цен токенов LLM](https://llmoney.ru)
- [Портал поддержки Comindware](https://support.comindware.com/)
- [VC.ru — гайд по тарифам Claude и доступу из России](https://vc.ru/ai/2757771-tarify-claude-2026-gayd-po-planam-i-dostupu-iz-rossii)

### Формат TOON и оптимизация токенов {: #app_a_toon_format_token_optimization }

- [Спецификация TOON](https://toonformat.dev/)
- [Tensorlake: бенчмарки](https://www.tensorlake.ai/blog/toon-vs-json)
- [Systenics: экономия токенов](https://systenics.ai/blog/2026-01-24-toon-vs-json-how-token-oriented-object-notation-reduces-llm-token-costs)

### Иллюстративные ориентиры нагрузки (публичные интервью, финсектор) {: #app_a_load_benchmarks_financial }

- [CIO — интервью: чат-бот, масштаб обращений и сценарии](https://cio.osp.ru/articles/5455)
- [«Открытые системы» — RAG и LLM для поддержки операционистов](https://www.osp.ru/articles/2025/0324/13059305)

### Инструменты разработки с ИИ (ориентиры) {: #app_a_ai_dev_tools_benchmarks }

- [OpenWork (different-ai/openwork)](https://github.com/different-ai/openwork)
- [OpenCode](https://opencode.ai/)
- [OpenCode — документация (Intro)](https://opencode.ai/docs)
- [OpenCode — Ecosystem](https://opencode.ai/docs/ecosystem/)
- [OpenCode Zen](https://opencode.ai/docs/zen)
- [OpenRouter](https://openrouter.ai/)
- [OpenRouter — журналирование и политики провайдеров](https://openrouter.ai/docs/guides/privacy/logging)

### Инференс и VRAM: бенчмарки, движок и калькуляторы {: #app_a_inference_vram_tools_sizing_nav }

- [apxml.com — VRAM calculator](https://apxml.com/tools/vram-calculator)
- [vLLM — документация](https://docs.vllm.ai/)
- [MLCommons — Inference Datacenter](https://mlcommons.org/benchmarks/inference-datacenter/)

### Финансовая и инфраструктурная база (FinOps/TCO/железо) {: #app_a_finops_tco_infrastructure }

- [Medium — Qwen 3.5 35B A3B (AgentNativeDev)](https://agentnativedev.medium.com/qwen-3-5-35b-a3b-why-your-800-gpu-just-became-a-frontier-class-ai-workstation-63cc4d4ebac1)
- [Hugging Face — Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Introl — планирование мощностей ИИ-инфраструктуры (прогнозы, McKinsey в обзоре)](https://introl.com/blog/ai-infrastructure-capacity-planning-forecasting-gpu-2025-2030)
- [Introl — финансирование CapEx/OpEx и инвестиции в GPU](https://introl.com/blog/ai-infrastructure-financing-capex-opex-gpu-investment-guide-2025)
- [PitchGrade — AI Infrastructure Primer](https://pitchgrade.com/research/ai-infrastructure-primer)
- [OpenAI — Prompt caching (снижение стоимости повторяющегося контекста)](https://platform.openai.com/docs/guides/prompt-caching)
- [Slyd — калькулятор TCO (on-prem и облако)](https://slyd.com/resources/tco-calculator)
- [Runpod — LLM inference optimization playbook (throughput)](https://www.runpod.io/articles/guides/llm-inference-optimization-playbook)
- [SWFTE — экономика частного AI / on-prem](https://www.swfte.com/blog/private-ai-enterprises-onprem-economics)

### Исследования рынка (зрелость GenAI, не технический сайзинг) {: #app_a_market_research_genai_maturity }

- [RB.RU — 93% команд в маркетинге используют ИИ (обзор исследования CMO Club × red_mad_robot, 2025)](https://rb.ru/news/93-komand-v-marketinge-ispolzuyut-ii-nejroseti-pomogayut-s-kontentom-no-im-pochti-ne-doveryayut-strategiyu-i-byudzhety/)
- [Telegram — CMO Club Russia: анонс исследования GenAI в маркетинге (red_mad_robot × CMO Club, 2025)](https://t.me/cmoclub/197)

### Наблюдаемость и телеметрия {: #app_a_observability_telemetry }

- [OpenInference — инструментирование ИИ для OpenTelemetry](https://arize-ai.github.io/openinference/)
- [Arize Phoenix — документация](https://docs.arize.com/phoenix)
- [LangSmith — документация](https://docs.smith.langchain.com/)
- [Langfuse — документация observability / tracing](https://langfuse.com/docs/observability/get-started)
- [OpenTelemetry — OpenTelemetry for Generative AI (блог)](https://opentelemetry.io/blog/2024/otel-generative-ai)
- [OpenTelemetry — Semantic conventions for generative AI metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)
- [OpenTelemetry — Semantic conventions for generative client AI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)

### Публичные материалы Ozon Tech {: #app_a_ozon_tech_materials }

- [GitHub — организация ozontech (открытые репозитории)](https://github.com/ozontech)
- [Хабр — Ozon Tech: анонс ML&DS Meetup (MLOps, программа докладов)](https://habr.com/ru/companies/ozontech/articles/768734/)
- [Хабр — Ozon Tech: пересборка конструктора чат-ботов (Bots Factory, no-code, масштаб)](https://habr.com/ru/companies/ozontech/articles/834812/)
- [Хабр — Ozon Tech: Query Prediction, ANN и обратный индекс](https://habr.com/ru/companies/ozontech/articles/990180/)

### Методологии внедрения и отраслевые практики {: #app_a_implementation_methodologies_industry_practices }

- [GitHub — redmadrobot-rnd/mcp-registry](https://github.com/redmadrobot-rnd/mcp-registry)
- [Habr — red_mad_robot: кейс RAG для ФСК](https://habr.com/ru/companies/redmadrobot/articles/892882/)
- [Habr — red_mad_robot: MCP Tool Registry и автоматизация RAG](https://habr.com/ru/companies/redmadrobot/articles/982004/)
- [InOrg — бесшовная передача (seamless handover) в модели BOT](https://inorg.com/blog/from-build-to-transfer-key-success-factors-a-seamless-bot-model-transition)
- [Just AI — корпоративный GenAI (упоминается как практикующий вендор)](https://just-ai.com/ru/)
- [Luxoft — модель Build–Operate–Transfer (BOT)](https://www.luxoft.com/blog/master-the-build-operate-transfer-bot-model-with-luxoft)
- [Ведомости — CTO AI red_mad_robot (Влад Шевченко)](https://www.vedomosti.ru/technologies/trendsrub/articles/2026/03/11/1181757-ii-uskoril-kod)

### Публичные материалы MWS / MTS AI {: #app_a_public_materials_mws_mts_ai }

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

### Публичные материалы финсектора (паттерны внедрения) {: #app_a_financial_sector_public_materials_patterns }

- [Хабр — MLOps и каскады моделей](https://habr.com/ru/companies/alfa/articles/801893/)
- [Хабр — автоматизация обучения и обновления моделей](https://habr.com/ru/companies/alfa/articles/852790/)
- [Хабр — классификация текстов диалогов на большом числе классов](https://habr.com/ru/companies/alfa/articles/900538/)
- [Хабр — обновление LLM: instruction following и tool calling](https://habr.com/ru/companies/tbank/articles/979650/)

### Telegram-каналы и посты {: #app_a_telegram_channels }

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

### Посты NeuralDeep {: #app_a_neuraldeep_posts }

- [Agentic RAG / SGR](https://t.me/neuraldeep/1605)
- [ETL, эмбеддинги, реранкеры, фреймворки RAG, eval, безопасность](https://t.me/neuraldeep/1758)

### Посты @ai_archnadzor {: #app_a_ai_archnadzor_posts }

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
- [Обзор локального стека наблюдаемости (канал @ai_archnadzor)](https://t.me/ai_archnadzor/177)
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

### Habr и статьи по инженерии RAG {: #app_a_habr_rag_engineering_articles }

- [Raft на Habr — чанкование](https://habr.com/ru/companies/raft/articles/954158/)

### Препринты (arXiv) {: #app_a_arxiv_preprints }

- [Google — Deep-Thinking Ratio (DTR), 2602.13517](https://arxiv.org/pdf/2602.13517)
- [Oppo AI — Search More, Think Less (SMTL), 2602.22675](https://arxiv.org/pdf/2602.22675)
- [Meta (Экстремистская организация, запрещена в РФ), OpenAI, xAI — непрерывное улучшение моделей (чаты), 2603.01973](https://arxiv.org/pdf/2603.01973)
- [Microsoft Research — безопасность агентов с внешними инструментами, 2603.03205](https://arxiv.org/pdf/2603.03205)
- [Accenture — Memex(RL), 2603.04257](https://arxiv.org/pdf/2603.04257)
- [SkillNet, 2603.04448](https://arxiv.org/pdf/2603.04448)
- [Databricks — KARL, 2603.05218](https://arxiv.org/pdf/2603.05218)
- [OpenAI — контроль рассуждения со скрытыми шагами, 2603.05706](https://arxiv.org/pdf/2603.05706)
- [Princeton — непрерывное обучение из взаимодействия с агентом, 2603.10165](https://arxiv.org/pdf/2603.10165)

### Продукты и блоги (эмбеддинги, M365; справочно) {: #app_a_products_blogs_embeddings_m365_reference }

- [Google — Gemini Embedding 2 (блог)](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
- [Microsoft — Copilot Cowork (блог Microsoft 365)](https://www.microsoft.com/en-us/microsoft-365/blog/2026/03/09/copilot-cowork-a-new-way-of-getting-work-done/)

### Открытые проекты {: #app_a_open_third_party_projects }

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

### Регулирование (проектный контур 2026) {: #app_a_regulation_project_context_2026 }

- [Портал НПА — проект федерального закона (ID 166424)](https://regulation.gov.ru/projects#npa=166424)

## Дополнительное чтение {: #app_a_additional_sources_backlog }

Для расширенного круга чтения, внешнего бенчмаркинга и обновления повестки используйте _Приложение F «[Дополнительное чтение](./20260325-research-appendix-f-extended-reading-ru.md)»_.

!!! note "Как пользоваться дополнительным чтением"

    Для ссылок и перепроверки тезисов внутри самого комплекта используйте **сводный реестр источников**.
    
    Для внешнего бенчмарка, дополнительного кейса или обновления повестки переходите в **Приложение F**.
