# Приложение C. Имеющиеся наработки CMW (состав и границы)

**Дата комплекта:** 2026-03-25  
**Статус:** утверждённый комплект материалов для руководства (v1)

## Обзор комплекта

Раздел отвечает на вопрос **что именно покрывает референс-стек CMW сегодня**: границы модулей, роли в архитектуре и различие между **поставляемыми артефактами** и **методологическими рекомендациями**. Критерии приёмки при передаче — в «Приложение B: отчуждение ИС и кода (KT, IP)»; модель внедрения и экономика — в двух основных отчётах. **Пересчёт валюты** для смет — [приложение A](./20260325-research-appendix-a-index-ru.md#app_a_fx_policy).

## Связанные документы

- «Приложение A: обзор и ведомость документов»
- «Основной отчёт: методология внедрения и разработки»
- «Основной отчёт: сайзинг и экономика»
- «Приложение B: отчуждение ИС и кода (KT, IP)»
- «Приложение D: безопасность, комплаенс и observability»

## Обзор текущей архитектуры CMW

В данном документе описывается методология внедрения и управления инфраструктурой ИИ в экосистеме CMW с фокусом на российские облачные провайдеры и локальный инференс. Архитектура основана на **модульном, контейнеризованном подходе**, объединяющем RAG-движок (**корпоративный RAG-контур**) с серверами инференса (**сервер инференса MOSEC**, **инференс на базе vLLM**), агентным слоем для CMW Platform (**агентный слой платформы (CMW Platform)**, по сценарию) и интеграцией с российскими облачными платформами.

Детальная экономика сайзинга, CapEx/OpEx и TCO — в сопутствующем резюме **Оценка сайзинга, КапЭкс и ОпЭкс для клиентов**; **профили on-prem GPU** (референсные consumer **24 ГБ**, **RTX 4090 48 ГБ** в коммерческой аренде, **RTX PRO 6000 Blackwell 96 ГБ** в проектах CMW) — в том же отчёте, подраздел **«Профиль on-prem GPU в проектах CMW»** (чтение **сначала** по **топологии ёмкости** и классам бенчмарков в том же разделе). Технические детали развёртывания конкретных модулей — в публичной документации соответствующих программных компонентов экосистемы.

**Сводные доли и барьеры** использования GenAI в маркетинге крупных брендов РФ (опрос **red_mad_robot × CMO Club Russia**, **2025**), включая **две различные** линии **~43 %** (галлюцинации vs утечка данных), зафиксированы в сопутствующем резюме **Оценка сайзинга, КапЭкс и ОпЭкс для клиентов** (раздел **«Российский рынок»**) и увязаны с _«[Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__org_barriers_risk_survey_2025)»_.

**Ключевые методологические принципы:**
- **Разделение ответственности:** Отдельные слои для обработки данных, поиска, инференса и доставки API.
- **Гибридный поиск:** Объединение векторного поиска (плотного) с ключевым (разреженным) для оптимальной точности.
- **Агентная архитектура:** Использование агентов LangChain для динамического вызова инструментов и рассуждения.
- **Гибкость инфраструктуры:** Поддержка как MOSEC (единый сервер), так и vLLM (распределенные инстансы) бэкендов инференса.
- **Российская суверенность:** Приоритет российских облачных провайдеров (Cloud.ru, Yandex Cloud, SberCloud, MWS GPT, Selectel и др. по контуру заказчика) для обеспечения соответствия требованиям о данных и инфраструктуре; дополнительно возможен путь **аренды GPU** у российских IaaS/dedicated-поставщиков — сводная матрица классов продуктов и ссылок на прайсы в сопутствующем резюме **Оценка сайзинга, КапЭкс и ОпЭкс для клиентов**, подраздел **[«Аренда GPU (IaaS РФ): дополнительные поставщики и ориентиры»](./20260325-research-report-sizing-economics-main-ru.md#sizing_gpu_rental_iaas_providers)**.
- **Недоверенное исполнение:** при сценариях с кодом и широким набором инструментов заказчик проектирует **изоляцию среды и сетевые политики** отдельно от перечня модулей в таблице ниже; ориентиры — в _«[Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__trust_boundary_agent_environment)»_ и _«[Приложение D — модель риска, паттерны среды и минимальный состав платформы](./20260325-research-appendix-d-security-observability-ru.md#app_d__risk_model_platform_patterns)»_.
- **Организационная зрелость:** наличие модулей **корпоративный RAG-контур**, **MOSEC**, **vLLM**, **агентный слой платформы (CMW Platform)** **не заменяет** оргпроцессы, обучение команд и операционную модель внедрения; см. _«[Основной отчёт: методология — Стратегия внедрения ИИ и организационная зрелость](./20260325-research-report-methodology-main-ru.md#method_ai_strategy_org_maturity)»_.

Публичные кейсы и платформенные описания крупных игроков рынка РФ (в том числе **МТС Web Services / MTS AI**) полезны как **каталог воспроизводимых инженерных и организационных идей** — гибридный ретрив, жизненный цикл агентов, интеграционные протоколы, аттестованные контуры для ПДн. Выбор поставщика и архитектуры остаётся за заказчиком по тендеру, требованиям ИБ и экономической модели; интегратор может опираться на такие материалы при выравнивании ожиданий и комплекта отчуждения.

### Связанные проекты

**Экосистема CMW включает ключевые программные проекты (имена условные, обозначают роли компонентов):**

| Проект | Назначение | Архитектура |
|-------------|-----------|--------------|
| **корпоративный RAG-контур** | RAG-движок для поиска и генерации ответов | LangChain, Gradio, ChromaDB |
| **сервер инференса MOSEC** | Унифицированный сервер инференса (MOSEC) | Эмбеддинг, Реранкер, Охранник на одном порту |
| **инференс на базе vLLM** | Распределённый сервер инференса (vLLM) | LLM, KV-кэш, непрерывная комплектная обработка |
| **агентный слой платформы (CMW Platform)** | AI-агент для управления CMW Platform | 49 инструментов (27 CMW + 22 утилиты) |

### CMW Platform Agent: Агент для управления сущностями

**Проект:** **агентный слой платформы (CMW Platform)** (отдельный компонент экосистемы CMW, см. таблицу выше).

**Назначение:** ИИ-агент для создания и управления сущностями в CMW Platform на естественном языке.

**Архитектура:**
```
Слой интерфейса (Gradio)
    ↓
Ядро агента (оркестрация, доступ к LLM, состояние сессий)
    ↓
Слой инструментов (49)
    ├── Инструменты CMW Platform (27)
    │   ├── Приложения и шаблоны (6)
    │   ├── Атрибуты (15)
    │   └── Шаблоны и записи (6)
    └── Утилитарные инструменты (22)
        ├── Поиск и исследования (веб, Wikipedia, arXiv)
        ├── Исполнение кода (Python, Bash, SQL)
        ├── Анализ файлов (CSV, Excel, PDF, OCR)
        └── Математические операции
```

**Поддерживаемые поставщики LLM:**
- OpenRouter (по умолчанию) — 100K–2M токенов, полная поддержка инструментов
- Google Gemini — 1M+ токенов, сильные рассуждения
- Groq — низкая задержка инференса, 131K токенов
- Hugging Face — локальные и облачные модели
- Mistral — европейские модели
- GigaChat — российские модели

**Контекст для внедрения у заказчика в РФ:** значение «по умолчанию» для OpenRouter отражает **удобство исходной конфигурации разработки** (единый API, быстрые эксперименты), а не рекомендацию **промышленного** контура при персональных данных и требованиях суверенитета. В таких случаях целевой выбор инференса — **API российских облаков**, **on-prem** или иной маршрут из блока Compliance ниже и из сопутствующего резюме **Оценка сайзинга, КапЭкс и ОпЭкс для клиентов**.

**Ключевые возможности:**
- **Многоходовый диалог** — управление памятью в стеке LangChain
- **Потоковая выдача** — ответ по токенам через потоковый API
- **Изоляция сессий** — разделение пользователей и освобождение ресурсов
- **Локализация** — полная поддержка английского и русского
- **Восстановление после ошибок** — векторная классификация сбоев
- **Учёт токенов и бюджета** — фактический расход токенов и оценка стоимости

В референс-стеке **агентный слой платформы (CMW Platform)** опирается на **краткосрочную** память диалога в стеке LangChain и на **корпоративный RAG-контур** для извлечения из индекса; это **не** заявляет паритета с исследовательскими системами **долговременной переносимой** памяти между задачами. Публикации уровня General Agentic Memory (GAM) задают **направление эволюции** (накопление знаний и исследовательский цикл поверх памяти), которое при проектировании согласуют с политикой данных и observability — см. «_[Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__pack_overview)_» и «_[Основной отчёт: методология внедрения и разработки](./20260325-research-report-methodology-main-ru.md#method_pack_overview)_»; первоисточник: [arXiv — GAM](https://arxiv.org/pdf/2511.18423). **Memex(RL)** (Accenture) описывает **другой** класс решений — **индексированную** память с **обучением с подкреплением** для решений о разгрузке контекста, заголовках и извлечении записей ([arXiv — Memex(RL)](https://arxiv.org/pdf/2603.04257); см. также «_[Основной отчёт: методология внедрения и разработки](./20260325-research-report-methodology-main-ru.md#method_ai_agent_memory_context)_»); это **не** компонент поставки **агентный слой платформы (CMW Platform)**.

**Инструменты для отчуждения:**
| Компонент | Артефакт | Назначение |
|-----------|----------|-----------|
| Документация | Руководство для AI-агентов | Инструкции для агентов и контрибьюторов |
| Код | **агентный слой платформы (CMW Platform)** | 49 инструментов (платформа CMW и утилиты) |
| Тесты | Пакет поведенческих тестов | Регрессия и контракты инструментов |
| Конфигурация | Шаблон переменных окружения | Параметры окружения без секретов |

---

## Источники

- Полный консолидированный реестр — см. [Приложение A: обзор и ведомость документов](./20260325-research-appendix-a-index-ru.md#app_a_sources_registry).

### Стек инференса (MOSEC, vLLM) и открытая документация

- [MOSEC — документация](https://mosecorg.github.io/mosec/index.html)
- [mosecorg/mosec (GitHub)](https://github.com/mosecorg/mosec)
- [vLLM — OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)
- [сервер инференса MOSEC — README проекта (пример публичного зеркала)](https://github.com/arterm-sedov/cmw-mosec)

### Облачные провайдеры и тарифы (РФ)

- [Cloud.ru — Evolution Foundation Models (продукт)](https://cloud.ru/products/evolution-foundation-models)
- [Cloud.ru — тарифы Evolution Foundation Models](https://cloud.ru/documents/tariffs/evolution/foundation-models)
- [Yandex AI Studio — доступные генеративные модели](https://aistudio.yandex.ru/docs/ru/ai-studio/concepts/generation/models.html)
- [Yandex AI Studio — прайсинг](https://aistudio.yandex.ru/docs/ru/ai-studio/pricing.html)
- [AKM.ru — Yandex B2B Tech и языковые модели на рынке РФ](https://www.akm.ru/eng/press/yandex-b2b-tech-has-opened-access-to-the-largest-language-model-on-the-russian-market/)
- [Сбер — портал GigaChat API](https://developers.sber.ru/portal/products/gigachat-api)
- [Сбер — юридические тарифы GigaChat](https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs)
- [Selectel — Foundation Models Catalog](https://selectel.ru/services/cloud/foundation-models-catalog)
- [MWS — MWS GPT (продукт)](https://mws.ru/mws-gpt/)
- [MWS — тарифы MWS GPT](https://mws.ru/docs/docum/cloud_terms_mwsgpt_pricing.html)
- [VK Cloud — машинное обучение в облаке (документация)](https://cloud.vk.com/docs/ru/ml)
- [Google — условия использования Gemma](https://ai.google.dev/gemma/terms)
- [Google — Gemini Embedding 2 (блог: нативно мультимодальные эмбеддинги)](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
- [NVIDIA — Nemotron 3 (обзор семейства)](https://research.nvidia.com/labs/nemotron/Nemotron-3/)
- [Hugging Face — nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)
- [Hugging Face — nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)
- [Hugging Face — zai-org/GLM-4.6](https://huggingface.co/zai-org/GLM-4.6)
- [Hugging Face — zai-org/GLM-4.7](https://huggingface.co/zai-org/GLM-4.7)
- [Hugging Face — zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
- [Hugging Face — zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5)
- [Hugging Face — openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- [Hugging Face — openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
- [Hugging Face — MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)
- [Hugging Face — moonshotai/Kimi-K2-Base](https://huggingface.co/moonshotai/Kimi-K2-Base)
- [Hugging Face — организация moonshotai](https://huggingface.co/moonshotai)
- [Hugging Face — deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- [Hugging Face — deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
- [Hugging Face — организация Qwen](https://huggingface.co/Qwen)

### Зарубежные frontier-модели (справочно для сравнения качества, не baseline КП РФ)

- [Anthropic — Introducing Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)
- [Anthropic — Introducing Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)
- [Claude Docs — What's new in Claude 4.6](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-6)
- [Claude Docs — Models overview](https://platform.claude.com/docs/en/about-claude/models/overview)

### Открытые проекты

Помимо перечисленных ниже фреймворков, в открытом доступе публикуют **референсные каркасы** под RAG и агентов: лаборатория AI R&D **red_mad_robot** описывает **MCP Tool Registry** (центральный реестр MCP-серверов для связки LLM с данными и инструментами) и выкладывает код на GitHub — это **ориентир для разведки архитектуры** и политики подключения MCP, а не компонент поставки **корпоративный RAG-контур** / **агентный слой платформы (CMW Platform)** без отдельного решения заказчика. Та же лаборатория совместно с **CMO Club Russia** публикует **бенчмарк зрелости GenAI в маркетинге** крупных брендов РФ (2025); контекст для TOM и переговоров — в _«[Основной отчёт: методология внедрения — GenAI в маркетинговых командах](./20260325-research-report-methodology-main-ru.md#method_genai_marketing_teams)_»_. Отдельно на рынке представлены **управляемые песочницы** для исполнения агентского кода (**E2B**, **Modal**, **Daytona** и др.) с разными моделями сессий, сети и размещения — для сравнения и бенчмарков см. _«[Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__managed_sandboxes_benchmarks)»_; в поставку референс-стека CMW они **не** входят по умолчанию.

**Рыночные RAG-цепочки интеграторов (вне SKU CMW):** в открытых обзорах и кейсах встречаются **собственные** конвейеры: декомпозиция запроса (**query decomposition**), гипотетические документы для ретрива (**HyDE**), **двойной** вызов с порогом сходства (**DCD**), **schema-guided** рассуждения (**SGR**), извлечение структуры из PDF (**Marker**, **Docling** и аналоги), хранение метаданных в **PostgreSQL** и векторный слой (**Qdrant**, **Chroma** и др.). Это **иллюстрация** зрелости рынка интеграции, а не требование воспроизвести все приёмы в **корпоративный RAG-контур**; пересечение с референс-стеком CMW оценивают по целевому threat model и TOM. Паттерны защиты и **Model Context Protocol (MCP)** — _«[Приложение D: паттерны промышленного RAG](./20260325-research-appendix-d-security-observability-ru.md#app_d__industrial_rag_protection_patterns)_»_ и _«[MCP, мультиагентная маршрутизация](./20260325-research-appendix-d-security-observability-ru.md#app_d__mcp_multiagent_routing_skills)_»_.

- [langchain-ai/langchain — text-splitters](https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters)
- [langgenius/dify](https://github.com/langgenius/dify/)
- [run-llama/llama_index](https://github.com/run-llama/llama_index)
- [GitHub — redmadrobot-rnd/mcp-registry](https://github.com/redmadrobot-rnd/mcp-registry)
- [Хабр — red_mad_robot: MCP Tool Registry и автоматизация RAG](https://habr.com/ru/companies/redmadrobot/articles/982004/)
- [Telegram — CMO Club Russia: анонс исследования GenAI в маркетинге (red_mad_robot × CMO Club, 2025)](https://t.me/cmoclub/197)
- [RB.RU — 93% команд в маркетинге используют ИИ (обзор исследования CMO Club × red_mad_robot, 2025)](https://rb.ru/news/93-komand-v-marketinge-ispolzuyut-ii-nejroseti-pomogayut-s-kontentom-no-im-pochti-ne-doveryayut-strategiyu-i-byudzhety/)
