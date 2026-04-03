---
title: 'Приложение C. Имеющиеся наработки Comindware (состав, границы, артефакты)'
date: 2026-03-29
status: 'Черновой комплект материалов для руководства (v1, март 2026)'
tags:
  - архитектура
  - GenAI
  - корпоративный
  - RAG
  - референс-стек
  - состав стека
  - KT
---

# Приложение C. Имеющиеся наработки **Comindware** (состав, границы, артефакты) {: #app_c_pack_overview }

## Обзор {: #app_c_overview }

Раздел отвечает на вопрос **что именно покрывает референс-стек Comindware сегодня**: границы модулей, роли в архитектуре и различие между **поставляемыми артефактами** и **методологическими рекомендациями**.

Критерии приёмки при передаче — в _«[Приложение B: отчуждение ИС и кода (KT, IP)](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_pack_overview)»_; модель внедрения и экономика — в двух основных отчётах.

**Курс USD** для смет — [приложение A](./20260325-research-appendix-a-index-ru.md#app_a_fx_policy).

## Как использовать в продажах и у руководства {: #app_c_how_to_use }

**Для обоснования инвестиций:**

- Дать покупателю быстрый ответ «что у **Comindware** уже реально есть» и где границы «референс-стека» vs «методология».
- **Comindware** поставляет не только внедрение, но и **передаваемые артефакты** (код/конфигурации/регламент эксплуатации (runbook)/eval/обучение) — см. _«[Приложение B: отчуждение ИС и кода](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_capability_transfer_overview)»_.

**Для переговоров с C-Level:**

- Подчёркивать, что имена модулей в комплекте — **роли компонентов** и иллюстративный референс, а коммерческий состав фиксируется договором.
- Снижать риск ожиданий: состав стека явно ограничен — нет «магии», каждый модуль имеет документацию и тесты.

!!! warning "Ограничения"

    Не используйте отчёт как:

    - Исчерпывающий перечень без сверки с договором (коммерческий состав определяется отдельно).
    - Гарантию совместимости со всеми версиями внешних компонентов без проверки.

## Связанные документы

- [Обзор и ведомость документов](./20260325-research-appendix-a-index-ru.md#app_a_pack_overview)
- [Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_pack_overview)
- [Отчёт. Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_pack_overview)
- [«Приложение B: отчуждение ИС и кода (KT, IP)»](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_pack_overview)
- [«Приложение D: безопасность, комплаенс и наблюдаемость»](./20260325-research-appendix-d-security-observability-ru.md#app_d__pack_overview)

## Обзор текущей архитектуры **Comindware**

Здесь описывается методология внедрения и управления инфраструктурой ИИ в экосистеме **Comindware** с фокусом на российские облачные провайдеры и локальный инференс. Архитектура основана на **модульном, контейнеризованном подходе**, объединяющем RAG-движок (**корпоративный RAG-контур**) с серверами инференса (**сервер инференса MOSEC**, **инференс на базе vLLM**), агентным слоем для **Comindware Platform** (агентный слой **Comindware Platform**, по сценарию) и интеграцией с российскими облачными платформами.

Экономика сайзинга, CapEx/OpEx и TCO — в отчёте _«[Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_pack_overview)»_; **профили on-prem-GPU** (потребительские **24 ГБ**, **RTX 4090 48 ГБ** в коммерческой аренде, **RTX PRO 6000 Blackwell 96 ГБ** в проектах **Comindware**) — в разделе _«[Профиль on-prem-GPU в проектах Comindware](./20260325-research-report-sizing-economics-main-ru.md#sizing_onprem_gpu_profile_cmw)»_.

Для корректного сравнения вариантов интерпретируйте эти данные через _[Топологию ёмкости GPU и типы источников цифр](./20260325-research-report-sizing-economics-main-ru.md#sizing_gpu_capacity_topology_bench_classes)_.

Технические детали развёртывания конкретных модулей — в публичной документации соответствующих программных компонентов экосистемы.

**Сводные доли и барьеры** использования GenAI: по глобальным кросс-валидированным данным (McKinsey, Stanford, EY, OECD) **40-50%** компаний отмечают проблемы качества и галлюцинаций как значимый барьер; **45-60%** выражают опасения по поводу утечки данных и безопасности. В РФ **71%** крупных компаний уже используют GenAI, но лишь **7-10%** пилотов достигают промышленного внедрения (_«[Исследование Яков и Partners + Яндекс, декабрь 2025](./20260325-research-report-sizing-economics-main-ru.md#sizing_russian_market)»_ и _Приложение D «[Безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__org_barriers_risk_survey_2025)»_).

**Ключевые методологические принципы:**
- **Разделение ответственности:** Отдельные слои для обработки данных, поиска, инференса и доставки API.
- **Гибридный поиск:** Объединение векторного поиска (плотного) с ключевым (разреженным) для оптимальной точности.
- **Агентная архитектура:** Использование агентов LangChain для динамического вызова инструментов и рассуждения.
- **Гибкость инфраструктуры:** Поддержка как MOSEC (единый сервер), так и vLLM (распределенные инстансы) бэкендов инференса.
- **Российская суверенность:** Приоритет российских облачных провайдеров (Cloud.ru, Yandex Cloud, SberCloud, MWS GPT, Selectel и др. по контуру заказчика) для обеспечения соответствия требованиям 152-ФЗ; с **июля 2025** года обязательна локализация всех персональных данных граждан РФ — см. _«[Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__compliance_rf_152fz)»_. Также возможен путь **аренды GPU** у российских IaaS/dedicated-поставщиков: сводная матрица классов продуктов и ссылок на прайсы приведена в отчёте _«[Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_russian_ai_cloud_tariffs)»_ и подразделе _«[Аренда GPU (IaaS РФ): дополнительные поставщики и ориентиры](./20260325-research-report-sizing-economics-main-ru.md#sizing_gpu_rental_iaas_providers)»_.
- **Недоверенное исполнение:** при сценариях с кодом и широким набором инструментов заказчик проектирует **изоляцию среды и сетевые политики** отдельно от перечня модулей в таблице ниже; ориентиры — в _«[Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__trust_boundary_agent_environment)»_ и _«[Приложение D — модель риска, паттерны среды и минимальный состав платформы](./20260325-research-appendix-d-security-observability-ru.md#app_d__risk_model_platform_patterns)»_.
- **Организационная зрелость:** наличие модулей **корпоративный RAG-контур**, **MOSEC**, **vLLM**, агентный слой **Comindware Platform** **не заменяет** оргпроцессы, обучение команд и операционную модель внедрения; см. _«[Отчёт. Методология — Стратегия внедрения ИИ и организационная зрелость](./20260325-research-report-methodology-main-ru.md#method_ai_strategy_org_maturity)»_.

Публичные кейсы и платформенные описания крупных игроков рынка РФ (в том числе **МТС Web Services / MTS AI**) полезны как **каталог воспроизводимых инженерных и организационных идей** — гибридный ретрив, жизненный цикл агентов, интеграционные протоколы, аттестованные контуры для ПДн. Выбор поставщика и архитектуры остаётся за заказчиком по тендеру, требованиям ИБ и экономической модели; интегратор может опираться на такие материалы при выравнивании ожиданий и комплекта отчуждения.

**Контекст рынка РФ (2025):** сводные цифры, прогнозы и доли игроков — _Приложение E «[Рыночные и технические сигналы](./20260325-research-appendix-e-market-technical-signals-ru.md#app_e_russian_market)»_ и в параграфе _«[ИИ-рынок России](./20260325-research-report-sizing-economics-main-ru.md#sizing_russia_ai_market_stats_forecasts)» отчёта «Отчёт. Сайзинг и экономика (CapEx / OpEx / TCO)»_.

### Связанные проекты

**Экосистема Comindware включает ключевые программные проекты:**

| Проект | Назначение | Архитектура |
|-------------|-----------|--------------|
| **Корпоративный RAG-контур** | RAG-движок для поиска и генерации ответов | LangChain, Gradio, ChromaDB |
| **Сервер инференса на базе MOSEC** | Сервер специализированных моделей (MOSEC) | Эмбеддинг, Реранкер, Охранник |
| **Сервер инференса на базе vLLM** | Универсальный сервер больших языковых моделей | LLM, KV-кэш, непрерывная пакетная обработка |
| Агентный слой **Comindware Platform** | AI-агент для управления **Comindware Platform** | 49 инструментов (27 **Comindware** + 22 утилиты) |

### **Comindware Platform** Agent: Агент для управления сущностями

**Проект:** агентный слой **Comindware Platform** (отдельный компонент экосистемы **Comindware**, см. таблицу выше).

**Назначение:** ИИ-агент для создания и управления сущностями в **Comindware Platform** на естественном языке.

**Архитектура:**

```
Слой интерфейса (Gradio)
    ↓
Ядро агента (оркестрация, доступ к LLM, состояние сессий)
    ↓
Слой инструментов (49)
    ├── Инструменты **Comindware Platform** (27)
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

**Контекст для внедрения у заказчика в РФ:** значение «по умолчанию» для OpenRouter отражает **удобство исходной конфигурации разработки** (единый API, быстрые эксперименты), а не рекомендацию **промышленного** контура при персональных данных и требованиях суверенитета. В таких случаях целевой выбор инференса — **API российских облаков**, **on-prem** или иной маршрут из блока Compliance ниже и из отчёта _«[Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_russian_ai_cloud_tariffs)»_.

**Ключевые возможности:**
- **Многоходовый диалог** — управление памятью в стеке LangChain
- **Потоковая выдача** — ответ по токенам через потоковый API
- **Изоляция сессий** — разделение пользователей и освобождение ресурсов
- **Локализация** — полная поддержка английского и русского
- **Восстановление после ошибок** — векторная классификация сбоев
- **Учёт токенов и бюджета** — фактический расход токенов и оценка стоимости

В референс-стеке агентный слой **Comindware Platform** опирается на **краткосрочную** память диалога в стеке LangChain и на **корпоративный RAG-контур** для извлечения из индекса; это **не** заявляет паритета с исследовательскими системами **долговременной переносимой** памяти между задачами. Публикации уровня General Agentic Memory (GAM) задают **направление эволюции** (накопление знаний и исследовательский цикл поверх памяти), которое при проектировании согласуют с политикой данных и observability — см. «_[Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__pack_overview)_» и «_[Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_pack_overview)_»; первоисточник: [arXiv — GAM](https://arxiv.org/pdf/2511.18423). **Memex(RL)** (Accenture) описывает **другой** класс решений — **индексированную** память с **обучением с подкреплением** для решений о разгрузке контекста, заголовках и извлечении записей ([arXiv — Memex(RL)](https://arxiv.org/pdf/2603.04257); см. также «_[Отчёт. Методология разработки и внедрения ИИ](./20260325-research-report-methodology-main-ru.md#method_ai_agent_memory_context)_»); это **не** компонент поставки агентный слой **Comindware Platform**.

**Инструменты для отчуждения:**
| Компонент | Артефакт | Назначение |
|-----------|----------|-----------|
| Документация | Руководство для AI-агентов | Инструкции для агентов и контрибьюторов |
| Код | агентный слой **Comindware Platform** | 49 инструментов (платформа **Comindware** и утилиты) |
| Тесты | Пакет поведенческих тестов | Регрессия и контракты инструментов |
| Конфигурация | Шаблон переменных окружения | Параметры окружения без секретов |

## Источники

- Полный консолидированный реестр — см. [Обзор и ведомость документов](./20260325-research-appendix-a-index-ru.md#app_a_sources_registry).

### Стек инференса (MOSEC, vLLM) и открытая документация

- [MOSEC — документация](https://mosecorg.github.io/mosec/index.html)
- [mosecorg/mosec (GitHub)](https://github.com/mosecorg/mosec)
- [vLLM — OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)
- [сервер инференса MOSEC — README проекта (пример публичного зеркала)](https://github.com/arterm-sedov/cmw-mosec)

### Модели на Hugging Face (Китай, РФ, США)

- [Selectel — Foundation Models Catalog](https://selectel.ru/services/cloud/foundation-models-catalog)
- [MWS — MWS GPT](https://mws.ru/mws-gpt/)
- [VK Cloud — машинное обучение](https://cloud.vk.com/docs/ru/ml)
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

### Передовые модели (США)

- [Anthropic — Introducing Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)
- [Anthropic — Introducing Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)
- [Claude Docs — What's new in Claude 4.6](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-6)
- [Claude Docs — Models overview](https://platform.claude.com/docs/en/about-claude/models/overview)

### Открытые проекты и фреймворки

- [langchain-ai/langchain — text-splitters](https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters)
- [langgenius/dify](https://github.com/langgenius/dify/)
- [run-llama/llama_index](https://github.com/run-llama/llama_index)
- [GitHub — redmadrobot-rnd/mcp-registry](https://github.com/redmadrobot-rnd/mcp-registry)
- [Хабр — red_mad_robot: MCP Tool Registry и автоматизация RAG](https://habr.com/ru/companies/redmadrobot/articles/982004/)
- [Telegram — CMO Club Russia: анонс исследования GenAI в маркетинге (red_mad_robot × CMO Club, 2025)](https://t.me/cmoclub/197)
- [RB.RU — 93% команд в маркетинге используют ИИ (обзор исследования CMO Club × red_mad_robot, 2025)](https://rb.ru/news/93-komand-v-marketinge-ispolzuyut-ii-nejroseti-pomogayut-s-kontentom-no-im-pochti-ne-doveryayut-strategiyu-i-byudzhety/)
