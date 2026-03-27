# Приложение C. Имеющиеся наработки CMW (состав и границы)

**Дата пакета:** 2026-03-25  
**Статус:** утверждённый комплект материалов для руководства (v1)

## Обзор пакета

Здесь фиксируется **что уже есть** в референс-стеке и смежных наработках: границы компонентов, роли в архитектуре и отличие «имеющегося артефакта» от «методологических рекомендаций». Критерии приёмки при передаче — в «Приложение B: отчуждение ИС и кода (KT, IP)»; методология и экономика — в основных отчётах.

## Связанные документы

- «Приложение A: витрина пакета»
- «Основной отчёт: методология внедрения и разработки»
- «Основной отчёт: сайзинг и экономика»
- «Приложение B: отчуждение ИС и кода (KT, IP)»
- «Приложение D: безопасность, комплаенс и observability»

## Обзор текущей архитектуры CMW

В данном документе описывается методология внедрения и управления инфраструктурой ИИ в экосистеме CMW с фокусом на российские облачные провайдеры и локальный инференс. Архитектура основана на **модульном, контейнеризованном подходе**, объединяющем RAG-движок (**корпоративный RAG-контур**) с серверами инференса (**сервер инференса MOSEC**, **инференс на базе vLLM**), агентным слоем для CMW Platform (**агентный слой платформы (CMW Platform)**, по сценарию) и интеграцией с российскими облачными платформами.

Детальная экономика сайзинга, CapEx/OpEx и TCO — в сопутствующем резюме **Оценка сайзинга, КапЭкс и ОпЭкс для клиентов**. Технические детали развёртывания конкретных модулей — в публичной документации соответствующих программных компонентов экосистемы.

**Ключевые методологические принципы:**
- **Разделение ответственности:** Отдельные слои для обработки данных, поиска, инференса и доставки API.
- **Гибридный поиск:** Объединение векторного поиска (плотного) с ключевым (разреженным) для оптимальной точности.
- **Агентная архитектура:** Использование агентов LangChain для динамического вызова инструментов и рассуждения.
- **Гибкость инфраструктуры:** Поддержка как MOSEC (единый сервер), так и vLLM (распределенные инстансы) бэкендов инференса.
- **Российская суверенность:** Приоритет российских облачных провайдеров (Cloud.ru, Yandex Cloud, SberCloud, MWS GPT, Selectel и др. по контуру заказчика) для обеспечения соответствия требованиям о данных и инфраструктуре.

Публичные кейсы и платформенные описания крупных игроков рынка РФ (в том числе **МТС Web Services / MTS AI**) полезны как **каталог воспроизводимых инженерных и организационных идей** — гибридный ретрив, жизненный цикл агентов, интеграционные протоколы, аттестованные контуры для ПДн. Выбор поставщика и архитектуры остаётся за заказчиком по тендеру, требованиям ИБ и экономической модели; интегратор может опираться на такие материалы при выравнивании ожиданий и пакета отчуждения.

### Связанные проекты

**Экосистема CMW включает ключевые программные проекты (имена условные, обозначают роли компонентов):**

| Проект | Назначение | Архитектура |
|-------------|-----------|--------------|
| **корпоративный RAG-контур** | RAG-движок для поиска и генерации ответов | LangChain, Gradio, ChromaDB |
| **сервер инференса MOSEC** | Унифицированный сервер инференса (MOSEC) | Эмбеддинг, Реранкер, Охранник на одном порту |
| **инференс на базе vLLM** | Распределённый сервер инференса (vLLM) | LLM, KV-кэш, непрерывная пакетная обработка |
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

В референс-стеке **агентный слой платформы (CMW Platform)** опирается на **краткосрочную** память диалога в стеке LangChain и на **корпоративный RAG-контур** для извлечения из индекса; это **не** заявляет паритета с исследовательскими системами **долговременной переносимой** памяти между задачами. Публикации уровня General Agentic Memory (GAM) задают **направление эволюции** (накопление знаний и исследовательский цикл поверх памяти), которое при проектировании согласуют с политикой данных и observability — см. «_[Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#research_pkg_d_obzor_paketa)_» и «_[Основной отчёт: методология внедрения и разработки](./20260325-research-report-methodology-main-ru.md#research_methodology_20260325_obzor_paketa)_»; первоисточник: [arXiv — GAM](https://arxiv.org/pdf/2511.18423).

**Инструменты для отчуждения:**
| Компонент | Артефакт | Назначение |
|-----------|----------|-----------|
| Документация | Руководство для AI-агентов | Инструкции для агентов и контрибьюторов |
| Код | **агентный слой платформы (CMW Platform)** | 49 инструментов (платформа CMW и утилиты) |
| Тесты | Пакет поведенческих тестов | Регрессия и контракты инструментов |
| Конфигурация | Шаблон переменных окружения | Параметры окружения без секретов |

---

## Источники

- Полный консолидированный реестр — см. [Приложение A: витрина пакета](./20260325-research-appendix-a-index-ru.md#research_pkg_a_polnyi_reestr_ispolzovannyh_istochnikov_tochnaya_konsolidatsiya).

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
- [NVIDIA — Nemotron 3 (обзор семейства)](https://research.nvidia.com/labs/nemotron/Nemotron-3/)
- [Hugging Face — nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)
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

Помимо перечисленных ниже фреймворков, в открытом доступе публикуют **референсные каркасы** под RAG и агентов: лаборатория AI R&D **red_mad_robot** описывает **MCP Tool Registry** (центральный реестр MCP-серверов для связки LLM с данными и инструментами) и выкладывает код на GitHub — это **ориентир для разведки архитектуры** и политики подключения MCP, а не компонент поставки **корпоративный RAG-контур** / **агентный слой платформы (CMW Platform)** без отдельного решения заказчика.

- [langchain-ai/langchain — text-splitters](https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters)
- [langgenius/dify](https://github.com/langgenius/dify/)
- [run-llama/llama_index](https://github.com/run-llama/llama_index)
- [GitHub — redmadrobot-rnd/mcp-registry](https://github.com/redmadrobot-rnd/mcp-registry)
- [Хабр — red_mad_robot: MCP Tool Registry и автоматизация RAG](https://habr.com/ru/companies/redmadrobot/articles/982004/)
