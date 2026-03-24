# Задача: Ревалидация и расширение executive-отчётов

## 1. Цель

Подготовить **два максимально подробных executive summary** на основе:

- текущего репозитория: `.`
- связанных репозиториев: `../cmw-mosec`, `../cmw-vllm`

Два итоговых документа:

1. **Методология внедрения и отчуждения ИИ**
2. **Оценка сайзинга, CapEx и OpEx для клиентов**

## 2. Объём анализа

Покрыть следующие темы (в обоих направлениях исследования, где применимо):

- **Внедрение ИИ**: как мы рекомендуем внедрять ИИ-ассистентов поддержки (например `.`, `../cmw-platform-agent`)
- **Отчуждение ИИ**: как передавать клиентам экспертизу, наработки, обучение и практику поставки
- **Сайзинг**: оценка для нас и для клиентов при внедрении и развёртывании ИИ-решений
- **CapEx / OpEx**: экономика для нас и для клиентов при внедрении и эксплуатации ИИ-решений
- **Российские правовые аспекты**: релевантные нормативные и правовые ограничения

## 3. Требования к методологии исследования

- Выполнить **глубокий ресерч** по всем темам.
- Использовать web research каналы:
  - стандартный web search
  - Tavily
  - SearXNG
  - Playwright-based browser research (поиск через Google/Yandex при необходимости)
- До начала работ использовать установленные planning/deep-research skills, где это релевантно.
- Задавать уточняющие вопросы перед принятием существенных допущений.

## 4. Правила редактирования контента

- **Не удалять релевантный контент.**
- Добавлять новый релевантный контент там, где есть пробелы.
- Удалять только явно нерелевантный/устаревший материал.
- Финальные документы должны быть executive-уровня, точными и доказательными.

## 5. Обязательные источники

### Внутренние reference-репозитории и документы

- `../awesome-llm-apps/rag_tutorials`
- `../awesome-llm-apps/mcp_ai_agents`
- `../awesome-llm-apps/awesome_agent_skills`
- `../awesome-llm-apps/ai_agent_framework_crash_course`
- `../awesome-llm-apps/advanced_llm_apps`
- `.opencode/README.md`
- `../cmw-mosec/.opencode/README.md`
- `../cmw-vllm/.opencode/README.md`

### Источник Telegram-экстракции

- `~/Documents/cmw-rag-channel-extractions`

## 6. Требования к обработке Telegram

При обработке экспортированного Telegram-контента:
- Извлекать **только текст** (без OCR для тел сообщений).
- Использовать ad hoc Python-скрипты для контроля полноты извлечения.
- Обеспечить **100% покрытие** содержимого чатов.
- После извлечения выполнить повторный скан всего извлечённого контента.
- Обработать **ВСЕ чаты от 0 до N**, включая ранее пропущенные.
- Сохранять артефакты в `~/Documents/cmw-rag-channel-extractions`.

## 7. Рабочий процесс (рекомендуемый)

1. Провести ревалидацию текущих версий отчётов и выявить пробелы.
2. Выполнить глубокий web + internal-repo ресерч.
3. Завершить полную Telegram-экстракцию и повторный скан.
4. Синтезировать результаты в два executive summary.
5. Провести повторную проверку на соответствие scope и ограничениям.
6. Подготовить финальные polished-версии.

## 8. Итоговые артефакты и критерии приёмки

### Deliverables

- Executive Summary #1: методология внедрения и отчуждения ИИ
- Executive Summary #2: модель сайзинга + CapEx/OpEx для клиентов

### Критерии приёмки

- Оба документа подробные, структурированные и прикладные.
- Покрыты внедрение, отчуждение, сайзинг, экономика и российские правовые аспекты.
- Все выводы опираются на глубокий ресерч и релевантные внутренние материалы.
- Инсайты из Telegram полностью интегрированы после 100% извлечения текста и повторного скана.
- Релевантный существующий контент сохранён; нерелевантный удалён.

## 9. Исследовательская повестка (agenda): как вести работу

### 9.1 Основная логика выполнения

Работать в 4 волны:
1. **Scoping**: зафиксировать, что уже есть в двух текущих отчётах и где пробелы.
2. **Evidence collection**: собрать внешние и внутренние источники по каждому блоку.
3. **Triangulation**: для каждого ключевого вывода иметь минимум 3 независимых источника.
4. **Synthesis**: превратить данные в управленческие решения (варианты, trade-offs, рекомендации).

### 9.2 Что искать по каждому направлению

#### A. Методология внедрения ИИ-ассистентов

Искать:
- operating model внедрения (pilot -> scale -> industrialization)
- governance и risk management для GenAI (политики, роли, контрольные процедуры)
- RAG production patterns (качество retrieval, evals, guardrails, мониторинг)
- change management: обучение пользователей, adoption, KPI/SLAs

Полезные ориентиры:
- NIST AI RMF + GenAI Profile (управление рисками и контрольные практики)
- LangChain/LangGraph production patterns и eval-подходы
- FinOps-практики для эксплуатации LLM-нагрузок

#### B. Методология отчуждения ИИ (передача клиенту)

Искать:
- модели передачи: managed service, co-build, build-operate-transfer, лицензирование IP
- пакет отчуждения: код, документация, runbooks, eval-наборы, security baselines
- модель обучения клиента: enablement tracks для бизнеса, ops, разработки, compliance
- контроль качества передачи: критерии готовности и приемки (readiness + handover checklist)

#### C. Сайзинг + CapEx/OpEx

Искать:

- драйверы стоимости: токены, embeddings, retrieval infra, observability, support
- размерность нагрузки: DAU/MAU, запросы на пользователя, средний контекст, latency SLO
- сценарные модели: conservative/base/aggressive
- unit economics: cost per user, cost per resolved ticket, cost per successful answer
- оптимизации стоимости: routing, caching, batching, prompt design, model tiering

#### D. Российские правовые аспекты

Искать:

- требования по персональным данным (152-ФЗ), локализация, согласия, процессы обработки
- требования к обезличиванию и внутренним ЛНА
- отраслевые рекомендации (например, Банк России для финсектора)
- ограничения по трансграничной передаче и хранению данных в контуре ИИ-решения

## 10. Где искать: приоритет источников

### 10.1 Tier-1 (обязательно, высокий приоритет)

- Официальные регуляторы/стандарты:
  - NIST (AI RMF 1.0, GenAI Profile)
  - EU AI Act official guidance (для comparative insight)
  - Банк России (доклады/рекомендации/кодексы)
  - Официальные тексты законов и подзаконных актов РФ
- Официальная документация вендоров/фреймворков:
  - LangChain/LangGraph docs
  - OpenAI docs (cost/latency/prompt caching)
  - Документация инфраструктурных платформ (где релевантно)

### 10.2 Tier-2 (поддерживающие)

- Профессиональные комьюнити и framework-организации (например FinOps Foundation).
- Технические разборы практиков with воспроизводимыми метриками и прозрачной методологией.

### 10.3 Tier-3 (использовать осторожно)

- Маркетинговые статьи, блоги без методологии, неподтверждённые цифры.
- Использовать только как вспомогательный сигнал, не как основу вывода.

## 11. Подсказки для web search: готовые поисковые запросы

### 11.1 Для методологии внедрения

- "NIST AI RMF 1.0 generative AI profile implementation"
- "AI governance operating model enterprise genai"
- "RAG production best practices retrieval evaluation hallucination guardrails"
- "LangChain LLM evals production monitoring regression tests"

### 11.2 Для отчуждения/передачи

- "build operate transfer AI platform model"
- "AI enablement playbook client handover checklist"
- "knowledge transfer framework software delivery enterprise"
- "AI CoE federated model centralized vs federated genai"

### 11.3 Для сайзинга и экономики

- "LLM cost model token economics enterprise"
- "FinOps unit economics cloud AI workloads"
- "prompt caching model routing batching inference cost optimization"
- "RAG infrastructure cost drivers embeddings vector database"

### 11.4 Для РФ правового блока (искать на русском)

- "152-ФЗ персональные данные локализация требования"
- "Роскомнадзор обезличивание персональных данных приказ"
- "Банк России искусственный интеллект рекомендации финансовый рынок"
- "трансграничная передача персональных данных требования РФ"

## 12. Операционная методика ресерча

### 12.1 Правило доказательности

- Для каждого важного тезиса: минимум **3 источника**, из них минимум **1 Tier-1**.
- Любые количественные оценки (ROI, CapEx/OpEx, SLA-эффекты) подтверждать первоисточниками.

### 12.2 Правило свежести

- Приоритет материалам за последние 12-24 месяца.
- Для стандартов и законов всегда проверять действующую редакцию/дату обновления.

### 12.3 Правило прослеживаемости

- Вести таблицу доказательств (evidence log): `Тезис | Источник | Тип источника | Дата | Надежность (H/M/L) | Комментарий`

### 12.4 Правило конфликтов источников

- Если источники противоречат: фиксировать расхождение отдельно и давать объяснение, какие допущения приводят к разнице.

## 13. Каркас содержания для двух отчётов

### 13.1 Отчёт 1: Методология внедрения и отчуждения ИИ

Рекомендуемый каркас:

1. Executive summary (для CEO/CTO/CFO)
2. Target operating model (роли, процессы, KPI)
3. Методология внедрения (этапы, артефакты, quality gates)
4. Методология отчуждения (handover-модель, обучение, контроль приемки)
5. Risk & compliance (в т.ч. РФ-аспекты)
6. Decision matrix (варианты внедрения/передачи и trade-offs)
7. Рекомендованный план 30/60/90 дней

### 13.2 Отчёт 2: Сайзинг + CapEx/OpEx

Рекомендуемый каркас:

1. Executive summary с диапазонами сценариев
2. Cost driver tree (что формирует стоимость)
3. Сценарный сайзинг (conservative/base/aggressive)
4. CapEx/OpEx-модель (для нас и для клиента отдельно)
5. Unit economics и чувствительность (sensitivity analysis)
6. Cost optimization playbook (quick wins + medium-term)
7. Риски бюджета и меры снижения

## 14. Контрольные точки качества (quality gates)

- **Gate 1 (после сбора данных):** закрыты все темы scope, нет "пустых" разделов.
- **Gate 2 (после синтеза):** каждый ключевой вывод подтвержден источниками.
- **Gate 3 (перед финализацией):** отчёты читаются на уровне руководства (decision-ready).
- **Gate 4 (финальный):** сохранены релевантные старые материалы, удалён шум.

## 15. Что считать результатом "готово"

Задача считается выполненной, если:

- Оба отчёта содержат actionable-рекомендации и альтернативы.
- Есть прозрачная логика расчётов сайзинга и экономики.
- Ясно описано, как внедрять и как отчуждать ИИ-решения клиентам.
- Правовой блок по РФ не формальный, а встроен в архитектурные и процессные решения.
- Все существенные тезисы имеют проверяемую доказательную базу.

## 16. Ссылки на источники

Глубоко исследуй приведённые ссылки, их содержимое и НАЙДИ НОВЫЕ

### 16.1 Международные стандарты и регулирование (Tier-1)

- [ISO/IEC 42001:2023 - Artificial Intelligence Management System (Official)](https://www.iso.org/standard/81230.html)
- [ISO/IEC 42001:2023 - PDF Sample and Core Requirements](https://cdn.standards.iteh.ai/samples/81230/4c1911ebc9a641fcb6ee21aa09c28ad3/ISO-IEC-42001-2023.pdf)
- [NIST AI Risk Management Framework 1.0 (Full Portal)](https://nist.gov/itl/ai-risk-management-framework)
- [NIST AI RMF: Generative AI Profile (NIST AI 600-1)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
- [NIST AI 600-1 Direct PDF Download](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=958388)
- [NIST AI RMF to ISO/IEC 42001 Crosswalk (Official)](https://airc.nist.gov/docs/NIST_AI_RMF_to_ISO_IEC_42001_Crosswalk.pdf)
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

### 16.2 Executive-методологии внедрения (Big Three & Big Four)

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
- [Everest Group: Enterprise Generative AI Adoption 2025 Playbook](https://www.everestgrp.com/report/generative-ai-playbook)
- [HFS Research: The Generative AI 2025 Horizon Report](https://www.hfsresearch.com/research/genai-horizon-2025/)

### 16.3 Технические паттерны и блоги инжиниринга (Production AI)

- [DoorDash: How We Built an Internal AI Platform That Works (2025)](https://www.getdot.ai/blog/doordash-ai-platform-agents)
- [DoorDash: Building a Collaborative Multi-Agent AI Ecosystem (2026)](https://www.zenml.io/llmops-database/building-a-collaborative-multi-agent-ai-ecosystem-for-enterprise-knowledge-access)
- [DoorDash: Building an Enterprise LLMOps Stack - Lessons (2026)](https://www.zenml.io/llmops-database/building-an-enterprise-llmops-stack-lessons-from-doordash)
- [Uber Engineering: Genie - GenAI On-Call Copilot Architecture (2025)](https://www.uber.com/en-CO/blog/genie-ubers-gen-ai-on-call-copilot/)
- [Uber Engineering: Raising the Bar on ML Model Deployment Safety (2025)](https://www.uber.com/en-CA/blog/raising-the-bar-on-ml-model-deployment-safety/)
- [Netflix Tech Blog: Foundation Models for Personalized Recommendations (2025)](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39)
- [Netflix Tech Blog: Scaling Generative Recommenders (2025)](https://netflixtechblog.medium.com/integrating-netflixs-foundation-model-into-personalization-applications-cf176b5860eb)
- [Airbnb Tech Blog: Reshaping Customer Support with GenAI (2025)](https://medium.com/airbnb-engineering/how-ai-text-generation-models-are-reshaping-customer-support-at-airbnb-a851db0b4fa3)
- [Airbnb Tech Blog: Agent-in-the-Loop (AITL) Framework Paper (2025)](https://aclanthology.org/2025.emnlp-industry.135.pdf)
- [Meta AI: Production Pipelines for Llama Deployments (Official 2025)](https://llama.meta.com/docs/deployment/production-deployment-pipelines)
- [vLLM: Performance Optimization and Tuning Guide (2025)](https://docs.vllm.ai/en/latest/performance/optimization.html)
- [vLLM Performance Tuning: The Ultimate Guide (2026)](https://cloud.google.com/blog/topics/developers-practitioners/vllm-performance-tuning-the-ultimate-guide-to-xpu-inference-configuration)
- [vLLM Production Stack Roadmap for 2025 Q2 (GitHub)](https://github.com/vllm-project/production-stack/issues/300)
- [vLLM Production Stack 2026 Roadmap (GitHub)](https://github.com/vllm-project/production-stack/issues/855)
- [vLLM: Practical strategies for performance tuning (Red Hat 2026)](https://developers.redhat.com/articles/2026/03/03/practical-strategies-vllm-performance-tuning)
- [LangGraph: Enterprise Multi-Agent Orchestration Patterns (2025)](https://sumanta9090.medium.com/langgraph-patterns-best-practices-guide-2025-38cc2abb8763)
- [LangGraph: Building Stateful, Multi-Agent AI Workflows (Checklist 2025)](https://bix-tech.com/ai-agents-orchestration-langgraph/)
- [Anthropic: Contextual Retrieval - Improving RAG Accuracy (2025)](https://www.anthropic.com/news/contextual-retrieval)
- [OpenAI: Production RAG Best Practices & Evaluation (Cookbook)](https://github.com/openai/openai-cookbook/blob/main/examples/RAG_with_Evaluation.ipynb)
- [Microsoft: RAG Architecture on Azure AI Search (2025 Update)](https://learn.microsoft.com/en-us/azure/architecture/guide/multimodal-rag/multimodal-rag-architecture)
- [RAG - A Deep Dive (System Design Newsletter 2025)](https://newsletter.systemdesign.one/p/how-rag-works)
- [RAG Evaluation with RAGAS and MLflow (Practical Guide 2026)](https://www.safjan.com/ragas-mlflow-rag-evaluation-tutorial/)
- [Giskard: Open-Source Evaluation for LLM Agents (2026 Docs)](https://github.com/Giskard-AI/giskard-oss)
- [Monitaur: AI Governance Software Platform Features (2026)](https://www.monitaur.ai/platform)
- [LiteLLM Docs: Production Gateway for 100+ Models (2026)](https://docs.litellm.ai/)
- [Portkey AI Gateway: Cost Management and Observability (2026)](https://docs.portkey.ai/)
- [Helicone: Provider Routing and AI Gateway Docs (2026)](https://docs.helicone.ai/)

### 16.4 Экономика ИИ, FinOps и сайзинг

- [FinOps Foundation: GenAI Working Group Hub (2025)](https://www.finops.org/wg/generative-ai/)
- [FinOps Foundation: Cost Estimation of AI Workloads (2026 Resource)](https://www.finops.org/wg/cost-estimation-of-ai-workloads)
- [FinOps Framework 2025: Unit Economics for AI Workloads](https://www.finops.org/framework/capabilities/unit-economics/)
- [FinOps Framework 2025: Cloud Cost Allocation PDF](https://www.finops.org/wp-content/uploads/2025/05/English-FinOps-Framework-2025.pdf)
- [FinOps in the AI Era: 2026 Survey Report (CloudZero)](https://www.cloudzero.com/guide/finops-in-the-ai-era-2026-survey-report/)
- [CloudZero: FinOps for AI - Why AI Alters Cloud Cost Management (2026)](https://www.cloudzero.com/blog/finops-for-ai/)
- [CloudZero: Cloud Unit Economics In 2026 Guide](https://www.cloudzero.com/guide/cloud-unit-economics-2026/)
- [CloudZero: FinOps Cost-Per-Unit Glossary (Feb 2026)](https://www.cloudzero.com/blog/finops-cost-per-unit-glossary/)
- [OpenAI: Official Prompt Caching Optimization Guide (2025)](https://platform.openai.com/docs/guides/prompt-caching)
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

### 16.5 Российские регуляторные и правовые источники (Tier-1)

- [Указ Президента РФ №490: Национальная стратегия развития ИИ до 2030 (Ред. 2024)](https://www.consultant.ru/document/cons_doc_LAW_470015/)
- [Указ Президента РФ от 15.02.2024: Изменения в стратегию ИИ (Актуальная версия)](https://ai.gov.ru/national-strategy/)
- [Минцифры РФ: Законопроект «Об основах госрегулирования ИИ» (ID 166424, 2026)](https://regulation.gov.ru/projects#npa=166424)
- [Минцифры РФ: Пояснительная записка к законопроекту об ИИ (Март 2026)](https://www.m24.ru/news/politika/18032026/883742)
- [Минцифры РФ: Правила маркировки ИИ-контента (Законопроект 2026)](https://www.infox.ru/news/299/375381-mincifry-rf-predstavilo-novye-pravila-dla-regulirovania-ii-s-markirovkoj-kontenta)
- [Банк России: Кодекс этики ИИ на финансовом рынке (Официальный PDF 2025)](https://www.cbr.ru/content/document/file/178667/code_09072025.pdf)
- [Банк России: Пять принципов ответственного использования ИИ (Июль 2025)](https://www.cbr.ru/press/event/?id=25755)
- [Банк России: Информационное письмо №ИН-016-13/91 (Июль 2025)](https://www.consultant.ru/document/cons_doc_LAW_509514/)
- [Банк России: Доклад о применении ИИ на финансовом рынке (Consultation Paper)](http://www.cbr.ru/analytics/d_ok/Consultation_Paper_03112023/)
- [Роскомнадзор: Приказ №140 «Об утверждении требований к обезличиванию ПД» (2025)](https://normativ.kontur.ru/document?documentId=500957&moduleId=1)
- [Постановление Правительства РФ №1154: Требования и методы обезличивания ПД (2025)](https://klerk.ru/doc/657888)
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

### 16.6 Российские прикладные исследования и бенчмарки

- [MERA: Open Independent Benchmark for Russian LLMs (Official)](https://mera.a-ai.ru/)
- [MERA Benchmark: GigaChat 2 MAX Ranking (Top-1 RU 2026)](https://setka.ru/posts/019592e7-54d7-4f94-af58-0b74d6968357)
- [ruMMLU: Benchmarking Russian LLM Intelligence (HSE/Sber)](https://github.com/ai-forever/ru-mmlu)
- [НИУ ВШЭ: Исследование точности RAG-систем на русском языке (2025)](https://www.hse.ru/edu/vkr/1053304649)
- [НИУ ВШЭ: Мультиагентная платформа для отраслевых задач (2026)](https://techpro.hse.ru/ai-solutions/description)
- [НИУ ВШЭ: План исследований мультиагентного ИИ 2025-2026](https://www.hse.ru/news/development/1053986394.html)
- [ИТМО: Мультиагентная система ProAGI для разработки ПО (2026)](https://iai.itmo.ru/news/v-itmo-sozdali-multiagentnuyu-ii-sistemu-proagi,-kotoraya-uskoryaet-sozdanie-promyishlennogo-po-ot-2-do-10-raz)
- [ИТМО: Лаборатория композитного ИИ - Фреймворк FEDOT (2025)](https://itmo.ru/ru/viewdepartment/507/laboratoriya_kompozitnogo_iskusstvennogo_intellekta.htm)
- [Сколково: Потенциал GenAI для инженерных задач (Июль 2025)](https://sk.ru/news/skolkovo-i-ano-ce-predstavili-obzor-potencial-primeneniya-generativnogo-ii-dlya-resheniya-inzhenernyh-zadach/)
- [АНО ЦЭ: Аналитический отчет «Будущее искусственного интеллекта» (2025)](https://d-economy.ru/news/ano-cje-vypustila-analiticheskij-otchet-budushhee-iskusstvennogo-intellekta/)
- [Иннополис: Применение ИИ в промышленности и строительстве (2025)](https://innopolis.university/news/)
- [Sber AI: ru-Gemma и open-source инициативы 2025-2026](https://developers.sber.ru/docs/ru/gigachat/models/updates)
- [Yandex Research: Оптимизация инференса LLM для русского языка (2025)](https://yandex.ru/company/research/)

### 16.7 Модели отчуждения и передачи (BOT & Handover)

- [Build-Operate-Transfer (BOT) Model: Full Guide 2025](https://build-operate-transfer.com/post/build-operate-transfer-bot-model-complete-guide-for-software-development-2025)
- [Luxoft: Master the BOT Model for Global AI Centers (2025)](https://www.luxoft.com/blog/master-the-build-operate-transfer-bot-model-with-luxoft)
- [InOrg: Seamless Handover Factors for AI Platforms (2025)](https://inorg.com/blog/from-build-to-transfer-key-success-factors-a-seamless-bot-model-transition)
- [Software Handover Checklist 2026: Documentation & IP Guide](https://www.tech4lyf.com/blog/software-handover-documentation-checklist-2026/)
- [InCommon: Why BOT Wins for AI Infrastructure](https://www.incommon.ai/blog/build-operate-transfer/)
- [Innowise: BOT Outsourcing Contract and IP Transfer Guide](https://innowise.com/blog/build-operate-transfer-bot-model-guide/)
- [Devico: Checklist for a seamless BOT transition (2025)](https://devico.io/blog/checklist-for-a-seamless-bot-transition)
- [Knowledge Transfer Framework for Enterprise Software Handover](https://www.knowledge-management-tools.net/knowledge-transfer-framework.html)

### 16.8 Кураторские списки и репозитории (Awesome Lists)

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

### 16.9 Кейсы внедрения в российском бизнесе (2025-2026)

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
- [МТС: RAG-система для техподдержки на базе Confluence и Jira (2025)](https://habr.com/ru/companies/ru_mts/articles/970476/)
- [Яндекс: Корпоративный DeepResearch по кодовой базе (Кейс 2025)](https://habr.com/ru/companies/yandex/articles/987388/)
- [X5 Group: ИИ в управлении цепочками поставок и прогнозировании спроса](https://www.tadviser.ru/index.php/Проект:X5_Retail_Group_(Искусственный_интеллект))
- [Северсталь: ИИ для контроля качества проката и оптимизации плавки](https://www.severstal.com/rus/media/news/)
- [Росатом: ИИ-системы для проектирования АЭС и анализа безопасности](https://rosatom.ru/journalist/news/)
- [Самолет: Кейс «Цифровой рабочий» и ИИ в управлении стройкой (2025)](https://samolet.ru/news/)

### 16.10 Технические статьи и инженерные блоги (Россия)

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
- [TAdviser: Рынок ИИ в России - цифры и тренды 2025-2026](https://www.tadviser.ru/index.php/Статья:Искусственный_интеллект_(рынок_России))
- [CNews: Технологии искусственного интеллекта 2025 - обзор](https://adobe.cnews.ru/reviews/tehnologii_iskusstvennogo_intellekta/)
- [RB.ru: Топ-100 ИИ-стартапов России 2025 - карта рынка](https://rb.ru/list/ai-100-2025/)
- [ComNews: Экономика автоматизации ИИ и точки экономии для бизнеса (2026)](https://www.comnews.ru/digital-economy/content/244350/2026-03-23/2026-w13/1016/ekonomika-avtomatizacii-ii-i-realnye-tochki-ekonomii-dlya-biznesa)

### 16.11 Российская экономика ИИ и отчеты консалтинга

- [Яков и Партнёры: Rewiring the Enterprise for GenAI - Russian Context (2025)](https://yakovpartners.ru/publications/ai-2025/)
- [Kept (ex-KPMG): ИИ-агенты KeptStore для корпоративного сектора (2026)](https://www.vedomosti.ru/press_releases/2026/01/14/kept-zapuskaet-platformu-s-ii-agentami-keptstore-dlya-avtomatizatsii-zadach-korporativnogo-segmenta-erid-2VfnxxbJAiD)
- [B1 (ex-EY): Использование ИИ в российских компаниях - опрос 2025](https://www.b1.ru/ru/insights/ai-survey-2025/)
- [Технологии Доверия (ex-PwC): ИИ как драйвер изменений экономики (2025)](https://ict.moscow/research/iskusstvennyi-intellekt-draiver-izmenenii-ekonomiki-i-finansov/)
- [Деловые Решения и Технологии (ex-Deloitte): ИИ в России 2026](https://delret.ru/insights/ai-russia-2026)
- [AIРассвет: Метрики ROI и стратегии внедрения ИИ в РФ 2025-2026](https://airassvet.ru/articles/effektivnost-iskusstvennogo-intellekta-v-rossiyskom-biznese-2025-2026-analiticheskiy-otchet-o-7-klyuchevyh-stsenariyah-metrikah-roi-i-strategiyah-vнедрения)
- [РБК: Во сколько обойдется ИИ-агент - подсчеты экспертов (2026)](https://www.rbc.ru/education/05/02/2026/697162269a794772c9cf9991)
- [CNews: Прогноз внедрения ИИ в BI-решения 2026](https://www.cnews.ru/news/line/2026-03-16_navikon_v_2026_gbolee_80)
- [TAdviser: ИТ-приоритеты 2026 - ИИ на первом месте](http://www.tadviser.ru/index.php/Статья:TAdviser%3A_%D0%98%D0%A2-%D0%BF%D1%80%D0%B8%D0%BE%D1%80%D0%B8%D1%82%D0%B5%D1%82%D1%8B_2026)
- [Cloud.ru: Тарифы GigaChat 2 MAX и кейсы для бизнеса (2026)](https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs)
- [Yandex Cloud: Стоимость YandexGPT 4 и кейсы интеграции (2026)](https://cloud.yandex.ru/services/yandexgpt)
- [Минцифры РФ: Национальный прогноз вклада ИИ в ВВП до 2030 года](https://digital.gov.ru/ru/activity/directions/1056/)
- [РБК: Тренды ИИ в медицине и госсекторе 2025-2026](https://www.rbc.ru/trends/innovation/)
- [Коммерсант: Сравнение цен на генерацию YandexGPT и GigaChat (2026)](https://ya-r.ru/2023/12/12/kommersant-sravnil-tseny-na-generatsiyu-yandexgpt-i-gigachat/)
