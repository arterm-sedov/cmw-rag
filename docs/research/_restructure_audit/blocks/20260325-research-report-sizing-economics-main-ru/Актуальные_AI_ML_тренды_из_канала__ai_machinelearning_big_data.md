## Актуальные AI/ML тренды из канала @ai_machinelearning_big_data

**Рыночный срез (не baseline для КП клиента):** ниже — дайджест глобальных продуктовых и инфраструктурных сигналов для контекста; обоснование CapEx/OpEx и SLA ведётся через дерево факторов стоимости, сценарный сайзинг и российские тарифы в начале документа.

Упоминания **coding agents** (в т.ч. Cursor, OpenCode, OpenWork, **OpenRouter**) в этом разделе отражают **рыночный контекст и OpEx разработки** на стороне заказчика; они **не** являются baseline **продакшн-RAG** и **не** входят в SKU референс-стека CMW — см. сопутствующее резюме **Методология внедрения и отчуждения ИИ**, подраздел «Ориентиры для заказчика: инструменты ускорения разработки».

Из канала **@ai_machinelearning_big_data** (323,407 подписчиков) — свежие новости и тренды AI/ML индустрии:

### Coding Agents

**NousResearch Hermes Agent Hackathon:**
*   187 заявок, призовой фонд ~998 750 руб. (эквивалент **~11 750 USD × 85 ₽/USD**)
*   Победители: Media Tool (ffmpeg integration), Hermes Agentic CAD Builder (FreeCAD), Hermes Sidecar (browser extension), Terminal World Map, HERMES Mars Rover
*   Hermes Agent также написал роман на 79,456 слов

**Cursor Composer 2:**
*   Конкурирует с Claude Opus 4.6 и GPT-5.4
*   Цена: от **~42,5 ₽/1M** входных и **~212,5 ₽/1M** выходных токенов (эквивалент **0,5 / 2,5 USD/1M × 85 ₽/USD**)
*   Внутренний бенчмарк: 61.3 балла (vs 44.2 у версии 1.5)

**OpenCode / OpenWork / OpenRouter:**
*   [OpenCode](https://opencode.ai/docs) — открытый AI coding agent; [OpenWork](https://github.com/different-ai/openwork) — опциональный UI/десктоп для команд; [OpenRouter](https://openrouter.ai/) — агрегирующий API для экспериментов и ассистентов. **OpEx разработки** может быть ниже, чем у платных IDE, при пути с **локальными или бесплатными** моделями; при подключении **[OpenCode Zen](https://opencode.ai/docs/zen)** или платных маршрутов OpenRouter сравнивать затраты с **официальными прайсами** (у Zen — USD за 1M токенов). Для КП в РФ такие цифры **иллюстративны**: пересчёт в руб. и оговорка по **юрисдикции данных** обязательны; **не** смешивать с тарифами **Cloud.ru / Yandex Cloud / SberCloud** из основного контура отчёта и **не** трактовать как продакшн-контур для ПД без ИБ/юридической оценки.

**Claude Code Channels (Anthropic):**
*   Подключение Claude Code к Telegram и Discord через MCP
*   Асинхронный формат работы с ИИ-агентами
*   "Убийца OpenClaw" по мнению сообщества

**OpenAI Codex + ChatGPT + Atlas → Суперприложение:**
*   Объединение продуктов в единую платформу
*   Агенты для автономной работы на компьютере

### AI Infrastructure

**NVIDIA Nemotron-Cascade 2:**
*   MoE 30B (3B активных) — золото на IMO, IOI, ICPC 2025
*   LiveCodeBench v6: 88.4 балла
*   Codeforces rating: 2345 (уровень моделей 300B+)
*   Лицензия: NVIDIA Open Model License

**Huawei Atlas 350:**
*   Ускоритель на Ascend 950PR — 2.87x быстрее Nvidia H20
*   FP4 вычисления, 112 ГБ HBM
*   Загрузка LLM до 70B параметров на одну карту

**GLM 5.1 — открытый код:**
*   Zixuan Li (ZAI) объявил о планах открыть код

**Mamba3:**
*   SSM-архитектура с приоритетом на инференс
*   SISO: лучшая суммарная латентность prefill + decode
*   MIMO: сопоставимая скорость, но заметно точнее

### Robotics & Hardware

**Unitree As2:**
*   Четвероногий робот в 3 версиях: AIR, PRO, EDU
*   18 кг, 12 степеней свободы, до 3.7 м/с
*   EDU: NVIDIA Jetson Orin NX support

**Huawei Atlas 350:**
*   AI-ускоритель для китайского рынка
*   2.87x быстрее Nvidia H20

**Pokemon Go → Robot Navigation:**
*   30 млрд фото от фанатов для обучения пространственного ИИ
*   Niantic Spatial: визуальная навигация с точностью до сантиметров
*   Coco Robotics: курьеры с 4 камерами

### Enterprise AI

**Google AI Studio — Vibe Coding:**
*   Antigravity Agent для автоматического развертывания Firebase
*   Next.js, React, Angular support
*   Gemini 3.1 Pro для полного цикла разработки

**ElevenLabs Music Marketplace:**
*   ИИ-музыка от ElevenCreative
*   14 млн сгенерированных песен
*   ~935 млн руб. заработано на маркетплейсе голосов (эквивалент **~11 млн USD × 85 ₽/USD**)

**Adobe Firefly:**
*   Custom AI models на пользовательских данных
*   Project Moonlight: agentic interface для всех приложений

### Российский рынок

**Agents Week от ШАДа (6-10 апреля):**
*   Интенсив по AI-агентам от экспертов Яндекса
*   Single-agent и multi-agent архитектуры
*   Продакшен-подходы: evaluation, monitoring, scaling

**Национальная технологическая олимпиада по ИИ:**
*   7,000 участников, 111 финалистов, 18 победителей
*   Победители получают стажировку в Сбере

**Yandex Prompt Hub:**
*   Промпт для генерации промптов (4-D методология)
*   Deconstruct → Diagnose → Develop → Deliver

**Сбер One Day Offer для Data Scientists:**
*   28 марта, возможность трудоустройства за 1 день

---
