---
title: 'Приложение A. Обзор и ведомость документов: навигация, реестр и источники'
date: 2026-03-25
status: 'утверждённый комплект материалов для руководства (v1)'
tags:
  - AI
  - CapEx
  - compliance
  - enterprise
  - methodology
  - observability
  - OpEx
  - RAG
  - research
  - TCO
  - исследование
  - методология
  - пакет
  - сайзинг
  - экономика
hide: tags
---

# Приложение A. Обзор и ведомость документов: навигация, реестр и источники {: #research_pkg_a }
## Обзор пакета {: #research_pkg_a_obzor_paketa }

Этот документ — **точка входа** в комплект из шести файлов: ведомость материалов, навигация «какой документ запросить по вопросу», единый реестр использованных источников и тематический перечень дополнительных источников из исходного задания на исследование. Два сводных материала от 23.03.2026 («методология внедрения и отчуждения ИИ» и «сайзинг, КапЭкс и ОпЭкс») сохраняются как неизменённые эталоны для внутреннего контроля полноты; при передаче вовне ориентируйтесь на документы этого комплекта.

## Как читать комплект (для руководства) {: #research_pkg_a_kak_chitat_komplekt_dlya_rukovodstva }

1. Начните с **этого файла (Приложение A)**: по таблице «вопрос → документ» выберите нужный материал.
2. Для **методологии внедрения, TOM, фаз и производственной модели** — основной отчёт по методологии; для **цифр, CapEx/OpEx/TCO и тарифов** — основной отчёт по сайзингу и экономике.
3. **Отчуждение ИС и кода** — Приложение B; **что уже есть в референс-стеке** — Приложение C; **безопасность, комплаенс и observability** — Приложение D.
4. Чтение **не требует** доступа к репозиториям или внутренним каталогам: все опоры — публичные ссылки в тексте и в разделе «Источники».

## Условные обозначения и границы примера {: #research_pkg_a_uslovnye_oboznacheniya_i_granitsy_primera }

В тексте комплекта **корпоративный RAG-контур**, **сервер инференса MOSEC**, **инференс на базе vLLM** и **агентный слой платформы (CMW Platform)** — это **условные имена ролей** иллюстративного референс-стека CMW, а не обязательный коммерческий продукт или фиксированная поставка. Продукт **[OpenCode](https://opencode.ai/)** (открытый coding agent) и **внутренние каталоги планирования** в репозиториях — **разные вещи**; подробнее формулировка для договоров и переговоров — в «Приложение B: отчуждение ИС и кода (KT, IP)».

| Условное обозначение в тексте | Смысл |
| --- | --- |
| корпоративный RAG-контур | RAG и доставка ответов в примере архитектуры |
| сервер инференса MOSEC | Унифицированный HTTP-сервис вспомогательных моделей (MOSEC) в примере |
| инференс на базе vLLM | Распределённый инференс LLM через vLLM в примере |
| агентный слой платформы (CMW Platform) | Агентные сценарии управления платформой в примере |

## Связанные документы {: #research_pkg_a_svyazannye_dokumenty }
- [«Основной отчёт: методология внедрения и разработки»](./20260325-research-report-methodology-main-ru.md#research_methodology_20260325_obzor_paketa)
- [«Основной отчёт: сайзинг и экономика»](./20260325-research-report-sizing-economics-main-ru.md#research_sizing_20260325_obzor_paketa)
- [«Приложение B: отчуждение ИС и кода (KT, IP)»](./20260325-research-appendix-b-ip-code-alienation-ru.md#research_pkg_b_obzor_paketa)
- «Приложение C: имеющиеся наработки CMW»
- [«Приложение D: безопасность, комплаенс и observability»](./20260325-research-appendix-d-security-observability-ru.md#research_pkg_d_obzor_paketa)

Исходные сводные материалы (март 2026), на которых основан комплект: «Краткое изложение: методология внедрения и отчуждения ИИ…», «Оценка сайзинга, КапЭкс и ОпЭкс для клиентов».

## Владельцы тем (канонические документы) {: #research_pkg_a_vladeltsy_tem_kanonicheskie_dokumenty }

| Тема | Канонический документ |
| --- | --- |
| Экономика: цифры, тарифы, сценарии сайзинга, CapEx / OpEx / TCO | [«Основной отчёт: сайзинг и экономика»](./20260325-research-report-sizing-economics-main-ru.md#research_sizing_20260325_obzor_paketa) |
| Методология: TOM, фазы внедрения, производственная модель без канона по таблицам затрат | [«Основной отчёт: методология внедрения и разработки»](./20260325-research-report-methodology-main-ru.md#research_methodology_20260325_obzor_paketa) |
| Отчуждение ИС и кода: KT / IP, лицензии, пакет передачи, приёмка | [«Приложение B: отчуждение ИС и кода (KT, IP)»](./20260325-research-appendix-b-ip-code-alienation-ru.md#research_pkg_b_obzor_paketa) |
| Наработки CMW: состав стека, границы «что есть сегодня» | «Приложение C: имеющиеся наработки CMW» |
| Безопасность, комплаенс, observability (углубление) | [«Приложение D: безопасность, комплаенс и observability»](./20260325-research-appendix-d-security-observability-ru.md#research_pkg_d_obzor_paketa) |
| Навигация по пакету, реестр документов и источников | [«Приложение A: обзор и ведомость документов»](#research_pkg_a_obzor_paketa) (этот файл) |

## Навигация «вопрос → документ» {: #research_pkg_a_navigatsiya_vopros_dokument }

- «Как внедрять и разрабатывать в пром контуре (роли, фазы, quality gates)?» → [Основной отчёт: методология внедрения и разработки](./20260325-research-report-methodology-main-ru.md#research_methodology_20260325_obzor_paketa)
- «Где в корпоративном ИИ формируется преимущество (данные, семантика, агенты)?» → [Основной отчёт: методология внедрения и разработки](./20260325-research-report-methodology-main-ru.md#research_methodology_20260325_obzor_paketa)
- «Какие цифры/диапазоны CapEx/OpEx/TCO заложить клиенту и как обосновать?» → [Основной отчёт: сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#research_sizing_20260325_obzor_paketa)
- «Где примерные расчёты расхода токенов по данным портала поддержки и допущениям?» → [Основной отчёт: сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#research_sizing_20260325_primernye_raschety_rashoda_tokenov_na_dostupnyh_dannyh_portal_podderzhki) (подраздел «Примерные расчёты расхода токенов на доступных данных (портал поддержки)»)
- «Как устроен пакет отчуждения ИС/кода и что именно передаём клиенту (KT/IP)?» → [Приложение B: отчуждение ИС и кода (KT, IP)](./20260325-research-appendix-b-ip-code-alienation-ru.md#research_pkg_b_obzor_paketa)
- «Как оформлять бизнес-процессы для KT (BPMN 2.0, генерация LLM, проверка)?» → [Основной отчёт: методология внедрения и разработки](./20260325-research-report-methodology-main-ru.md#research_methodology_20260325_spravochno_bpmn_20_i_generatsiya_llm) и [Приложение B: пакет отчуждения](./20260325-research-appendix-b-ip-code-alienation-ru.md#research_pkg_b_paket_otchuzhdeniya_minimalno_tselostnyi)
- «Что есть в CMW сегодня (состав стека, границы ‘что есть’ vs ‘методология’)?» → Приложение C: имеющиеся наработки CMW
- «Как обеспечить security, комплаенс и промышленную observability (контур контроля, data minimization posture)?» → [Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#research_pkg_d_obzor_paketa)
- «Как проектировать изоляцию и сеть для агентского исполнения (граница доверия, egress, краткоживущие учётные данные)?» → [Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#research_pkg_d_spravochno_granitsa_doveriya_set_i_sreda_ispolneniya_agenta)
- «Какие паттерны среды для агента в PR и долгоживущей dev, модель риска по сценарию и минимальный состав платформы задач?» → [Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#research_pkg_d_spravochno_model_riska_patterny_sredy_i_minimalnyi_sostav_platformy); для KT/IP и PR — [Приложение B: отчуждение ИС и кода (KT, IP)](./20260325-research-appendix-b-ip-code-alienation-ru.md#research_pkg_b_spravochno_agent_v_pr_i_artifacty_vmesto_pryamoi_zapisi_v_is)
- «Как сравнивать E2B / Modal / Daytona и бенчмаркать песочницы (сеть, сессии, метрики прода)?» → [Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#research_pkg_d_spravochno_upravlyaemye_pesochnitsy_sravnenie_modelei_i_benchmarki) и [Основной отчёт: методология внедрения и разработки](./20260325-research-report-methodology-main-ru.md#research_methodology_20260325_spravochno_otsenka_upravlyaemyh_pesochnits_i_benchmarki)
- «Как за ~30 дней вывести безопасный MVP контура исполнения агента, какие враждебные сценарии и критерии готовности?» → [Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#research_pkg_d_spravochno_bezopasnyi_mvp_kontura_ispolneniya_diskussiya_sredy_vyvody) и [Основной отчёт: методология внедрения и разработки](./20260325-research-report-methodology-main-ru.md#research_methodology_20260325_spravochno_uzkii_bezopasn_mvp_kontura_ispolneniya_agenta_orientir_30_dnei)
- «Нужен единый реестр источников и дополнительные источники из исходного задания по темам?» → [Приложение A: обзор и ведомость документов](#research_pkg_a_obzor_paketa) (этот документ)

## Соответствие разделов исходных материалов документам пакета {: #research_pkg_a_sootvetstvie_razdelov_ishodnyh_materialov_dokumentam_paketa }

Ниже для прозрачности показано, в каком документе комплекта закреплён каждый крупный раздел (заголовок второго уровня) исходных сводных материалов. Вложенные подразделы следуют за родительским разделом в том же файле; полный перечень ссылок — в разделе «Источники» ниже.

### Из «Методология внедрения и отчуждения ИИ» (исходный сводный материал, март 2026) {: #research_pkg_a_iz_metodologiya_vnedreniya_i_otchuzhdeniya_ii_ishodnyi_svodnyi_material_mart_202 }

| Заголовок оригинала | Канонический документ в пакете |
| --- | --- |
| `## Назначение документа и границы применения` | Основной отчёт: методология внедрения и разработки |
| `## Резюме для руководства` | Основной отчёт: методология внедрения и разработки |
| `## Целевая операционная модель (Target Operating Model)` | Основной отчёт: методология внедрения и разработки |
| `## Методология внедрения (Этапы и Качество)` | Основной отчёт: методология внедрения и разработки |
| `## Промышленная наблюдаемость LLM, RAG и агентов` | Приложение D: безопасность, комплаенс и observability |
| `## Обзор текущей архитектуры CMW` | Приложение C: имеющиеся наработки CMW |
| `## Детальная архитектура внедрения` | Основной отчёт: методология внедрения и разработки |
| `## Детальная методология отчуждения` | Приложение B: отчуждение ИС и кода (KT, IP) |
| `## Рекомендации по производственной эксплуатации (2026)` | Основной отчёт: методология внедрения и разработки |
| `## Общие рекомендации` | Основной отчёт: методология внедрения и разработки |
| `## Практики и архитектуры RAG: NeuralDeep и продвинутая ретривальная инженерия` | Основной отчёт: методология внедрения и разработки |
| `## Паттерны промышленного RAG и защитных контуров` | Приложение D: безопасность, комплаенс и observability |
| `## Агенты, инструменты, память и обсервабильность (справочный обзор @ai_archnadzor)` | Приложение D: безопасность, комплаенс и observability |
| `## MCP, мультиагентная маршрутизация и воспроизводимые навыки` | Приложение D: безопасность, комплаенс и observability |
| `## Инженерия обвязки для агентов` | Основной отчёт: методология внедрения и разработки |
| `## Практический опыт внедрения ИИ (red_mad_robot)` | Основной отчёт: методология внедрения и разработки |
| `## Российский рынок ИИ: Текущее состояние и Прогнозы (2024-2026)` | Основной отчёт: методология внедрения и разработки |
| `## Методология Enterprise AI (Global Best Practices)` | Основной отчёт: методология внедрения и разработки |
| `## Практические кейсы из каналов` | Основной отчёт: методология внедрения и разработки |
| `## Рекомендации по внедрению ИИ для клиентов` | Основной отчёт: методология внедрения и разработки |
| `## Управление рисками и соответствие (Compliance)` | Приложение D: безопасность, комплаенс и observability |
| `## Матрица принятия решений и экономика (РБК 2026)` | Основной отчёт: сайзинг и экономика |
| `## Рекомендованный план 30/60/90 дней` | Основной отчёт: методология внедрения и разработки |
| `## Выводы по отчуждению ИИ` | Приложение B: отчуждение ИС и кода (KT, IP) |
| `## Обоснование рекомендаций (метод исследования)` | Основной отчёт: методология внедрения и разработки |
| `## Источники` | Приложение A: обзор и ведомость документов |

### Из «Сайзинг, CapEx и OpEx для клиентов» (исходный сводный материал, март 2026) {: #research_pkg_a_iz_saizing_capex_i_opex_dlya_klientov_ishodnyi_svodnyi_material_mart_2026 }

| Заголовок оригинала | Канонический документ в пакете |
| --- | --- |
| `## Назначение документа и границы применения` | Основной отчёт: сайзинг и экономика |
| `## Резюме для руководства` | Основной отчёт: сайзинг и экономика |
| `## Обзор` | Основной отчёт: сайзинг и экономика |
| `## Дерево факторов стоимости (Cost Factor Tree)` | Основной отчёт: сайзинг и экономика |
| `## Сценарный сайзинг (Scenario Sizing)` | Основной отчёт: сайзинг и экономика |
| `## CapEx и OpEx: роли интегратора и заказчика` | Основной отчёт: сайзинг и экономика |
| `## CapEx / OpEx Модель (Данные РБК 2026)` | Основной отчёт: сайзинг и экономика |
| `## Юнит-экономика и анализ чувствительности` | Основной отчёт: сайзинг и экономика |
| `## Сборник мер по оптимизации стоимости (Cost Optimization Suite)` | Основной отчёт: сайзинг и экономика |
| `## Риски бюджета и меры снижения` | Основной отчёт: сайзинг и экономика |
| `## Тарифы российских облачных провайдеров ИИ` | Основной отчёт: сайзинг и экономика |
| `## Детальный анализ аппаратных требований` | Основной отчёт: сайзинг и экономика |
| `### Профиль on-prem GPU в проектах CMW` | [Основной отчёт: сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#research_sizing_20260325_profil_onprem_gpu_v_proektah_cmw) |
| `## Детальные капитальные затраты (CapEx)` | Основной отчёт: сайзинг и экономика |
| `## Детальные операционные затраты (OpEx)` | Основной отчёт: сайзинг и экономика |
| `## Анализ общей стоимости владения (TCO)` | Основной отчёт: сайзинг и экономика |
| `## Рекомендации по сайзингу для клиентов` | Основной отчёт: сайзинг и экономика |
| `## Дополнительные стратегии оптимизации затрат` | Основной отчёт: сайзинг и экономика |
| `## Промежуточное заключение по сайзингу` | Основной отчёт: сайзинг и экономика |
| `### Бенчмарки RTX 4090 (реф. 24 ГБ GeForce; у CMW — 48 ГБ кастом)` | [Основной отчёт: сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#research_sizing_20260325_benchmarki_rtx_4090_24_gb) |
| `### Ориентиры сообщества: Qwen3.5-35B-A3B и потребительское железо (март 2026)` | [Основной отчёт: сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#research_sizing_20260325_orientiry_soobschestva_qwen3_5_35b_a3b_i_potrebitelskoe_zhelezo_mart_2026) |
| `## Актуальные AI/ML тренды из канала @ai_machinelearning_big_data` | Основной отчёт: сайзинг и экономика |
| `## Оптимизация затрат на инференс (практический опыт)` | Основной отчёт: сайзинг и экономика |
| `## Локальный инференс: практические кейсы` | Основной отчёт: сайзинг и экономика |
| `## Рынок AI: статистика a16z (March 2026)` | Основной отчёт: сайзинг и экономика |
| `## Модели и ценообразование (March 2026)` | Основной отчёт: сайзинг и экономика |
| `## Глобальный рынок ИИ-инфраструктуры: CapEx и OpEx (2025-2026)` | Основной отчёт: сайзинг и экономика |
| `## VRAM Requirements для LLM Inference` | Основной отчёт: сайзинг и экономика |
| `## Корректировка TCO для российского рынка` | Основной отчёт: сайзинг и экономика |
| `## ИИ-рынок России: Статистика и прогнозы` | Основной отчёт: сайзинг и экономика |
| `## Практические рекомендации по сайзингу (Decision Tree)` | Основной отчёт: сайзинг и экономика |
| `## Новые тренды 2026 (Дополнительно)` | Основной отчёт: сайзинг и экономика |
| `## Планирование мощности ИИ-инфраструктуры (2025-2030)` | Основной отчёт: сайзинг и экономика |
| `## Методология отчуждения ИИ-активов` | Приложение B: отчуждение ИС и кода (KT, IP) |
| `## Методология ROI для ИИ-проектов` | Основной отчёт: методология внедрения и разработки |
| `## Резюме по методологии отчуждения` | Приложение B: отчуждение ИС и кода (KT, IP) |
| `## Заключение` | Основной отчёт: сайзинг и экономика |
| `## Обоснование рекомендаций и границы документа` | Основной отчёт: сайзинг и экономика |
| `## Источники` | Приложение A: обзор и ведомость документов |

### Справка о структуре {: #research_pkg_a_spravka_o_strukture }

Каждый раздел исходных материалов с заголовком второго уровня (`##`) в комплекте имеет ровно один канонический документ; подзаголовки `###` и `####` остаются внутри соответствующего раздела и переносятся вместе с ним.

Отдельные разделы второго уровня в **основном отчёте по методологии** могут добавляться **после** выпуска исходного сводного материала (март 2026) как уточнения рамки комплекта без изменения соответствия таблицы выше для строк исходника; так, раздел «Источник преимущества в корпоративном ИИ (2026)…» не имел одноимённого заголовка в исходном файле и закреплён канонически в основном отчёте по методологии.

## Источники {: #research_pkg_a_istochniki }

### Источники из «Методология внедрения и отчуждения ИИ» (использовано в пакете) {: #research_pkg_a_istochniki_iz_metodologiya_vnedreniya_i_otchuzhdeniya_ii_ispolzovano_v_pakete }

#### Инженерия обвязки и мультиагентная разработка {: #research_pkg_a_inzheneriya_obvyazki_i_multiagentnaya_razrabotka }

- [OpenAI — Harness engineering](https://openai.com/ru-RU/index/harness-engineering/)
- [Anthropic — Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [Anthropic — Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Martin Fowler — Harness Engineering (Thoughtworks)](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)
- [Хабр — Инженер будущего строит обвязку для агентов](https://habr.com/ru/articles/1005032/)

#### OWASP GenAI Security, тестирование и адаптации на русском {: #research_pkg_a_owasp_genai_security_testirovanie_i_adaptatsii_na_russkom }

- [OWASP Gen AI Security Project — Introduction](https://genai.owasp.org/introduction-genai-security-project/)
- [OWASP — проект Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [GenAI Security — OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/)
- [GenAI Security — OWASP Top 10 for Agentic Applications for 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
- [OWASP — Web Security Testing Guide (WSTG), stable](https://owasp.org/www-project-web-security-testing-guide/stable/)
- [GitHub — OWASP Application Security Verification Standard 5.0.0 (PDF, RU)](https://github.com/OWASP/ASVS/blob/master/5.0/OWASP_Application_Security_Verification_Standard_5.0.0_ru.pdf)
- [GitHub — OWASP www-project-ai-testing-guide](https://github.com/OWASP/www-project-ai-testing-guide)
- [Habr — OWASP: LLM TOP 10 2025 (адаптация)](https://habr.com/ru/companies/owasp/articles/893712/)
- [Habr — OWASP (вводные по тестированию и материалам сообщества)](https://habr.com/ru/companies/owasp/articles/817241/)
- [Habr — OWASP (смежные публикации сообщества)](https://habr.com/ru/companies/owasp/articles/896328/)
- [Habr — OWASP (смежные публикации сообщества)](https://habr.com/ru/companies/owasp/articles/900276/)
- [Habr — OWASP (смежные публикации сообщества)](https://habr.com/ru/companies/owasp/articles/896328/)

#### Угрозы GenAI и иллюстративные материалы третьих лиц (не реклама) {: #research_pkg_a_ugrozy_genai_i_illyustrativnye_materialy_tretih_lits_ne_reklama }

- [Kaspersky Resource Center — What Is Prompt Injection?](https://www.kaspersky.com/resource-center/threats/prompt-injection)
- [Kaspersky Blog — How LLMs can be compromised in 2025](https://www.kaspersky.com/blog/new-llm-attack-vectors-2025/54323/)
- [Kaspersky Blog — Agentic AI security measures and OWASP ASI Top 10](https://www.kaspersky.com/blog/top-agentic-ai-risks-2026/29988/)
- [Securelist — webinar: AI agents vs. prompt injections](https://securelist.com/webinars/ai-agents-vs-prompt-injections/)
- [Kaspersky — press release: training Large Language Models Security (описание программы)](https://www.kaspersky.com/about/press-releases/kaspersky-introduces-a-new-training-large-language-models-security)
- [CodeWall — How we hacked McKinsey’s AI platform (разбор red team)](https://codewall.ai/blog/how-we-hacked-mckinseys-ai-platform)

#### Нормативные и стратегические материалы {: #research_pkg_a_normativnye_i_strategicheskie_materialy }

- [ISO/IEC 42001:2023 — Artificial intelligence management system](https://www.iso.org/standard/81230.html)
- [NIST — AI RMF to ISO/IEC 42001 Crosswalk (PDF)](https://airc.nist.gov/docs/NIST_AI_RMF_to_ISO_IEC_42001_Crosswalk.pdf)
- [NIST — AI RMF: Generative AI Profile (NIST.AI.600-1, 2024)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
- [NIST AIRC — Roadmap for the AI Risk Management Framework](https://airc.nist.gov/airmf-resources/roadmap)
- [Официальное опубликование — Приказ Роскомнадзора от 19.06.2025 № 140 (обезличивание ПДн)](http://publication.pravo.gov.ru/document/0001202508010002)

#### Данные и стратегические сигналы {: #research_pkg_a_dannye_i_strategicheskie_signaly }

- [Gartner — пресс-релиз: нехватка AI-ready data подрывает ИИ-проекты (26.02.2025)](https://www.gartner.com/en/newsroom/press-releases/2025-02-26-lack-of-ai-ready-data-puts-ai-projects-at-risk)

#### Стек инференса (MOSEC, vLLM) и открытая документация {: #research_pkg_a_stek_inferensa_mosec_vllm_i_otkrytaya_dokumentatsiya }

- [MOSEC — документация](https://mosecorg.github.io/mosec/index.html)
- [mosecorg/mosec (GitHub)](https://github.com/mosecorg/mosec)
- [vLLM — OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)
- [сервер инференса MOSEC — README проекта (пример публичного зеркала)](https://github.com/arterm-sedov/cmw-mosec)

#### Экономика, рынок, enterprise AI {: #research_pkg_a_ekonomika_rynok_enterprise_ai }

- [Yakov & Partners — публикация AI 2025](https://yakovpartners.com/publications/ai-2025/)
- [MarketsandMarkets — Russia AI Inference PaaS Market](https://www.marketsandmarkets.com/ResearchInsight/russia-ai-inference-platform-as-a-service-paas-market.asp)
- [IMARC — Russia Artificial Intelligence Market](https://www.imarcgroup.com/russia-artificial-intelligence-market)
- [Larridin — State of Enterprise AI in 2025](https://larridin.com/blog/state-of-enterprise-ai-in-2025)
- [Dataoorts — GPU cloud providers in Russia](https://dataoorts.com/top-5-plus-gpu-cloud-providers-in-russia/)
- [ITNext — GPU infrastructure as foundational to enterprise AI strategy](https://itnext.io/why-gpu-infrastructure-is-foundational-to-an-enterprise-ai-strategy-5b574ef1eebc)
- [РБК Education — во сколько обойдётся ИИ-агент: подсчёты экспертов (2026)](https://www.rbc.ru/education/05/02/2026/697162269a794772c9cf9991)
- [a16z — Top 100 Gen AI Apps (6)](https://a16z.com/100-gen-ai-apps-6/)
- [FinOps Foundation — Generative AI / Unit Economics](https://www.finops.org/wg/generative-ai/)
- [FinOps Foundation — Framework: Unit Economics (Capability)](https://www.finops.org/framework/capabilities/unit-economics/)
- [Anthropic — Introducing Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)
- [Anthropic — Introducing Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)
- [Claude Docs — What's new in Claude 4.6](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-6)
- [Claude Docs — Models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Хабр — Релиз Claude Opus 4.6](https://habr.com/ru/news/993322/)
- [Microsoft Research — Fara-7B: An Efficient Agentic Model for Computer Use (PDF)](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/11/Fara-7B-An-Efficient-Agentic-Model-for-Computer-Use.pdf)

#### Оценка качества и мониторинг (LangSmith) {: #research_pkg_a_otsenka_kachestva_i_monitoring_langsmith }

- [LangChain Docs — Evaluation concepts (LangSmith)](https://docs.langchain.com/langsmith/evaluation-concepts)
- [LangSmith — Online evaluations (how-to)](https://docs.smith.langchain.com/observability/how_to_guides/online_evaluations)

#### Исследования (edge–cloud routing, агентная память и обучение; ориентиры НИОКР) {: #research_pkg_a_issledovaniya_edge_cloud_routing_agentnaya_pamyat_i_obuchenie_orientiry_niokr }

- [arXiv — HybridFlow: Resource-Adaptive Subtask Routing for Edge-Cloud LLM Inference](https://arxiv.org/html/2512.22137v4)
- [arXiv — PRISM: Privacy-Aware Routing for Cloud-Edge LLM Inference](https://arxiv.org/html/2511.22788v1)
- [arXiv — Agent0: co-evolving curriculum and executor agents](https://arxiv.org/pdf/2511.16043)
- [arXiv — General Agentic Memory (GAM)](https://arxiv.org/pdf/2511.18423)
- [arXiv — MoE на стеке AMD (IBM, Zyphra и др.)](https://arxiv.org/pdf/2511.17127)
- [arXiv — Moonshot AI: ускорение синхронного RL](https://arxiv.org/pdf/2511.14617)

#### Облачные провайдеры и тарифы (РФ) {: #research_pkg_a_oblachnye_provaidery_i_tarify_rf }

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

#### Иллюстративная лицензия и примеры (YandexGPT-5-Lite-8B) {: #research_pkg_a_illyustrativnaya_litsenziya_i_primery_yandexgpt_5_lite_8b }

- [Hugging Face — LICENSE (YandexGPT-5-Lite-8B), сырой текст соглашения](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct/raw/main/LICENSE)
- [Hugging Face — карточка модели YandexGPT-5-Lite-8B-instruct](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct)

#### Открытые модели ai-sage (GigaChat и спутники, TCO) {: #research_pkg_a_otkrytye_modeli_ai_sage_gigachat_i_sputniki_tco }

- [Хабр — GigaChat-3.1: большое обновление больших моделей (блог Сбера)](https://habr.com/ru/companies/sberbank/articles/1014146/)
- [Hugging Face — организация ai-sage](https://huggingface.co/ai-sage)
- [Hugging Face — коллекция GigaChat 3.1](https://huggingface.co/collections/ai-sage/gigachat-31)
- [Hugging Face — ai-sage/GigaChat3.1-702B-A36B (Ultra)](https://huggingface.co/ai-sage/GigaChat3.1-702B-A36B)
- [Hugging Face — ai-sage/GigaChat3.1-10B-A1.8B (Lightning 3.1)](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B)
- [Hugging Face — ai-sage/GigaChat3-10B-A1.8B (Lightning 3.0)](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B)

#### Операционные и «рядом» материалы {: #research_pkg_a_operatsionnye_i_ryadom_materialy }

- [OpenTelemetry — Semantic conventions for generative client AI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- [OpenTelemetry — Semantic conventions for generative AI metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)
- [Langfuse — документация observability / tracing](https://langfuse.com/docs/observability/get-started)
- [Arize Phoenix — документация](https://docs.arize.com/phoenix)
- [LangSmith — документация](https://docs.smith.langchain.com/)

### Источники из «Оценка сайзинга, КапЭкс и ОпЭкс для клиентов» (использовано в пакете) {: #research_pkg_a_istochniki_iz_otsenka_saizinga_kapeks_i_opeks_dlya_klientov_ispolzovano_v_pakete }

#### Примерные расчёты токенов (портал поддержки, агрегаторы и обзоры прайсов) {: #research_pkg_a_primernye_raschety_tokenov_portal_podderzhki_agregatory_i_obzory_praisov }

- [Портал поддержки Comindware](https://support.comindware.com/)
- [LLMoney — калькулятор цен токенов LLM](https://llmoney.ru)
- [Хабр — обзор цен на токены](https://habr.com/ru/articles/1000058/)
- [Хабр — гид по топ-20 нейросетям для текстов (в т.ч. цены)](https://habr.com/ru/articles/948672/)
- [VC.ru — гайд по тарифам Claude и доступу из России](https://vc.ru/ai/2757771-tarify-claude-2026-gayd-po-planam-i-dostupu-iz-rossii)

#### Инженерия обвязки и мультиагентная разработка {: #research_pkg_a_inzheneriya_obvyazki_i_multiagentnaya_razrabotka_2 }

- [OpenAI — Harness engineering](https://openai.com/ru-RU/index/harness-engineering/)
- [Anthropic — Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [Anthropic — Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Martin Fowler — Harness Engineering (Thoughtworks)](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)
- [Хабр — Инженер будущего строит обвязку для агентов](https://habr.com/ru/articles/1005032/)

#### Агентная память и модели (ориентиры НИОКР и прайсинга, не строка КП) {: #research_pkg_a_agentnaya_pamyat_i_modeli_orientiry_niokr_i_praisinga_ne_stroka_kp }

- [arXiv — General Agentic Memory (GAM)](https://arxiv.org/pdf/2511.18423)
- [Anthropic — Introducing Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)
- [Anthropic — Introducing Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)
- [Claude Docs — What's new in Claude 4.6](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-6)
- [Claude Docs — Models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Хабр — Релиз Claude Opus 4.6](https://habr.com/ru/news/993322/)
- [Anthropic — Pricing](https://www.anthropic.com/pricing)
- [Microsoft Research — Fara-7B (PDF)](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/11/Fara-7B-An-Efficient-Agentic-Model-for-Computer-Use.pdf)

#### Безопасность GenAI, OWASP и сигналы рынка (TCO / риски) {: #research_pkg_a_bezopasnost_genai_owasp_i_signaly_rynka_tco_riski }

- [OWASP GenAI Security — Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/)
- [OWASP GenAI Security — Top 10 for Agentic Applications for 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
- [GitHub — NVIDIA Garak (сканер для LLM, только изолированные стенды)](https://github.com/NVIDIA/garak)
- [CodeWall — разбор red team: McKinsey AI platform](https://codewall.ai/blog/how-we-hacked-mckinseys-ai-platform)
- [Коммерсантъ — рынок и атаки на ИИ-системы (журналистский контекст)](https://www.kommersant.ru/doc/8363105)
- [OpenAI — приобретение PromptFoo (контекст рынка тестирования)](https://openai.com/index/openai-to-acquire-promptfoo/)
- [Kaspersky — пресс-релиз: угрозы под видом популярных ИИ-сервисов (бенчмарк тренда)](https://www.kaspersky.com/about/press-releases/kaspersky-chatgpt-mimicking-cyberthreats-surge-115-in-early-2025-smbs-increasingly-targeted)

#### Иллюстративные ориентиры нагрузки (публичные интервью, финсектор) {: #research_pkg_a_illyustrativnye_orientiry_nagruzki_publichnye_intervyu_finsektor }

- [CIO — интервью: чат-бот, масштаб обращений и сценарии](https://cio.osp.ru/articles/5455)
- [«Открытые системы» — RAG и LLM для поддержки операционистов](https://www.osp.ru/articles/2025/0324/13059305)

#### Облачные провайдеры и тарифы (РФ) {: #research_pkg_a_oblachnye_provaidery_i_tarify_rf_2 }

- [Cloud.ru — Evolution Foundation Models (продукт, перечень моделей)](https://cloud.ru/products/evolution-foundation-models)
- [Cloud.ru — Evolution Foundation Models, тарифы (2026)](https://cloud.ru/documents/tariffs/evolution/foundation-models)
- [Yandex AI Studio — доступные генеративные модели](https://aistudio.yandex.ru/docs/ru/ai-studio/concepts/generation/models.html)
- [Yandex AI Studio — правила тарификации](https://aistudio.yandex.ru/docs/ru/ai-studio/pricing.html)
- [AKM.ru — доступ к крупнейшей языковой модели на рынке РФ (Yandex B2B)](https://www.akm.ru/eng/press/yandex-b2b-tech-has-opened-access-to-the-largest-language-model-on-the-russian-market/)
- [Сбер — GigaChat API: юридические тарифы](https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs)
- [Selectel — Foundation Models Catalog](https://selectel.ru/services/cloud/foundation-models-catalog)
- [MWS — MWS GPT](https://mws.ru/mws-gpt/)
- [MWS — тарифы MWS GPT](https://mws.ru/docs/docum/cloud_terms_mwsgpt_pricing.html)
- [МТС Cloud — виртуальная инфраструктура с GPU](https://cloud.mts.ru/services/virtual-infrastructure-gpu/)
- [MWS — GPU On‑premises](https://mws.ru/services/mws-gpu-on-prem/)
- [CNews — кейс: MTS AI и экономия инвестиций за счёт облака MWS (обзор)](https://www.cnews.ru/reviews/provajdery_gpu_cloud_2025/cases/kak_mts_ai_sekonomila_bolee_milliarda)
- [CIO — MTS AI перенесла обучение моделей в облако](https://cio.osp.ru/articles/140525-MTS-AI-perenesla-obuchenie-modeley-v-oblako)
- [VK Cloud — машинное обучение в облаке](https://cloud.vk.com/docs/ru/ml)
- [Google — условия использования Gemma](https://ai.google.dev/gemma/terms)

#### Публичные веса с нестандартной лицензией {: #research_pkg_a_publichnye_vesa_s_nestandartnoi_litsenziei }

- [Hugging Face — LICENSE (YandexGPT-5-Lite-8B), сырой текст соглашения](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct/raw/main/LICENSE)
- [Hugging Face — карточка модели YandexGPT-5-Lite-8B-instruct](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct)

#### Открытые проекты ai-sage (GigaChat и спутники) {: #research_pkg_a_otkrytye_proekty_ai_sage_gigachat_i_sputniki }

- [Хабр — GigaChat-3.1: большое обновление больших моделей (блог Сбера)](https://habr.com/ru/companies/sberbank/articles/1014146/)
- [Hugging Face — организация ai-sage](https://huggingface.co/ai-sage)
- [Hugging Face — коллекция GigaChat 3.1](https://huggingface.co/collections/ai-sage/gigachat-31)
- [Hugging Face — ai-sage/GigaChat3.1-702B-A36B (Ultra)](https://huggingface.co/ai-sage/GigaChat3.1-702B-A36B)
- [Hugging Face — ai-sage/GigaChat3.1-10B-A1.8B (Lightning 3.1)](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B)
- [Hugging Face — ai-sage/GigaChat3.1-10B-A1.8B (Lightning 3.0)](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B)

#### Инструменты разработки с ИИ (ориентиры) {: #research_pkg_a_instrumenty_razrabotki_s_ii_orientiry }

- [OpenCode](https://opencode.ai/)
- [OpenCode — документация (Intro)](https://opencode.ai/docs)
- [OpenCode — Ecosystem](https://opencode.ai/docs/ecosystem/)
- [OpenCode Zen](https://opencode.ai/docs/zen)
- [OpenWork (different-ai/openwork)](https://github.com/different-ai/openwork)
- [OpenRouter](https://openrouter.ai/)
- [OpenRouter — Logging и политики провайдеров](https://openrouter.ai/docs/guides/privacy/logging)

#### Финансовая и инфраструктурная база (FinOps/TCO/железо) {: #research_pkg_a_finansovaya_i_infrastrukturnaya_baza_finops_tco_zhelezo }

- [FinOps Foundation — Generative AI (Unit Economics)](https://www.finops.org/wg/generative-ai/)
- [FinOps Framework — Unit Economics (capability)](https://www.finops.org/framework/capabilities/unit-economics/)
- [OpenAI — Prompt caching (снижение стоимости повторяющегося контекста)](https://platform.openai.com/docs/guides/prompt-caching)
- [Slyd — TCO Calculator: On-Prem vs Cloud](https://slyd.com/resources/tco-calculator)
- [Introl — финансирование CapEx/OpEx и инвестиции в GPU](https://introl.com/blog/ai-infrastructure-financing-capex-opex-gpu-investment-guide-2025)
- [SWFTE — экономика частного AI / on-prem](https://www.swfte.com/blog/private-ai-enterprises-onprem-economics)
- [Runpod — LLM inference optimization playbook (throughput)](https://www.runpod.io/articles/guides/llm-inference-optimization-playbook)
- [Introl — планирование мощностей ИИ-инфраструктуры (прогнозы, McKinsey в обзоре)](https://introl.com/blog/ai-infrastructure-capacity-planning-forecasting-gpu-2025-2030)
- [PitchGrade — AI Infrastructure Primer](https://pitchgrade.com/research/ai-infrastructure-primer)
- [Medium — Qwen 3.5 35B A3B (AgentNativeDev)](https://agentnativedev.medium.com/qwen-3-5-35b-a3b-why-your-800-gpu-just-became-a-frontier-class-ai-workstation-63cc4d4ebac1)
- [apxml.com — VRAM calculator](https://apxml.com/tools/vram-calculator)
- [Hugging Face — Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

#### Наблюдаемость и телеметрия {: #research_pkg_a_nablyudaemost_i_telemetriya }

- [OpenTelemetry — Semantic conventions for generative client AI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- [OpenTelemetry — Semantic conventions for generative AI metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)
- [OpenTelemetry — OpenTelemetry for Generative AI (блог)](https://opentelemetry.io/blog/2024/otel-generative-ai)
- [OpenInference — инструментирование ИИ для OpenTelemetry](https://arize-ai.github.io/openinference/)
- [Arize Phoenix — документация](https://docs.arize.com/phoenix)
- [LangSmith — документация](https://docs.smith.langchain.com/)
- [LangChain Docs — Evaluation concepts (LangSmith)](https://docs.langchain.com/langsmith/evaluation-concepts)
- [Официальное опубликование — Приказ Роскомнадзора от 19.06.2025 № 140 (обезличивание ПДн)](http://publication.pravo.gov.ru/document/0001202508010002)
- [Langfuse — документация observability](https://langfuse.com/docs/observability/get-started)

## Полный реестр использованных источников (точная консолидация) {: #research_pkg_a_polnyi_reestr_ispolzovannyh_istochnikov_tochnaya_konsolidatsiya }

### Источники из «Методология внедрения и отчуждения ИИ» (без сокращений) {: #research_pkg_a_istochniki_iz_metodologiya_vnedreniya_i_otchuzhdeniya_ii_bez_sokraschenii }

#### Инженерия обвязки и мультиагентная разработка {: #research_pkg_a_inzheneriya_obvyazki_i_multiagentnaya_razrabotka_3 }

- [OpenAI — Harness engineering](https://openai.com/ru-RU/index/harness-engineering/)
- [Anthropic — Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [Anthropic — Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Martin Fowler — Harness Engineering (Thoughtworks)](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)
- [Хабр — Инженер будущего строит обвязку для агентов](https://habr.com/ru/articles/1005032/)

#### OWASP GenAI Security, тестирование и адаптации на русском {: #research_pkg_a_owasp_genai_security_testirovanie_i_adaptatsii_na_russkom_2 }

- [OWASP Gen AI Security Project — Introduction](https://genai.owasp.org/introduction-genai-security-project/)
- [OWASP — проект Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [GenAI Security — OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/)
- [GenAI Security — OWASP Top 10 for Agentic Applications for 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
- [OWASP — Web Security Testing Guide (WSTG), stable](https://owasp.org/www-project-web-security-testing-guide/stable/)
- [GitHub — OWASP Application Security Verification Standard 5.0.0 (PDF, RU)](https://github.com/OWASP/ASVS/blob/master/5.0/OWASP_Application_Security_Verification_Standard_5.0.0_ru.pdf)
- [GitHub — OWASP www-project-ai-testing-guide](https://github.com/OWASP/www-project-ai-testing-guide)
- [Habr — OWASP: LLM TOP 10 2025 (адаптация)](https://habr.com/ru/companies/owasp/articles/893712/)
- [Habr — OWASP (вводные по тестированию и материалам сообщества)](https://habr.com/ru/companies/owasp/articles/817241/)
- [Habr — OWASP (смежные публикации сообщества)](https://habr.com/ru/companies/owasp/articles/896328/)
- [Habr — OWASP (смежные публикации сообщества)](https://habr.com/ru/companies/owasp/articles/900276/)

#### Угрозы GenAI и иллюстративные материалы третьих лиц (не реклама) {: #research_pkg_a_ugrozy_genai_i_illyustrativnye_materialy_tretih_lits_ne_reklama_2 }

- [Kaspersky Resource Center — What Is Prompt Injection?](https://www.kaspersky.com/resource-center/threats/prompt-injection)
- [Kaspersky Blog — How LLMs can be compromised in 2025](https://www.kaspersky.com/blog/new-llm-attack-vectors-2025/54323/)
- [Kaspersky Blog — Agentic AI security measures and OWASP ASI Top 10](https://www.kaspersky.com/blog/top-agentic-ai-risks-2026/29988/)
- [Securelist — webinar: AI agents vs. prompt injections](https://securelist.com/webinars/ai-agents-vs-prompt-injections/)
- [Kaspersky — press release: training Large Language Models Security (описание программы)](https://www.kaspersky.com/about/press-releases/kaspersky-introduces-a-new-training-large-language-models-security)
- [CodeWall — How we hacked McKinsey’s AI platform (разбор red team)](https://codewall.ai/blog/how-we-hacked-mckinseys-ai-platform)

#### Нормативные и стратегические материалы {: #research_pkg_a_normativnye_i_strategicheskie_materialy_2 }

- [ISO/IEC 42001:2023 — Artificial intelligence management system](https://www.iso.org/standard/81230.html)
- [NIST — AI RMF to ISO/IEC 42001 Crosswalk (PDF)](https://airc.nist.gov/docs/NIST_AI_RMF_to_ISO_IEC_42001_Crosswalk.pdf)
- [NIST — AI RMF: Generative AI Profile (NIST.AI.600-1, 2024)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
- [NIST AIRC — Roadmap for the AI Risk Management Framework](https://airc.nist.gov/airmf-resources/roadmap)
- [Официальное опубликование — Приказ Роскомнадзора от 19.06.2025 № 140 (обезличивание ПДн)](http://publication.pravo.gov.ru/document/0001202508010002)
- [Фонтанка — проект закона о госрегулировании ИИ (Минцифры, 18.03.2026)](https://www.fontanka.ru/2026/03/18/76318717/)
- [ACSOUR — обязанность операторов передавать анонимизированные ПД в ГИС (152-ФЗ)](https://acsour.com/en/news-and-articles/tpost/2g13ahnab1-mandatory-anonymized-personal-data-shari)
- [DataGuidance — поправки к национальной стратегии развития ИИ РФ](https://www.dataguidance.com/news/russia-president-issues-amendments-national-ai)
- [Известия (EN) — создание офисов внедрения ИИ](https://en.iz.ru/en/node/1985740)

#### Данные и стратегические сигналы {: #research_pkg_a_dannye_i_strategicheskie_signaly_2 }

- [Gartner — пресс-релиз: нехватка AI-ready data подрывает ИИ-проекты (26.02.2025)](https://www.gartner.com/en/newsroom/press-releases/2025-02-26-lack-of-ai-ready-data-puts-ai-projects-at-risk)

#### Стек инференса (MOSEC, vLLM) и открытая документация {: #research_pkg_a_stek_inferensa_mosec_vllm_i_otkrytaya_dokumentatsiya_2 }

- [MOSEC — документация](https://mosecorg.github.io/mosec/index.html)
- [mosecorg/mosec (GitHub)](https://github.com/mosecorg/mosec)
- [vLLM — OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)
- [сервер инференса MOSEC — README проекта (пример публичного зеркала)](https://github.com/arterm-sedov/cmw-mosec)

#### Экономика, рынок, enterprise AI {: #research_pkg_a_ekonomika_rynok_enterprise_ai_2 }

- [Yakov & Partners — публикация AI 2025](https://yakovpartners.com/publications/ai-2025/)
- [MarketsandMarkets — Russia AI Inference PaaS Market](https://www.marketsandmarkets.com/ResearchInsight/russia-ai-inference-platform-as-a-service-paas-market.asp)
- [Larridin — State of Enterprise AI in 2025](https://larridin.com/blog/state-of-enterprise-ai-in-2025)
- [ITNext — GPU infrastructure as foundational to enterprise AI strategy](https://itnext.io/why-gpu-infrastructure-is-foundational-to-an-enterprise-ai-strategy-5b574ef1eebc)
- [РБК Education — во сколько обойдётся ИИ-агент: подсчёты экспертов (2026)](https://www.rbc.ru/education/05/02/2026/697162269a794772c9cf9991)
- [FinOps Foundation — Generative AI / Unit Economics](https://www.finops.org/wg/generative-ai/)
- [FinOps Foundation — Framework: Unit Economics (Capability)](https://www.finops.org/framework/capabilities/unit-economics/)
- [Anthropic — Introducing Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)
- [Anthropic — Introducing Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)
- [Claude Docs — What's new in Claude 4.6](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-6)
- [Claude Docs — Models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Хабр — Релиз Claude Opus 4.6](https://habr.com/ru/news/993322/)
- [Microsoft Research — Fara-7B: An Efficient Agentic Model for Computer Use (PDF)](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/11/Fara-7B-An-Efficient-Agentic-Model-for-Computer-Use.pdf)

#### Оценка качества и мониторинг (LangSmith) {: #research_pkg_a_otsenka_kachestva_i_monitoring_langsmith_2 }

- [LangChain Docs — Evaluation concepts (LangSmith)](https://docs.langchain.com/langsmith/evaluation-concepts)
- [LangSmith — Online evaluations (how-to)](https://docs.smith.langchain.com/observability/how_to_guides/online_evaluations)

#### Исследования (edge–cloud routing, агентная память и обучение; ориентиры НИОКР) {: #research_pkg_a_issledovaniya_edge_cloud_routing_agentnaya_pamyat_i_obuchenie_orientiry_niokr_2 }

- [arXiv — HybridFlow: Resource-Adaptive Subtask Routing for Edge-Cloud LLM Inference](https://arxiv.org/html/2512.22137v4)
- [arXiv — PRISM: Privacy-Aware Routing for Cloud-Edge LLM Inference](https://arxiv.org/html/2511.22788v1)
- [arXiv — Agent0: co-evolving curriculum and executor agents](https://arxiv.org/pdf/2511.16043)
- [arXiv — General Agentic Memory (GAM)](https://arxiv.org/pdf/2511.18423)
- [arXiv — MoE на стеке AMD (IBM, Zyphra и др.)](https://arxiv.org/pdf/2511.17127)
- [arXiv — Moonshot AI: ускорение синхронного RL](https://arxiv.org/pdf/2511.14617)

#### Облачные провайдеры и тарифы (РФ) {: #research_pkg_a_oblachnye_provaidery_i_tarify_rf_3 }

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

#### Публичные веса с нестандартной лицензией {: #research_pkg_a_publichnye_vesa_s_nestandartnoi_litsenziei_2 }

- [Hugging Face — LICENSE (YandexGPT-5-Lite-8B), сырой текст соглашения](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct/raw/main/LICENSE)
- [Hugging Face — карточка модели YandexGPT-5-Lite-8B-instruct](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct)
- [Yandex Research — обзор направлений работ (2025)](https://research.yandex.com/blog/yandex-research-in-2025)
- [Yandex Research — принятые к ICML 2025 (список, в т.ч. KV-cache)](https://research.yandex.com/blog/papers-accepted-to-icml-2025)
- [arXiv — Cache Me If You Must (KV-quantization), 2501.19392](https://arxiv.org/abs/2501.19392)

#### Открытые модели ai-sage (GigaChat и спутники) {: #research_pkg_a_otkrytye_modeli_ai_sage_gigachat_i_sputniki }

- [Хабр — GigaChat-3.1: большое обновление больших моделей (блог Сбера)](https://habr.com/ru/companies/sberbank/articles/1014146/)
- [Hugging Face — организация ai-sage](https://huggingface.co/ai-sage)
- [Hugging Face — коллекция GigaChat 3.1](https://huggingface.co/collections/ai-sage/gigachat-31)
- [Hugging Face — ai-sage/GigaChat3.1-702B-A36B (Ultra)](https://huggingface.co/ai-sage/GigaChat3.1-702B-A36B)
- [Hugging Face — ai-sage/GigaChat3.1-10B-A1.8B (Lightning 3.1)](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B)
- [Hugging Face — ai-sage/GigaChat3.1-10B-A1.8B (Lightning 3.0)](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B)
- [Hugging Face — коллекция GigaEmbeddings](https://huggingface.co/collections/ai-sage/gigaembeddings)
- [Hugging Face — коллекция GigaAM](https://huggingface.co/collections/ai-sage/gigaam)
- [Hugging Face — коллекция GigaChat Lite](https://huggingface.co/collections/ai-sage/gigachat-lite)
- [GitVerse — GigaTeam/gigachat3.1](https://gitverse.ru/GigaTeam/gigachat3.1)
- [GitHub — sgl-project/sglang, PR #18802](https://github.com/sgl-project/sglang/pull/18802)

#### Публичные материалы Ozon Tech {: #research_pkg_a_publichnye_materialy_ozon_tech }

- [Хабр — Ozon Tech: пересборка конструктора чат-ботов (Bots Factory, no-code, масштаб)](https://habr.com/ru/companies/ozontech/articles/834812/)
- [Хабр — Ozon Tech: Query Prediction, ANN и обратный индекс](https://habr.com/ru/companies/ozontech/articles/990180/)
- [Хабр — Ozon Tech: анонс ML&DS Meetup (MLOps, программа докладов)](https://habr.com/ru/companies/ozontech/articles/768734/)
- [GitHub — организация ozontech (открытые репозитории)](https://github.com/ozontech)

#### Методологии внедрения и отраслевые практики {: #research_pkg_a_metodologii_vnedreniya_i_otraslevye_praktiki }

- [Just AI — корпоративный GenAI (упоминается как практикующий вендор)](https://just-ai.com/ru/)
- [Luxoft — модель Build–Operate–Transfer (BOT)](https://www.luxoft.com/blog/master-the-build-operate-transfer-bot-model-with-luxoft)
- [InOrg — seamless handover в модели BOT](https://inorg.com/blog/from-build-to-transfer-key-success-factors-a-seamless-bot-model-transition)
- [Habr — red_mad_robot: кейс RAG для ФСК](https://habr.com/ru/companies/redmadrobot/articles/892882/)
- [Ведомости — CTO AI red_mad_robot (Влад Шевченко)](https://www.vedomosti.ru/technologies/trendsrub/articles/2026/03/11/1181757-ii-uskoril-kod)

#### Публичные материалы MWS / MTS AI {: #research_pkg_a_publichnye_materialy_mws_mts_ai }

- [Хабр — МТС: RAG для поддержки (Confluence, Jira, гибридный поиск)](https://habr.com/ru/companies/ru_mts/articles/970476/)
- [Хабр — МТС: RAG-помощник для саппорта (смежная публикация)](https://habr.com/ru/companies/ru_mts/articles/970392/)
- [Хабр — MTS AI: граф в RAG](https://habr.com/ru/companies/mts_ai/articles/915276/)
- [Хабр — МТС: MWS Octapi и AI-агенты](https://habr.com/ru/companies/ru_mts/articles/932382/)
- [Хабр — МТС: архитектура LLM-платформы MWS GPT](https://habr.com/ru/companies/ru_mts/articles/967478/)
- [Хабр — MTS AI: взаимная оценка LLM при улучшении Cotype](https://habr.com/ru/companies/mts_ai/articles/892176/)
- [MWS — MWS Octapi (продукт)](https://mws.ru/dev-tools/octapi/)
- [MWS — новость: Octapi и создание ИИ-агентов](https://mws.ru/news/mts-web-services-na-30-uskorila-sozdanie-ii-agentov-pri-pomoshhi-platformy-mws-octapi/)
- [MWS AI — MWS AI Agents Platform (описание модулей)](https://mts.ai/product/ai-agents-platform/)
- [MERA — бенчмарк русскоязычных моделей](https://mera.a-ai.ru/)
- [Альянс в сфере искусственного интеллекта](https://a-ai.ru/)
- [MWS Docs — лицензионные условия ПО «MWS GPT»](https://mws.ru/docs/docum/lic_terms_mwsgpt.html)
- [МТС Cloud — IaaS 152-ФЗ УЗ-1](https://cloud.mts.ru/services/iaas-152-fz/)
- [MWS Docs — условия облачного сегмента 152-ФЗ](https://mws.ru/docs/docum/cloud_terms_152fz.html)
- [MWS — новость: хранение персональных данных в облаке](https://mws.ru/news/mts-web-services-zapustila-servis-dlya-hraneniya-personalnyh-dannyh-v-oblake/)

#### Публичные материалы финсектора (паттерны внедрения) {: #research_pkg_a_publichnye_materialy_finsektora_patterny_vnedreniya }

- [Хабр — автоматизация обучения и обновления моделей](https://habr.com/ru/companies/alfa/articles/852790/)
- [Хабр — классификация текстов диалогов на большом числе классов](https://habr.com/ru/companies/alfa/articles/900538/)
- [Хабр — MLOps и каскады моделей](https://habr.com/ru/companies/alfa/articles/801893/)
- [Хабр — обновление LLM: instruction following и tool calling](https://habr.com/ru/companies/tbank/articles/979650/)
- [CIO — интервью: чат-бот, масштаб обращений и сценарии](https://cio.osp.ru/articles/5455)
- [«Открытые системы» — RAG и LLM для поддержки операционистов](https://www.osp.ru/articles/2025/0324/13059305)

#### Telegram-каналы {: #research_pkg_a_telegram_kanaly }

- [NeuralDeep](https://t.me/neuraldeep)
- [@ai_archnadzor](https://t.me/ai_archnadzor)
- [@Redmadnews (red_mad_robot)](https://t.me/Redmadnews)
- [@rmr_rnd — R&D red_mad_robot](https://t.me/rmr_rnd)
- [AGORA — Industrial AI](https://t.me/AGORA)
- [«ITипичные аспекты Артёма» (Артём Лысенко)](https://t.me/virrius_tech_chat)

#### Посты NeuralDeep {: #research_pkg_a_posty_neuraldeep }

- [ETL, эмбеддинги, реранкеры, фреймворки RAG, eval, безопасность](https://t.me/neuraldeep/1758)
- [Agentic RAG / SGR](https://t.me/neuraldeep/1605)

#### Посты @ai_archnadzor {: #research_pkg_a_posty_ai_archnadzor }

- [GraphOS для RAG](https://t.me/ai_archnadzor/151)
- [Semantic Gravity Framework](https://t.me/ai_archnadzor/155)
- [Nested Learning](https://t.me/ai_archnadzor/157)
- [LEANN](https://t.me/ai_archnadzor/161)
- [OpenClaw (ex-Moltbot)](https://t.me/ai_archnadzor/165)
- [Perplexica](https://t.me/ai_archnadzor/166)
- [Guardrails как архитектурный паттерн](https://t.me/ai_archnadzor/168)
- [EffGen / agentic SLM](https://t.me/ai_archnadzor/171)
- [Типы AI-агентов](https://t.me/ai_archnadzor/173)
- [GenAI в продакшене: технологический манифест](https://t.me/ai_archnadzor/175)
- [Локальный стек обсервабильности](https://t.me/ai_archnadzor/177)
- [REFRAG](https://t.me/ai_archnadzor/178)
- [Cog-RAG](https://t.me/ai_archnadzor/179)
- [HippoRAG 2](https://t.me/ai_archnadzor/180)
- [Topo-RAG](https://t.me/ai_archnadzor/182)
- [Disco-RAG](https://t.me/ai_archnadzor/183)
- [DSPy 3 и GEPA](https://t.me/ai_archnadzor/184)
- [OCR: NEMOTRON-PARSE, Chandra, DOTS.OCR](https://t.me/ai_archnadzor/185)
- [BitNet](https://t.me/ai_archnadzor/189)
- [CLI вместо MCP](https://t.me/ai_archnadzor/190)
- [Doc-to-LoRA; память агентов (пост /191)](https://t.me/ai_archnadzor/191)
- [Multimodal LLM](https://t.me/ai_archnadzor/192)

#### Habr и статьи по инженерии RAG {: #research_pkg_a_habr_i_stati_po_inzhenerii_rag }

- [Raft на Habr — чанкование](https://habr.com/ru/companies/raft/articles/954158/)

#### Препринты (arXiv) {: #research_pkg_a_preprinty_arxiv }

- [Google — Deep-Thinking Ratio (DTR), 2602.13517](https://arxiv.org/pdf/2602.13517)
- [Oppo AI — Search More, Think Less (SMTL), 2602.22675](https://arxiv.org/pdf/2602.22675)
- [Meta (Экстремистская организация, запрещена в РФ), OpenAI, xAI — непрерывное улучшение моделей (чаты), 2603.01973](https://arxiv.org/pdf/2603.01973)
- [Microsoft Research — безопасность агентов с внешними инструментами, 2603.03205](https://arxiv.org/pdf/2603.03205)
- [Accenture — Memex(RL), 2603.04257](https://arxiv.org/pdf/2603.04257)
- [SkillNet, 2603.04448](https://arxiv.org/pdf/2603.04448)
- [Databricks — KARL, 2603.05218](https://arxiv.org/pdf/2603.05218)
- [OpenAI — контроль рассуждения со скрытыми шагами, 2603.05706](https://arxiv.org/pdf/2603.05706)
- [Princeton — непрерывное обучение из взаимодействия с агентом, 2603.10165](https://arxiv.org/pdf/2603.10165)

#### Продукты и блоги (эмбеддинги, M365; справочно) {: #research_pkg_a_produkty_i_blogi_emveddingi_m365_spravochno }

- [Google — Gemini Embedding 2 (блог)](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
- [Microsoft — Copilot Cowork (блог Microsoft 365)](https://www.microsoft.com/en-us/microsoft-365/blog/2026/03/09/copilot-cowork-a-new-way-of-getting-work-done/)

#### Открытые проекты третьих сторон {: #research_pkg_a_otkrytye_proekty_tretih_storon }

- [microsoft/markitdown](https://github.com/microsoft/markitdown)
- [datalab-to/marker](https://github.com/datalab-to/marker)
- [docling-project/docling](https://github.com/docling-project/docling)
- [chonkie-inc/chonkie](https://github.com/chonkie-inc/chonkie)
- [langchain-ai/langchain — text-splitters](https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters)
- [vamplabAI/sgr-agent-core](https://github.com/vamplabAI/sgr-agent-core) (ветка tool-confluence)
- [vamplabAI/sgr-agent-core — ветка tool-confluence](https://github.com/vamplabAI/sgr-agent-core/tree/tool-confluence)
- [GitHub — ozontech/seq-db](https://github.com/ozontech/seq-db)
- [GitHub — ozontech/file.d](https://github.com/ozontech/file.d)
- [GitHub — ozontech/framer](https://github.com/ozontech/framer)
- [langgenius/dify](https://github.com/langgenius/dify/)
- [Marker-Inc-Korea/AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG)
- [run-llama/llama_index](https://github.com/run-llama/llama_index)
- [mastra-ai/mastra](https://github.com/mastra-ai/mastra)
- [RAGAS — документация](https://docs.ragas.io/en/stable/)
- [stanford-futuredata/ARES](https://github.com/stanford-futuredata/ARES)
- [NVIDIA-NeMo/Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)
- [Lakera — платформа](https://platform.lakera.ai/)
- [protectai/rebuff](https://github.com/protectai/rebuff)
- [NVIDIA/garak](https://github.com/NVIDIA/garak)
- [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)
- [EvilFreelancer/openapi-to-cli](https://github.com/EvilFreelancer/openapi-to-cli)
- [SkillsBD.ru](https://skillsbd.ru/)
- [OpenCode](https://opencode.ai/)
- [OpenCode — документация (Intro)](https://opencode.ai/docs)
- [OpenCode — Ecosystem](https://opencode.ai/docs/ecosystem/)
- [OpenCode Zen](https://opencode.ai/docs/zen)
- [OpenWork (different-ai/openwork)](https://github.com/different-ai/openwork)
- [OpenRouter](https://openrouter.ai/)
- [OpenRouter — Logging и политики провайдеров](https://openrouter.ai/docs/guides/privacy/logging)

#### Регулирование (проектный контур 2026) {: #research_pkg_a_regulirovanie_proektnyi_kontur_2026 }

- [Портал НПА — проект федерального закона (ID 166424)](https://regulation.gov.ru/projects#npa=166424)

#### Наблюдаемость и телеметрия: стандарты и стек {: #research_pkg_a_nablyudaemost_i_telemetriya_standarty_i_stek }

- [OpenTelemetry — Semantic conventions for generative client AI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- [OpenTelemetry — Semantic conventions for generative AI metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)
- [OpenTelemetry — OpenTelemetry for Generative AI (блог)](https://opentelemetry.io/blog/2024/otel-generative-ai)
- [OpenInference — инструментирование ИИ для OpenTelemetry](https://arize-ai.github.io/openinference/)
- [Langfuse — документация observability / tracing](https://langfuse.com/docs/observability/get-started)
- [Arize Phoenix — документация](https://docs.arize.com/phoenix)
- [LangSmith — документация](https://docs.smith.langchain.com/)

### Источники из «Оценка сайзинга, КапЭкс и ОпЭкс для клиентов» (без сокращений) {: #research_pkg_a_istochniki_iz_otsenka_saizinga_kapeks_i_opeks_dlya_klientov_bez_sokraschenii }

#### Примерные расчёты токенов (портал поддержки, агрегаторы и обзоры прайсов) {: #research_pkg_a_primernye_raschety_tokenov_portal_podderzhki_agregatory_i_obzory_praisov_2 }

- [Портал поддержки Comindware](https://support.comindware.com/)
- [LLMoney — калькулятор цен токенов LLM](https://llmoney.ru)
- [Хабр — обзор цен на токены](https://habr.com/ru/articles/1000058/)
- [Хабр — гид по топ-20 нейросетям для текстов (в т.ч. цены)](https://habr.com/ru/articles/948672/)
- [VC.ru — гайд по тарифам Claude и доступу из России](https://vc.ru/ai/2757771-tarify-claude-2026-gayd-po-planam-i-dostupu-iz-rossii)

#### Инженерия обвязки и мультиагентная разработка {: #research_pkg_a_inzheneriya_obvyazki_i_multiagentnaya_razrabotka_4 }

- [OpenAI — Harness engineering](https://openai.com/ru-RU/index/harness-engineering/)
- [Anthropic — Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [Anthropic — Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Martin Fowler — Harness Engineering (Thoughtworks)](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)
- [Хабр — Инженер будущего строит обвязку для агентов](https://habr.com/ru/articles/1005032/)

#### Агентная память и модели (ориентиры НИОКР и прайсинга, не строка КП) {: #research_pkg_a_agentnaya_pamyat_i_modeli_orientiry_niokr_i_praisinga_ne_stroka_kp_2 }

- [arXiv — General Agentic Memory (GAM)](https://arxiv.org/pdf/2511.18423)
- [Anthropic — Introducing Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)
- [Anthropic — Introducing Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)
- [Claude Docs — What's new in Claude 4.6](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-6)
- [Claude Docs — Models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Хабр — Релиз Claude Opus 4.6](https://habr.com/ru/news/993322/)
- [Anthropic — Pricing](https://www.anthropic.com/pricing)
- [Microsoft Research — Fara-7B (PDF)](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/11/Fara-7B-An-Efficient-Agentic-Model-for-Computer-Use.pdf)

#### Безопасность GenAI, OWASP и сигналы рынка (TCO / риски) {: #research_pkg_a_bezopasnost_genai_owasp_i_signaly_rynka_tco_riski_2 }

- [OWASP GenAI Security — Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/)
- [OWASP GenAI Security — Top 10 for Agentic Applications for 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
- [GitHub — NVIDIA Garak (сканер для LLM, только изолированные стенды)](https://github.com/NVIDIA/garak)
- [CodeWall — разбор red team: McKinsey AI platform](https://codewall.ai/blog/how-we-hacked-mckinseys-ai-platform)
- [Коммерсантъ — рынок и атаки на ИИ-системы (журналистский контекст)](https://www.kommersant.ru/doc/8363105)
- [OpenAI — приобретение PromptFoo (контекст рынка тестирования)](https://openai.com/index/openai-to-acquire-promptfoo/)
- [Kaspersky — пресс-релиз: угрозы под видом популярных ИИ-сервисов (бенчмарк тренда)](https://www.kaspersky.com/about/press-releases/kaspersky-chatgpt-mimicking-cyberthreats-surge-115-in-early-2025-smbs-increasingly-targeted)

#### Иллюстративные ориентиры нагрузки (публичные интервью, финсектор) {: #research_pkg_a_illyustrativnye_orientiry_nagruzki_publichnye_intervyu_finsektor_2 }

- [CIO — интервью: чат-бот, масштаб обращений и сценарии](https://cio.osp.ru/articles/5455)
- [«Открытые системы» — RAG и LLM для поддержки операционистов](https://www.osp.ru/articles/2025/0324/13059305)

#### Облачные провайдеры и тарифы (РФ) {: #research_pkg_a_oblachnye_provaidery_i_tarify_rf_4 }

- [Cloud.ru — Evolution Foundation Models (продукт, перечень моделей)](https://cloud.ru/products/evolution-foundation-models)
- [Cloud.ru — Evolution Foundation Models, тарифы (2026)](https://cloud.ru/documents/tariffs/evolution/foundation-models)
- [Yandex AI Studio — доступные генеративные модели](https://aistudio.yandex.ru/docs/ru/ai-studio/concepts/generation/models.html)
- [Yandex AI Studio — правила тарификации](https://aistudio.yandex.ru/docs/ru/ai-studio/pricing.html)
- [AKM.ru — доступ к крупнейшей языковой модели на рынке РФ (Yandex B2B)](https://www.akm.ru/eng/press/yandex-b2b-tech-has-opened-access-to-the-largest-language-model-on-the-russian-market/)
- [Сбер — GigaChat API: юридические тарифы](https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs)
- [Selectel — Foundation Models Catalog](https://selectel.ru/services/cloud/foundation-models-catalog)
- [MWS — MWS GPT](https://mws.ru/mws-gpt/)
- [MWS — тарифы MWS GPT](https://mws.ru/docs/docum/cloud_terms_mwsgpt_pricing.html)
- [МТС Cloud — виртуальная инфраструктура с GPU](https://cloud.mts.ru/services/virtual-infrastructure-gpu/)
- [MWS — GPU On‑premises](https://mws.ru/services/mws-gpu-on-prem/)
- [CNews — кейс: MTS AI и экономия инвестиций за счёт облака MWS (обзор)](https://www.cnews.ru/reviews/provajdery_gpu_cloud_2025/cases/kak_mts_ai_sekonomila_bolee_milliarda)
- [CIO — MTS AI перенесла обучение моделей в облако](https://cio.osp.ru/articles/140525-MTS-AI-perenesla-obuchenie-modeley-v-oblako)
- [VK Cloud — машинное обучение в облаке](https://cloud.vk.com/docs/ru/ml)
- [Google — условия использования Gemma](https://ai.google.dev/gemma/terms)

#### Публичные веса с нестандартной лицензией {: #research_pkg_a_publichnye_vesa_s_nestandartnoi_litsenziei_3 }

- [Hugging Face — LICENSE (YandexGPT-5-Lite-8B), сырой текст соглашения](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct/raw/main/LICENSE)
- [Hugging Face — карточка модели YandexGPT-5-Lite-8B-instruct](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct)

#### Открытые модели ai-sage (GigaChat и спутники) {: #research_pkg_a_otkrytye_modeli_ai_sage_gigachat_i_sputniki_2 }

- [Хабр — GigaChat-3.1: большое обновление больших моделей (блог Сбера)](https://habr.com/ru/companies/sberbank/articles/1014146/)
- [Hugging Face — организация ai-sage](https://huggingface.co/ai-sage)
- [Hugging Face — коллекция GigaChat 3.1](https://huggingface.co/collections/ai-sage/gigachat-31)
- [Hugging Face — ai-sage/GigaChat3.1-702B-A36B (Ultra)](https://huggingface.co/ai-sage/GigaChat3.1-702B-A36B)
- [Hugging Face — ai-sage/GigaChat3.1-10B-A1.8B (Lightning 3.1)](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B)
- [Hugging Face — ai-sage/GigaChat3-10B-A1.8B (Lightning 3.0)](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B)
- [Hugging Face — коллекция GigaEmbeddings](https://huggingface.co/collections/ai-sage/gigaembeddings)
- [Hugging Face — коллекция GigaAM](https://huggingface.co/collections/ai-sage/gigaam)
- [Hugging Face — коллекция GigaChat Lite](https://huggingface.co/collections/ai-sage/gigachat-lite)
- [GitVerse — GigaTeam/gigachat3.1](https://gitverse.ru/GigaTeam/gigachat3.1)
- [GitHub — sgl-project/sglang, PR #18802](https://github.com/sgl-project/sglang/pull/18802)

#### Telegram-каналы и посты {: #research_pkg_a_telegram_kanaly_i_posty }

- [NeuralDeep — бенчмарки vLLM / RTX 4090](https://t.me/neuraldeep/1476)
- [NeuralDeep — рекомендации по кластерам](https://t.me/neuraldeep/1627)
- [NeuralDeep — экономика LLM-решений](https://t.me/neuraldeep/1366)
- [Канал NeuralDeep](https://t.me/neuraldeep)
- [Канал @ai_archnadzor — RAG и архитектуры](https://t.me/ai_archnadzor)
- [@ai_archnadzor — локальные модели для кодинга и снижения затрат](https://t.me/ai_archnadzor/167)
- [@ai_archnadzor — CLI вместо MCP](https://t.me/ai_archnadzor/190)
- [@Redmadnews (red_mad_robot)](https://t.me/Redmadnews)
- [@llm_under_hood](https://t.me/llm_under_hood)
- [Канал @ai_machinelearning_big_data](https://t.me/ai_machinelearning_big_data)

#### Инструменты разработки с ИИ {: #research_pkg_a_instrumenty_razrabotki_s_ii }

- [OpenCode](https://opencode.ai/)
- [OpenCode — документация (Intro)](https://opencode.ai/docs)
- [OpenCode — Ecosystem](https://opencode.ai/docs/ecosystem/)
- [OpenCode Zen](https://opencode.ai/docs/zen)
- [OpenWork (different-ai/openwork)](https://github.com/different-ai/openwork)
- [OpenRouter](https://openrouter.ai/)
- [OpenRouter — Logging и политики провайдеров](https://openrouter.ai/docs/guides/privacy/logging)

#### Финансовая и инфраструктурная база (FinOps/TCO/железо) {: #research_pkg_a_finansovaya_i_infrastrukturnaya_baza_finops_tco_zhelezo_2 }

- [FinOps Foundation — Generative AI (Unit Economics)](https://www.finops.org/wg/generative-ai/)
- [FinOps Framework — Unit Economics (capability)](https://www.finops.org/framework/capabilities/unit-economics/)
- [OpenAI — Prompt caching (снижение стоимости повторяющегося контекста)](https://platform.openai.com/docs/guides/prompt-caching)
- [Slyd — TCO Calculator: On-Prem vs Cloud](https://slyd.com/resources/tco-calculator)
- [Introl — финансирование CapEx/OpEx и инвестиции в GPU](https://introl.com/blog/ai-infrastructure-financing-capex-opex-gpu-investment-guide-2025)
- [SWFTE — экономика частного AI / on-prem](https://www.swfte.com/blog/private-ai-enterprises-onprem-economics)
- [Runpod — LLM inference optimization playbook (throughput)](https://www.runpod.io/articles/guides/llm-inference-optimization-playbook)
- [Introl — планирование мощностей ИИ-инфраструктуры (прогнозы, McKinsey в обзоре)](https://introl.com/blog/ai-infrastructure-capacity-planning-forecasting-gpu-2025-2030)
- [PitchGrade — AI Infrastructure Primer](https://pitchgrade.com/research/ai-infrastructure-primer)
- [Medium — Qwen 3.5 35B A3B (AgentNativeDev)](https://agentnativedev.medium.com/qwen-3-5-35b-a3b-why-your-800-gpu-just-became-a-frontier-class-ai-workstation-63cc4d4ebac1)
- [apxml.com — VRAM calculator](https://apxml.com/tools/vram-calculator)
- [Hugging Face — Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

#### Наблюдаемость и телеметрия {: #research_pkg_a_nablyudaemost_i_telemetriya_2 }

- [OpenTelemetry — Semantic conventions for generative client AI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- [OpenTelemetry — Semantic conventions for generative AI metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)
- [OpenTelemetry — OpenTelemetry for Generative AI (блог)](https://opentelemetry.io/blog/2024/otel-generative-ai)
- [OpenInference — инструментирование ИИ для OpenTelemetry](https://arize-ai.github.io/openinference/)
- [Arize Phoenix — документация](https://docs.arize.com/phoenix)
- [LangSmith — документация](https://docs.smith.langchain.com/)
- [LangChain Docs — Evaluation concepts (LangSmith)](https://docs.langchain.com/langsmith/evaluation-concepts)
- [Официальное опубликование — Приказ Роскомнадзора от 19.06.2025 № 140 (обезличивание ПДн)](http://publication.pravo.gov.ru/document/0001202508010002)
- [Langfuse — документация observability](https://langfuse.com/docs/observability/get-started)

## Дополнительные источники из ТЗ (категории) {: #research_pkg_a_dopolnitelnye_istochniki_iz_tz_kategorii }

### 16.1 Международные стандарты и регулирование (Приоритет 1) {: #research_pkg_a_16_1_mezhdunarodnye_standarty_i_regulirovanie_prioritet_1 }

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

### 16.2 Executive-методологии внедрения (Big Three & Big Four) {: #research_pkg_a_16_2_executive_metodologii_vnedreniya_big_three_big_four }

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

### 16.3 Технические паттерны и инженерные блоги (Промышленный ИИ / Production AI) {: #research_pkg_a_16_3_tehnicheskie_patterny_i_inzhenernye_blogi_promyshlennyi_ii_production_ai }

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

### 16.4 Экономика ИИ, FinOps и сайзинг {: #research_pkg_a_16_4_ekonomika_ii_finops_i_saizing }

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

### 16.5 Российские регуляторные и правовые источники (Приоритет 1) {: #research_pkg_a_16_5_rossiiskie_regulyatornye_i_pravovye_istochniki_prioritet_1 }

- [Указ Президента РФ №490: Национальная стратегия развития ИИ до 2030 (Ред. 2024)](https://www.consultant.ru/document/cons_doc_LAW_470015/)
- [Указ Президента РФ от 15.02.2024: Изменения в стратегию ИИ (Актуальная версия)](https://ai.gov.ru/national-strategy/)
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

### 16.6 Российские прикладные исследования и бенчмарки {: #research_pkg_a_16_6_rossiiskie_prikladnye_issledovaniya_i_benchmarki }

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

### 16.7 Модели отчуждения и передачи (BOT и Handover) {: #research_pkg_a_16_7_modeli_otchuzhdeniya_i_peredachi_bot_i_handover }

- [Build-Operate-Transfer (BOT) Model: Full Guide 2025](https://build-operate-transfer.com/post/build-operate-transfer-bot-model-complete-guide-for-software-development-2025)
- [Software Handover Checklist 2026: Documentation & IP Guide](https://www.tech4lyf.com/blog/software-handover-documentation-checklist-2026/)
- [InCommon: Why BOT Wins for AI Infrastructure](https://www.incommon.ai/blog/build-operate-transfer/)
- [Innowise: BOT Outsourcing Contract and IP Transfer Guide](https://innowise.com/blog/build-operate-transfer-bot-model-guide/)
- [Devico: Checklist for a seamless BOT transition (2025)](https://devico.io/blog/checklist-for-a-seamless-bot-transition)
- [Knowledge Transfer Framework for Enterprise Software Handover](https://www.knowledge-management-tools.net/knowledge-transfer-framework.html)

### 16.8 Кураторские списки и репозитории (Awesome Lists) {: #research_pkg_a_16_8_kuratorskie_spiski_i_repozitorii_awesome_lists }

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

### 16.9 Кейсы внедрения в российском бизнесе (2025-2026) {: #research_pkg_a_16_9_keisy_vnedreniya_v_rossiiskom_biznese_2025_2026 }

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

### 16.10 Технические статьи и инженерные блоги (Россия) {: #research_pkg_a_16_10_tehnicheskie_stati_i_inzhenernye_blogi_rossiya }

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

### 16.11 Российская экономика ИИ и отчеты консалтинга {: #research_pkg_a_16_11_rossiiskaya_ekonomika_ii_i_otchety_konsaltinga }

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
