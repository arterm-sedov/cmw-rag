---
title: 'Приложение B. Отчуждение ИС и кода (KT, IP, лицензии, приёмка)'
date: 2026-03-25
status: 'утверждённый комплект материалов для руководства (v1)'
tags:
  - compliance
  - handover
  - IP
  - KT
  - licensing
  - research
  - отчуждение
  - передача
  - приёмка
hide: tags
---

# Приложение B. Отчуждение ИС и кода (KT, IP, лицензии, приёмка) {: #research_pkg_b }
## Обзор комплекта {: #research_pkg_b_obzor_paketa }

Здесь сосредоточены модели передачи, состав передаваемых артефактов, требования к учёту лицензий и критерии приёмки при отчуждении кода и сопутствующей ИС. Количественные эффекты для P&L не дублируются: на них указывает «Основной отчёт: сайзинг и экономика».

## Связанные документы {: #research_pkg_b_svyazannye_dokumenty }

- [«Приложение A: обзор и ведомость документов»](./20260325-research-appendix-a-index-ru.md#research_pkg_a_obzor_paketa)
- [«Основной отчёт: методология внедрения и разработки»](./20260325-research-report-methodology-main-ru.md#research_methodology_20260325_obzor_paketa)
- [«Основной отчёт: сайзинг и экономика»](./20260325-research-report-sizing-economics-main-ru.md#research_sizing_20260325_obzor_paketa)
- [Профиль on-prem GPU в проектах CMW](./20260325-research-report-sizing-economics-main-ru.md#research_sizing_20260325_profil_onprem_gpu_v_proektah_cmw) (реф. consumer 24 ГБ, кастом 4090 48 ГБ, PRO 6000 96 ГБ)
- «Приложение C: имеющиеся наработки CMW»
- [«Приложение D: безопасность, комплаенс и observability»](./20260325-research-appendix-d-security-observability-ru.md#research_pkg_d_obzor_paketa)

## Детальная методология отчуждения {: #research_pkg_b_detalnaya_metodologiya_otchuzhdeniya }

### Ориентиры для заказчика: инструменты ускорения разработки (вне поставки CMW) {: #research_pkg_b_orientiry_dlya_zakazchika_instrumenty_uskoreniya_razrabotki_vne_postavki_cmw }

Ниже перечисленные продукты — **подсказки заказчику** для ускорения прототипирования и освоения практик разработки с ИИ; они **не** входят в коммерческую поставку референс-стека **корпоративный RAG-контур**, **сервер инференса MOSEC**, **инференс на базе vLLM**, **агентный слой платформы (CMW Platform)** и **не** заменяют промышленный контур RAG и инференса, описанный в этом документе.

**Важно:** продукт [OpenCode](https://opencode.ai/) (открытый AI coding agent) и внутренние каталоги планирования в репозиториях CMW — **разные сущности**; не смешивать их в переговорах и в договорной документации.

- **[OpenCode](https://opencode.ai/)** — открытый агент для кода; провайдеры и модели задаются конфигурацией. Каталог плагинов и интеграций сообщества: [Ecosystem](https://opencode.ai/docs/ecosystem/).
- **[OpenWork](https://github.com/different-ai/openwork)** — десктоп/UI-слой для команд поверх OpenCode (также перечислен в [Ecosystem](https://opencode.ai/docs/ecosystem/)).
- **[OpenCode Zen](https://opencode.ai/docs/zen)** — опциональный **платный** шлюз с отобранными моделями (beta); для контуров с **152-ФЗ** и суверенитетом данных **не** следует принимать как дефолт без оценки: хостинг и политики обработки данных определяются провайдерами шлюза (в т.ч. юрисдикция США). Бесплатные линейки на Zen могут иметь **ограниченный срок** и **особые условия использования данных** — см. официальный текст Zen.
- **[OpenRouter](https://openrouter.ai/)** — агрегирующий **API-шлюз** к множеству зарубежных провайдеров; типичное применение — **IDE, coding agents, прототипирование** (в т.ч. совместимо с конфигурацией **агентный слой платформы (CMW Platform)** в upstream). Для **продакшн-развёртывания ИИ-решений** у заказчиков в РФ с ПД и ожиданием локализации OpenRouter **не** является подразумеваемым baseline: маршрутизация к исполнителям за рубежом, биллинг и политики логирования/удержания данных задаются цепочкой провайдеров ([документация OpenRouter — logging и политики](https://openrouter.ai/docs/guides/privacy/logging)); без отдельной **юридической и ИБ-оценки** не подменяет API **Cloud.ru / Yandex Cloud / SberCloud / MWS GPT / Selectel** или закрытый контур.
- **Cursor** — коммерческая IDE с подпиской; ориентиры по токенам для сравнения — в сопутствующем резюме **Оценка сайзинга, КапЭкс и ОпЭкс для клиентов**.

**Практика для РФ:** снижение зависимости от зарубежного биллинга возможно за счёт **локальных моделей** и/или **API в РФ** там, где это допускает конфигурация выбранного инструмента и политика ИБ заказчика; доступность сервисов и условия использования нужно **проверять на дату** по официальным источникам и TOS. Итоговый контур согласовывается с комплаенсом и владельцем данных.

### Теневой GenAI в маркетинге и маршрутизация моделей (ориентир опроса CMO, 2025) {: #research_pkg_b_tenevoi_genai_v_marketinge_i_marshrutizatsiya_modelei_orientir_oprosa_cmo_2025 }

Публичные материалы опроса **red_mad_robot × CMO Club Russia** фиксируют **высокую концентрацию** на универсальных зарубежных чат- и визуальных сервисах среди маркетинговых директоров (в сводке комплекта — порядка **91%** для **ChatGPT** и **59%** для **Midjourney**, с широким разрывом до следующих инструментов; канонические доли и контекст — в _«[Основной отчёт: сайзинг и экономика — Российский рынок](./20260325-research-report-sizing-economics-main-ru.md#research_sizing_20260325_rossiiskii_rynok)_»_).

Для комплекта отчуждения и договоров о **ИС** это аргумент за явный **каталог допустимых моделей и маршрутов данных**, учёт **TOS/API** зарубежных SaaS и разделение **промышленного** контура (**корпоративный RAG-контур**, API РФ, on-prem) от **самостоятельного** использования маркетингом глобальных сервисов; управленческий смысл и перекрёстные ссылки — в _«[Основной отчёт: методология — GenAI в маркетинговых командах](./20260325-research-report-methodology-main-ru.md#research_methodology_20260325_genai_v_marketingovyh_komandah_krupnyh_brendov_rf_opros_cmo_2025)_»_.

### Отчуждение данных {: #research_pkg_b_otchuzhdenie_dannyh }
-   **Поддержка ChromaDB:** Штатные утилиты сопровождения (обслуживание коллекций, инспекция схемы) позволяют диагностировать, очищать и мигрировать.
-   **Удаление векторного хранилища:** Коллекции можно удалить через HTTP API ChromaDB или Python-клиент.
-   **Архивация документов:** Исходные документы (Markdown) остаются в файловой системе; векторные данные не теряются, если сохранен источник.

### Отчуждение моделей {: #research_pkg_b_otchuzhdenie_modelei }
-   **Обновление конфигурации:** Смена идентификаторов моделей через переменные окружения и файл конфигурации моделей (YAML).
-   **Горячая перезагрузка:** MOSEC поддерживает динамическую загрузку/выгрузку моделей (для vLLM требуется перезапуск).
-   **Версионирование:** Модели отслеживаются через HuggingFace Hub; откат изменением ID модели.
- **Открытые веса и лицензия:** при self-hosted чекпойнтах (в т.ч. GigaChat-3.1 под MIT — [Хабр, Сбер](https://habr.com/ru/companies/sberbank/articles/1014146/)) в комплект передачи входят идентификаторы релиза (HF/GitVerse), текст лицензии, политика фиксации версий, регрессионные eval при смене весов; интеграция с **инференс на базе vLLM** / **сервер инференса MOSEC** фиксируется в runbook.
- **Кастомные лицензии на публичные веса:** помимо permissive-лицензий хранить пороги по **выходным токенам**, календарные сроки уведомления правообладателя и условия атрибуции; иллюстративный полный текст — [Лицензионное соглашение YandexGPT-5-Lite-8B (файл на Hugging Face)](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct/raw/main/LICENSE).
-   **Реестр доверенных моделей:** публикация открытых весов **не заменяет** проверку допуска модели для госсектора и КИИ (см. раздел Compliance ниже).

### Отчуждение инфраструктуры {: #research_pkg_b_otchuzhdenie_infrastruktury }

- **Завершение сервисов инференса:** штатная остановка через CLI соответствующего комплекта (**сервер инференса MOSEC** или **инференс на базе vLLM**); детали — в поставляемой документации.
- **Завершение контейнера:** при развёртывании через Docker — стандартные процедуры остановки и удаления контейнеров.
- **Очистка ресурсов:** память GPU освобождается при завершении процесса; данные ChromaDB сохраняются на диске до явного удаления.

### Справочно: аренда GPU и лицензирование NVIDIA (GeForce vs datacenter) {: #research_pkg_b_spravochno_arenda_gpu_i_litsenzirovanie_nvidia }

При **аренде ВМ или сервера с GPU** юридический контур дополняет open-source лицензии на веса моделей: для потребительских линеек (GeForce / RTX и аналоги) и для продуктов, классифицируемых как **datacenter**, действуют **разные** рамки [лицензионных условий NVIDIA](https://www.nvidia.com/en-us/drivers/geforce-license/) и сопутствующих ограничений на ПО и сценарии использования — **due diligence** по текстам на дату сделки и по профилю нагрузки (коммерческий инференс, колокация, облако). Каталоги аренды публично смешивают классы железа (иллюстрации: [Intelion Cloud](https://intelion.cloud/), [HOSTKEY — GPU dedicated servers](https://hostkey.ru/gpu-dedicated-servers/)); **наличие SKU в каталоге не заменяет** юридическую и ИБ-проверку сценария заказчика. Количественные **₽/час** для таких каналов — только в сопутствующем резюме **Оценка сайзинга…**, подраздел **[«Аренда GPU (IaaS РФ): дополнительные поставщики и ориентиры»](./20260325-research-report-sizing-economics-main-ru.md#research_sizing_20260325_arenda_gpu_iaas_rf_dopolnitelnye_postavshchiki)**.

### Модели поставки и передачи (интеллектуальная собственность (ИС) и передача знаний) {: #research_pkg_b_modeli_postavki_i_peredachi_intellektualnaya_sobstvennost_is_i_peredacha_znanii }

| Модель | Суть | Типичный комплект на выходе | Риски для заказчика |
| :--- | :--- | :--- | :--- |
| **Управляемый сервис** | Эксплуатация и развитие у интегратора | SLA, отчёты, доступ к API; ограниченный доступ к коду | Зависимость от поставщика, границы ИС по договору |
| **Совместная разработка** | Команды заказчика и интегратора в одном контуре | Репозиторий, CI, совместные регламенты | Согласование скорости и приоритетов |
| **Построение — эксплуатация — передача (BOT, Build–Operate–Transfer)** | Сначала ввод в промышленную эксплуатацию силами интегратора, затем передача заказчику | Регламент эксплуатации (runbook), обучение, интенсивное сопровождение сразу после передачи (hypercare), права на код и конфигурацию по договору | Качество передачи и полнота документации |
| **Создание и передача (create–transfer)** | Разработка и передача заказчику «под ключ» без длительной эксплуатации у интегратора | Код, тесты, документация, сессии передачи знаний (KT — knowledge transfer) | Нужна внутренняя эксплуатационная готовность |

Модель **BOT** и факторы успешной передачи обобщены, в частности, в материалах [Luxoft — Build–Operate–Transfer](https://www.luxoft.com/blog/master-the-build-operate-transfer-bot-model-with-luxoft) и [InOrg — seamless handover](https://inorg.com/blog/from-build-to-transfer-key-success-factors-a-seamless-bot-model-transition).

При выборе **управляемой LLM-платформы** у инфраструктурного провайдера типовой комплект для юридической и закупочной сверки включает **лицензионные и иные условия ПО**, описание режимов **SaaS / hybrid / on-prem** и границ ответственности — иллюстрация: [специальные условия для ПО «MWS GPT»](https://mws.ru/docs/docum/lic_terms_mwsgpt.html). Публичная декомпозиция **платформы корпоративных агентов** (разметка, RAG, ops, интеграции) задаёт **чеклист владельцев** и артефактов передачи, переносимый на стек **корпоративный RAG-контур** / **агентный слой платформы (CMW Platform)** независимо от бренда ([MWS AI Agents Platform](https://mts.ai/product/ai-agents-platform/)).

### Пакет отчуждения (минимально целостный) {: #research_pkg_b_paket_otchuzhdeniya_minimalno_tselostnyi }

| Артефакт | Назначение |
| :--- | :--- |
| Исходный код и манифест зависимостей | Воспроизводимая сборка |
| Конфигурация без секретов + описание переменных окружения | Развёртывание у заказчика |
| Runbook эксплуатации (старт, стоп, бэкап, масштабирование) | Снижение bus factor |
| Наборы для оценки качества (eval) и регрессии | Контроль деградации после релизов |
| Политика наблюдаемости (сэмплинг, ретенция, маскирование ПДн) и схема экспорта телеметрии | Согласованность с 152-ФЗ и воспроизводимость разборов инцидентов |
| Дашборды и правила алертов (латентность, ошибки, токены, guardrails) | Эксплуатация и FinOps в одном контуре метрик |
| Описание данных и политика индексации RAG | Повторяемость ingestion |
| Политики ИБ и guardrails (черновик под ЛНА заказчика) | Согласование с комплаенсом |
| Матрица ролей и эскалаций | Эксплуатация и аудит |
| Регламент и реестр **Agent Skills** (версии, условия вызова) | Воспроизводимая агентская разработка и сопровождение |
| Конфигурация **MCP**, **CI** и **CD** для агентов (allowlist, секреты, политика веток) | Контролируемая среда исполнения |
| **Рубрики и промпты** для **модели-контролёра**, **эталонные примеры в промпте** | Меньше завышенных вердиктов, если вердикт выставляет **только модель-контролёр** без инструментальных проверок |
| **Шаблоны промптов** для структурированных не-кодовых артефактов (пример: **BPMN 2.0 XML** с согласованными `id` семантики и диаграммы) | Воспроизводимая формализация процессов при KT; меньше правок после генерации LLM при явных правилах и проверке в редакторе |
| Регламент **синхронизации док ↔ код** (периодические прогоны, ответственный) | Борьба с устареванием знаний в репозитории |

Практика по **BPMN 2.0**, шаблонам промптов и валидации XML — в «_[Основной отчёт: методология внедрения и разработки](./20260325-research-report-methodology-main-ru.md#research_methodology_20260325_spravochno_bpmn_20_i_generatsiya_llm)_» (подраздел «Справочно: формализация процессов (BPMN 2.0) и генерация с помощью LLM»).

### Справочно: агент в PR и артефакты вместо прямой записи в ИС {: #research_pkg_b_spravochno_agent_v_pr_i_artifacty_vmesto_pryamoi_zapisi_v_is }

Для сценариев **анализа и предложения правок** по **pull request** снижает риск для ИС и приёмки, когда среда исполнения выдаёт **наружу артефакты** (**diff**, отчёты тестов, текст ревью), а **прямая запись** в защищаемую ветку или «истинный» репозиторий выполняется только после **человеческого** или **согласованного CI**-решения. Типовая песочница: репозиторий **только чтение**, временная рабочая область, сеть с **deny-by-default** и allowlist на зеркала и артефакты, **краткоживущие** токены с минимальным scope. Вопрос «разрешать ли запись **напрямую** в репозиторий или ограничиться **артефактом** на проверку» имеет смысл явно вынести в решение владельца продукта и ИБ; см. также _«[Приложение D — вопросы для дискуссии и выводы по исполнению](./20260325-research-appendix-d-security-observability-ru.md#research_pkg_d_spravochno_bezopasnyi_mvp_kontura_ispolneniya_diskussiya_sredy_vyvody)»_. Детали и соседние паттерны (долгоживущая dev-среда, регулируемый контур) — в _«[Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#research_pkg_d_spravochno_model_riska_patterny_sredy_i_minimalnyi_sostav_platformy)»_.

### Уровни обучения при передаче {: #research_pkg_b_urovni_obucheniya_pri_peredache }

| Аудитория | Фокус |
| :--- | :--- |
| Бизнес / владельцы продукта | Сценарии, KPI, ограничения и ответственность |
| Эксплуатация (DevOps / SRE) | Развёртывание, мониторинг, инциденты |
| Разработка / ML | Код, пайплайны, доработка промптов и инструментов |
| Комплаенс / ИБ | ПД, журналирование, доступы, требования регуляторов |

### Организационные условия после передачи {: #research_pkg_b_organizatsionnye_usloviya_posle_peredachi }

Устойчивость эффекта после KT и hypercare зависит не только от runbook и eval, но и от **оргмеханики** заказчика: **внутренняя мобильность** между функциями для кросс-функциональных сценариев с ИИ; **поддержка экспериментов** (песочницы, лимиты, этика использования); роль **руководителей** в **масштабировании** повторяемых практик и в снятии блокеров по данным и доступам. Эти меры **дополняют** таблицу уровней обучения и типичные модели BOT/create–transfer выше.

### Критерии приёмки передачи (чек-лист) {: #research_pkg_b_kriterii_priemki_peredachi_chek_list }

- Сборка из переданных артефактов воспроизводится на стенде заказчика без «скрытых» шагов.
- Пройдены согласованные сценарии eval; зафиксированы baseline-метрики.
- Runbook покрывает типовые сбои и контакты эскалации.
- Определены владельцы компонентов на стороне заказчика и дата окончания hypercare.
- По ИС: зафиксированы лицензии, сторонние компоненты и ограничения использования.

### Справочно: открытые стандарты OWASP и внешние программы обучения (не входят в поставку по умолчанию) {: #research_pkg_b_spravochno_otkrytye_standarty_owasp_i_vneshnie_programmy_obucheniya_ne_vhodyat_v }

В комплект **отчуждения знаний** целесообразно включать **ссылочный каркас**: канонические URL [OWASP GenAI Security Project](https://genai.owasp.org/introduction-genai-security-project/) (LLM Top 10 2025, Agentic Top 10 2026, [AI Testing Guide](https://github.com/OWASP/www-project-ai-testing-guide)), при необходимости — [WSTG](https://owasp.org/www-project-web-security-testing-guide/stable/) и [ASVS 5.0 RU](https://github.com/OWASP/ASVS/blob/master/5.0/OWASP_Application_Security_Verification_Standard_5.0.0_ru.pdf). Русскоязычные дайджесты сообщества (например, [Habr — OWASP LLM TOP 10 2025](https://habr.com/ru/companies/owasp/articles/893712/)) удобны для онбординга, но **не** заменяют официальные тексты.

Коммерческие **курсы безопасности LLM** у третьих лиц (иллюстративный пример публичной программы — «Large Language Models Security» у [«Лаборатории Касперского»](https://www.kaspersky.com/about/press-releases/kaspersky-introduces-a-new-training-large-language-models-security), расписание и стоимость — только по сайту поставщика на дату закупки) могут дополнять подготовку ИБ и разработки заказчика; это **опция**, а не часть базовой поставки **корпоративный RAG-контур** / **сервер инференса MOSEC** / **инференс на базе vLLM** / **агентный слой платформы (CMW Platform)** без отдельного соглашения.

Публичная программа Школы управления СКОЛКОВО _«[Переход в ИИ: трансформация бизнес-процессов](https://www.skolkovo.ru/programmes/cdto/)»_ (модули, сроки и заявленные результаты — по странице программы) иллюстрирует рынок **обучения руководителей** внедрению ИИ в процессы; **не** входит в поставку референс-стека без отдельного договора. Управленческий контекст — в _«[Основной отчёт: методология — Стратегия внедрения ИИ и организационная зрелость](./20260325-research-report-methodology-main-ru.md#research_methodology_20260325_strategiya_vnedreniya_ii_i_organizatsionnaya_zrelost)»_.

---

## Выводы по отчуждению ИИ {: #research_pkg_b_vyvody_po_otchuzhdeniyu_ii }

### Передача экспертизы {: #research_pkg_b_peredacha_ekspertizy }
**Методы:**
-   **Документация:** руководство для агентов с инструкциями, RFC для дизайн-документов.
-   **Обучение:** Воркшопы, курс «ШАД: Интенсив по AI-агентам» (Яндекс).
-   **Код:** исходный код под открытой лицензией (при применимости), примеры использования.

### Создание агентов для клиентов {: #research_pkg_b_sozdanie_agentov_dlya_klientov }
**Подходы:**
-   **SkillsBD.ru** — база навыков для российских сервисов (Яндекс, Битрикс24, 1С).
-   **SGR Agent Core** — Schema-Guided Reasoning для агентов.
-   **OpenClaw** — self-hosted AI-агент (170K звезд за неделю).

### Национальная стратегия развития ИИ (Россия) {: #research_pkg_b_natsionalnaya_strategiya_razvitiya_ii_rossiya }
**Указ Президента РФ №124 (февраль 2024):**
-   Поправки к Национальной стратегии развития ИИ до 2030 года.
-   Цель: более 11 трлн руб. влияния ИИ на ВВП к 2030 году.
-   7.7 млрд руб. на федеральный проект «ИИ» (2025).

**Тренд 2025-2026:**
-   Массовое открытие офисов внедрения ИИ.
-   Рост вакансий с ИИ-скиллами: +62% за янв-окт 2024.
-   86% компаний используют open-source модели и fine-tuning.

---

## Методология отчуждения ИИ-активов {: #research_pkg_b_metodologiya_otchuzhdeniya_ii_aktivov }

При приобретении у инфраструктурного вендора **платформенного SKU** (управляемый LLM-шлюз, GPU как сервис, интеграционная платформа для агентов) в TCO заказчика обычно входят **лицензии или абонентская плата**, **сопровождение** (в т.ч. on-prem), **интеграции** с корпоративными системами и строки на обучение персонала — см. чеклисты и комплект артефактов в сопутствующем резюме **Методология внедрения и отчуждения ИИ** (в т.ч. юридические условия ПО и режимы развёртывания).

### Передача экспертизы {: #research_pkg_b_peredacha_ekspertizy_2 }

| Компонент | Метод отчуждения | Формат |
|-----------|-----------------|--------|
| **Документация** | Руководство для агентов, RFC | Markdown |
| **Код** | Исходный код, VCS, CI/CD | Открытая лицензия или условия поставки |
| **Модели** | HuggingFace Hub | Model weights |
| **Данные** | Экспорт ChromaDB | SQLite/Parquet |
| **Инфраструктура** | Docker Compose, K8s | YAML |
| **Мониторинг** | Arize Phoenix, LangSmith | SaaS/Self-hosted |

### Обучение клиентов {: #research_pkg_b_obuchenie_klientov }

**Рекомендуемые программы:**

- ШАД: Интенсив по AI-агентам (Яндекс)
- Google AI Studio: Vibe Coding
- a16z Top 100 AI Apps: анализ рынка
- Самообслуживание команд: [документация OpenCode](https://opencode.ai/docs) и при необходимости [OpenWork](https://github.com/different-ai/openwork) — **дополнение** к корпоративным программам обучения, не замена

**Сертификации:**
- Yandex Cloud: облачные технологии, DevOps, ИБ
- ISO/IEC 42001:2023 — система менеджмента ИИ (AIMS); **число сертифицированных организаций на дату сметы** запрашивать у органа сертификации или официальной статистики аккредитации, без опоры на устаревшие вторичные цифры в прессе
- AIUC-1: «SOC-2 для ИИ-агентов»

### Создание агентов для клиентов {: #research_pkg_b_sozdanie_agentov_dlya_klientov_2 }

**Подходы:**
1. **SkillsBD.ru** — база навыков для российских сервисов
2. **SGR Agent Core** — Schema-Guided Reasoning для агентов
3. **OpenClaw** — self-hosted AI-агент с ACP

**Экономика:**
- Замена подписки на облачный coding-агент (**~127 500 – 212 500 руб./мес** на 5–10 инженеров, эквив. **1 500 – 2 500 USD/мес × 85**) на локальные модели
- Qwen3-Coder-Next MoE: 24 ГБ VRAM, 64K+ контекст
- GLM-4.7-Flash MoE: 16-24 ГБ VRAM, 200K контекст

---

## Резюме по методологии отчуждения {: #research_pkg_b_rezyume_po_metodologii_otchuzhdeniya }

### Чек-лист отчуждения {: #research_pkg_b_chek_list_otchuzhdeniya }

| Этап | Действие | Результат |
|------|----------|-----------|
| **1. Документация** | Руководство для агентов + RFC | Инструкции для агентов |
| **2. Код** | VCS + CI/CD | Воспроизводимость |
| **3. Модели** | Export weights | Независимость от API |
| **4. Данные** | Backup + Schema | Миграция |
| **5. Инфраструктура** | Docker/K8s | Деплой |
| **6. Обучение** | Воркшопы | Передача знаний |
| **7. Поддержка** | SLA | Помощь при проблемах |

### Риски при отчуждении {: #research_pkg_b_riski_pri_otchuzhdenii }

| Риск | Митигация |
|------|----------|
| **Потеря знаний** | Документация, руководство для агентов |
| **Зависимость от API** | Локальные модели |
| **Данные в облаке** | On-premise векторное хранилище |
| **Регуляторные риски** | 152-ФЗ комплаенс |
| **Безопасность** | Guardrails, аудит |

### Финальные рекомендации {: #research_pkg_b_finalnye_rekomendatsii }

1. **Для новых внедрений:**
   - Начните с POC на облачном инференсе
   - Перейдите on-premise для продакшена
   - Документируйте все артефакты

2. **Для отчуждения:**
   - Используйте open-source модели
   - Храните данные локально
   - Обучите команду клиента

3. **Для масштабирования:**
   - Мониторьте утилизацию
   - Оптимизируйте TCO
   - Планируйте на 3-5 лет

### Модули передачи знаний и поведенческая готовность при KT (справочно) {: #research_pkg_b_moduli_peredachi_znanii_i_povedencheskaya_gotovnost_pri_kt }

При планировании **передачи знаний (KT)** после внедрения полезно явно закладывать не только техническую документацию, но и **модули**, снижающие сопротивление и ошибочное использование GenAI: психологическая безопасность экспериментов, разделение ролей «промпт / ответ / эскалация», треки зрелости для бизнес- и ИТ-ролей. Ориентир по формализованному обучению безопасной работе с LLM — программа [Kaspersky — Large Language Models Security](https://www.kaspersky.com/about/press-releases/kaspersky-introduces-a-new-training-large-language-models-security); программы повышения квалификации руководителей по трансформации процессов — например, [СКОЛКОВО — «Переход в ИИ: трансформация бизнес-процессов»](https://www.skolkovo.ru/programmes/cdto/) (структура и заявленные результаты — на странице программы; **не** часть поставки референс-стека CMW).

---

## Источники {: #research_pkg_b_istochniki }

- Полный консолидированный реестр — см. [Приложение A: обзор и ведомость документов](./20260325-research-appendix-a-index-ru.md#research_pkg_a_polnyi_reestr_ispolzovannyh_istochnikov_tochnaya_konsolidatsiya).

### BPMN 2.0 и шаблоны промптов (справочно) {: #research_pkg_b_bpmn_20_i_shablony_promptov_spravochno }

- [GitHub — yksi12/prompts: generate-bpmn-prompt.md](https://github.com/yksi12/prompts/blob/main/generate-bpmn-prompt.md)

### Нормативные и стратегические материалы {: #research_pkg_b_normativnye_i_strategicheskie_materialy }

- [ISO/IEC 42001:2023 — Artificial intelligence management system](https://www.iso.org/standard/81230.html)
- [NIST — AI RMF to ISO/IEC 42001 Crosswalk (PDF)](https://airc.nist.gov/docs/NIST_AI_RMF_to_ISO_IEC_42001_Crosswalk.pdf)
- [NIST — AI RMF: Generative AI Profile (NIST.AI.600-1, 2024)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
- [NIST AIRC — Roadmap for the AI Risk Management Framework](https://airc.nist.gov/airmf-resources/roadmap)
- [Официальное опубликование — Приказ Роскомнадзора от 19.06.2025 № 140 (обезличивание ПДн)](http://publication.pravo.gov.ru/document/0001202508010002)
- [Фонтанка — проект закона о госрегулировании ИИ (Минцифры, 18.03.2026)](https://www.fontanka.ru/2026/03/18/76318717/)
- [ACSOUR — обязанность операторов передавать анонимизированные ПД в ГИС (152-ФЗ)](https://acsour.com/en/news-and-articles/tpost/2g13ahnab1-mandatory-anonymized-personal-data-shari)
- [DataGuidance — поправки к национальной стратегии развития ИИ РФ](https://www.dataguidance.com/news/russia-president-issues-amendments-national-ai)
- [Известия (EN) — создание офисов внедрения ИИ](https://en.iz.ru/en/node/1985740)

### Проекты OWASP по безопасности ИИ: GenAI Security, WSTG, ASVS и руководство по тестированию {: #research_pkg_b_proekty_owasp_po_bezopasnosti_ii_genai_security_wstg_asvs_i_rukovodstvo_po_testi }

- [OWASP GenAI Security Project — Introduction](https://genai.owasp.org/introduction-genai-security-project/)
- [GitHub — OWASP www-project-ai-testing-guide](https://github.com/OWASP/www-project-ai-testing-guide)
- [OWASP — Web Security Testing Guide (WSTG), stable](https://owasp.org/www-project-web-security-testing-guide/stable/)
- [GitHub — OWASP ASVS 5.0.0 (PDF, RU)](https://github.com/OWASP/ASVS/blob/master/5.0/OWASP_Application_Security_Verification_Standard_5.0.0_ru.pdf)
- [Habr — OWASP LLM TOP 10 2025 (адаптация)](https://habr.com/ru/companies/owasp/articles/893712/)

### Программы обучения по безопасности LLM {: #research_pkg_b_programmy_obucheniya_po_bezopasnosti_llm }

- [Kaspersky — программа «Large Language Models Security» (пресс-релиз, описание)](https://www.kaspersky.com/about/press-releases/kaspersky-introduces-a-new-training-large-language-models-security)

### Платформа корпоративных агентов MTS AI {: #research_pkg_b_platforma_korporativnyh_agentov_mts_ai }

- [MWS AI — MWS AI Agents Platform (описание модулей)](https://mts.ai/product/ai-agents-platform/)

### Публичные веса с нестандартной лицензией {: #research_pkg_b_publichnye_vesa_s_nestandartnoi_litsenziei }

- [Hugging Face — LICENSE (YandexGPT-5-Lite-8B), сырой текст соглашения](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct/raw/main/LICENSE)
- [Hugging Face — карточка модели YandexGPT-5-Lite-8B-instruct](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct)
- [Yandex Research — обзор направлений работ (2025)](https://research.yandex.com/blog/yandex-research-in-2025)
- [Yandex Research — принятые к ICML 2025 (список, в т.ч. KV-cache)](https://research.yandex.com/blog/papers-accepted-to-icml-2025)
- [arXiv — Cache Me If You Must (KV-quantization), 2501.19392](https://arxiv.org/abs/2501.19392)

### Открытые модели ai-sage (GigaChat и спутники) {: #research_pkg_b_otkrytye_modeli_ai_sage_gigachat_i_sputniki }

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

### Лицензионные материалы и 152-ФЗ (MWS / МТС) {: #research_pkg_b_litsenzionnye_materialy_i_152_fz_mws_mts }

- [MWS Docs — лицензионные условия ПО «MWS GPT»](https://mws.ru/docs/docum/lic_terms_mwsgpt.html)
- [МТС Cloud — IaaS 152-ФЗ УЗ-1](https://cloud.mts.ru/services/iaas-152-fz/)
- [MWS Docs — условия облачного сегмента 152-ФЗ](https://mws.ru/docs/docum/cloud_terms_152fz.html)
- [MWS — новость: хранение персональных данных в облаке](https://mws.ru/news/mts-web-services-zapustila-servis-dlya-hraneniya-personalnyh-dannyh-v-oblake/)

### Модель Build–Operate–Transfer (передача знаний) {: #research_pkg_b_model_build_operate_transfer_peredacha_znanii }

- [Luxoft — модель Build–Operate–Transfer (BOT)](https://www.luxoft.com/blog/master-the-build-operate-transfer-bot-model-with-luxoft)
- [InOrg — seamless handover в модели BOT](https://inorg.com/blog/from-build-to-transfer-key-success-factors-a-seamless-bot-model-transition)

### Регулирование (проектный контур 2026) {: #research_pkg_b_regulirovanie_proektnyi_kontur_2026 }

- [Портал НПА — проект федерального закона (ID 166424)](https://regulation.gov.ru/projects#npa=166424)

### Опрос CMO Club × red_mad_robot (маркетинг, концентрация SaaS и ИС) {: #research_pkg_b_opros_cmo_club_red_mad_robot_marketing_kontsentratsiya_saas_i_is }

- [Telegram — CMO Club Russia: анонс исследования GenAI в маркетинге (пост 197)](https://t.me/cmoclub/197)
- [RB.RU — 93% команд в маркетинге используют ИИ (обзор исследования CMO Club × red_mad_robot, 2025)](https://rb.ru/news/93-komand-v-marketinge-ispolzuyut-ii-nejroseti-pomogayut-s-kontentom-no-im-pochti-ne-doveryayut-strategiyu-i-byudzhety/)

### Инструменты разработки с ИИ (ориентиры для заказчика, вне SKU CMW) {: #research_pkg_b_instrumenty_razrabotki_s_ii_orientiry_dlya_zakazchika_vne_sku_cmw }

- [Хабр — red_mad_robot: MCP Tool Registry (реестр MCP для RAG/агентов, открытый код)](https://habr.com/ru/companies/redmadrobot/articles/982004/)
- [GitHub — redmadrobot-rnd/mcp-registry](https://github.com/redmadrobot-rnd/mcp-registry)
- [OpenCode](https://opencode.ai/)
- [OpenCode — документация (Intro)](https://opencode.ai/docs)
- [OpenCode — Ecosystem](https://opencode.ai/docs/ecosystem/)
- [OpenCode Zen](https://opencode.ai/docs/zen)
- [OpenWork (different-ai/openwork)](https://github.com/different-ai/openwork)
- [OpenRouter](https://openrouter.ai/)
- [OpenRouter — Logging и политики провайдеров](https://openrouter.ai/docs/guides/privacy/logging)
- [Claude Platform — Models overview](https://platform.claude.com/docs/en/about-claude/models/overview) (актуальная линейка **Claude 4.6**; см. также [What's new in Claude 4.6](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-6)) — справочно при сверке зарубежного API с контрактной и комплаенс-моделью заказчика
