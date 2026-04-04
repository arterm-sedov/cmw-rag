# Report-Pack Drift: Focused Diff Extracts

## Scope

- Comparison window: `7b35683..dc3d554`
- Canonical baseline for this audit: `7b35683`
- Current tip under review: `dc3d554`
- Related worktrees in the same window:
  - `.worktrees/20260404-smart-replay`
  - `.worktrees/20260404-smart-resync`
  - `.worktrees/20260404-appendix-de-polish`
- Related temp branches in the same window:
  - `cursor/-bc-52d2a31d-e222-44c0-aab8-959845c4b1aa-4fef`
  - `cursor/-bc-619d5694-21bc-4cb8-8e65-6165e24f94f9-eca1`
  - `cursor/-bc-9b2928cb-788e-4076-a537-964e42c1aad2-16ee`
  - `cursor/-bc-9d52bc1e-640f-4120-973e-59922a52f74c-5545`
  - `cursor/-bc-954187d7-1ab6-4e64-a08e-836c2255a284-943d`

## 1. Methodology: malformed markdown links introduced

```diff
@@ -427,7 +431,7 @@
-**Yandex Cloud (Yandex AI Studio / YandexGPT)** · [модели](https://aistudio.yandex.ru/docs/ru/ai-studio/concepts/generation/models.html) · [тарификация](https://aistudio.yandex.ru/docs/ru/ai-studio/pricing.html)
+**Yandex Cloud (Yandex AI Studio / YandexGPT)** · [модели]](https://aistudio.yandex.ru/docs/ru/ai-studio/concepts/generation/models.html) · [[тарификация](https://aistudio.yandex.ru/docs/ru/ai-studio/pricing.html)

@@ -435,19 +439,17 @@
-**SberCloud (GigaChat API)** · [портал](https://developers.sber.ru/portal/products/gigachat-api) · [юридические тарифы](https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs)
+**SberCloud (GigaChat API)** [портал]](https://developers.sber.ru/portal/products/gigachat-api) · [[юридические тарифы](https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs)

@@ -442,7 +444,7 @@
-**MWS GPT (МТС Web Services)** · [продукт](https://mws.ru/mws-gpt/) · [тарифы](https://mws.ru/docs/docum/cloud_terms_mwsgpt_pricing.html)
+**MWS GPT (МТС Web Services)** [продукт]](https://mws.ru/mws-gpt/) · [[тарифы](https://mws.ru/docs/docum/cloud_terms_mwsgpt_pricing.html)
```

## 2. Appendix A: anchor fix was overwritten back toward dead targets

```diff
@@ -204,8 +204,8 @@
-| **Ориентиры для углублённого аппаратного сайзинга (официальные бенчмарки и документация)** | [Отчёт «Сайзинг и экономика (CapEx / OpEx / TCO)»](./20260325-research-report-sizing-economics-main-ru.md#sizing_inference_benchmarks_vram_tools) |
-| **Бенчмарки RTX 4090 (потребительская 24 ГБ GeForce; 48 ГБ — коммерческая аренда)** | [Отчёт «Сайзинг и экономика (CapEx / OpEx / TCO)»](./20260325-research-report-sizing-economics-main-ru.md#sizing_inference_benchmarks_vram_tools) |
+| **Ориентиры для углублённого аппаратного сайзинга (официальные бенчмарки и документация)** | [Отчёт «Сайзинг и экономика (CapEx / OpEx / TCO)»](./20260325-research-report-sizing-economics-main-ru.md#sizing_hardware_deep_research_pointers) |
+| **Бенчмарки RTX 4090 (реф. 24 ГБ GeForce; 48 ГБ — коммерческая аренда)** | [Отчёт «Сайзинг и экономика (CapEx / OpEx / TCO)»](./20260325-research-report-sizing-economics-main-ru.md#sizing_rtx_4090_benchmarks) |
```

## 3. Sizing: Appendix E cross-link regressed to a non-existent anchor

```diff
@@ -1076,7 +1072,7 @@
 ### Актуальные тренды AI/ML {: #sizing_ai_ml_trends_channel }

-Актуальные тренды AI/ML для стратегического планирования — см. _[Приложение E](./20260325-research-appendix-e-market-technical-signals-ru.md#app_e_trends_2026_summary)_.
+Актуальные тренды AI/ML для стратегического планирования — см. _[Приложение E](./20260325-research-appendix-e-market-technical-signals-ru.md#sizing_ai_ml_trends_channel)_.
```

## 4. Sizing: provider-pricing precision was flattened

```diff
@@
-    Для **Sber GigaChat API** использованы публичные тарифы для юрлиц (pay-as-you-go / sync, действуют с **1 февраля 2026**); асинхронные ставки вынесены в примечания.
-    Для **Yandex AI Studio** первичный биллинг публикуется как **₽ за 1000 токенов с НДС** в таблице Model Gallery; здесь значения пересчитаны в **₽/млн**.
-    Для **Cloud.ru Evolution Foundation Models** в КП фиксируйте **конкретный SKU и дату прайса**: официальный документ версии **260316** вступил в силу **26 марта 2026**.
-| **Cloud.ru** | GigaChat3-10B-A1.8B | 12,2 | 12,2 | Evolution FM; с НДС; прайс от 26.03.2026 |
+| **Cloud.ru** | Evolution | 35 | 70 | |

-!!! note "Сбер и Cloud.ru: не смешивать строки"
-    Строки **Sber** в таблице — тарифы **GigaChat API** для юрлиц. Строка **Cloud.ru** — это тариф **конкретного SKU** в **Evolution Foundation Models**, а не общий эквивалент тарифов Сбера по всем моделям.
+    Актуальны на март 2026. Цены едины для Cloud.ru и SberCloud.
```

## 5. Appendix D: detailed EU AI Act context was removed

```diff
@@
-### Международный регуляторный контекст: EU AI Act (справочно) {: #app_d__eu_ai_act_context }
-
-Раздел — **сравнительный фон** для сделок, где продукт или его компоненты используются клиентами в ЕС.
-Приоритет для резидентного контура РФ остаётся 152-ФЗ, 123-ФЗ и законопроектом об ИИ (см. выше).
-
-!!! warning "Юридическая оговорка"
-    Применимость EU AI Act к конкретному продукту зависит от роли (поставщик / оператор / импортёр / дистрибьютор), факта вывода на рынок и характера использования в ЕС.
-    Данный раздел не является юридическим заключением — для квалификации привлекайте уполномоченного юриста.
-
-#### Хронология вступления в силу {: #app_d__eu_ai_act_timeline }
-#### Территориальный охват {: #app_d__eu_ai_act_scope }
-#### Структура штрафов (ст. 99) {: #app_d__eu_ai_act_penalties }
```

## 6. Appendix E: useful new additions that appear aligned with goals

```diff
@@
+#### MWS Octapi {: #app_e_ent_mws_octapi }
+- **Концепция:** ускорить подключение ИИ-агентов к корпоративным системам без отдельного слоя «самодельных» интеграций.
+- **Решение:** **MWS Octapi** — интеграционная платформа из реестра российского ПО с **MCP**-поддержкой...

+#### Yandex Agent Atelier {: #app_e_ent_yandex_agent_atelier }
+- **Концепция:** собрать корпоративного агента не как отдельный поисковый инструмент, а как управляемую платформу...
+- **Решение:** **Agent Atelier** в **Yandex AI Studio** объединяет **Agent Tools**, **MCP Hub** и **Workflows**...

+#### Yandex SpeechKit / AI Speech {: #app_e_ent_yandex_speechkit }
+#### SaluteSpeech (Сбер) {: #app_e_ent_salutespeech }
+#### ElevenLabs {: #app_e_ent_elevenlabs_music }
```

## 7. Executive summary: new evidence and framing that look useful

```diff
@@ -36,8 +36,8 @@
-**Ситуация:** спрос на GenAI растёт, но у заказчиков дефицит управляемого внедрения и масштабирования.
-**Проблема:** без прозрачной экономики, контроля качества и формализованной передачи пилоты не капитализируются в устойчивый внутренний актив.
+**Ситуация:** спрос на GenAI растёт, но у заказчиков дефицит управляемого внедрения и масштабирования; при этом международные исследования фиксируют разрыв между масштабом использования ИИ и реально подтверждённым эффектом на уровне бизнеса.
+**Проблема:** без прозрачной экономики, контроля качества и формализованной передачи пилоты не капитализируются в устойчивый внутренний актив: по McKinsey регулярное использование ИИ есть уже у 88% организаций хотя бы в одной функции, но только 39% фиксируют enterprise-level EBIT impact; по BCG ИИ входит в top-3 приоритетов у 75% руководителей, тогда как значимую ценность видят лишь 25%.
```

## 8. Methodology: useful new grounding additions

```diff
@@
+Это уже отражается и в внешней статистике: Gartner указывает, что **63%** организаций либо не имеют, либо не уверены, что имеют корректные практики управления данными для ИИ...
+
+Это согласуется с правилом **10-20-70**, которое BCG использует для AI-трансформаций: около **10%** эффекта определяют алгоритмы, **20%** — данные и технологии, **70%** — люди, процессы и организационные изменения...
```

## Interpretation

- Sections 1-5 are candidate regressions or harmful compressions.
- Sections 6-8 are candidate enrichments that are broadly aligned with the task goals and should be preserved if they survive a cleanup pass.
