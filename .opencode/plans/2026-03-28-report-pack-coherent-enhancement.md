# Report-Pack Coherent Enhancement & Cross-Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 10-document `report-pack` act as a single coherent system — consistent evidence, accurate cross-links, sharp narrative, and complete market context — so any C-level executive (CEO, CRO, CFO, CIO, CISO, CPO) can enter from any document and trust what they read.

**Architecture:** Four phases executed sequentially: (0) validated defect ledger (already complete — embedded below), (1) structural repairs to broken YAML/anchors/links, (2) cross-validation of the broken canonical-source chain for the CMO survey data, (3) narrative enrichment of the two weakest customer-facing documents.

**Tech Stack:** Markdown, Kramdown-style heading anchors (`{: #anchor }`), YAML front matter, Russian-language prose per `docs/research/AGENTS.md` conventions.

---

## Defect Ledger (validated, read-only — do not edit)

> All defects below were confirmed by direct file inspection on 2026-03-28. Line numbers are stable as of that date; re-verify before editing.

### D1 — YAML: `comindware-ai-commercial-offer-ru.md` — malformed `tags` (CONFIRMED)

**File:** `20260325-comindware-ai-commercial-offer-ru.md`
**Line 6:** `- продажипродажи` — two words fused without a space; should be `- продажи`.
**Impact:** Tag deduplication/filtering broken; cosmetic in rendered Markdown but wrong in YAML.

### D2 — Missing `## Источники` section in commercial offer (CONFIRMED)

**File:** `20260325-comindware-ai-commercial-offer-ru.md`
**Issue:** The doc implicitly references market data (43%×2, 93%, etc.) but has no `## Источники` section — the only doc in the pack missing it per AGENTS.md requirement.
**Impact:** Executives using this as a pitch doc have no citation trail.

### D3 — Appendix A: Appendix C listed without hyperlink (CONFIRMED)

**File:** `20260325-research-appendix-a-index-ru.md`
**Line 63:** `- «Приложение C: имеющиеся наработки CMW»` — plain text, no link.
All other 9 docs in the pack are hyperlinked here. Same defect on line 74 (topic table), line 93 (Q→doc), and line 128 (source mapping table).
**Impact:** Navigation dead-end from the master index.

### D4 — Appendix C: no YAML front matter (CONFIRMED)

**File:** `20260325-research-appendix-c-cmw-existing-work-ru.md`
**Lines 1-4:** No `---` YAML block — only bold text for date/status.
Only document in the pack without proper YAML front matter.
**Impact:** Metadata inconsistency; date/status not machine-readable.

### D5 — Appendix C: "Related documents" has no hyperlinks (CONFIRMED)

**File:** `20260325-research-appendix-c-cmw-existing-work-ru.md`
**Lines 16-22:** All five sibling doc references are plain text — no links.
Every other doc in the pack uses hyperlinks for its sibling list.
**Impact:** Navigation dead-end from Appendix C.

### D6 — Appendix C: no heading anchor on H1 or H2 pack-overview (CONFIRMED)

**File:** `20260325-research-appendix-c-cmw-existing-work-ru.md`
**Line 1:** `# Приложение C. Доказательство готовности...` — no `{: #app_c_pack_overview }`.
**Line 6:** `## Обзор комплекта` — no anchor.
The commercial offer links to Appendix C without an anchor (`./20260325-research-appendix-c-cmw-existing-work-ru.md` — valid file, but no deep-link possible).
**Impact:** External references can't deep-link to the pack overview section.

### D7 — CRITICAL: `#sizing_russian_market` anchor is in Appendix E, but 4 docs link to it as if it were in the Sizing report (CONFIRMED)

**Anchor defined in:** `20260325-research-appendix-e-market-technical-signals-ru.md`, line 152 (`### Российский рынок {: #sizing_russian_market }`).

**4 documents incorrectly reference `sizing-economics-main#sizing_russian_market`:**
- `20260325-research-appendix-a-index-ru.md` line 99
- `20260325-research-appendix-b-ip-code-alienation-ru.md` lines 68, 72
- `20260325-research-appendix-d-security-observability-ru.md` line 97
- `20260325-research-report-methodology-main-ru.md` line 695

**Impact:** These 4 cross-links point to a non-existent anchor in the wrong file. Any reader following the link in a rendered MkDocs site lands at the top of the sizing report, not at the Russian market survey section. This is the most structurally important defect: it breaks the navigation chain for the CMO survey data (43%×2, 93%, etc.) for all 6 C-level audiences.

**Resolution strategy (per user preference — avoid duplication):**
Add a minimal stub section `## Российский рынок GenAI (опрос CMO Club × red_mad_robot, 2025) {: #sizing_russian_market }` to `sizing-economics-main-ru.md` containing 3-4 bullet key figures + a pointer to Appendix E for the full breakdown. This makes the anchor real, satisfies all 4 existing references, and keeps the full data in one place (Appendix E).

### D8 — Appendix C: reference to sizing report uses non-linked bold text for section name (CONFIRMED)

**File:** `20260325-research-appendix-c-cmw-existing-work-ru.md`, line 28-30.
Refers to «Оценка сайзинга, КапЭкс и ОпЭкс для клиентов» (раздел «**Российский рынок**») in bold plain text — no hyperlink to the actual section.
After D7 is fixed, this should link to `sizing-economics-main#sizing_russian_market`.

### D9 — Commercial offer: no market evidence (CMO survey) despite being the primary pitch doc (CONFIRMED)

**File:** `20260325-comindware-ai-commercial-offer-ru.md`
Zero mentions of CMO Club survey, 93%, 43%, or any market demand signal.
Every other doc in the pack includes or cross-references this data.
**Impact:** The most customer-facing document is the weakest on market justification — exactly what a CRO or CEO needs in a first conversation.

### D10 — Executive summaries: no role-specific "so what" aligned to commercial offer role matrix (CONFIRMED)

**Files:** `20260325-research-executive-methodology-ru.md`, `20260325-research-executive-sizing-ru.md`
Both docs have a "Содержательные тезисы" section but no role-specific takeaway mapping to the matrix (CEO/CRO/CFO/CIO/CISO/CPO) defined in the commercial offer.
**Impact:** Executives reading the summaries have to self-navigate; the pitch alignment is lost.

---

## Phase 1 — Structural Repairs

> **Checkpoint after Phase 1:** Verify with a read-only review before proceeding to Phase 2.

### Task 1.1 — Fix YAML tag typo in commercial offer

**Files:**
- Modify: `report-pack/20260325-comindware-ai-commercial-offer-ru.md` line 6

- [ ] **Step 1: Open file and confirm defect**
  Read line 6: confirm `- продажипродажи`.

- [ ] **Step 2: Fix the tag**
  Replace `- продажипродажи` with two separate tags:
  ```yaml
    - продажи
    - продукт
  ```
  (Remove the duplicate `- продукт` on line 7 since it would then appear twice. Final tags block should be: `продажи`, `продукт`, `корпоративный ИИ`, `GenAI`, `RAG`, `TOM`, `KT`, `BOT`.)

- [ ] **Step 3: Verify YAML is valid**
  Confirm the front matter opens with `---` on line 1 and closes with `---` on line 14. Re-read lines 1–15 to confirm.

---

### Task 1.2 — Add YAML front matter to Appendix C

**Files:**
- Modify: `report-pack/20260325-research-appendix-c-cmw-existing-work-ru.md` lines 1-4

- [ ] **Step 1: Confirm current state**
  Read lines 1–5. Confirm no `---` YAML block; only `# Приложение C...` heading and bold text.

- [ ] **Step 2: Prepend YAML block**
  Insert before line 1:
  ```yaml
  ---
  title: 'Приложение C. Доказательство готовности: референс-стек Comindware (состав, границы, артефакты)'
  date: 2026-03-25
  status: 'Утверждённый комплект материалов для руководства (v1, март 2026)'
  tags:
    - архитектура
    - GenAI
    - корпоративный
    - RAG
    - референс-стек
    - состав стека
    - KT
  ---
  ```

- [ ] **Step 3: Remove redundant bold-text metadata**
  Delete lines (now shifted) `**Дата комплекта:** 2026-03-25` and `**Статус:** утверждённый комплект материалов для руководства (v1)` — this info is now in YAML.

- [ ] **Step 4: Add heading anchor to H1**
  Change `# Приложение C. Доказательство готовности (Proof of Capability): референс-стек Comindware (состав, границы, артефакты)`
  to `# Приложение C. Доказательство готовности (Proof of Capability): референс-стек Comindware (состав, границы, артефакты) {: #app_c_pack_overview }`

- [ ] **Step 5: Add anchor to pack-overview H2**
  Change `## Обзор комплекта` to `## Обзор комплекта {: #app_c_overview }`

- [ ] **Step 6: Verify file starts correctly**
  Re-read lines 1–12 to confirm YAML block + H1 with anchor.

---

### Task 1.3 — Fix Appendix C "Related documents" section (add hyperlinks)

**Files:**
- Modify: `report-pack/20260325-research-appendix-c-cmw-existing-work-ru.md` lines 16-22

- [ ] **Step 1: Confirm current state**
  Read lines 16–23. Confirm all 5 entries are plain text without hyperlinks.

- [ ] **Step 2: Replace with linked list**
  Replace the entire `## Связанные документы` block (lines 16-22) with:
  ```markdown
  ## Связанные документы

  - [«Приложение A: обзор и ведомость документов»](./20260325-research-appendix-a-index-ru.md#app_a_pack_overview)
  - [«Основной отчёт: методология внедрения и разработки»](./20260325-research-report-methodology-main-ru.md#method_pack_overview)
  - [«Основной отчёт: сайзинг и экономика»](./20260325-research-report-sizing-economics-main-ru.md#sizing_pack_overview)
  - [«Приложение B: отчуждение ИС и кода (KT, IP)»](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_pack_overview)
  - [«Приложение D: безопасность, комплаенс и наблюдаемость (observability)»](./20260325-research-appendix-d-security-observability-ru.md#app_d__pack_overview)
  ```

- [ ] **Step 3: Verify all 5 anchors exist**
  Grep for `#app_a_pack_overview`, `#method_pack_overview`, `#sizing_pack_overview`, `#app_b_pack_overview`, `#app_d__pack_overview` in the respective files to confirm each target anchor exists.

---

### Task 1.4 — Fix Appendix A: add Appendix C hyperlinks in all 4 locations

**Files:**
- Modify: `report-pack/20260325-research-appendix-a-index-ru.md` lines 63, 74, 93, 128

- [ ] **Step 1: Confirm defects**
  Read the 4 lines to confirm they are all plain text without links.

- [ ] **Step 2: Fix line 63 — Related documents list**
  Replace `- «Приложение C: имеющиеся наработки CMW»`
  with `- [«Приложение C: доказательство готовности — референс-стек Comindware»](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_pack_overview)`

- [ ] **Step 3: Fix line 74 — Topic table**
  In the table row for `Наработки Comindware`:
  Replace `«Приложение C: имеющиеся наработки Comindware»`
  with `[«Приложение C: доказательство готовности — референс-стек Comindware»](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_pack_overview)`

- [ ] **Step 4: Fix line 93 — Q→doc navigation**
  Replace the plain-text reference `Приложение C: имеющиеся наработки Comindware`
  with `[Приложение C: доказательство готовности — референс-стек Comindware](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_pack_overview)`

- [ ] **Step 5: Fix line 128 — Source mapping table**
  In the row for `Обзор текущей архитектуры Comindware`:
  Replace `Приложение C: имеющиеся наработки Comindware`
  with `[Приложение C: доказательство готовности — референс-стек Comindware](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_pack_overview)`

- [ ] **Step 6: Re-read all 4 fixed lines to confirm**

---

**PHASE 1 CHECKPOINT — stop here for review before Phase 2.**

Verify:
1. Commercial offer YAML is valid (tags fixed, `---` present top and bottom)
2. Appendix C has YAML front matter + H1 anchor + H2 anchor
3. Appendix C related-docs list has 5 working hyperlinks
4. Appendix A has Appendix C hyperlinked in all 4 locations
5. No other content was changed

---

## Phase 2 — Cross-Validation & Broken Anchor Resolution

> **Checkpoint after Phase 2.** This phase makes structural changes with semantic impact; review each task output carefully.

### Task 2.1 — Fix D7: Create `#sizing_russian_market` stub in sizing-economics report

**Files:**
- Modify: `report-pack/20260325-research-report-sizing-economics-main-ru.md`

**Context:** 4 documents point to `sizing-economics-main#sizing_russian_market`. That anchor does not exist in the sizing report. The anchor IS defined in Appendix E (`#sizing_russian_market`, line 152), which is the right home for the full data. The fix: add a minimal stub section in the sizing report that (a) makes the anchor real, (b) gives key figures inline, (c) points to Appendix E for full breakdown. This satisfies all 4 existing references without duplicating Appendix E's data block.

- [ ] **Step 1: Find the right insertion point**
  Search for `sizing_russian_ai_cloud_tariffs` or `sizing_russian_recommendations` in the sizing report to find a natural "Russian market" area. The section `## Тарифы российских облачных провайдеров ИИ {: #sizing_russian_ai_cloud_tariffs }` (line 372) is the nearest. Insert the new stub **before** this section.

- [ ] **Step 2: Write the stub section**
  Insert at line ~371 (before the tariffs H2):
  ```markdown
  ## Российский рынок GenAI: зрелость и барьеры (опрос CMO Club × red_mad_robot, 2025) {: #sizing_russian_market }

  Ключевые доли из публичных материалов исследования **red_mad_robot × CMO Club Russia** (2025) — сигнал спроса со стороны владельцев маркетингового бюджета:

  - **93%** компаний используют GenAI в рабочих процессах маркетинга; системно интегрировали технологию **около трети** респондентов.
  - **64%** выделяют на GenAI **только 1–5%** маркетингового бюджета — разрыв между использованием и инвестицией.
  - Барьеры: **53%** — необходимость доработки контента; **49%** — шаблонность; **43%** — галлюцинации и ошибки; **отдельно 43%** — риски утечки данных (это **не** та же доля, что про галлюцинации — разбор в _«[Приложение D: безопасность, комплаенс и observability](./20260325-research-appendix-d-security-observability-ru.md#app_d__org_barriers_risk_survey_2025)»_).
  - **85%** российских CMO считают GenAI ключевым фактором трансформации на горизонте трёх лет.

  Полный набор долей и формулировок (91% ChatGPT, 59% Midjourney, эффекты, бюджетный разрыв, взгляд вперёд) — в _«[Приложение E: рыночные и технические сигналы — Российский рынок](./20260325-research-appendix-e-market-technical-signals-ru.md#sizing_russian_market)»_.

  ```

- [ ] **Step 3: Verify the 4 broken references now resolve correctly**
  Re-read the stub. Confirm: anchor `#sizing_russian_market` is present, key figures are correct (match Appendix E lines 173-182), 43% disambiguation is explicit, pointer to Appendix E is present, pointer to Appendix D for the security split is present.

- [ ] **Step 4: Add `## Источники` entry for the CMO survey**
  At the end of the sizing report's `## Источники` section (around line 1575), confirm these two entries already exist:
  - `[Telegram — CMO Club Russia: анонс исследования GenAI в маркетинге (red_mad_robot × CMO Club, 2025)](https://t.me/cmoclub/197)`
  - `[RB.RU — 93% команд в маркетинге используют ИИ...](https://rb.ru/news/...)`
  If either is missing, add it.

---

### Task 2.2 — Fix D8: Update Appendix C reference to use the new anchor

**Files:**
- Modify: `report-pack/20260325-research-appendix-c-cmw-existing-work-ru.md` lines 28-30

- [ ] **Step 1: Confirm current state**
  Read lines 28-31. Confirm the bold-text reference «Оценка сайзинга, КапЭкс и ОпЭкс для клиентов» (раздел «**Российский рынок**») is not hyperlinked.

- [ ] **Step 2: Replace bold text with hyperlink**
  In the sentence on line 30, replace the bold-text reference to the survey data:
  - Change: `зафиксированы в сопутствующем резюме **Оценка сайзинга, КапЭкс и ОпЭкс для клиентов** (раздел «**Российский рынок**»)`
  - To: `зафиксированы в _«[Основной отчёт: сайзинг и экономика — Российский рынок](./20260325-research-report-sizing-economics-main-ru.md#sizing_russian_market)»_`

- [ ] **Step 3: Verify the link and surrounding sentence are grammatically correct**
  Re-read lines 28-32.

---

### Task 2.3 — Verify all other major named anchors referenced across files

This is a systematic spot-check. For each anchor below, confirm it exists in the target file.

- [ ] **Step 1: Check `#app_b_pack_overview`**
  Grep `app_b_pack_overview` in `appendix-b-ip-code-alienation-ru.md`. Confirm defined.

- [ ] **Step 2: Check `#app_d__pack_overview`**
  Grep `app_d__pack_overview` in `appendix-d-security-observability-ru.md`. Confirm defined.

- [ ] **Step 3: Check `#method_pack_overview`**
  Grep `method_pack_overview` in `research-report-methodology-main-ru.md`. Confirm defined.

- [ ] **Step 4: Check `#app_a_fx_policy`**
  Grep `app_a_fx_policy` in `appendix-a-index-ru.md`. Confirm defined.

- [ ] **Step 5: Check `#method_genai_marketing_teams`**
  Grep in `research-report-methodology-main-ru.md`. Confirm defined.

- [ ] **Step 6: Check `#app_b_shadow_genai_marketing_model_routing`**
  Grep in `appendix-b-ip-code-alienation-ru.md`. Confirm defined.

- [ ] **Step 7: Check `#app_b_capability_transfer_overview`** (referenced from Appendix C line 13)
  Grep in `appendix-b-ip-code-alienation-ru.md`. Confirm defined. If not found, downgrade the link to `#app_b_pack_overview` which is confirmed to exist.

- [ ] **Step 8: Document results**
  Record any missing anchors found. If any are missing, add minimal anchor stubs (a bare `{: #anchor_name }` on the nearest relevant heading). Do NOT restructure content.

---

**PHASE 2 CHECKPOINT — stop here for review before Phase 3.**

Verify:
1. `sizing-economics-main` now contains `#sizing_russian_market` with correct stub content
2. Key figures in stub match Appendix E exactly (spot-check: 93%, 64%, 53%, 49%, 43%×2, 85%)
3. Appendix C line 30 now links to `sizing-economics-main#sizing_russian_market`
4. All 7 spot-checked anchors from Task 2.3 confirmed or fixed
5. No data was changed — only structure and links

---

## Phase 3 — Narrative Enrichment

> **Checkpoint after Phase 3.** Content additions — review for tone, accuracy, and non-duplication.

### Task 3.1 — Enrich commercial offer with market demand evidence and `## Источники`

**Files:**
- Modify: `report-pack/20260325-comindware-ai-commercial-offer-ru.md`

**Principle:** The commercial offer is the first document a C-level executive sees. It currently has zero market demand data. Add a lean, factual demand signal section and a proper `## Источники` section. Do NOT write new research — pull from existing validated figures in Appendix E.

- [ ] **Step 1: Find insertion point**
  The section `## Зачем это бизнесу (P&L и контроль)` (line 27) is the right place to add market context. Insert a new short section immediately after the `## Зачем это бизнесу` section (before `## Что мы продаём`).

- [ ] **Step 2: Write the demand evidence block**
  Insert between the `## Зачем это бизнесу` and `## Что мы продаём` sections:
  ```markdown
  ## Рыночный контекст (опрос CMO Club × red_mad_robot, 2025)

  По публичным материалам исследования **red_mad_robot × CMO Club Russia** среди директоров по маркетингу крупнейших брендов РФ:

  - **93%** компаний уже используют GenAI в рабочих процессах; системно интегрировали — лишь **около трети**.
  - **64%** тратят на GenAI **только 1–5%** маркетингового бюджета — разрыв между личной цифровой зрелостью и корпоративным управлением.
  - Главные барьеры: отсутствие стратегии, **43%** отмечают галлюцинации и ошибки, **отдельно 43%** — риски утечки данных (разные доли опроса).
  - **85%** считают GenAI ключевым фактором трансформации на горизонте **трёх лет**.

  Это рынок, где спрос уже есть, а **управляемый контур** (данные, eval, комплаенс, передача) остаётся полем конкуренции. Именно это предлагает Comindware — подробнее в _«[Приложение E: рыночные и технические сигналы](./20260325-research-appendix-e-market-technical-signals-ru.md#sizing_russian_market)»_ и _«[Основной отчёт: сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_russian_market)»_.
  ```

- [ ] **Step 3: Add `## Источники` section at end of file**
  After the last line of the navigation section, append:
  ```markdown
  ## Источники

  - [Telegram — CMO Club Russia: анонс исследования GenAI в маркетинге (red_mad_robot × CMO Club, 2025)](https://t.me/cmoclub/197)
  - [RB.RU — 93% команд в маркетинге используют ИИ (обзор исследования CMO Club × red_mad_robot, 2025)](https://rb.ru/news/93-komand-v-marketinge-ispolzuyut-ii-nejroseti-pomogayut-s-kontentom-no-im-pochti-ne-doveryayut-strategiyu-i-byudzhety/)
  - [РБК Education — во сколько обойдётся ИИ-агент: подсчёты экспертов (2026)](https://www.rbc.ru/education/05/02/2026/697162269a794772c9cf9991)
  - [OpenAI — The state of enterprise AI (обзор, декабрь 2025)](https://openai.com/index/the-state-of-enterprise-ai-2025-report)
  - [Банк России — официальные курсы валют](https://www.cbr.ru/currency_base/daily/)
  ```

- [ ] **Step 4: Verify no duplication and correct tone**
  Re-read the full commercial offer. Confirm: (a) market section is lean (≤8 lines), (b) figures match Appendix E exactly, (c) 43%×2 disambiguation is clear, (d) tone is business-factual not marketing-fluff, (e) no internal tool names or repo paths appear in body text.

---

### Task 3.2 — Add role-specific "so what" to executive summaries

**Files:**
- Modify: `report-pack/20260325-research-executive-methodology-ru.md`
- Modify: `report-pack/20260325-research-executive-sizing-ru.md`

**Principle:** Both executive summary docs are accurate but end abruptly without connecting back to the role matrix defined in the commercial offer. Add a lean "Что важно каждому" block that mirrors the commercial offer's role matrix — no new content, just routing.

- [ ] **Step 1: Add role routing to executive-methodology summary**
  In `20260325-research-executive-methodology-ru.md`, before `## Где искать полноту`, insert:

  ```markdown
  ## Что важно каждому руководителю

  - **CEO / Генеральный директор:** фазы PoC → Pilot → Scale с измеримыми KPI; критерии выхода из «вечного пилота»; суверенность как управленческий аргумент.
  - **CRO / Директор по продажам:** упаковка предложения (пакеты 1–5); отчуждение как ценность, а не «техническая деталь»; питч «маркетинг / shadow SaaS» — в «Приложение B».
  - **CPO / Продукт:** TOM, сдвиг роли оператора-оркестратора, управление жизненным циклом знаний (RAG/процессы/правила).
  - **CIO / CTO:** архитектурные варианты и TOM — в «Основном отчёте: методология»; наблюдаемость — в «Приложение D».
  - **CISO / Комплаенс:** периметр до LLM, минимизация данных в логах, политика телеметрии, контроль инструментов агента — в «Приложение D».
  - **CFO:** юнит-экономика (₽/диалог, ₽/тикет), TCO и вилки CapEx/OpEx — в «Резюме для руководства: сайзинг».
  ```

- [ ] **Step 2: Add role routing to executive-sizing summary**
  In `20260325-research-executive-sizing-ru.md`, before `## Где искать полноту`, insert:

  ```markdown
  ## Что важно каждому руководителю

  - **CFO:** используйте вилки как «порядок величин»; в КП переносите после сверки прайса на дату и (on-prem) замеров на стенде; ±10% чувствительности по USD-статьям — ориентир, не прогноз.
  - **CIO / CTO:** матрица SaaS / on-prem / гибрид (РБК 2026) и дерево факторов — в «Основном отчёте: сайзинг»; аппаратный профиль Comindware (24 ГБ / 48 ГБ / 96 ГБ) — там же.
  - **CEO / Генеральный директор:** «скрытые» затраты после инцидента (ИБ, eval, миграции моделей) часто превышают годовой токенный бюджет — закладывать отдельной статьёй.
  - **CRO / Директор по продажам:** сценарные вилки — готовый аргумент для переговоров о бюджете; матрица РБК — публичный ориентир, не внутренний расчёт.
  - **CISO / Комплаенс:** OpEx безопасности GenAI (отдельная переменная статья) и TCO инцидента — в «Основном отчёте: сайзинг».
  - **CPO / Продукт:** юнит-экономика сценариев (₽/диалог, ₽/тикет, утилизация) — базис для приоритизации и road map.
  ```

- [ ] **Step 3: Verify both summaries remain ≤2 pages equivalent**
  Re-read both files top to bottom. Confirm they stay lean (the role blocks are ~8 lines each), no duplication with body text above, cross-links are correct.

---

### Task 3.3 — Add commercial offer entry to Appendix A Q→doc navigation

**Files:**
- Modify: `report-pack/20260325-research-appendix-a-index-ru.md`

**Issue:** The Q→doc navigation (lines 84-103) has no entry pointing to the commercial offer as the "first stop" for C-level pitch context. The doc exists and is linked in the Related Documents list, but is not referenced in the navigation by a question.

- [ ] **Step 1: Confirm current state**
  Read lines 84-86. Confirm the first Q→doc entry is about implementation (not the commercial offer).

- [ ] **Step 2: Prepend the commercial offer entry**
  Before the first Q→doc bullet (line 84), insert:
  ```markdown
  - «Нужно быстро понять суть предложения Comindware — что продаём, пакеты (PoC→Пилот→Масштаб→BOT), артефакты передачи, матрица по ролям?» → [«Comindware: корпоративный ИИ для промышленного предприятия — предложение для руководства»](./20260325-comindware-ai-commercial-offer-ru.md)
  ```

- [ ] **Step 3: Verify the entry is grammatically correct and follows the established Q→doc style**
  Re-read lines 83-87 to confirm the new entry blends in with the pattern.

---

**PHASE 3 CHECKPOINT — final review.**

Verify across all modified files:
1. Commercial offer: market section added (lean, ≤8 lines, correct figures), `## Источники` added (5 entries)
2. Executive summaries: both have role routing blocks (≤8 lines each), correct tone
3. Appendix A: commercial offer is now the first Q→doc entry
4. No repo paths, internal tool names, or raw scraping content appear in any customer-facing section
5. 43%×2 disambiguation is consistent in all newly added text
6. All new cross-links are valid (targets confirmed to exist in earlier phases)

---

## Summary: Files Modified and Expected Net Change

| File | Phase | Change type | Lines delta (approx.) |
|------|-------|-------------|----------------------|
| `comindware-ai-commercial-offer-ru.md` | 1+3 | YAML fix, market section, Источники | +~20 |
| `research-appendix-c-cmw-existing-work-ru.md` | 1 | YAML front matter, anchors, linked sibling list | +~15 |
| `research-appendix-a-index-ru.md` | 1+3 | 4 Appendix C links fixed, 1 Q→doc entry added | +~5 |
| `research-report-sizing-economics-main-ru.md` | 2 | `#sizing_russian_market` stub section | +~12 |
| `research-appendix-c-cmw-existing-work-ru.md` | 2 | Line 30 hyperlink fix | ±0 |
| `research-executive-methodology-ru.md` | 3 | Role routing block | +~10 |
| `research-executive-sizing-ru.md` | 3 | Role routing block | +~10 |

**Files NOT touched:** `research-report-methodology-main-ru.md`, `research-appendix-b-ip-code-alienation-ru.md`, `research-appendix-d-security-observability-ru.md`, `research-appendix-e-market-technical-signals-ru.md` — all cross-links INTO these files are being fixed from the other side; no internal changes needed.

---

## Governing Rules (from `docs/research/AGENTS.md`)

- All text Russian; number format `1 000`; decimal `,`; currency руб.; em dash `—`
- Inline doc/source mentions: `_«[Title](./file.md#anchor)»_`
- References sections: plain `- [Title](url)` (no guillemets, no italic)
- No heading links; heading anchors on same line as heading
- No new files created
- No commits unless explicitly requested
- Never invent figures — all numbers must trace to existing validated content in the pack
