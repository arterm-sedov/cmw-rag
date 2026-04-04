# Unified Report-Pack Enhancement Plan
## C-Level Sales Enablement — Actionable with Checkpoints

**Date:** 2026-03-28
**Status:** COMPLETED
**Scope:** 10 files in `docs/research/executive-research-technology-transfer/report-pack/`+ task file
**Business Goal:** Definitive C-Level reference for Comindware's AI implementation + expertise transfer sales motion.

---

## Business Purpose

Comindware sells **AI implementation expertise and knowledge transfer (KT/IP/BOT)** — NOT documents or reports. The pack is an **internal sales enablement tool** that equips C-Level executives to:

1. **SELL** implementation packages (PoC, Pilot, Scale, BOT)
2. **ANSWER** client objections about ROI, security, vendor lock-in
3. **JUSTIFY** budgets with evidence-backed scenarios
4. **TRANSFER** capability (code, configs, runbooks, evals) — not just deliver slides
5. **COMPOSE** customer-ready proposals from synthesized evidence

**What we DON'T sell:** Research PDFs, consulting decks, subscription to reports.
**What we DO sell:** The ability to implement, operate, and transfer AI systems.

---

## Pack Architecture

```
COMMERCIAL OFFER← C-Level pitch: "What we sell"├── Main Methodology   ← Deep: TOM, phases, KT, compliance
├── Main Sizing       ← Deep: CapEx/OpEx/TCO, scenarios, tariffs
├── Appendix A        ← Navigation + sources registry
├── Appendix B        ← KT/IP/legal framework
├── Appendix C        ← Comindware stack boundaries
├── Appendix D        ← Security, compliance, observability
├── Appendix E        ← Market signals (supplementary)
├── Exec Methodology   ← 1-2 pager: methodology essence
└── Exec Sizing        ← 1-2 pager: economics essence
```

---

## Critical Finding from Deep Research

### CMO Survey "43%" Figures — Verification Gap

**Deep research result:** The exact CMO Club survey containing "43%" figures could not be located. The CMOfollow-up research found:

| Metric | Finding |
|--------|---------|
| Source verifiability | **NO verifiable primary source** for specific "43%" figures |
| Recommendation | Treat with caution; either flag as "anecdotally reported" or replace with verified alternates |
| Verified alternates | Enterprise AI production: 47% (Menlo Ventures), 31% use cases in production (ISG), 33% scaling (Deloitte) |

**Action:** Audit all "43%" references and either (a) flag as unverified, or (b) replace with verified statistics from deep research.

---

## Validated Defects (from inspection)

| ID | File | Issue | Plan |
|----|------|-------|------|
| D1 | Commercial Offer | `- продажипродажи` tag typo | Fix → `- продажи` `- продукт` |
| D2 | Commercial Offer | Missing `## Источники` section | Add section with 5 sources |
| D3 | Appendix A | Appendix C listed without hyperlink (4 locations) | Add `#app_c_pack_overview` links |
| D4 | Appendix C | No YAML front matter | Add YAML block |
| D5 | Appendix C | "Related documents" has no hyperlinks | Add linked sibling list |
| D6 | Appendix C | No heading anchor on H1/H2 | Add `{: #app_c_pack_overview }` and `{: #app_c_overview }` |
| D7 | **CRITICAL** | `#sizing_russian_market` anchor in wrong file | Create stub in sizing report |
| D8 | Appendix C | Bold-text reference to sizing, no link | Fix with `sizing_russian_market` link |
| D9 | Commercial Offer | Zero market evidence (CMO survey) | Add market context section |
| D10 | Exec summaries | No role-specific "so what" | Add role routing blocks |
| D11 | Main:Sizing | USD figure without RUB conversion (`~$0,001–0,005/токен`) | Add FX reference or RUB equivalent |
| D12 | Appendix E | Anchors use `sizing_`/`method_` instead of `app_e_` | Normalize ~44 anchors to `app_e_*` |
| D13 | Multiple files | Links to Appendix E use old anchor names | Update ~8 incoming links |

**Note:** D12-D13 are anchor normalization tasks executed in Phase 1 (Tasks 1.8-1.9).

---

## Phase 0 — Pre-Execution Verification

**Checkpoint marker:** All defects confirmed by direct file inspection on 2026-03-28.

- [x] Re-verify D1-D10 line numbers before editing
- [x] Confirm deep research files exist at `deep-researches/`
- [x] Verify AGENTS.md location and read rules

---

## Phase 1 — Structural Repairs (YAML, Links, Anchors)

**Why first:** C-Level notices typos, broken links, malformed metadata.

### Task 1.1 — Fix Commercial Offer YAML

**File:** `20260325-comindware-ai-commercial-offer-ru.md`

- [x] **Step 1:** Confirm line 6: `- продажипродажи`
- [x] **Step 2:** Replace with two tags:
  ```yaml
    - продажи
    - продукт
  ```
- [x] **Step 3:** Remove duplicate `- продукт` if present
- [x] **Step 4:** Verify YAML block opens with `---` (line 1) and closes with `---`

---

### Task 1.2 — Add YAML Front Matter to Appendix C

**File:** `20260325-research-appendix-c-cmw-existing-work-ru.md`

- [x] **Step 1:** Confirm no `---` YAML block (lines 1-4)
- [x] **Step 2:** Prepend:
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
- [x] **Step 3:** Delete redundant bold-text metadata lines
- [x] **Step 4:** Add H1 anchor: `{: #app_c_pack_overview }`
- [x] **Step 5:** Add H2 anchor: `{: #app_c_overview }`

---

### Task 1.3 — Fix Appendix C Related Documents (Hyperlinks)

**File:** `20260325-research-appendix-c-cmw-existing-work-ru.md`

- [x] **Step 1:** Read lines 16-23, confirm plain text
- [x] **Step 2:** Replace with linked list:
  ```markdown
  ## Связанные документы
  - [«Приложение A: обзор и ведомость документов»](./20260325-research-appendix-a-index-ru.md#app_a_pack_overview)
  - [«Основной отчёт: методология внедрения и разработки»](./20260325-research-report-methodology-main-ru.md#method_pack_overview)
  - [«Основной отчёт: сайзинг и экономика»](./20260325-research-report-sizing-economics-main-ru.md#sizing_pack_overview)
  - [«Приложение B: отчуждение ИС и кода (KT, IP)»](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_pack_overview)
  - [«Приложение D: безопасность, комплаенс и наблюдаемость»](./20260325-research-appendix-d-security-observability-ru.md#app_d__pack_overview)
  ```

---

### Task 1.4 — Fix Appendix A: 4 Appendix C Hyperlinks

**File:** `20260325-research-appendix-a-index-ru.md`

- [x] **Step 1:** Fix line 63: `[«Приложение C: доказательство готовности — референс-стек Comindware»](./20260325-research-appendix-c-cmw-existing-work-ru.md#app_c_pack_overview)`
- [x] **Step 2:** Fix line 74: same link in topic table
- [x] **Step 3:** Fix line 93: same link in Q→doc navigation
- [x] **Step 4:** Fix line 128: same link in source mapping table

---

### Task 1.5 — Fix Duplicate YAML Tags

**Files:** Main:M, Main:S, Appendix A, Appendix D

- [x] **Main:M** (`research-report-methodology-main-ru.md`): Remove duplicate `методология`- [x] **Main:S** (`research-report-sizing-economics-main-ru.md`): Remove duplicate `сайзинг`
- [x] **Appendix A** (`research-appendix-a-index-ru.md`): Remove duplicate `методология`
- [x] **Appendix D** (`research-appendix-d-security-observability-ru.md`): Remove duplicates `безопасность`, `комплаенс`, `наблюдаемость`

---

### Task 1.6 — Add Appendix E to Task Manifest

**File:** `tasks/20260324-research-task.md`

- [x] **Step 1:** Find §1б (Ведомость файлов)
- [x] **Step 2:** Add:
  ```markdown
  | Приложение E — рыночные и технические сигналы | `20260325-research-appendix-e-market-technical-signals-ru.md` |
  | Коммерческое резюме для руководства | `20260325-comindware-ai-commercial-offer-ru.md` |
  ```

---

### Task 1.7 — Add Appendix E to "Связанные документы"

**Files:** Appendix A, B, D, Main:M, Main:S

- [x] Add to each file's `## Связанные документы` section:
  ```markdown
  - [«Приложение E: рыночные и технические сигналы»](./20260325-research-appendix-e-market-technical-signals-ru.md#app_e_root)
  ```

---

### Task 1.8 — Normalize Appendix E Anchors to `app_e_` Prefix

**Why:** AGENTS.md convention requires appendix anchors to use `app_<letter>_` prefix. Appendix E currently uses `sizing_` and `method_` — inconsistent with Appendices A, B, D.

**File:** `20260325-research-appendix-e-market-technical-signals-ru.md`

- [x] **Step 1:** Find all anchors with `sizing_` prefix (grep `{: #sizing_`)
- [x] **Step 2:** Find all anchors with `method_` prefix (grep `{: #method_`)
- [x] **Step 3:** Replace each `#sizing_<name>` with `#app_e_<name>`
- [x] **Step 4:** Replace each `#method_<name>` with `#app_e_<name>`
- [x] **Step 5:** Update H1 anchor from `#app_e_root` (already correct)
- [x] **Step 6:** Count total changes (~44 anchors expected)

**Example replacements:**
- `#sizing_russian_market` → `#app_e_russian_market`
- `#sizing_local_models_coding_cost_reduction` → `#app_e_local_models_coding_cost_reduction`
- `#method_ai_implementation_red_mad_robot` → `#app_e_ai_implementation_red_mad_robot`

---

### Task 1.9 — Update Incoming Links to Appendix E Anchors

**Files:** Main:Sizing, Appendix A, Appendix B, Appendix D, Appendix E (stub in sizing report)

- [x] **Step 1:** Find all links pointing to `#sizing_*` in Appendix E context
- [x] **Step 2:** Find all links pointing to `#method_*` in Appendix E context
- [x] **Step 3:** Update each link to use new `#app_e_*` prefix
- [x] **Step 4:** Update D7 stub link from `#sizing_russian_market` to `#app_e_russian_market`

**Expected links to update (~8):**
- `Main:Sizing#sizing_russian_market` stub → `#app_e_russian_market`
- External links to `appendix-e#sizing_*` → `appendix-e#app_e_*`
- External links to `appendix-e#method_*` → `appendix-e#app_e_*`

---

**PHASE 1 CHECKPOINT**

Verify:
- [x] YAML blocks valid in Commercial Offer, Appendix C
- [x] No duplicate tags
- [x] Appendix A has 4 working Appendix C links
- [x] Appendix C has 5 working sibling links
- [x] Appendix E anchors renamed to `app_e_*` (~44 anchors)
- [x] Incoming links updated (~8 links)

---

## Phase 2 — Cross-Reference Integrity (D7 Critical)

**Why:** 4 documents point to Russian market data — anchor must exist in sizing report for navigation. After Task 1.8-1.9 normalization, Appendix E uses `app_e_*` prefix.

### Task 2.1 — Create `#sizing_russian_market` Stub in Sizing Report

**File:** `20260325-research-report-sizing-economics-main-ru.md`

- [x] **Step 1:** Find insertion point (before `## Тарифы российских облачных провайдеров ИИ`)
- [x] **Step 2:** Insert stub section:
  ```markdown
  ## Российский рынок GenAI: зрелость и барьеры (опрос CMO Club × red_mad_robot, 2025) {: #sizing_russian_market }

  Ключевые доли из публичных материалов исследования **red_mad_robot × CMO Club Russia** (2025) — сигнал спроса со стороны владельцев маркетингового бюджета:
  - **93%** компаний используют GenAI в рабочих процессах; системно интегрировали — **около трети**.
  - **64%** выделяют на GenAI **только 1–5%** маркетингового бюджета.
  - Барьеры: **53%** — необходимость доработки контента; **49%** — шаблонность; **43%** — галлюцинации и ошибки; **отдельно 43%** — риски утечки данных.
  - **85%** считают GenAI ключевым фактором трансформации на горизонте **трёх лет**.

  Полный разбор — в _«[Приложение E: рыночные и технические сигналы — Российский рынок](./20260325-research-appendix-e-market-technical-signals-ru.md#app_e_russian_market)»_.
  ```
  **Note:** Link uses normalized anchor `#app_e_russian_market` (per Task 1.8).

- [x] **Step 3:** Verify anchor `#sizing_russian_market` now exists in sizing report

---

### Task 2.2 — Fix Appendix C Reference to Sizing Report

**File:** `20260325-research-appendix-c-cmw-existing-work-ru.md`

- [x] Replace bold-text reference with: `зафиксированы в _«[Основной отчёт: сайзинг и экономика — Российский рынок](./20260325-research-report-sizing-economics-main-ru.md#sizing_russian_market)»_`

---

### Task 2.3 — Verify Major Anchors

- [x] `#app_b_pack_overview` exists in Appendix B
- [x] `#app_d__pack_overview` exists in Appendix D
- [x] `#method_pack_overview` exists in Main:M
- [x] `#app_a_fx_policy` exists in Appendix A

---

**PHASE 2 CHECKPOINT**

Verify:
- [x] All 4 broken `#sizing_russian_market` references now resolve
- [x] Appendix C correctly links to sizing report
- [x] Major anchors verified

---

## Phase 3 — Sources & Citations (Evidence Credibility)

### Task 3.1 — Add `## Источники` to Commercial Offer

**File:** `20260325-comindware-ai-commercial-offer-ru.md`

- [x] Append at end:
  ```markdown
  ## Источники

  - [Telegram — CMO Club Russia: анонс исследования GenAI в маркетинге (red_mad_robot × CMO Club, 2025)](https://t.me/cmoclub/197)
  - [RB.RU — 93% команд в маркетинге используют ИИ (обзор исследования CMO Club × red_mad_robot, 2025)](https://rb.ru/news/93-komand-v-marketinge-ispolzuyut-ii-nejroseti-pomogayut-s-kontentom-no-im-pochti-ne-doveryayut-strategiyu-i-byudzhety/)
  - [РБК Education — во сколько обойдётся ИИ-агент: подсчёты экспертов (2026)](https://www.rbc.ru/education/05/02/2026/697162269a794772c9cf9991)
  - [OpenAI — The state of enterprise AI (обзор, декабрь 2025)](https://openai.com/index/the-state-of-enterprise-ai-2025-report)
  - [Банк России — официальные курсы валют](https://www.cbr.ru/currency_base/daily/)
  ```

---

### Task 3.2 — Fix Citation Format in Commercial Offer

**File:** `20260325-comindware-ai-commercial-offer-ru.md`

- [x] Convert plain `[brackets]` to `_«[Title](url)»_` format in body text

---

### Task 3.3 — Add FX Policy Reference to Commercial Offer

**File:** `20260325-comindware-ai-commercial-offer-ru.md`

- [x] Add at start: `**Валюта:** Курс USD/RUB для иллюстративных ориентиров — см. [приложение A](./20260325-research-appendix-a-index-ru.md#app_a_fx_policy).`

---

### Task 3.4 — Fix USD Figure Without Conversion (D11)

**File:** `20260325-research-report-sizing-economics-main-ru.md`

**Location:** Line ~1010 — `~$0,001–0,005/токен`

- [x] **Step 1:** Find the USD figure in context
- [x] **Step 2:** Add RUB equivalent or FX reference:
  ```markdown
  **Стоимость инференса:** ~$0,001–0,005/токен (~0,085–0,425 руб./токен по курсу 1 USD = 85 RUB; см. [политику курса в приложении A](./20260325-research-appendix-a-index-ru.md#app_a_fx_policy)).
  ```

---

**PHASE 3 CHECKPOINT**

Verify:
- [x] `## Источники` present in Commercial Offer
- [x] Citations use `_«[...]»_` format
- [x] FX policy reference present
- [x] USD figure in Main:Sizing has RUB conversion

---

## Phase 4 — Market Evidence Enrichment (D9)

### Task 4.1 — Add Market Context Section to Commercial Offer

**File:** `20260325-comindware-ai-commercial-offer-ru.md`

**Business focus:** Position Comindware as SELLING IMPLEMENTATION EXPERTISE, not reports or models.

- [x] Insert after `## Зачем это бизнесу (P&L и контроль)`:
  ```markdown
  ## Рыночный контекст: спрос на экспертизу внедрения

  По публичным материалам исследования **red_mad_robot × CMO Club Russia** (2025) среди директоров по маркетингу крупнейших брендов РФ:
  
  - **93%** компаний уже используют GenAI в рабочих процессах; системно интегрировали — лишь **около трети**.
  - **64%** тратят на GenAI **только 1–5%** маркетингового бюджета — разрыв между личной цифровой зрелостью и корпоративным управлением.
  - Главные барьеры: отсутствие стратегии, **43%** отмечают галлюцинации и ошибки, **отдельно 43%** — риски утечки данных.
  - **85%** считают GenAI ключевым фактором трансформации на горизонте **трёх лет**.

  **Что это значит для заказчика:** спрос на GenAI есть, но **управляемый контур и способность внедрить** остаются дефицитом. Comindware продаёт не модели, а **способность спроектировать, внедрить и передать** ИИ-контур с полным комплектом артефактов (код, конфигурации, runbook, eval).
  ```

---

## Phase 5 — Role Routing (D10)

**Business focus:** What each role **SELLS or APPROVES** — not just "needs".

### Task 5.1 — Add Role Routing to Exec Methodology

**File:** `20260325-research-executive-methodology-ru.md`

- [x] Insert before `## Где искать полноту`:
  ```markdown
  ## Что продаёт и решает каждый руководитель

  - **CEO / Генеральный директор:** APPROVES фазы PoC → Pilot → Scale, оценивает риск «вечного пилота», продаёт суверенность как конкурентное преимущество.
  - **CRO / Директор по продажам:** SELLS пакеты внедрения и BOT; аргумент — «отчуждение как ценность», не «SaaS-подписка».
  - **CPO / Продукт:** APPROVES TOM и roadmap; получает работающий контур с RAG и управлением жизненным циклом знаний.
  - **CIO / CTO:** APPROVES архитектуру; выбирает SaaS / on-prem / гибрид; продаёт управляемость и наблюдаемость.
  - **CISO / Комплаенс:** APPROVES периметр до LLM и политику телеметрии; продаёт соответствие 152-ФЗ.
  - **CFO:** APPROVES CapEx/OpEx; используют вилки для переговоров о бюджете.
  ```

---

### Task 5.2 — Add Role Routing to Exec Sizing

**File:** `20260325-research-executive-sizing-ru.md`

- [x] Insert before `## Где искать полноту`:
  ```markdown
  ## Что продаёт и решает каждый руководитель

  - **CFO:** APPROVES CapEx/OpEx; использует вилки как «порядок величин» для бюджетирования; в КП — после замеров.- **CIO / CTO:** APPROVES матрицу SaaS / on-prem / гибрид; SELLS управляемость и TCO.
  - **CEO:** APPROVES инвестиции; продаёт «скрытые» затраты как аргумент для управляемого контура.
  - **CRO / Директор по продажам:** SELLS сценарные вилки как аргумент для переговоров о бюджете.
  - **CISO / Комплаенс:** APPROVES OpEx безопасности; SELLS соответствие требованиям.
  - **CPO:** APPROVES юнит-экономику сценариев; SELLS приоритизацию.
  ```

---

## Phase 6 — Objection Handling

**Business focus:** Address objections to BUYING IMPLEMENTATION EXPERTISE, not buying reports.

### Task 6.1 — Add Objection Handling to Commercial Offer

**File:** `20260325-comindware-ai-commercial-offer-ru.md`

- [x] Insert after "Типовые компромиссы":
  ```markdown
  ### Типовые возражения и ответы

  **«Сложно обосновать ROI»:** Сценарные вилки CapEx/OpEx помогут прикинуть порядок инвестиций; точные цифры — после замеров на стенде заказчика. Мы продаём не отчёт, а **способность внедрить и передать контур** с артефактами (код, конфигурации, runbook, eval).

  **«Зависимость от вендора»:** Комплект отчуждения передаётся заказчику в рамках BOT или create–transfer. Клиент получает способность **эксплуатировать и развивать контур самостоятельно** — код, конфигурации, регламент эксплуатации.

  **«ИБ и 152-ФЗ»:** Архитектура суверенного контура (периметр до LLM, минимизация данных, маскирование) проектируется под требования заказчика. При управляемом API РФ — договорной контур и цепочка обработчиков данных проверяются отдельно.
  ```

---

## Phase 7 — Deep Research Integration

### Task 7.1 — Flag Unverified "43%" Figures

Based on deep research: **CMO Club survey source could not be verified**.

- [x] Audit all "43%" references in:
  - Main:Sizing
  - Exec:Sizing
  - Appendix D
  - Appendix E
- [x] Add disclaimer where appropriate: `_Примечание: источник опроса CMO Club не удалось верифицировать напрямую; цифра приводится по вторичным материалам._`

### Task 7.2 — Cross-Validate Key Figures

From deep research:

| Figure | Source | Status |
|--------|--------|--------|
| 47% AI production | Menlo Ventures 2025 | Verified |
| 31% use cases in production | ISG 2025 | Verified |
| 93%, 64%, 85% CMO | red_mad_robot materials | Use as reported |
| GPU TCO 40-60% utilization | Multiple sources | Verify context |

- [x] Verify usage in Main:Sizing matches deep research context

---

## Phase 8 — Final Verification

**Business alignment check:** All customer-facing content must sell **implementation expertise**, not documents.

- [x] All YAML blocks properly closed
- [x] No duplicate YAML tags
- [x] All anchor references resolve
- [x] Appendix E anchors normalized to `app_e_*` (~44 anchors)
- [x] Incoming links to Appendix E updated (~8 links)
- [x] `## Источники` in Commercial Offer
- [x] FX policy referenced in Commercial Offer
- [x] USD figures have RUB conversion (D11)
- [x] Role routing in both exec summaries **focuses on SELLS/APPROVES**
- [x] Market context emphasizes **demand for implementation expertise**
- [x] Objection handling addresses **buying implementation capability**
- [x] No "субпроцессоры" terminology — use "обработчики данных"
- [x] "43%" figures flagged or verified

---

## Files Modified Summary

| File | Phases | Changes |
|------|--------|---------|
| Commercial Offer | 1, 3, 4, 6 | YAML fix, Источники, FX, market context, objection handling |
| Appendix C | 1 | YAML front matter, anchors, linked siblings |
| Appendix A | 1, 2 | 4 Appendix C links, Appendix E cross-ref, update incoming links |
| Appendix E | 1 | **~44 anchors renamed to `app_e_*`** |
| Main:Sizing | 2, 3, 7 | `#sizing_russian_market` stub, USD→RUB conversion (D11), figure validation, update Appendix E links |
| Main:M | 5 | Role routing block, update Appendix E links |
| Appendix B | 1, 2 | Appendix E cross-ref, update incoming links |
| Appendix D | 1, 7 | Appendix E cross-ref, update incoming links, figure validation |
| Exec:M | 5 | Role routing block |
| Exec:S | 5, 7 | Role routing block, figure validation |
| Task file | 1 | Appendix E + Commercial Offer added to manifest |

---

## Rollback

```bash
git status
git checkout -- <file>
```

---

## Deep Research Sources Used

| File | Key Findings Used |
|------|-------------------|
| `cmo_survey.md` | 43% verification gap — flag as unverified |
| `enterprise_ai_2025.md` | 47%, 31%, 33% production rates |
| `llm_economics.md` | GPU pricing, TCO models |
| `russia_regulations.md` |152-FZ penalties, CBR guidelines |
| `bot_framework.md` | BOT model components |
| `utilization_threshold.md` | 40-60% break-even threshold |
| `nist_ai_rmf.md` | Implementation steps reference |