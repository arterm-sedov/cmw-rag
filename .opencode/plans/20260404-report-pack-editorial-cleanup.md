# План: Editorial Cleanup & PDF-Ready Polish Report Pack

**Дата:** 2026-04-04
**Цель:** Привести комплект отчётов `report-pack/` к уровню типографики, вёрстки и структуры, готовому для компиляции в многостраничный PDF через репозиторий `cbap-mkdocs-ru`.
**Принципы:** Business-focused, executive, coherent, grounded, deduplicated. No editorial bloat. C-level best practices. Nothing lost — useful content perfectly structured, useless removed.

---

## Scope

9 файлов в `docs/research/executive-research-technology-transfer/report-pack/`:

| # | Файл | Роль |
|---|------|------|
| 1 | `20260331-research-executive-unified-ru.md` | Единое резюме для руководства (C-level entry point) |
| 2 | `20260325-research-appendix-a-index-ru.md` | Навигация, реестр, политики KPI/FX |
| 3 | `20260325-research-report-methodology-main-ru.md` | Основной отчёт: методология внедрения |
| 4 | `20260325-research-report-sizing-economics-main-ru.md` | Основной отчёт: сайзинг и экономика |
| 5 | `20260325-research-appendix-b-ip-code-alienation-ru.md` | Приложение B: отчуждение ИС/кода |
| 6 | `20260325-research-appendix-c-cmw-existing-work-ru.md` | Приложение C: наработки Comindware |
| 7 | `20260325-research-appendix-d-security-observability-ru.md` | Приложение D: безопасность, комплаенс, наблюдаемость |
| 8 | `20260325-research-appendix-e-market-technical-signals-ru.md` | Приложение E: рыночные сигналы |
| 9 | `20260325-research-appendix-f-extended-reading-ru.md` | Приложение F: расширенное чтение |

---

## Phase 1: Typography & Russian Standards (All 9 files)

### 1.1 Normalize dashes, arrows, and ranges

**Rule:** Each character has a single semantic role in Russian executive text.

| Pattern | Replace with | Context | Example |
|---------|-------------|---------|---------|
| `->` in prose | `→` | Stage transitions | `PoC → Пилот → Масштабирование` |
| `-->` in Mermaid | **DO NOT TOUCH** | Mermaid flowchart operators | `userInput --> preLLM` |
| `-` between numbers (hyphen) | `–` | Ranges: years, percentages, numbers | `2019–2025`, `30–40%`, `40–50%` |
| Standalone `-` in prose | `—` | Explanations, appositions | `три оси — резидентность, ...` |
| `--` | `—` | Em dash (if any remain) | — |

**Known instances to fix:**
- **Unified:** lines 23, 28 — `PoC -> Пилот -> Масштабирование -> BOT` (×2)
- **Methodology:** lines 214, 231, 238 — `PoC -> пилот -> масштабирование` (×3 prose); line 769 — `2025-2026` → `2025–2026`
- **Sizing:** line 559 — `Простые вопросы -> SLM, сложные -> LLM` (×1)
- **Appendix A:** line 35 — `вопрос -> документ` (×1)
- **Appendix B:** lines 140, 152, 205 — `2025-2026` → `2025–2026` (×3)
- **Appendix E:** line 333 — `2025—2026` (em dash — wrong!) → `2025–2026` (en dash); lines 441, 602 — `2025-2026` → `2025–2026` (×2)
- **Appendix F:** lines 79, 87 — `2025-2026` in link text → `2025–2026` (×2; fix visible text only, not URL path)
- **Ranges to normalize:** `40-50%` → `40–50%`, `45-60%` → `45–60%`, `1-3` → `1–3`, `2-4` → `2–4`, `3-12` → `3–12`, `3-5` → `3–5`
- **Already correct:** `январь–март 2026` (Sizing:959) — en dash, leave as-is

**Checkpoint 1.1:** Zero `->` in prose. Zero hyphen-minus in numeric ranges. All appositions use em dash. Mermaid diagrams untouched.

### 1.2 Non-breaking spaces (NBSP)

**Rule:** Russian typography requires NBSP (`&nbsp;`) in specific contexts to prevent widow/orphan breaks.

| Context | Pattern | Example |
|---------|---------|---------|
| Before `%` | NO space | `>60%`, `99,9%`, `3,6%` |
| Before `руб.` | `1000&nbsp;руб.` | `1000&nbsp;руб.` |
| Before `ФЗ` | `152‑ФЗ` (U+2011 narrow NB hyphen) | `152-ФЗ` |
| Thousands separator | `&nbsp;` | `10&nbsp;000`, `1&nbsp;000&nbsp;000`, `50&nbsp;000` — BUT `1000`, `5000`, `9000` (no separator for 1000–9999) |
| Before abbreviations | `т.&nbsp;д.`, `и&nbsp;т.&nbsp;д.` | `т. д.` |
| Initials | `А.&nbsp;С.&nbsp;Пушкин` | `А. С. Пушкин` |
| `г.` before year | `в&nbsp;2026&nbsp;г.` | `в 2026 г.` |
| `см.` before reference | `см.&nbsp;` | `см. Приложение` |

**Known issues:**
- `%` spacing inconsistent: `99.9%` (no space), `3,6%` (no space), `>60 %` (space) — **standardize all to NO space before `%`**: `99,9%`, `3,6%`, `>60%`
- `152-ФЗ` uses regular hyphen — replace with U+2011 narrow no-break hyphen

**Checkpoint 1.2:** No space before `%`. All `руб.` preceded by `&nbsp;`. All `ФЗ` uses narrow NB hyphen. All 5+ digit numbers have `&nbsp;` thousands separators.

### 1.3 Quotation marks and bold-in-quotes

**Rule:** Russian guillemets `«»` for all quotations. Bold goes **inside** guillemets. English/Latin acronyms do not need quotations. Russian proper names use guillemets. English proper names in Russian text, like in English, usually do not need quotes.

| Pattern | Rule | Example |
|---------|------|---------|
| `**«Термин»**` | Bold inside quotes | `«**Термин**»` |
| `"простой текст в кавычках, не код"` | Russian quotes | `«простой текст в кавычках, не код»` |
| `API`, `LLM`, `RAG`, `CapEx` | No quotes needed | `API`, `LLM`, `RAG` |
| `English Proper Name` | No quotes needed | `GigaChat`, `Yandex Cloud` |
| `Русское имя собственное` | Russian quotes | `«Сбер»`, `«Яндекс»` |

**Checkpoint 1.3:** No `**«` pattern. No straight quotes in Russian prose (except code literals). English/Latin acronyms and English proper names are unquoted. Russian proper names use `«»`.

### 1.4 Decimal separators and number formatting

**Rule:** Comma for decimals in Russian text. NBSP for thousands. No comma in 4-digit numbers unless decimal.

| Wrong | Correct | Location |
|-------|---------|----------|
| `99.9%` | `99,9%` | Methodology:282 |
| `84.5%` | `84,5%` | Methodology:633 |
| `70.7%` | `70,7%` | Appendix E:651 |
| `11.2%` | `11,2%` | Appendix D:335 |
| `2.4%` | `2,4%` | Appendix D:336 |
| `1,000,000` | `1&nbsp;000&nbsp;000` | (if any found) |
| `3,6%` | `3,6%` (already correct — no space) | Sizing:132, Appendix E:460 |

**Checkpoint 1.4:** Zero `.` as decimal separator in Russian prose. All decimal percentages use comma, no space before `%`. All 5+ digit numbers have `&nbsp;` thousands separators.

### 1.5 Currency consistency

**Rule:** All financial values in `руб.` with NBSP. No bare `$` in Russian text.

**Known exceptions (deliberate — AWS reference table):**
- Sizing lines 950–953: AWS pricing table with `$0,526`, `$1,006`, `$3,060`, `$32,773` — **KEEP as-is** (this is a reference table showing USD pricing with ruble equivalents already calculated in adjacent columns). The table header says `Стоимость/час ($)` and has `Эквивалент ₽/час*` — this is intentional comparison data.
- Appendix E line 249: ElevenLabs pricing `$0,06-0,12` and `$0,22 / час` — **KEEP as-is** (global benchmark with explicit note to convert per FX policy).
- Appendix E line 449: `$37 млрд` Menlo Ventures — **KEEP as-is** (global market stat with attribution).

**Checkpoint 1.5:** All standalone currency values use `руб.` format. Deliberate USD reference tables preserved with attribution. FX policy referenced consistently.

**Note:** When fixing typography issues, scan for and fix similar problems in the surrounding context without rewriting entire documents. Apply the same pattern consistently (e.g., if you fix one `->` arrow, check nearby lines for the same pattern).

---

## Phase 2: PDF-Readiness & Structural Anchors

### 2.1 Add missing H1 anchors

**Issue:** Sizing report H1 lacks anchor.

**Fix:** Add `{: #sizing }` to line 18:
```
# Отчёт. Сайзинг и экономика (CapEx / OpEx / TCO) {: #sizing }
```

**Duplicate anchor fix:** `sizing_russian_ai_cloud_tariffs` is used for BOTH H2 (`## Тарифы и провайдеры РФ`, line 249) AND H3 (`### Российские модели (март 2026)`, line 253). Rename H3 anchor to `sizing_russian_models_march_2026`.

**Checkpoint 2.1:** Every H1 has a `{: #root_anchor }` attribute. No duplicate anchors within any file.

### 2.1b H2-H6 anchor prefix consistency

**Rule:** All H2-H6 anchors should use the `#root_anchor_` prefix convention (e.g., `#method_phases`, `#sizing_cloud_tariffs`, `#app_a_fx_policy`).

**Important:** Do NOT refactor existing anchors if they are already correct, consistent, and follow the naming convention. Only fix:
- Missing anchors on headings that should have them
- Duplicate anchors (same ID used twice)
- Plain wrong anchors (non-semantic, legacy patterns like `research_pkg_*` or overly long `research_methodology_20260325_*`)

**Checkpoint 2.1b:** All anchors use `#root_anchor_` prefix where applicable. Existing correct anchors left untouched. Only missing, duplicate, or wrong anchors are fixed.

### 2.2 Page break markers

**Rule:** Use `{: .pageBreakBefore }` on headings that should start a new PDF page. Matches cbap-mkdocs-ru convention (`pdf_templates/styles.scss`).

**Apply to:**
- All H1 headings (9 files — each document starts on a new page)
- Major H2 section breaks within long reports (Methodology, Sizing):
  - `## Описание документа` — no break (follows H1)
  - `## Источник преимущества в корпоративном ИИ` — break
  - `## Стратегия внедрения ИИ и организационная зрелость` — break
  - `## Целевая операционная модель` — break
  - `## Методология внедрения` — break
  - `## Детальная архитектура внедрения` — break
  - `## Рыночный контекст` — break (Sizing)
  - `## Тарифы и провайдеры РФ` — break (Sizing)
  - `## Экономический каркас` — break (Sizing)
  - `## Модель затрат` — break (Sizing)
  - `## Детальные затраты и TCO` — break (Sizing)
  - `## Рекомендации по сайзингу` — break (Sizing)
  - `## Заключение и обоснование` — break (Sizing)
- Each appendix H1 (already starts new doc)

**Files:** All 9
**Checkpoint 2.2:** All major section H2s have `{: .pageBreakBefore }` where appropriate. No orphan headings at page bottom.

### 2.3 YAML front matter enrichment

**Add to all files:**

```yaml
---
title: '...'
date: 'YYYY-MM-DD'
status: '...'
tags:
  - ...
hide:
  - tags
---
```

**Changes:**
- Add `hide: - tags` to prevent tag display in web output (matching reference repo pattern — 274+ files use this)
- Ensure `date` is quoted string for YAML consistency
- Ensure `title` matches H1 text exactly (without anchor suffix)
- **No `description` field** — not needed, `with-pdf` doesn't use it

**Titles to set (match H1 exactly, without anchor suffix):**
1. **Unified:** «Принципы внедрения и передачи технологий корпоративного ИИ»
2. **Appendix A:** «Ведомость документов, реестр источников, политики KPI и курса валют»
3. **Methodology:** «Методология разработки и внедрения ИИ»
4. **Sizing:** «Отчёт. Сайзинг и экономика (CapEx / OpEx / TCO)»
5. **Appendix B:** «Отчуждение ИС и кода: KT, IP, лицензии, критерии приёмки передачи»
6. **Appendix C:** «Корпоративный ИИ Comindware: состав стека, границы, артефакты»
7. **Appendix D:** «Безопасность, комплаенс, наблюдаемость»
8. **Appendix E:** «Рыночные и технические сигналы»
9. **Appendix F:** «Дополнительное чтение»

**Checkpoint 2.3:** All 9 files have complete front matter with `title`, `date`, `status`, `tags`, `hide: tags`. Titles match H1 exactly.

### 2.4 Table width hints for PDF

**Rule:** Key tables should have `{: style="width:100%;" }` for proper PDF rendering (matching cbap-mkdocs-ru pattern).

**Apply to:** All tables with 4+ columns, especially:
- Role matrices (C-level decision tables) — Unified, Methodology, Sizing
- Tariff/pricing tables — Sizing (Russian, Chinese, Global models)
- Comparison tables — Sizing (on-prem vs cloud, TCO)
- OWASP tables — Appendix D (LLM Top 10, Agentic Top 10)
- Decision matrix — Sizing (РБК 2026)
- Regulatory timeline — Methodology
- Hardware pricing — Sizing
- Speech/TTS pricing — Appendix E
- Market stats — Appendix E

**Checkpoint 2.4:** All wide tables have width hints. Tables render without overflow in PDF.

### 2.5 Heading hierarchy discipline

**Rule:** No skipped heading levels (H1 → H2 → H3, not H1 → H3). H4–H6 are acceptable in body text where structurally justified — the reference repo (`cbap-mkdocs-ru`) renders deep heading hierarchies correctly. The `toc_level: 3` setting in `with-pdf` only affects TOC depth, not body rendering.

**Action:** Audit for skipped levels only (e.g., H2 → H4 without H3). Fix any skips by inserting the missing intermediate level or promoting the heading.

**Checkpoint 2.5:** No heading level skips. H4–H6 remain where structurally justified.

### 2.6 Citation and Admonition Formatting

**Rule 1 (Citations):**
- Inline mentions in body text must use quoted italics: `_«[Название документа](link)»_`
- Grouped lists under `## Источники` must use plain links: `- [Название](link)` without guillemets or italics.

**Rule 2 (Admonitions):**
- Convert bolded callouts to MkDocs admonitions **strategically** — use where they add visual hierarchy and executive clarity, not to bloat the PDF:
  - `!!! tip "Рекомендация"` — for actionable recommendations that deserve visual emphasis
  - `!!! warning "Важно"` — for risks, limitations, or compliance requirements
  - `!!! note "Примечание"` — for supplementary context that doesn't fit the main flow
  - `!!! important` — for critical business decisions or KPIs
  - `!!! danger` — for security/compliance blockers
- **Do NOT convert** every bold label — use admonitions where they genuinely improve readability and executive scanning. Simple inline bold labels are fine for minor notes.

```markdown
!!! tip "Рекомендация"

    Текст с отступом 4 пробела.
```

**Rule 3 (Positional Navigation):**
- Audit and remove relative locators like «см. ниже» or «таблица выше». Replace them with explicit named anchor links: `_«[Заголовок](#anchor)»_`. Semantic use (e.g., «риск ниже») is acceptable.

**Rule 4 (Spacing):**

Blank lines are required in the following contexts:

- **After headings (H1–H6):** Always add a blank line after any heading before starting paragraph content or lists.
- **After bold pseudo-headings:** Always add a blank line after `**Ситуация:**`, `**Проблема:**`, etc. before content.
- **Before bullet lists:** Add a blank line before a bulleted list, EXCEPT when the list is a continuation of a uniform bulleted list (same semantic level, no intervening text).
- **Before bullet lists inside numbered lists:** Add a blank line before a bulleted list nested inside a numbered list item.
- **Before numbered lists:** Add a blank line before any numbered list.
- **Between paragraphs:** Add a blank line between every paragraph — paragraphs must be separated by exactly one empty line.

**Checkpoint 2.6:** All inline citations use `_«»_`. `## Источники` uses plain links. Bold callouts converted to `!!!` strategically (not all — only where they improve executive scanning). No positional navigation. Spacing rules enforced: blank lines after headings, after pseudo-headings, before lists (with exceptions), and between all paragraphs.

### 2.7 Cross-Reference Normalization

**Goal:** Unify all ~130+ cross-references across 9 files into a consistent, canonical pattern. Currently there are **18+ different formatting variants** for the same type of reference.

**Rule: Appendix name first for whole-appendix refs; appendix name last for paragraph refs. Main reports referenced by H1 title alone (no "Отчёт").**

#### 2.7.1 Canonical Patterns

| Scope | Pattern | Example |
|-------|---------|---------|
| Whole appendix (H1) | `см. _Приложение X «[H1 title](link#anchor)»_` | `см. _Приложение D «[Безопасность, комплаенс и наблюдаемость](...)»_` |
| Single paragraph (H2/H3) | `см. _«[H2/H3 title](link#anchor)»_ в Приложении X` | `см. _«[Граница доверия](...)»_ в Приложении D` |
| Multiple paragraphs | `см. _«[A](link)»_, _«[Б](link)»_ и _«[В](link)»_ в Приложении X` | `см. _«[Персональные данные](...)»_, _«[Периметр до LLM](...)»_ в Приложении D` |
| Main report (inline, nominative) | `_«[H1 title](link#anchor)»_` | `_«[Методология разработки и внедрения ИИ](...)»_` |
| Main report (inline, conjugated) | `_[Conjugated H1 title](link#anchor)_` | `см. _[Методологию разработки и внедрения ИИ](...)_`, `согласно _[Методологии...](...)_` |
| Main report (plain list) | `- [H1 title](link#anchor)` | `- [Методология разработки и внедрения ИИ](...)` |

**Disambiguation logic (built into word order):**
- `Приложение X` **prefix** → whole appendix (H1)
- `в Приложении X` **suffix** → specific paragraph(s) within appendix (H2/H3)
- **No appendix mention, title only** → main report (H1 title is unique enough)

#### 2.7.2 Not Acceptable — Fix on Sight

| Bad Pattern | Why | Fix |
|-------------|-----|-----|
| `[текст в Приложении A](link)` | Appendix name buried in link text | `см. _«[текст](link)»_ в Приложении A` |
| `[Приложение D](link)` without anchor or title | No anchor, no title | Add H1 anchor + title |
| `см. ниже`, `выше в документе` | Positional navigation | Replace with explicit anchor link |
| `того же приложения` | Vague reference | Name the appendix explicitly |
| `Отчёт «Методология...»` | Redundant — H1 title is self-identifying | `_«[Методология...](link)»_` |
| `_Приложение X, параграф «...»_` | Outdated — use suffix pattern | `см. _«[Title](link)»_ в Приложении X` |

#### 2.7.3 Execution Method

1. Extract all `./20260325-research-appendix-[a-f]*` and `./20260325-research-report-*` links from all 9 files
2. Classify each reference: whole appendix vs specific paragraph vs main report
3. Normalize to canonical pattern based on scope
4. Verify all target anchors exist (broken anchor → fix or remove)
5. Verify grammatical case matches sentence context (nominative, prepositional, accusative, etc.)

**Known instances to normalize (sample):**
- **Sizing:** ~20 cross-refs — mix of "Приложении D, параграф", "Приложение E", bare links
- **Methodology:** ~25 cross-refs — mix of "Приложении D", "Приложение B", "см. ниже"
- **Appendix A:** ~5 cross-refs — navigation block + inline refs
- **Appendix C:** ~8 cross-refs — "Приложении B", "Приложении D"
- **Appendix D:** ~10 cross-refs — "Приложении B", "Приложении E", "того же приложения"
- **Appendix E:** ~8 cross-refs — "Приложении D", "Приложении B"
- **Appendix B:** ~5 cross-refs — "Приложении D", "Приложении A"

**Checkpoint 2.7:** All cross-references use canonical patterns. Zero positional navigation. Zero vague references. All anchors resolve. Grammatical case matches context.

---

## Phase 3: Content Deduplication & Coherence

### 3.1 Deduplicate overlapping content

**Areas of overlap to audit:**

| Topic | Appears in | Action |
|-------|-----------|--------|
| SCQA | Unified, Methodology, Sizing | Keep full in each; ensure consistent numbers |
| Role matrix (C-level) | Unified, Methodology, Sizing | Unified = condensed; Methodology = canonical; Sizing = financial focus only |
| FX policy | Unified, Appendix A, Methodology, Sizing | Canonical in Appendix A; stubs + links elsewhere |
| KPI thresholds (>60%, >95%, 30-40%) | Unified, Methodology, Appendix A | Single canonical definition in Appendix A; consistent values everywhere |
| Regulatory timeline | Methodology, Appendix D | Methodology = brief; Appendix D = detailed |
| Market signals | Sizing, Appendix E | Sizing = economic context; Appendix E = full radar |
| Comindware stack description | Methodology, Appendix C, Sizing | Appendix C = canonical; others = brief reference + link |
| OWASP tables | Appendix D only | Ensure not duplicated elsewhere |

**Rule:** When the same concept appears in multiple documents, use **identical wording and values**. Contradictions undermine executive trust (per AGENTS.md). Cross-check **dates**, **terminology**, and **unit conventions** across all 9 files. Resolve or flag contradictions.

**Checkpoint 3.1:** No contradictory values across files. No unnecessary duplication. Canonical sources clearly linked. Dates and terminology align.

### 3.2 Executive Structural Formats

**Rule 1 (Technology & Innovation Profiles):**
- Preference across the report pack: listed tools, frameworks, and architectural patterns should use a uniform executive triplet (eg., `Проблема` / `Решение` / `Результат` or an appropriate uniform pattern) instead of unstructured or "flat" bullet points.

**Rule 2 (Validation):**
- Run `python rag_engine/scripts/validate_token_calculations.py` (if it exists, or manually verify) to programmatically verify all token-to-pricing conversions and unit economics in the Sizing report before finalizing.

**Checkpoint 3.2:** All tools/frameworks use triplet. Pricing calculations verified.

### 3.3 Remove editorial bloat

**Guiding principle:** Make sure nothing is lost — only streamlined and consolidated. Our goal is NOT to rewrite the reports from draft, but to perfect them. Tighter if needed, but we do NOT want a complete rewrite. We're making the reports perfect, streamlined, but NOT a new doc completely.

**Remove:**
- Any meta-text about "how this document was written"
- Redundant "Связанные документы" sections that list ALL siblings — move to end of article (after `## Источники`) with only 3-5 most relevant cross-links. This improves readability and keeps the main flow focused.
- Repeated full lists of sources when a canonical registry exists (Appendix A)
- Overly verbose warnings that repeat the same limitation 3+ times

**Preserve:**
- All facts, figures, citations
- All business recommendations
- All architectural descriptions
- All compliance/legal content
- All economic data and ranges

**Checkpoint 3.3:** Each file is tighter where needed. No content loss. No complete rewrites. Word count reduced only by removing repetition, not substance.

### 3.4 Executive register review (Russian prose)

**Guiding principle:** Tighter if needed, we do not want a complete rewrite. We're making the reports perfect, streamlined, but not a new doc completely.

**Audit for:**
- **Calques from English** («в терминах» → «с точки зрения»)
- **Hedging filler** («возможно», «в целом», «достаточно»)
- **Pleonasms** («свободная вакансия» → «вакансия»)
- **Weak sentence starters** («Это является…», «Данный подход…»)
- **Passive voice** where active works better — prefer active voice
- **Sentences with >1 subordinate clause** → split
- **Case government** (управление, e.g., `согласно + дательный падеж`)
- **Tautologies**
- **Unnatural English-style SVO word order** (theme-rheme)
- **Imperative Voice:** Strongly prefer recommendations to use direct imperative verbs (e.g., «утвердите», «используйте»), rather than passive/soft forms («следует использовать», «рекомендуется» unless it's really a soft recommendation).
- **Pyramid Principle:** Enforce the "one idea per paragraph" rule and ensure paragraphs are front-loaded with the conclusion/answer first.

**Files:** All 9, prioritized: Unified > Methodology > Sizing > Appendices

**Checkpoint 3.4:** Prose reads as native executive editor. No AI-generated anti-patterns.

---

## Phase 4: YAML Config for PDF Build

### 4.1 MkDocs config structure

**Location:** One file in `cmw-rag` repo: `docs/research/executive-research-technology-transfer/report-pack/mkdocs_executive_report_pdf.yml`

This config inherits from `cbap-mkdocs-ru` via relative path, so all PDF cooking stays in `cmw-rag` while themes/templates come from the sibling repo:

```yaml
INHERIT: ../../../../cbap-mkdocs-ru/mkdocs_ru.yml

site_name: "Comindware. Коммерческое обоснование внедрения корпоративного ИИ"
site_description: "Комплект отчётов по методологии, экономике и комплаенсу внедрения корпоративного GenAI в резидентном контуре РФ"
docs_dir: .
site_dir: ../../../../pdf/executive-report/
site_url: ''

extra:
  productName: "Коммерческое обоснование внедрения корпоративного ИИ"
  productVersion: "2026-03"
  publicationDate: "Апрель 2026"
  pdfOutput: true

nav:
  - Резюме для руководства: 20260331-research-executive-unified-ru.md
  - Обзор и ведомость документов: 20260325-research-appendix-a-index-ru.md
  - Методология разработки и внедрения ИИ: 20260325-research-report-methodology-main-ru.md
  - Сайзинг и экономика (CapEx / OpEx / TCO): 20260325-research-report-sizing-economics-main-ru.md
  - Приложение B. Отчуждение ИС и кода: 20260325-research-appendix-b-ip-code-alienation-ru.md
  - Приложение C. Наработки Comindware: 20260325-research-appendix-c-cmw-existing-work-ru.md
  - Приложение D. Безопасность, комплаенс, наблюдаемость: 20260325-research-appendix-d-security-observability-ru.md
  - Приложение E. Рыночные сигналы: 20260325-research-appendix-e-market-technical-signals-ru.md
  - Приложение F. Дополнительное чтение: 20260325-research-appendix-f-extended-reading-ru.md

markdown_extensions:
  toc:
    permalink_title: Скопируйте адрес этой ссылки, чтобы поделиться параграфом

plugins:
  minify:
    minify_html: false
  glightbox:
    manual: true
  mermaid-to-svg:
    # Converts Mermaid diagrams to SVG during build so WeasyPrint can render them in PDF
    # WeasyPrint (with-pdf engine) does NOT execute JavaScript, so Mermaid won't render natively
    # Tested: produces crisp SVG diagrams in PDF (infinite scaling, no pixelation)
    # Requires: npm install -g @mermaid-js/mermaid-cli (mmdc)
    output_dir: _mermaid_assets
  with-pdf:
    cover_subtitle: "Методология, экономика и комплаенс внедрения корпоративного GenAI"
    output_path: ../../../Comindware. Коммерческое обоснование внедрения ИИ.pdf
    enabled_if_env: ''
```

**Note on Mermaid in PDF:**
- `cbap-mkdocs-ru` uses `with-pdf` (orzih), which relies on WeasyPrint — **no JavaScript execution**
- `render_js: true` flag in `with-pdf` **does NOT work** — bug in v0.9.3 (`AttributeError: property 'text' of 'Tag' object has no setter`), incompatible with modern BeautifulSoup
- **Solution:** `mkdocs-mermaid-to-svg` plugin converts Mermaid → SVG at build time via `mmdc` (Node.js Mermaid CLI), which WeasyPrint includes natively
- Install: `pip install mkdocs-mermaid-to-svg` + `npm install -g @mermaid-js/mermaid-cli`
- **Tested:** PDF with Mermaid diagrams rendered successfully (2.7 MB, SVG 14.9 KB, build time 36s)
- See full test results: `.opencode/plans/20260404-mermaid-pdf-test.md`

**Key design decisions:**
- Single config file lives in `cmw-rag` alongside the report pack — no cross-repo config management
- Inherits from `cbap-mkdocs-ru/mkdocs_ru.yml` via relative path to get: Russian key mappings, `toc_title: Оглавление`, `with-pdf` logo/copyright/author defaults, PDF templates
- `docs_dir: .` — the config sits in the same directory as the docs
- No `pymdownx.snippets` — report pack doesn't use snippets, so the config doesn't override `base_path`
- Follows the PDF-specific override pattern from existing `mkdocs_guide_complete_ru_pdf.yml` (minify disabled, glightbox manual, `enabled_if_env: ''`)
- Uses `mermaid-to-svg` (not `mermaid-to-image`) — SVG output tested and working, infinite scaling in PDF

### 4.2 PDF build script

**Location:** `cmw-rag` repo, same directory as the config: `docs/research/executive-research-technology-transfer/report-pack/build_executive_report_pdf.ps1`

```powershell
# Build executive report PDF
# Requires: cbap-mkdocs-ru venv activated, GTK3 installed
# Usage: .\build_executive_report_pdf.ps1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$configPath = Join-Path $scriptDir "mkdocs_executive_report_pdf.yml"

Write-Host "Building executive report PDF..." -ForegroundColor Cyan
py -m mkdocs build -f $configPath
Write-Host "PDF output: Comindware. Коммерческое обоснование внедрения ИИ.pdf" -ForegroundColor Green
```

### 4.3 Test PDF output

**Verify:**
- Cover page renders correctly (logo from `mkdocs_ru.yml`, subtitle, productName, productVersion, publicationDate, copyright)
- TOC shows all 9 documents with correct hierarchy (3 levels deep)
- Page breaks occur at intended locations
- Tables don't overflow
- Cross-references are clickable
- Russian typography renders correctly (guillemets, dashes, NBSP via `&nbsp;`)
- Total page count is reasonable (target: 40-80 pages)
- Deep heading hierarchy (H4-H6) renders correctly in body text

**Checkpoint 4.3:** PDF builds successfully. Output is visually polished and readable.

---

## Phase 5: Mermaid Rendering Test — COMPLETED

**Results documented in:** `.opencode/plans/20260404-mermaid-pdf-test.md`

**Verdict:** Use `mkdocs-mermaid-to-svg` + `mmdc` — tested, works perfectly.
- `render_js: true` in `with-pdf` does NOT work (bug in v0.9.3)
- `mkdocs-mermaid-to-image` not available on PyPI for Python 3.13
- SVG output: infinite scaling, crisp at any resolution, 14.9 KB per diagram
- Build time: 36 seconds for full methodology document
- Dependencies already added to `cbap-mkdocs-ru/install/requirements.txt`
- Documentation updated in `cbap-mkdocs-ru/readme.md`

**Config already updated in Phase 4.1** to use `mermaid-to-svg` plugin.

---

## Execution Order & Checkpoints

```
Phase 1 (Typography) → Phase 2 (PDF-Readiness) → Phase 3 (Content) → Phase 4 (YAML Config) → Phase 5 (Mermaid Test)
      ↓                      ↓                        ↓                    ↓                        ↓
    CP 1.1-1.5            CP 2.1-2.7              CP 3.1-3.4           CP 4.1-4.3              CP 5.1-5.4
```

### Checkpoint Gates

| Gate | Verification | Pass Criteria |
|------|-------------|---------------|
| **CP 1** | Typography scan | Zero `->` in prose, zero hyphen-minus in ranges, no space before `%`, all decimals use comma, `&nbsp;` for thousands |
| **CP 2** | Anchor & front matter audit | All H1s have anchors, no duplicate anchors, all files have `title`, `date`, `status`, `tags`, `hide: tags`, page breaks set, no heading level skips, citations/admonitions formatted, all cross-references normalized to canonical patterns |
| **CP 3** | Dedup + prose audit | Zero contradictory values, no editorial bloat, prose reads as native executive editor |
| **CP 4** | PDF build test | PDF builds, renders correctly, clickable links, proper page breaks, cover page complete |

---

## Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| `&nbsp;` may not render in some editors | MkDocs/WeasyPrint handle `&nbsp;` natively; visible in raw markdown for easy editing |
| Cross-reference breakage during edits | Verify all anchors after each file edit; run final audit |
| YAML `INHERIT` relative path breakage | Config lives in `report-pack/`; relative path to `cbap-mkdocs-ru/mkdocs_ru.yml` is `../../../../cbap-mkdocs-ru/mkdocs_ru.yml` — verify at build time |
| Content loss during deduplication | Git diff before/after; verify all facts preserved |
| PDF page count too high | Tighten content in Phase 3; adjust page break markers |
| Mermaid diagrams broken by typography changes | Explicit exclusion rule in 1.1 — never touch `-->` inside ```mermaid blocks |
| `mmdc` (Node.js) not installed | Prerequisite for `mermaid-to-svg` — document in README, add to build script check |
| `render_js: true` attempted by mistake | Documented as broken in plan — use `mermaid-to-svg` only |

---

## Definition of Done

- [ ] All 9 files pass typography audit (CP 1)
- [ ] All 9 files are PDF-ready with anchors, front matter, page breaks (CP 2)
- [ ] Content is deduplicated, coherent, no contradictions (CP 3)
- [ ] YAML config exists in `cmw-rag/report-pack/` and PDF builds successfully (CP 4)
- [ ] No content loss — all facts, figures, citations preserved
- [ ] Russian prose reads as native executive editor quality
- [ ] Cross-references all resolve
- [ ] AGENTS.md rules satisfied per Definition of Done section
