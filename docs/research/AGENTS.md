# Research Agents Guidelines

## Scope

This file defines research workflow, document standards, and formatting requirements for all files in `docs/research/`.

## Instruction Priority

- Apply this file as the default rulebook for research documents in `docs/research/`.
- If a specific task file introduces stricter requirements, follow the stricter rule.
- Do not relax or omit rules from this file unless explicitly requested.

## Agent role: four-persona consensus (internal reasoning)

**Internally in English**, weigh substantive choices through four lenses—**technical writer** (clarity, structure, citations), **AI systems engineer** (models, RAG/agents, eval, safety), **DevOps** (run, deploy, observability, cost), **business analyst** (scope, value, risk, task fit)—**debate, align, then act** on consensus. **Search the web** when facts are thin or contested; **cite authoritative sources** (see **Current data requirement (Web search mandatory)**). Personas stay **internal**; **deliverables** follow **Target Language and Audience**.

## Agent modes: plans and business alignment

### Any mode

- **Business goals:** derive scope, priorities, and acceptance from [`executive-research-technology-transfer/tasks/`](./executive-research-technology-transfer/tasks/) — authoritative `*-research-task.md`; `*_original.md` only as non-conflicting history (see **Business goals and task authority**).

### Plan mode

- **Drafting:** ask when scope, goals, or evidence bar are unclear; put resolved assumptions and scope into the plan.
- **Plan shape:** ordered steps; each step → one primary **artifact** + which task-file objective it serves.
- **Checkpoints:** after named steps, self-check evidence, task fit, and consistency — continue or realign. For **agent self-control**, not routine user stops (escalate per **Permission Boundaries** or real blockers).
- **Execution:** run the plan **autonomously**; involve the user only where this file, the task, or a blocker requires it.
- **Progress:** while executing, add **dated** notes under [`docs/progress_reports/`](../progress_reports/) (`YYYYMMDD-topic.md` or one log per thread): done, checkpoint results, blockers, remaining work — brief ops trail, not the final deliverable.

## Business goals and task authority

**Read this before any edits** to the **consolidated research pack** under `docs/research/executive-research-technology-transfer/report-pack/` (full methodology and sizing reports, akappendices, and the executive summaries). The corpus serves **Comindware’s commercial work**: evidence and narrative to **win and deliver** customer AI programs—**budget, architecture, compliance, handover (KT/IP), and roadmap**—not a standalone SKU for “selling research” or “curating publications.”

**When you edit, optimize for:** decision-ready prose, **RF-resident defaults** (152-FZ, local clouds, data residency), and **honest limits** on global/vendor telemetry (sample scope; not an automatic baseline for RF production). Readers **reuse** claims in **their own** board packs and proposals; avoid meta-text about “how we wrote this for executives.”

### Reader-facing pack: business, technology, and expertise transfer

The consolidated **20260325-** research pack is **not** internal authoring documentation. Write for **commercial and technical decision-makers** who will **sell and scope** AI delivery: budget, architecture, compliance, KT/IP, and risk—grounded in **Comindware’s documented practice** (open repos where relevant) and **attributed** public sources (pricing, benchmarks, regulation).

- **Tone:** **business + technology**; concise; no patronizing “how to read this file,” no references to source-file mechanics (YAML, front matter, template filenames), and no pack-maintenance asides (split history, “canonical row” editorial rules).
- **Cross-cutting rules** (e.g. **FX**): **one canonical block** — [_Валюта и правила для коммерческих предложений_](./20260331-research-executive-unified-ru.md#exec_unified_fx_policy) (`#exec_unified_fx_policy`); all other documents use a **single stub line + link** pointing to `#exec_unified_fx_policy` — no separate FX section or anchor in sizing, methodology, or appendices.
- **Hardware facts:** do **not** describe **RTX 4090 48 GB** as a “Comindware custom” GPU. It is a **commercially offered** dedicated-GPU server configuration (e.g. providers such as [1dedic — GPU servers](https://1dedic.ru/gpu-servers)); Comindware or Clients may buy or rent such SKUs like any customer-facing integrator.

- [Research pack task (authoritative)](./20260324-research-task.md) — **authoritative** scope, acceptance, workflow, evidence and FX: **Russian** body (§§1–8). It points agents here for English workflow rules and non-duplication.
- [Original task snapshot](./20260324-research-task_original.md) — historical manager brief; if it conflicts with the main task file, **follow the main task file**.

### Agent execution contract (`report-pack/`)

Treat `tasks/` as **authoring instructions** and `report-pack/` as **reader-facing executive content**.

- **Primary purpose:** Enable Comindware C-level leaders to make decisions and shape commercial AI offers.
- **Audience outcome:** Executives can reuse the text directly in board briefs and proposals without edits.
- **Hard prohibition:** NO authoring remarks, workflow commentary, or repository-mechanics narrative in report text.
- **Reference boundaries:** Repo mechanics stay in tasks/plans; report text retains only business/technical meaning and evidence.
- **If uncertain:** Prefer decision usefulness over methodological self-description.

## Permission Boundaries

### Always Allowed

- Read and analyze files in `docs/research/`.
- Improve structure, clarity, and evidence quality without changing intended meaning.
- Add citations, references, and formatting fixes required by this guide.

### Ask First

- Remove large content blocks that may be important for historical traceability.
- Change scope, objectives, or acceptance criteria in task files.
- Introduce new mandatory sections that affect downstream workflows.

### Never

- Invent facts, metrics, legal claims, or references.
- Keep uncited high-impact claims in final executive summaries.
- Place raw scraping dumps in repository summaries.

## Repository Layout

```text
docs/research/
├── AGENTS.md              # This file - research workflow guidelines
├── YYYYMMDD-topic.md      # Russian executive summary (primary)
└── YYYYMMDD-topic-en.md   # **Optional** English executive summary
```

### Naming Convention

```text
YYYYMMDD-topic-[lang].md
```

Example: `20260323-ai-implementation-methodology-ru.md`

## Data Storage and Raw Materials

### Source Material Location

- Raw extracted materials must be stored in: `~/Documents/cmw-rag-channel-extractions/`.

### Purpose of External Storage

- Store raw scraped data from external sources (channels, websites, APIs).
- Keep large regenerable artifacts outside the repository.
- Keep this repository focused on curated, processed content.

### Ignored Local Artifacts

```text
.playwright-cli/
channel_snapshot.yml
```

## Executive Summary Content Standards

### Language and Audience

- Use a single language per file (preferably **Russian** for this project’s customer-facing and leadership deliverables).
- Keep content executive-oriented: concise, decision-ready, and evidence-based.
- Structure final summaries as one-pagers (1–2 pages of concentrated meaning).

### Logic and Storyline

- Use SCQA logic:
  - **Ситуация:** Market context and current state.
  - **Вызов:** Key challenge or opportunity.
  - **Задача:** Core business question.
  - **Решение:** Concrete recommendation and implications.
- Ensure recommendations are MECE.
- Focus on "So what?" (business meaning and impact).
- **SCQA acronym expansion:** define it **once** in the pack glossary (`intro-ru.md`, entry `| SCQA (Situation–Complication–Question–Answer) | ситуация → проблема → вопрос → ответ … |`).

### Required Analytical Content

- Key facts with citations.
- Pricing, benchmarks, and metrics (ROI, unit economics, etc.).
- Architecture comparisons.
- Market analysis.
- Actionable recommendations (for example, a 30/60/90 day plan).

### Content to Exclude

- Raw scraping outputs, intermediate artifacts, and unverified marketing fluff.
- Duplicate language versions unless explicitly requested.

### Authoring inputs vs published executive text

- Internal repositories, slide decks, and local paths are **authoring inputs only**. Published executive summaries (`20260323-*-ru.md`) state **generic, transferable practice** and must not require readers to open the codebase.

## Russian-Language Documents: Numbers, Currency, Typography

These rules apply to Russian research articles in `docs/research/`.

### Numbers and Values

- Use a non-breaking space as the thousands separator (e.g., `1 000 000`, `10 000`, but `5000`).
- Use a comma as the decimal separator (e.g., `2,5%`).
- **No space before `%`**: `99,9%`, `>60%`, `3,6%` (not `99,9 %`).
- **NBSP (`&nbsp;`)** before: `руб.` (`1000 руб.`), `ФЗ` (`152‑ФЗ` with U+2011 narrow NB hyphen), abbreviations (`т. д.`, `и т. д.`), initials (`А. С. Пушкин`), `г.` before year (`в 2026 г.`), `см.` before reference (`см. Приложение`).

### Currency

- Express financial values in **Russian rubles** (`руб.`).
- Convert USD using the pack’s fixed rate (typically `1 USD = 85 RUB`) unless the task specifies otherwise.
- Example: `$1,200` → `102 000 руб.`

### Typography and punctuation (Russian)

- **Quotation marks for proper names:** English/Latin acronyms (`API`, `LLM`, `CapEx`) and English proper names (`GigaChat`, `Yandex Cloud`) do NOT need quotes. Well-known Russian company names used informally (`Сбер`, `Яндекс`, `Мегафон`) also do NOT need quotes. Legal entity forms require guillemets: `АО «Яндекс»`, `ПАО «Сбер»`.
- **Bold inside quotes:** when formatting a quoted term in bold, place the asterisks **inside** the guillemets: `«**Термин**»`, not `**«Термин»**`.
- **Em dash (`—`):** appositions, breaks in sense, dialogue-style breaks; do not use hyphen-minus `-` where a long dash is intended.
- **En dash (`–`):** ranges (pages, years), e.g. `2019–2025`, no extra spaces inside the range.
- **Arrow (`→`):** stage transitions and process flows, e.g. `PoC → Пилот → Масштабирование`. Do NOT use `->` in prose. Mermaid `-->` inside code blocks is exempt.
- **Colon:** if text continues on the same line after `:`, start with lowercase (not English sentence case), except proper names, acronyms, or a clearly new sentence.
- **Hyphen:** compounds and hyphenated words; keep minus and code literals as needed in technical snippets.

### Terminology

- Prefer established Russian business and technical terms.
- Use English acronyms or terms only when strictly necessary or industry-standard (for example, `RAG`, `LLM`, `CapEx`).
- At first use, provide translation or short gloss in parentheses.
- Example: `RAG (Retrieval-Augmented Generation — генерация с дополненной выборкой)`.
- **Engineering acronyms** such as `CI`, `CD`, `TCO` may appear without expansion when the audience and context are clear.

### Russian prose quality and executive register

Every Russian deliverable must read as if written by a native-speaking executive editor. Grammar, morphology, case government (управление), and word order must be flawless. The standard is **деловой стиль высшего уровня** — the register used in board-level memos and strategy documents, not academic papers or bureaucratic correspondence.

**AI-generated anti-patterns to eliminate:**

- **Calques from English** (literal structural borrowings): do not transplant English syntax into Russian. Examples: avoid «в терминах» (calque of "in terms of") — use «с точки зрения» or rephrase; avoid «имеет место быть» — use «существует» or a direct verb.
- **Unnatural word order:** Russian word order is flexible and meaning-driven; do not force English SVO (subject-verb-object) when theme-rheme or emphasis demands a different order.
- **Wrong prepositional and case government:** verify that each verb, noun, and preposition governs the correct case (e.g. «согласно + дат.», «благодаря + дат.», «вопреки + дат.», not genitive).
- **Unidiomatic collocations:** use established Russian collocations (e.g. «принять решение», not «сделать решение»; «провести аудит», not «выполнить аудит»).

**Executive register rules:**

- **Active, direct constructions.** Minimize passive voice and reflexive passives (`-ся`). Prefer «компания внедрила» over «внедрение было осуществлено компанией».
- **No канцелярит.** Eliminate chains of отглагольные существительные: «осуществление реализации внедрения» → «внедрить». Prefer verbs to heavy nominalizations: «внедрить» over «произвести внедрение», «оценить» over «провести оценку».
- **Short, front-loaded sentences.** State the conclusion or action before the explanation (pyramid principle applied to sentence structure). Split any sentence with more than one subordinate clause.
- **No hedging filler.** Cut gratuitous «возможно», «в целом», «определённым образом», «как бы», «достаточно», «в некотором роде» unless they carry genuine epistemic meaning.
- **Imperatives for recommendations.** Use direct imperative: «используйте», «зафиксируйте», «утвердите» — not «следует использовать» or «рекомендуется зафиксировать».

**Lexical precision:**

- Eliminate **pleonasms**: «свободная вакансия» → «вакансия», «главный приоритет» → «приоритет», «полностью завершить» → «завершить».
- Eliminate **tautologies**: do not repeat the same meaning in adjacent words or clauses.
- Cut **semantically empty words**: «данный» when «этот» suffices, «является» when a dash or a concrete verb works, «определённый» without specific reference.
- Use **concrete verbs and nouns**, not abstract bureaucratic chains. Replace «произвести расчёт стоимости» with «рассчитать стоимость».

**Sentence-level discipline:**

- Maximum one subordinate clause per sentence in executive text; split complex constructions into separate sentences.
- Every sentence must pass the "so what?" test — if it adds no decision value, cut it.
- Avoid starting sentences with weak demonstratives («Это является…», «Данный подход…») — lead with the subject or action.

**Reference benchmark:** Розенталь Д.Э. ("Справочник по правописанию и литературной правке") and Мильчин А.Э. / Чельцова Л.К. ("Справочник издателя и автора") as linguistic authority; ГОСТ Р 7.0.97-2016 for formal business document structure where applicable.

### Executive communication discipline

- **Pyramid principle:** state the answer or recommendation first, then the supporting evidence. Never bury the conclusion at the end of a paragraph or section.
- **One idea per paragraph.** Each paragraph opens with its topic sentence; supporting detail follows. If a paragraph covers two ideas, split it.
- **Action-oriented language.** Every recommendation or next-step section uses imperative voice addressed to the reader: «утвердите план», «зафиксируйте KPI», «проведите аудит».
- **Quantify or qualify.** No vague claims. Attach a number, a range, or an explicit qualifier: «по данным X», «в диапазоне Y–Z», «по оценке Yakov Partners (2025)». If a number is approximate, say so explicitly: «порядка», «~», or «ориентировочно».
- **No self-referential text.** Do not describe the document, its structure, or how it was written. Let content speak. Avoid meta openers («в данном разделе…», «этот документ предназначен для…», «ниже приводится обзор…»). **Positional navigation** («ниже/выше», «см. ниже», «перечень/таблица ниже по документу», «Ниже — …» meaning scroll order only): replace with a **named anchor**—`_«[Заголовок](#anchor)»_` in-file or `./sibling.md#anchor` (see **Consolidated packs**). **OK:** semantic «ниже/выше» («риск ниже», «порог выше»).
- **Coherence across the pack.** When the same concept (KPI threshold, cost range, compliance requirement) appears in multiple documents, use identical wording and values. Contradictions undermine executive trust.

## Markdown and document formatting

### Prose and line breaks

- No hard wraps mid-sentence in prose or list body text; one sentence per source line, soft-wrap in the editor.
- Break lines only between sentences, list items, or blocks (headings, lists, fences).

### Heading rules

- Do not use numbered headings in final reports.
- Do not place hyperlinks inside headings.
- Avoid English words in **Russian** headings unless strictly necessary.
- Convert standalone label-style lines (for example, `**Законопроект об ИИ:**`) into proper Markdown headings instead of bold paragraphs ending with a colon.

### Admonitions (callout blocks)

Use MkDocs admonitions (`!!! type "Title"`) instead of bold text for recommendations, warnings, and notes:

| Instead of | Use |
|:---|:---|
| `**Рекомендация:** текст` | `!!! tip "Рекомендация"` |
| `**Важно:** текст` | `!!! warning "Важно"` |
| `**Примечание:** текст` | `!!! note "Заголовок"` |

**Format:**
```markdown
!!! tip "Рекомендация"

    Текст с отступом 4 пробела.
```

**Types:** `note`, `tip`, `warning`, `important`, `danger` — choose based on urgency and action required.

### List rules

- Add an empty line before every bulleted or numbered list.
- Use `-` as the standard bullet marker.
- Do not use `*` as the primary bullet marker.

### Spacing Between Headings and Content

- Always add a blank line after headings (H1-H6) before starting paragraph content or bullet lists
- Always add a blank line after bold pseudo-headings (like **Situation:**) before starting paragraph content or bullet lists
- This improves readability and ensures consistent formatting

### Citations and references

- **Traceability modes** (pick what fits the document; sources remain mandatory):
  - **Inline** — prefer when you name a specific publication or report **by title**, cite **law or regulator** text, or introduce **pricing or quantitative norms** as a baseline (per task file: first occurrence of a figure). External web sources may use `_«[Source title](https://...)»_` in body text when the sentence hinges on that source.
  - **Grouped** — for **multi-item signal digests** (e.g. Appendix E-style market/technical bullets), body text may stay scannable without a link on every line; collect **plain** URLs in the **centralized appendix** (`20260325-research-appendix-a-index-ru.md`), optionally under a **thematic subheading** (e.g. «Инфраструктура и модели»), so the cluster remains verifiable. Do not use grouped mode to hide uncited high-impact numbers in **short** C-level summaries (those still need explicit support per **Permission Boundaries**).
- **Inline mentions** (body text and ordinary lists; not inside the references section): when you name an internal document or external source by title, use a **quoted italic** Markdown link—guillemets `«»` around an italic link built with underscores `_..._`:
  - **Internal** `.md` in this repo (typically under `docs/research/`): `_«[Document title](relative-path/file-name.md)»_`. Path **relative to the current file** (for example `./20260323-topic-ru.md` or `../other/file.md`).
  - **External** web sources: `_«[Source title](https://...)»_` with the real URL as the link target.
- **Examples (inline, Russian prose):**
  - See _«[Методология внедрения ИИ](./20260323-ai-implementation-methodology-ru.md)»_ for context.
  - See also _«[Research Agents Guidelines](./AGENTS.md)»_.
- **No per-document `## Источники` sections.** Cite sources **inline** where they matter (see **Traceability modes** above), and collect them in the **centralized appendix** (`20260325-research-appendix-a-index-ru.md`). This avoids duplication and keeps executive documents lean.
- Under the centralized appendix sources list (or any bullet list that is links only), use **plain** Markdown links—**no** guillemets, **no** italic underscores around the link:
  - `- [Source title](https://...)`
  - Do not format reference-list entries as `_«[...](...)»_`.

### Consolidated packs (optional)

Use for multi-file research sets under `docs/research/` when you want stable deep links and consistent metadata.

**Heading anchors/IDs** — On the same line as the heading, Kramdown-style attribute list: `## Heading text {: #anchor_slug }`. Pick one stable English **prefix** per file as the H1 `#root_anchor` from the **subject** of the **document**; H2+ use `#root_anchor_concise_english_snake_case` from the **meaning** of the heading (not Cyrillic transliteration). Characters: lowercase letters, digits, underscores; anchors **unique within the file** (duplicate titles: `_2`, `_3`, …). No links inside heading text (see **Heading rules**).

**YAML front matter** — When the pack uses it, at the very top: `title` (same as H1 without the `{: #… }` suffix), `date` (ISO), `status`, `tags` (about 5–12; English alphabetically, then Russian alphabetically), `` if tags are for filtering/search only. Optional: `description` (one line). If `date` / `status` are in YAML, drop redundant **Дата пакета** / **Статус** lines under H1.

**Cross-links** — `./sibling.md#anchor` from the current file’s directory. Body mentions of titled internals: **Citations and references**. In the centralized appendix (`20260325-research-appendix-a-index-ru.md`): plain `[title](url)` only. No path for a document that is not a real file—title in guillemets only. Long tables/lists: link the **section heading** (or a subheading with its own `#anchor`), not “the list below”—see **Executive communication discipline** (positional navigation).

**Cross-references vs C-level summaries:** All documents in the pack **should** include cross-references (`.md#anchor` links) to sibling documents for navigation and coherence. C-level executive summaries may link to other documents within the same pack using human-readable names with hyperlinks (e.g., `_«[Методология внедрения](./20260325-research-report-methodology-main-ru.md)»_`). However, C-level summaries must **not** contain paths to **external repositories** or internal code paths (e.g., `../cmw-mosec/README.md`, `rag_engine/`)—these are authoring inputs only.

**Split-pack heading anchors (methodology + sizing + appendices):** In `docs/research/`, explicit heading IDs use prefixes `method_`, `sizing_`, `app_a_`, `app_b_`, `app_d__`. Do **not** use legacy patterns `research_pkg_*` or long `research_methodology_20260325_*` in pack body files; if they appear in older `.cursor/plans/`, treat as historical and reconcile with the live document (ledger: [Research pack task](./20260324-research-task.md), §1б).

**Canonical cross-reference patterns** — use consistently across multi-file packs:

| Scope | Pattern |
|-------|---------|
| Whole appendix (H1) | `см. _Приложение X «[H1 title](link)»_` |
| Single paragraph (H2/H3) | `см. _«[H2/H3 title](link)»_ в Приложении X` |
| Multiple paragraphs | `см. _«[A](link)»_, _«[Б](link)»_ и _«[В](link)»_ в Приложении X` |
| Main report (inline) | `_«[H1 title](link)»_` or conjugated: `_[Методологию...](link)_` |
| Main report (plain list) | `- [H1 title](link)` |

Disambiguation by word order: `Приложение X` prefix → whole appendix; `в Приложении X` suffix → paragraph(s); no appendix mention → main report. Not acceptable: `[текст в Приложении A](link)`, `см. ниже`, `того же приложения`, `Отчёт «...»`.

**Optional** — `{: #id .pageBreakBefore }` on a heading where the export toolchain should force a page break.

## Minimal article template (Russian deliverables)

Use this compact structure for consistency. **Headings below are Russian** because final articles target a Russian executive audience.

```text
# <Document title>

## Резюме для руководства

<SCQA in brief>

## Ключевые выводы

- ...

## Рекомендации

- ...

## Риски и ограничения

- ...
```

Note: sources are cited inline in the body text and collected in the centralized appendix (`20260325-research-appendix-a-index-ru.md`). No per-document `## Источники` section is needed.

## Research workflow

- Extract raw data to designated external storage (e.g. `~/Documents/cmw-rag-channel-extractions/`; if needed ask once the user where to store or use temp folder): keep repository focused on curated content.
- Process findings into executive summaries in the research documents directory (e.g. `docs/research`).
- Commit only processed summaries, keeping raw data and regenerable artifacts external.
- When reusable methodology updates are discovered, document in this AGENTS.md file.

### Core Research Principles (Agnostic to Domain)

These principles apply regardless of specific topic, industry, or organization:

**Evidence-Based Decision Support**

- Research exists to enable actionable decisions, not as an academic exercise
- All claims must be traceable to verifiable sources
- Focus on "So what?" - the business meaning and impact of findings

**Traceability and Verifiability**

- Significant external claims must be **traceable**: **inline citations** and/or entries in the centralized appendix (`20260325-research-appendix-a-index-ru.md`) with enough grouping or labels that a reader can map section/table → URLs. Inline is preferred for law, regulator, first-use pricing, and named publications.
- Maintain a complete references section in the centralized appendix with all sources used (including those supporting grouped digest blocks).
- Document validation processes for time-sensitive data (versions, pricing, etc.)

**Progressive Refinement with Checkpoints**

- Research evolves through structured iteration with defined verification points
- Use version control to trace evolution of research plans and findings
- Validate and ground findings before synthesis and enhancement

**Cross-Validation Discipline**

- Verify key findings against multiple independent sources
- Resolve contradictions explicitly or flag them for attention
- Apply consistent conventions (currency, units, terminology) across related research

**Audience-Centric Communication**

- Tailor content to specific decision-maker needs (executives, technical leads, etc.)
- Structure for easy consumption (SCQA: Situation-Complication-Question-Answer)
- Focus on knowledge transfer, not scenario building

**Execution Principles**

| Principle | Application |
| :--- | :--- |
| **Better, not bigger** | Perfect coherence over volume; clarify, don't inflate |
| **Grounded synthesis** | Generate original research rather than copy-pasting |
| **C-Level enablement** | Support decision-making, don't teach executives their job |
| **Worldwide scope** | Explore global research, reports, and surveys |
| **Autonomous refinement** | Iterate until perfect without requiring approval for each step |

### Current data requirement (Web search mandatory)

**Always use Web search for model versions, pricing, and vendor data.**

LLM versions, API pricing, and model capabilities change monthly. Your training data is outdated by months. **Do not rely on memory** for:

- Model version numbers (e.g., GPT-5.4 vs GPT-5.2, Claude 4.6 vs 3.7, GigaChat 3.1 vs 2.x)
- Pricing tiers and token costs
- Release dates and feature availability
- Regional provider catalogs (Yandex, Cloud.ru, SberCloud)

**Procedure:**

1. Before citing any model or price, run Web search with year 2026 in the query: `"Claude 4.6 2026 pricing"`, `"GigaChat 3.1 March 2026"`, `"MiniMax M2.7 latest"`.
2. Verify the top 2–3 sources agree; if conflict, prefer official vendor docs or reputable aggregators (modelpricing.ai, llmoney.ru).
3. Document the search date in the article’s sources or as a note: `_Проверено: март 2026._`
4. If no reliable source found, flag the claim: `_(Требует уточнения: веб-поиск не дал результатов)_`.

**Anti-pattern to avoid:** citing "latest" models from memory without verification, e.g., calling MiniMax-Text-01 or Kimi k1.5 "current" when they are 2025 versions replaced by M2.7 and K2.5.

### Validation tools

**Token and pricing calculations:** `validate_token_calculations.py` — Python script for validating word-to-token conversions and pricing calculations using actual Russian cloud provider tariffs. Use when updating sizing tables or verifying token economics.

## Cross-validation of related research

Before finalizing or materially revising an article in `docs/research/`, **cross-check** other research files that belong to the same thread of work. Treat as related any document that matches on at least one of:

- **Semantic linkage** — shared themes, entities, markets, technologies, or recommendations (including cross-references, overlapping titles, or clearly parallel subject matter).
- **Business line** — same product, offering, customer segment, or value stream the research supports.
- **Current scope or task** — objectives, acceptance criteria, or explicit scope in an active task file (for example `docs/research/*-research-task.md` or a linked plan under `.cursor/plans/`).

**Cross-validation means:** compare conclusions, figures, dates, currency and unit conventions, and terminology across those related files; resolve or explicitly flag contradictions (for example under `## Риски и ограничения` with a short note pointing to the sibling document); avoid duplicate contradictory “single truths” without reconciliation. When this file and a task file disagree on process, follow the stricter requirement; when task scope defines what is in or out of bounds for the engagement, respect that scope when aligning related summaries.

## Definition of done (per article)

- Document follows naming convention and single-language rule.
- Executive structure is clear (SCQA), concise, and decision-oriented.
- Russian numeric and currency standards are applied consistently (for Russian files).
- Terminology rules are applied (Russian-first, translated first use for English terms).
- Markdown formatting rules are followed (headings, lists, links placement).
- All significant claims are traceable via **inline citations** (per **Traceability modes**); sources are also collected in the **centralized appendix** (`20260325-research-appendix-a-index-ru.md`). Inline document/source titles follow **Citations and references** (quoted italic links in body text) when cited by name.
- No per-document `## Источники` sections — sources stay inline and in the centralized appendix only.
- No critical statement remains without a source.
- Related research in `docs/research/` has been cross-validated per **Cross-validation of related research** when semantic overlap, business line, or the active task scope applies.
- If the output is a **consolidated pack** (several linked articles under `docs/research/`), satisfy **Consolidated packs (optional)** in full; ordinary single articles ignore that subsection.
- Russian prose has been self-reviewed for grammar, case government, word order, канцелярит, calques, and executive register per **Russian prose quality and executive register** and **Executive communication discipline**.

## Operating principles

- Keep the repository lean (raw data stays outside).
- Process once, document once.
- Maintain **Russian market** focus and sovereign-default framing per the task file.
- Keep full source traceability via inline citations **where appropriate** and the centralized appendix (`20260325-research-appendix-a-index-ru.md`).
- Reuse abstract patterns and avoid unnecessary duplication.
- Cross-validate sibling research so the corpus stays internally consistent where topics intersect.

## Reusable research patterns

### Plan → Execute → Validate → Iterate

- Plan under `.opencode/plans/`
- Subagents for parallel work → `deep-researches/`
- Iterate plan based on results

### Git diff patches for planning and execution

Use `git diff` patch format when planning edits to existing documents. This is the most token-efficient and deterministic way to communicate, review, and apply changes.

**When to use:**

- Planning structural or prose changes to any file in `report-pack/`
- Reviewing agent-proposed edits before applying them
- Communicating multi-file changes in a single reviewable block

**How to produce a plan as a patch:**

Save prepared patch files to your plans/patches folder.

Make sure the patches are perfectly valid for git application.

```diff
--- a/docs/research/executive-research-technology-transfer/report-pack/FILENAME.md
+++ b/docs/research/executive-research-technology-transfer/report-pack/FILENAME.md
@@ -LINE,COUNT +LINE,COUNT @@
 context line (unchanged)
-old line to remove
+new line to add
 context line (unchanged)
```

**Rules:**

- Always include 3 lines of unchanged context above and below each hunk — this uniquely locates the change and prevents misapplication.
- One hunk per logical change; split unrelated changes into separate hunks.
- Propose the patch first; apply only after explicit user approval.
- After applying, run `git diff --stat` to verify insertions/deletions match the plan exactly.

**Application:**

If possible and feasible, use git to apply the planned patch instead of manual changes.

Prefer `git apply` over manual edits when feasible — it is atomic, auditable, and eliminates transcription errors:

```bash
# Write the patch to a temp file and apply
git apply patch.diff

# Dry-run first to catch mismatches without touching files
git apply --check patch.diff

# If context lines don't match exactly (e.g. line endings), use fuzzy matching
git apply --whitespace=fix patch.diff
```

If `git apply` fails (context mismatch, encoding, or Windows line endings), fall back to the Edit tool with `oldString`/`newString` — the patch still serves as the reviewable plan.

**Why this works:**

- **Token-efficient:** reviewer sees only what changes, not the full file.
- **Deterministic:** `oldString` is pinned by context; no ambiguity about location.
- **Loss-proof:** deleted lines are explicit (`-`); nothing disappears silently.
- **Auditable:** the patch itself is the record of intent — copy it to the plan file for traceability.

### Validation

- Web search mandatory for pricing, versions, any figures
- Cross-validate 2-3 sources
- Distinguish cloud vs hybrid vs on-prem

### Harmonization

- Compare timelines, pricing, terminology across files
- Remove any authoring remarks from the actual target documents
- Add cross-references

### Three-tier categorization (sort by local relevance)

- Cloud (SaaS) → Hybrid → On-prem

### Market priority for technology research

When researching solutions, vendors, or technologies, apply this priority order:

1. **Russia solutions** — Russia-native vendors, products, and services
2. **Open-source solutions** that work in Russia without VPN
3. **Proprietary solutions** that work in Russia without VPN
4. **International/global solutions** (outside Russia) — baseline for best practices, top-tier frontier research and models, comparison, foreign clients, or last resort when Russia options are unavailable

This ordering reflects the business context: Russia-resident defaults, 152-FZ compliance, and local deployment feasibility come first. International solutions serve as technology baseline or for foreign client engagements.

### Technology & Innovation Profiles

When listing tools, frameworks, or architectural patterns, avoid unstructured "flat" bullet points or mixed metadata. Apply a strict executive triplet (a micro-SCQA format):

- **Проблема** (or **Концепция**): The specific business or technical pain point being solved.
- **Решение**: The core mechanism, architecture, or tool used to address it.
- **Результат**: The measurable business or technical impact (speed, cost reduction, quality improvement) and the target audience.

This ensures clarity, eliminates tautology, and immediately answers "So What?" for C-level decision-makers, making the text ready for direct reuse in commercial offers and presentations.

### Real-World Case Integration

When adding case studies to frameworks:

- **Inline, not standalone** — integrate into relevant doc
- **Ground with numbers** — specific metrics over vague statements
- **Cite sources** — traceable via inline links **where appropriate** and/or in the centralized appendix (see **Traceability modes**)

### Russian Technical Terms

Use established Russian equivalents. Keep English acronyms (LLM, RAG, API) when industry-standard.

## Comindware research positioning

- **What Comindware sells:** **implementation and guidance** on customer AI programs—not “research PDFs” or “publication curation” as an offering. `docs/research/` is **internal** enablement so teams can **compose** customer-ready narratives with **traceable** evidence.
- **Comindware first-party practice:** engineering reality in **cmw-rag**, **cmw-mosec**, **cmw-vllm**, **cmw-platform-agent**. Say **“we measured”** only where **documented** in those repos; do **not** imply customer-site benchmarks without evidence.
- **Harvested market and vendor material:** conferences, regulation, surveys, community—**attribute** sources and state **where it does / does not** apply to a resident RF contour. External benchmarks are **inputs**, not automatic baselines for customer proposals.
- **Sovereign default (RF):** default architecture and compliance story to **residency**, **152-FZ**, and **local** clouds/APIs where the customer requires it. **US/CN/global** content is **scoped background** unless legal and contract review say otherwise.
- **Global telemetry and ecosystem surveys** (e.g. large-provider reports): pair numbers with a **one-line sample/scope** so repurposed slides stay honest.
- **Depth vs brevity:** the pack’s **short executive summaries** carry the same **business intent** as the long reports but **no repository paths** in body text (task §4, §8); deep reports and appendices keep `.md#anchor` links for maintenance. Details: [Research pack task](./20260324-research-task.md).

## Agentic Research Execution Protocol

This section defines the step-by-step workflow for conducting deep research on technology transfer and AI markets. Follow this protocol when assigned research tasks in `docs/research/executive-research-technology-transfer/`.

### Pre-Research: Read Required Context

Read before starting:

1. `docs/research/executive-research-technology-transfer/tasks/*-research-task.md`
2. Comindware's business goals (implementation/guidance on AI programs).
3. `deep-researches/` files.
4. Raw materials in `~/Documents/cmw-rag-channel-extractions/`.

### Step 1: Create and Version the Master Plan

- [ ] Create a **single** master plan in `.opencode/plans/YYYYMMDD-descriptive-name.md`.
- [ ] Define scope and identify deep research gaps.
- [ ] Set **self-review checkpoints** (agent verification gates).
- [ ] Use subagents recursively to compile/refine the plan.
- [ ] **Version with git** to trace evolution.

### Step 2: Execute Parallel Research

- [ ] Assign subagents to specific tracks (competitors, pricing, etc.).
- [ ] Use skills: `agent-browser`, `playwright`, `tavily`, etc. Note: `webfetch` may fail sometimes, `tavily` has limits.
- [ ] Write findings to `deep-researches/`.
- [ ] Collect worldwide reports and surveys.

### Step 3: Validate and Ground Findings

- [ ] Mandatory web search for versions, pricing, vendor data.
- [ ] Cross-validate 2-3 independent sources for key figures.
- [ ] Resolve or flag contradictions.
- [ ] Use rounded values for macro figures.

### Step 4: Synthesize and Enhance

- [ ] Refine master plan using subagent reports.
- [ ] **Commit plan iterations** via git.
- [ ] Combine figures into original conclusions, group data.
- [ ] Clarify conflicting content and add missing insights.

### Step 5: Produce C-Level Output

- [ ] Write deeply worked, grounded C-Level material.
- [ ] Enable decision-making (don't teach executives their job).
- [ ] Focus on knowledge transfer, not sales scenarios.
- [ ] Ensure perfect coherence.
- [ ] Write final reports in **Russian**.

### Step 6: Final Review and Iteration

- [ ] Re-read all deep-research files and cross-validate
- [ ] Verify all figures are sourced and validated
- [ ] Iterate until the research is perfect and grounded in truth
- [ ] Do **not** commit research output changes (research is autonomous refinement)

### Master Plan Versioning

Use git to version the **single** master plan file instead of creating multiple dated copies. This maintains a clear trace of how the plan recursively evolved.

| Action | Git Command | Commit Message Pattern |
| :--- | :--- | :--- |
| Initial plan creation | `git add .opencode/plans/YYYYMMDD-*.md && git commit -m "..."` | `plan: init research plan for [topic]` |
| Plan iteration after subagent findings | `git commit -am "..."` | `plan: refine [topic] — added [specific gap addressed]` |
| Checkpoint before major phase | `git commit -am "..."` | `plan: checkpoint — [phase] complete, ready for [next phase]` |
| Final evolved plan | `git commit -am "..."` | `plan: finalize — research scope locked for execution` |

**Why version the master plan:**

- Single source of truth (one file, not `plan-v1.md`, `plan-v2.md`, `plan-final.md`)
- Full traceability of how research scope evolved through recursive refinement
- Easy diff between iterations (`git diff HEAD~2`)
- Recovery to any previous checkpoint if research direction needs adjustment
