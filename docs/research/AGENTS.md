# Research Agents Guidelines

## Scope

This file defines research workflow, document standards, and formatting requirements for all files in `docs/research/`.

## Instruction Priority

- Apply this file as the default rulebook for research documents in `docs/research/`.
- If a specific task file introduces stricter requirements, follow the stricter rule.
- Do not relax or omit rules from this file unless explicitly requested.

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

- Raw extracted materials must be stored in: `D:/Documents/cmw-rag-channel-extractions/`.

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

- Use a single language per file (preferably Russian for this project).
- Keep content executive-oriented: concise, decision-ready, and evidence-based.
- Structure final summaries as one-pagers (1-2 pages of concentrated meaning).

### Logic and Storyline

- Use SCQA logic:
  - Situation (Ситуация): market context and current state.
  - Complication (Проблема): key challenge or opportunity.
  - Question (Вопрос): core business question.
  - Answer (Ответ/Решение): concrete recommendation and implications.
- Ensure recommendations are MECE where applicable.
- Focus on "So What?" (business meaning and impact).

### Required Analytical Content

- Key facts with citations.
- Pricing, benchmarks, and metrics (ROI, unit economics, etc.).
- Architecture comparisons.
- Market analysis.
- Actionable recommendations (for example, a 30/60/90 day plan).

### Content to Exclude

- Raw scraping outputs.
- Duplicate language versions unless explicitly requested.
- Intermediate processing artifacts.
- Unverified marketing fluff or "water" content.

### Authoring inputs vs published executive text

- Internal repositories, slide decks, and local paths are **authoring inputs only**. Published executive summaries (for example `20260323-*-ru.md`) state **generic, transferable practice**; they must not treat repositories as a product SKU or require readers to open the codebase.

## Russian Language and Numeric Standards

### Numbers and Values

- Use a space as the thousands separator (for example, `1 000 000`).
- Use a comma as the decimal separator (for example, `2,5%`).

### Currency Rules

- Express all financial values in Russian rubles (`руб.`).
- Recalculate USD values using fixed conversion `1 USD = 85 RUB`.
- Example conversion: `$1,200` -> `102 000 руб.`

### Russian typography and punctuation

- **Quotation marks:** use `«` and `»` for Russian quotations. Reserve straight `"` for code, APIs.
- **Em dash (`—`):** use for appositions, breaks in sense, and dialogue-style breaks; do not use hyphen-minus `-` where a long dash is intended.
- **En dash (`–`):** use for ranges (pages, years), e.g. `2019–2025`, with no extra spaces inside the range unless your style guide requires thin spaces.
- **Colon:** if text continues on the same line after `:`, start with lowercase (not English sentence case), except proper names, acronyms, or a clearly new sentence.
- **Hyphen:** use for compounds and hyphenated words; keep minus and code literals as needed in technical snippets.

### Terminology Rules

- Prefer established Russian business and technical terms.
- Use English acronyms or terms only when strictly necessary or industry-standard (for example, `RAG`, `LLM`, `CapEx`).
- At first use, provide translation or short explanation in parentheses.
- Example: `RAG (Retrieval-Augmented Generation — генерация с дополненной выборкой)`.
- **Engineering acronyms** such as `CI`, `CD`, `TCO`, and the like may appear in Russian report prose without expansion when the audience and context make the meaning clear.

## Markdown and Document Formatting Standards

### Prose and line breaks

- No hard wraps mid-sentence in prose or list body text; one sentence per source line, soft-wrap in the editor.
- Break lines only between sentences, list items, or blocks (headings, lists, fences).

### Heading Rules

- Do not use numbered headings in final reports.
- Do not place hyperlinks inside headings.
- Avoid English words in headings unless strictly necessary.
- Convert standalone label-style lines (for example, `**Законопроект об ИИ:**`) into proper Markdown headings instead of bold paragraphs ending with a colon.

### List Rules

- Add an empty line before every bulleted or numbered list.
- Use `-` as the standard bullet marker.
- Do not use `*` as the primary bullet marker.

### Citations and References

- Cite sources inline with hyperlinks where claims are made.
- **Inline mentions** (body text and ordinary lists; not inside `## Источники`): when you name an internal document or an external source by title, use a **quoted italic** Markdown link—guillemets `«»` around an italic link built with underscores `_..._`:
  - **Internal** `.md` in this repo (typically under `docs/research/`): `_«[Название документа](relative-path/file-name.md)»_`. Use a path **relative to the current file** (for example `./20260323-topic-ru.md` or `../other/file.md`).
  - **External** web sources: `_«[Название источника](https://...)»_` with the real URL as the link target.
- **Examples (inline):**
  - Sentence: См. _«[Методология внедрения ИИ](./20260323-ai-implementation-methodology-ru.md)»_ для контекста.
  - List item:
    - См. также _«[Руководство для авторов research](./AGENTS.md)»_.
- At the end of each final article, add a section with exact heading `## Источники`.
- Under `## Источники` or any bulleted lists containing only links, include references with **plain** Markdown links only—**no** guillemets, **no** italic underscores around the link:
  - `- [Название источника](https://...)`
  - Do not format reference-list entries as `_«[...](...)»_`.

### Consolidated packs (optional)

Use for multi-file research sets under `docs/research/` when you want stable deep links and consistent metadata.

**Heading ids** — On the same line as the heading, attribute-list syntax: `## Заголовок {: #prefix_slug }`. Pick one stable English **prefix** per file; H1 uses only the prefix, H2+ use `prefix_transliterated_slug`. Characters: lowercase letters, digits, underscores; ids **unique within the file** (duplicate titles: `_2`, `_3`, …). No links inside heading text (**Heading Rules**).

**YAML front matter** — At the very top when the pack uses it: `title` (same as H1 without the `{: #… }` suffix), `date` (ISO), `status`, `tags` (about 5–12; English alphabetically, then Russian alphabetically), `hide: tags` if tags are for filtering/search only. Optional: `description` (one line). If `date` / `status` are in YAML, drop redundant **Дата пакета** / **Статус** lines under H1.

**Cross-links** — `./sibling.md#anchor` from the current file’s directory. Body mentions of titled internals: **Citations and References**. Under `## Источники`: plain `[title](url)` only. No path for a document that is not a real file—title in guillemets only.

**Optional** — `{: #id .pageBreakBefore }` on a heading where the export toolchain should force a page break.

## Minimal Article Template

Use this compact structure for consistency:

```text
# <Название документа>

## Резюме для руководства
<SCQA в кратком виде>

## Ключевые выводы
- ...

## Рекомендации
- ...

## Риски и ограничения
- ...

## Источники
- [Название источника](https://...)
```

## Research Workflow

- Extract raw data to `D:/Documents/cmw-rag-channel-extractions/`.
- Process findings into executive summaries in `docs/research/`.
- Commit only processed summaries.
- Document reusable methodology updates in this file when needed.

## Cross-validation of related research

Before finalizing or materially revising an article in `docs/research/`, **cross-check** other research files that belong to the same thread of work. Treat as related any document that matches on at least one of:

- **Semantic linkage** — shared themes, entities, markets, technologies, or recommendations (including cross-references, overlapping titles, or clearly parallel subject matter).
- **Business line** — same product, offering, customer segment, or value stream the research supports.
- **Current scope or task** — objectives, acceptance criteria, or explicit scope in an active task file (for example `docs/research/*-research-task.md` or a linked plan under `.cursor/plans/`).

**Cross-validation means:** compare conclusions, figures, dates, currency and unit conventions, and terminology across those related files; resolve or explicitly flag contradictions (for example under `## Риски и ограничения` with a short note pointing to the sibling document); avoid duplicate contradictory “single truths” without reconciliation. When this file and a task file disagree on process, follow the stricter requirement; when task scope defines what is in or out of bounds for the engagement, respect that scope when aligning related summaries.

## Definition of Done (Per Article)

- Document follows naming convention and single-language rule.
- Executive structure is clear (SCQA), concise, and decision-oriented.
- Russian numeric and currency standards are applied consistently.
- Terminology rules are applied (Russian-first, translated first use for English terms).
- Markdown formatting rules are followed (headings, lists, links placement).
- All claims are traceable with inline citations; inline document/source titles follow **Citations and References** (quoted italic links in body text).
- Final `## Источники` section is present with all used references as plain bullet links (no guillemets or italic wrapper on list entries).
- No critical statement remains without a source.
- Related research in `docs/research/` has been cross-validated per **Cross-validation of related research** when semantic overlap, business line, or the active task scope applies.
- If the output is a **consolidated pack** (several linked articles under `docs/research/`), satisfy **Consolidated packs (optional)** in full; ordinary single articles ignore that subsection.

## Operating Principles

- Keep the repository lean (raw data stays outside).
- Process once, document once.
- Maintain Russian market focus.
- Keep full source traceability via inline citations and final references list.
- Reuse abstract patterns and avoid unnecessary duplication.
- Cross-validate sibling research so the corpus stays internally consistent where topics intersect.
