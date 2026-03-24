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

## Russian Language and Numeric Standards

### Numbers and Values

- Use a space as the thousands separator (for example, `1 000 000`).
- Use a comma as the decimal separator (for example, `2,5%`).

### Currency Rules

- Express all financial values in Russian rubles (`руб.`).
- Recalculate USD values using fixed conversion `1 USD = 85 RUB`.
- Example conversion: `$1,200` -> `102 000 руб.`

### Russian typography and punctuation

- **Quotation marks:** use `«` and `»` for Russian quotations. Reserve straight `"` for code, APIs, or when the medium cannot render guillemets.
- **Em dash (`—`):** use for appositions, breaks in sense, and dialogue-style breaks; do not use hyphen-minus `-` where a long dash is intended.
- **En dash (`–`):** use for ranges (pages, years), e.g. `2019–2025`, with no extra spaces inside the range unless your style guide requires thin spaces.
- **Colon:** if text continues on the same line after `:`, start with lowercase (not English sentence case), except proper names, acronyms, or a clearly new sentence.
- **Hyphen:** use for compounds and hyphenated words; keep minus and code literals as needed in technical snippets.

### Terminology Rules

- Prefer established Russian business and technical terms.
- Use English acronyms or terms only when strictly necessary or industry-standard (for example, `RAG`, `LLM`, `CapEx`).
- At first use, provide translation or short explanation in parentheses.
- Example: `RAG (Retrieval-Augmented Generation — генерация с дополненной выборкой)`.

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
- At the end of each final article, add a section with exact heading `## Источники`.
- Under `## Источники`, include all used references as a bulleted list with links to original materials.

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
- All claims are traceable with inline citations.
- Final `## Источники` section is present with all used references as bullet links.
- No critical statement remains without a source.
- Related research in `docs/research/` has been cross-validated per **Cross-validation of related research** when semantic overlap, business line, or the active task scope applies.

## Operating Principles

- Keep the repository lean (raw data stays outside).
- Process once, document once.
- Maintain Russian market focus.
- Keep full source traceability via inline citations and final references list.
- Reuse abstract patterns and avoid unnecessary duplication.
- Cross-validate sibling research so the corpus stays internally consistent where topics intersect.
