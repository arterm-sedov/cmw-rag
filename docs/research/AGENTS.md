# Research Agents Guidelines

## Scope

This file defines research workflow, document standards, and formatting requirements for all files in `docs/research/`.

## Instruction Priority

- Apply this file as the default rulebook for research documents in `docs/research/`.
- If a specific task file introduces stricter requirements, follow the stricter rule.
- Do not relax or omit rules from this file unless explicitly requested.

## Business goals and task authority

**Read this before any edits** to the **consolidated research pack** in `docs/research/` (full methodology and sizing reports, appendices A–D, and the two short executive summaries). The corpus serves **Comindware’s commercial work**: evidence and narrative to **win and deliver** customer AI programs—**budget, architecture, compliance, handover (KT/IP), and roadmap**—not a standalone SKU for “selling research” or “curating publications.”

**When you edit, optimize for:** decision-ready prose, **RF-resident defaults** (152-FZ, local clouds, data residency), and **honest limits** on global/vendor telemetry (sample scope; not an automatic baseline for RF production). Readers **reuse** claims in **their own** board packs and proposals; avoid meta-text about “how we wrote this for executives.”

### Reader-facing pack: business, technology, and expertise transfer

The consolidated **20260325-** research pack is **not** internal authoring documentation. Write for **commercial and technical decision-makers** who will **sell and scope** AI delivery: budget, architecture, compliance, KT/IP, and risk—grounded in **Comindware’s documented practice** (open repos where relevant) and **attributed** public sources (pricing, benchmarks, regulation).

- **Tone:** **business + technology**; concise; no patronizing “how to read this file,” no references to source-file mechanics (YAML, front matter, template filenames), and no pack-maintenance asides (split history, “canonical row” editorial rules).
- **Cross-cutting rules** (e.g. **FX**): **one canonical block** — [Appendix A, «Валюта: пересчёт USD…»](./20260325-research-appendix-a-index-ru.md#app_a_fx_policy) (`#app_a_fx_policy`); other documents **short stub + link**; sizing/methodology use `#sizing_fx_policy` / `#method_fx_policy` and point in-report conversions to those anchors.
- **Hardware facts:** do **not** describe **RTX 4090 48 GB** as a “Comindware custom” GPU. It is a **commercially offered** dedicated-GPU server configuration (e.g. providers such as [1dedic — GPU servers](https://1dedic.ru/gpu-servers)); Comindware or Clients may buy or rent such SKUs like any customer-facing integrator.

- [Research pack task (authoritative)](./20260324-research-task.md) — **authoritative** scope, acceptance, workflow, evidence and FX: **Russian** body (§§1–8). It points agents here for English workflow rules and non-duplication.
- [Original task snapshot](./20260324-research-task_original.md) — historical manager brief; if it conflicts with the main task file, **follow the main task file**.

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
  - **Situation:** market context and current state.

  - **Complication:** key challenge or opportunity.

  - **Question:** core business question.

  - **Answer:** concrete recommendation and implications.

- Ensure recommendations are MECE where applicable.
- Focus on “So what?” (business meaning and impact).

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
- Unverified marketing fluff or filler.

### Authoring inputs vs published executive text

- Internal repositories, slide decks, and local paths are **authoring inputs only**. Published executive summaries (for example `20260323-*-ru.md`) state **generic, transferable practice**; they must not treat repositories as a product SKU or require readers to open the codebase.

## Russian-language documents: numbers, currency, typography

These rules apply to **Russian** research articles under `docs/research/` (the default language for this corpus).

### Numbers and Values

- Use a space as the thousands separator (for example, `1 000 000`).
- Use a comma as the decimal separator (for example, `2,5%`).

### Currency

- Express financial values in **Russian rubles** (`руб.`).
- Recalculate USD using the pack’s fixed conversion unless the task states otherwise (typically **`1 USD = 85 RUB`** for internal comparison; proposals use CBR or contract rate per task file).
- Example: `$1,200` → `102 000 руб.`

### Typography and punctuation (Russian)

- **Quotation marks:** use `«` and `»` for Russian quotations. Reserve straight `"` for code and APIs.
- **Bold inside quotes:** when formatting a quoted term in bold, place the asterisks **inside** the guillemets: `«**Термин**»`, not `**«Термин»**`.
- **Em dash (`—`):** appositions, breaks in sense, dialogue-style breaks; do not use hyphen-minus `-` where a long dash is intended.
- **En dash (`–`):** ranges (pages, years), e.g. `2019–2025`, no extra spaces inside the range unless your style guide requires thin spaces.
- **Colon:** if text continues on the same line after `:`, start with lowercase (not English sentence case), except proper names, acronyms, or a clearly new sentence.
- **Hyphen:** compounds and hyphenated words; keep minus and code literals as needed in technical snippets.

### Terminology

- Prefer established Russian business and technical terms.
- Use English acronyms or terms only when strictly necessary or industry-standard (for example, `RAG`, `LLM`, `CapEx`).
- At first use, provide translation or short gloss in parentheses.
- Example: `RAG (Retrieval-Augmented Generation — генерация с дополненной выборкой)`.
- **Engineering acronyms** such as `CI`, `CD`, `TCO` may appear without expansion when the audience and context are clear.

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

- Cite sources inline with hyperlinks where claims are made.
- **Inline mentions** (body text and ordinary lists; not inside the references section): when you name an internal document or external source by title, use a **quoted italic** Markdown link—guillemets `«»` around an italic link built with underscores `_..._`:
  - **Internal** `.md` in this repo (typically under `docs/research/`): `_«[Document title](relative-path/file-name.md)»_`. Path **relative to the current file** (for example `./20260323-topic-ru.md` or `../other/file.md`).
  - **External** web sources: `_«[Source title](https://...)»_` with the real URL as the link target.
- **Examples (inline, Russian prose):**
  - See _«[Методология внедрения ИИ](./20260323-ai-implementation-methodology-ru.md)»_ for context.
  - See also _«[Research Agents Guidelines](./AGENTS.md)»_.
- At the end of each final Russian article, add a section with the **exact** heading `## Источники`.
- Under `## Источники` (or any bullet list that is links only), use **plain** Markdown links—**no** guillemets, **no** italic underscores around the link:
  - `- [Source title](https://...)`
  - Do not format reference-list entries as `_«[...](...)»_`.

### Consolidated packs (optional)

Use for multi-file research sets under `docs/research/` when you want stable deep links and consistent metadata.

**Heading anchors/IDs** — On the same line as the heading, Kramdown-style attribute list: `## Heading text {: #anchor_slug }`. Pick one stable English **prefix** per file as the H1 `#root_anchor` from the **subject** of the **document**; H2+ use `#root_anchor_concise_english_snake_case` from the **meaning** of the heading (not Cyrillic transliteration). Characters: lowercase letters, digits, underscores; anchors **unique within the file** (duplicate titles: `_2`, `_3`, …). No links inside heading text (see **Heading rules**).

**YAML front matter** — When the pack uses it, at the very top: `title` (same as H1 without the `{: #… }` suffix), `date` (ISO), `status`, `tags` (about 5–12; English alphabetically, then Russian alphabetically), `` if tags are for filtering/search only. Optional: `description` (one line). If `date` / `status` are in YAML, drop redundant **Дата пакета** / **Статус** lines under H1.

**Cross-links** — `./sibling.md#anchor` from the current file’s directory. Body mentions of titled internals: **Citations and references**. Under `## Источники`: plain `[title](url)` only. No path for a document that is not a real file—title in guillemets only.

**Cross-references vs C-level summaries:** All documents in the pack **should** include cross-references (`.md#anchor` links) to sibling documents for navigation and coherence. C-level executive summaries may link to other documents within the same pack using human-readable names with hyperlinks (e.g., `_«[Методология внедрения](./20260325-research-report-methodology-main-ru.md)»_`). However, C-level summaries must **not** contain paths to **external repositories** or internal code paths (e.g., `../cmw-mosec/README.md`, `rag_engine/`)—these are authoring inputs only.

**Split-pack heading anchors (methodology + sizing + appendices):** In `docs/research/`, explicit heading IDs use prefixes `method_`, `sizing_`, `app_a_`, `app_b_`, `app_d__`. Do **not** use legacy patterns `research_pkg_*` or long `research_methodology_20260325_*` in pack body files; if they appear in older `.cursor/plans/`, treat as historical and reconcile with the live document (ledger: [Research pack task](./20260324-research-task.md), §1б).

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

## Источники
- [Source title](https://...)
```

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

- Every significant claim requires inline citations to sources
- Maintain a complete references section with all sources used
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
- All claims are traceable with inline citations; inline document/source titles follow **Citations and references** (quoted italic links in body text).
- Final `## Источники` section is present with all used references as plain bullet links (no guillemets or italic wrapper on list entries).
- No critical statement remains without a source.
- Related research in `docs/research/` has been cross-validated per **Cross-validation of related research** when semantic overlap, business line, or the active task scope applies.
- If the output is a **consolidated pack** (several linked articles under `docs/research/`), satisfy **Consolidated packs (optional)** in full; ordinary single articles ignore that subsection.

## Operating principles

- Keep the repository lean (raw data stays outside).
- Process once, document once.
- Maintain **Russian market** focus and sovereign-default framing per the task file.
- Keep full source traceability via inline citations and final references list.
- Reuse abstract patterns and avoid unnecessary duplication.
- Cross-validate sibling research so the corpus stays internally consistent where topics intersect.

## Reusable research patterns

### Plan → Execute → Validate → Iterate

- Plan under `.opencode/plans/`
- Subagents for parallel work → `deep-researches/`
- Iterate plan based on results

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

### Real-World Case Integration

When adding case studies to frameworks:

- **Inline, not standalone** — integrate into relevant doc
- **Ground with numbers** — specific metrics over vague statements
- **Cite sources** — sources hyperlinked inline and in bibliography

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

Before starting any research task, you **must** read:

1. **Task definition:** `docs/research/executive-research-technology-transfer/tasks/*-research-task.md`
2. **Business goals:** Refer to Comindware's positioning (implementation and guidance on customer AI programs)
3. **Existing deep-research files:** `docs/research/executive-research-technology-transfer/deep-researches/`
4. **Raw materials:** `~/Documents/cmw-rag-channel-extractions/`

### Step 1: Create and Version the Master Plan

- [ ] Create **single** master plan under `.opencode/plans/YYYYMMDD-descriptive-name.md`
- [ ] Define research scope based on task requirements
- [ ] Identify specific gaps requiring deep research (competitor analysis, pricing models, market sizing)
- [ ] Set verification checkpoints
- [ ] Use subagents recursively to compile and refine the plan
- [ ] **Version the plan with git** — commit iterations to trace evolution (see **Master Plan Versioning** below)

### Step 2: Execute Parallel Research

Use subagents to parallelize work across multiple phases:

- [ ] Assign subagents to specific research tracks (competitors, pricing, market stats)
- [ ] Direct subagents to use skills: `agent-browser`, `playwright`, `exa`, `searxng`, `tavily` for web scraping. Note: `webfetch` may fail sometimes, `tivaly` has limits.
- [ ] Subagents write findings to `docs/research/executive-research-technology-transfer/deep-researches/`
- [ ] Collect worldwide research, reports, and surveys

### Step 3: Validate and Ground Findings

- [ ] Web search mandatory for all model versions, pricing, and vendor data (see **Current data requirement** above)
- [ ] Cross-validate 2-3 independent sources for every key figure
- [ ] Resolve contradictions or flag them explicitly
- [ ] Use rounded values for macro figures (avoid 4+ decimal places)

### Step 4: Synthesize and Enhance

- [ ] Use subagent reports to refine the master plan
- [ ] **Commit plan iterations** — each significant evolution of the master plan gets a git commit with descriptive message
- [ ] Combine figures from multiple sources into original conclusions
- [ ] Group scattered information into coherent sections
- [ ] Clarify confused or conflicting content
- [ ] Add missing valuable insights

### Step 5: Produce C-Level Output

- [ ] Target: deeply worked, grounded material for C-Level decision making
- [ ] Enable executives to make decisions easier—do not teach them their job
- [ ] Focus on knowledge transfer, not sales scenarios
- [ ] Ensure perfect coherence: better, not bigger
- [ ] Write final reports in **Russian** (internal thinking/plans in English)

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
