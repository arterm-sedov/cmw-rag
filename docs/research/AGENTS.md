# Research Agents Guidelines

## Directory Structure

```
docs/research/
├── AGENTS.md           # This file - research workflow guidelines
├── YYYYMMDD-topic.md   # Russian executive summary (primary)
└── YYYYMMDD-topic-en.md   # (Optional) English executive summary
```

## Source Material Storage

**Location:** `D:/Documents/cmw-rag-channel-extractions/`

**Purpose:**
- Raw scraped data from external sources (channels, websites, APIs)
- Large files that can be regenerated on demand
- Keeps repository focused on curated, processed content

**Git Ignore:**
```
.playwright-cli/
channel_snapshot.yml
```

## Executive Summary Guidelines

**Naming Convention:**
```
YYYYMMDD-topic-[lang].md
```
Example: `20260323-ai-implementation-methodology-ru.md`

**Format & Style (McKinsey/BCG Style):**
- **Language**: Single language per file (preferably Russian for this project).
- **Executive focus**: Processed, curated content only, structured as a "one-pager" (1-2 pages of concentrated meaning).
- **Citations**: Cite all sources inline with links.
- **Logic (SCQA)**:
  1.  **Situation (Ситуация)**: Market context and current state.
  2.  **Complication (Проблема)**: Key challenge or opportunity requiring a solution.
  3.  **Question (Вопрос)**: Core question the report answers.
  4.  **Answer (Ответ/Решение)**: Specific recommendations backed by data and calculations. Focus on the "So What?" (the meaning and implications) and ensure recommendations are **MECE** (Mutually Exclusive, Collectively Exhaustive).
- **No numbered headings**: Do not use numbered headings in final reports (e.g., avoid 1., 1.1., etc.). Use clear, descriptive text headings only.
- **No links in headings**: Do not place hyperlinks within heading text. Instead, place them in the following descriptive text or as citations.
- **Avoid English in headings**: Do not use English words in headings unless strictly necessary (e.g., fixed industry abbreviations with no practical Russian equivalent).
- **List Spacing**: Always add an empty newline before any bulleted or numbered list for better readability.
- **Tone**: Business-like, concise, decision-oriented. Focus on figures and metrics (ROI, savings, sizing).

**Russian Standards for Numbers & Values:**
- **Number Formatting**:
  - Use space as a thousands separator (e.g., `1 000 000`, not `1,000,000`).
  - Use comma as a decimal separator (e.g., `2,5%`, not `2.5%`).
- **Currency & Pricing**:
  - All financial metrics must be expressed in **Russian Rubles (руб.)**.
  - Recalculate USD figures to RUB at a fixed rate: **1 USD = 85 RUB**.
  - Example: `$1,200` -> `102 000 руб.`
- **Terminology**:
  - **Priority**: Use established Russian business and technical terms whenever possible.
  - **English Terms**: Use English acronyms/terms (e.g., *RAG*, *LLM*, *CapEx*) only when absolutely necessary or when they are industry standards.
  - **First Use**: Provide a translation or brief explanation in parentheses at first mention.
  - Example: `RAG (Retrieval-Augmented Generation — генерация с дополненной выборкой)`.

**Include:**
- Key facts with citations
- Pricing, benchmarks, metrics (ROI, unit economics, etc.)
- Architecture comparisons
- Market analysis
- Actionable recommendations (e.g., 30/60/90 day plan)

**Exclude:**
- Raw scraping data
- Duplicate language versions (unless explicitly requested)
- Intermediate processing artifacts
- Unverified marketing fluff or "water" content

## Workflow

1. **Extract** raw data to `D:/Documents/cmw-rag-channel-extractions/`
2. **Process** into executive summary in `docs/research/`
3. **Commit** only the processed summaries
4. **Document** methodology in this AGENTS.md file

## Key Principles

- Keep repository lean (raw data outside)
- Process once, document once
- Russian market focus (primary language)
- Cite all sources inline
- Abstract patterns for reusability