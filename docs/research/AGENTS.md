# Research Agents Guidelines

## Directory Structure

```
docs/research/
├── AGENTS.md           # This file - research workflow guidelines
├── 20260323-topic-ru.md   # Russian executive summary
├── 20260323-topic.md      # (Optional) English executive summary
└── extractions/        # (Optional) Raw extraction files
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

**Format:**
- Single language per file (preferably Russian for this project)
- Processed, curated content only
- Cite sources inline with links
- Structure for executive consumption (not raw data)

**Include:**
- Key facts with citations
- Pricing, benchmarks, metrics
- Architecture comparisons
- Market analysis

**Exclude:**
- Raw scraping data
- Duplicate language versions (unless explicitly requested)
- Intermediate processing artifacts

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