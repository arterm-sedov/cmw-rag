# Reranking Boost Configuration - March 22, 2026

## Summary

Made reranking metadata boosts configurable via `.env` and reduced default values. The previous hardcoded defaults (`1.2/1.15/1.1`) were aggressive — scores could inflate 2-4x when multiple boosts stacked. New defaults are `0.2/0.15/0.1`, providing modest 10-20% boosts.

## Problem

Boost formula `score * (1.0 + boost)` with defaults `tag_match=1.2, code_presence=1.15, section_match=1.1`:

- Single boost: `0.8 * (1.0 + 1.15) = 1.72` (2.15x)
- All three stacked: `0.8 * (1.0 + 3.45) = 3.56` (4.45x)

These values were never configurable — baked into `retriever.py` with no way to tune without code changes.

## Changes

### Config (`rag_engine/config/settings.py`)

Added 3 settings fields (default `0.0`):

```python
rerank_boost_code: float = 0.0
rerank_boost_tag: float = 0.0
rerank_boost_section: float = 0.0
```

### Retriever (`rag_engine/retrieval/retriever.py:51-55`)

Hardcoded defaults changed from `1.2/1.15/1.1` to `0.0/0.0/0.0`. Actual values now come from settings.

### App (`rag_engine/api/app.py:376-387`)

Passes boost settings to `RAGRetriever` constructor via `metadata_boost_weights` dict.

### Environment (`.env`, `.env-example`)

```bash
RERANK_BOOST_CODE=0.15      # Docs containing code blocks
RERANK_BOOST_TAG=0.2        # Docs with matching tags
#RERANK_BOOST_SECTION=0.0   # Reserved — not yet implemented
```

`RERANK_BOOST_SECTION` is commented out in `.env` — code falls back to `0.0` default in `settings.py`. Kept visible in `.env-example` for documentation.

### Formula

Kept `score * (1.0 + boost)` — multiplicative, proportional to relevance. The issue was magnitude of defaults, not the formula.

## How Boost Criteria Are Identified

### `has_code` → `code_presence` boost

Set during **indexing** in `rag_engine/core/metadata_enricher.py:21`:

```python
CODE_BLOCK_PATTERN = re.compile(r"```(\w+)?[\s\S]*?```",  re.MULTILINE)
has_code = bool(CODE_BLOCK_PATTERN.search(content))
```

Detects markdown code fences (`` ``` ``). Applied per-chunk during `enrich_metadata()`.

### `tags` → `tag_match` boost

Set from **YAML frontmatter** in source markdown files. `document_processor.py:149` passes `extra=fm` into `_normalize_base_metadata()`, which merges it into metadata. If a `.md` file has:

```yaml
---
kbId: 12345
tags: [api, authentication]
---
```

...then `meta.get("tags")` is truthy and the boost applies.

### `section_heading` → `section_match` boost

**Never explicitly set by the codebase.** The processor splits by H1 headings and stores the result as `title` (`document_processor.py:145`), but not as `section_heading`. This boost can only trigger if a frontmatter field named `section_heading` exists in the source `.md`, or if an external process sets it during ingestion. Currently dead weight.

### Summary

| Boost | Source | Currently active? |
|---|---|---|
| `code_presence` | Regex on chunk content | Yes |
| `tag_match` | Frontmatter `tags` field | Yes (if docs have tags) |
| `section_match` | Frontmatter `section_heading` field | No (nothing sets it) |

### Where Boosts Are Applied

Three places in `rag_engine/retrieval/reranker.py`, all identical logic:
- `CrossEncoderReranker.rerank()` (line 168)
- `InfinityReranker.rerank()` (line 228)
- `RerankerAdapter.rerank()` (line 347)

## Test Updates

- `test_reranker_factory.py` — boost values changed to `0.05`, expected scores recalculated
- `test_reranker_contracts.py` — boost value changed to `0.05`, expected score recalculated

## Breaking Change

Deployments relying on the old implicit boosts (`1.2/1.15/1.1`) will see reduced boost effects unless they set the env vars. The new `.env` values (`0.2/0.15/0.1`) provide ~5-10% of the old boost magnitude.

## Verification

- 30 tests passed
- Lint clean
