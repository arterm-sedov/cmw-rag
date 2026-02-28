# Endpoint Configuration Refactor: Single Source of Truth

**Date:** 2026-02-18  
**Branch:** `20260217_sgr_tool_choice`  
**Commits:** `1c51222`, `b11d6ba`

---

## Summary

Consolidated endpoint configuration across embedder, reranker, and guardian to use complete URLs from `.env` as the single source of truth. Eliminated path concatenation bugs and orphaned configuration data.

---

## Problem Statement

### Path Duplication Bug
Endpoints in `.env` included paths (e.g., `http://localhost:7998/v1/embeddings`), but code appended paths again:
```
# .env
MOSEC_EMBEDDING_ENDPOINT=http://localhost:7998/v1/embeddings

# Code (BUG)
f"{self.config.endpoint}/embeddings"  
# Result: http://localhost:7998/v1/embeddings/embeddings ❌
```

### Orphan Data in models.yaml
- `endpoint_path` entries in provider_formats were defined but never used
- Reranker had unused `path` field in `ServerRerankerConfig`

### Inconsistent Guardian Config
Guardian used three separate fields (`url`, `port`, `path`) instead of single endpoint like embedder/reranker.

---

## Solution

### 1. Complete URLs in .env
All endpoints now include full path:
```bash
# Embedder
MOSEC_EMBEDDING_ENDPOINT=http://localhost:7998/v1/embeddings
INFINITY_EMBEDDING_ENDPOINT=http://localhost:7997/v1/embeddings

# Reranker
MOSEC_RERANKER_ENDPOINT=http://localhost:7998/v1/rerank
INFINITY_RERANKER_ENDPOINT=http://localhost:7998/v1/rerank

# Guardian
GUARD_MOSEC_ENDPOINT=http://localhost:7999/api/v1/guard
```

### 2. Removed Path Concatenation
```python
# Before (BUG)
resp = requests.post(f"{self.config.endpoint}/embeddings", ...)

# After (FIXED)
resp = requests.post(self.config.endpoint, ...)
```

### 3. Schema Updates

**ServerRerankerConfig:**
- Removed: `path` field
- Added: `provider`, `timeout`, `max_retries`
- Removed: `local` (HTTP-only, no SDK option for rerankers)

**Guardian settings:**
- Removed: `guard_mosec_url`, `guard_mosec_port`, `guard_mosec_path`
- Added: `guard_mosec_endpoint` (single field)

### 4. Cleaned models.yaml
- Removed all `endpoint_path` entries from reranker models
- Model metadata only, no runtime configuration

---

## Files Changed

| File | Changes |
|------|---------|
| `rag_engine/retrieval/embedder.py` | Fixed path duplication in `_embed_local()` and `_embed_documents_local()` |
| `rag_engine/retrieval/reranker.py` | Removed path arg from `_post()`, use complete endpoint |
| `rag_engine/core/guard_client.py` | Use `guard_mosec_endpoint` directly |
| `rag_engine/config/schemas.py` | Updated `ServerRerankerConfig` schema |
| `rag_engine/config/settings.py` | Consolidated guardian fields |
| `rag_engine/config/models.yaml` | Removed `endpoint_path` from rerankers |
| `.env` / `.env-example` | Updated to single endpoint format |
| `README.md` | Updated guardian config docs |

---

## Tests

All 27 tests pass:
- `test_embedder_factory.py`: 9 passed
- `test_reranker_factory.py`: 18 passed (2 skipped - integration tests)

---

## Key Principle

**Single source of truth**: `.env` provides all runtime values, `models.yaml` provides model metadata only. No defaults in code, no path concatenation.
