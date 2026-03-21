# Reranker Refactoring Progress Report - March 21, 2026

## Summary

Successfully refactored cmw-rag reranker to use industry-standard vLLM/Cohere API contracts from cmw-mosec. All formatting (prefix, suffix, instruction) moved to client-side. Server remains agnostic.

## Key Changes

### New Components

1. **`RerankerAdapter`** (`rag_engine/retrieval/reranker.py`)
   - Handles `/v1/score` (vLLM format) and `/v1/rerank` (Cohere format)
   - Client-side formatting for cross-encoder and LLM rerankers
   - Supports Qwen3 (ChatML) and BGE-Gemma formatting

2. **`RerankerFormatting`** (`rag_engine/config/schemas.py`)
   - Pydantic model for formatting templates
   - `query_template`, `doc_template`, `prefix`, `suffix`, `prompt`

3. **Test Fixture** (`rag_engine/tests/fixtures/test_rerankers.yaml`)
   - Aligned with cmw-mosec test cases
   - Cross-encoder and LLM reranker test cases

### Updated Components

1. **`models.yaml`**
   - Added `reranker_type: cross_encoder | llm_reranker`
   - Added `formatting` for Qwen3 rerankers

2. **`create_reranker()` factory**
   - Returns `RerankerAdapter` for server providers
   - Gets `reranker_type` and `formatting` from registry

3. **`loader.py`** (deprecated)
   - Fixed for new schema changes
   - Added `provider` field for `ServerRerankerConfig`

### Deprecated

- `InfinityReranker` - Still functional, use `RerankerAdapter` instead

## Contract Compliance

### `/v1/score` (vLLM Format)

```json
// Request
{"query": "What is Python?", "documents": ["doc1", "doc2"]}

// Response
{"data": [{"index": 0, "object": "score", "score": 0.88}, {"index": 1, "object": "score", "score": 0.12}]}
```

- Returns scores in **original document order**
- Scores identical to `/v1/rerank`

### `/v1/rerank` (Cohere Format)

```json
// Request
{"query": "What is Python?", "documents": ["doc1", "doc2"], "top_n": 2}

// Response
{"results": [{"index": 0, "document": {"text": "doc1"}, "relevance_score": 0.88}, {"index": 1, "document": {"text": "doc2"}, "relevance_score": 0.12}]}
```

- Returns results **sorted by relevance** (highest first)
- Optional `top_n` parameter

## Test Results

```
rag_engine/tests/test_reranker_factory.py      17 passed, 4 skipped
rag_engine/tests/test_retrieval_reranker.py    5 passed
rag_engine/tests/test_reranker_contracts.py    17 passed
rag_engine/tests/test_config_loader.py         17 passed
---------------------------------------------------------
Total: 56 passed, 4 skipped
```

## Behavior-Based Tests (TDD)

### Contract Tests
- `test_returns_data_array` - vLLM format
- `test_preserves_original_order` - `/v1/score` order
- `test_returns_results_array` - Cohere format
- `test_sorted_by_relevance` - `/v1/rerank` sorting
- `test_top_n_limits_results` - pagination

### Formatting Tests
- `test_no_query_formatting` - cross-encoder passthrough
- `test_no_document_formatting` - cross-encoder passthrough
- `test_instruction_ignored_with_warning` - cross-encoder warns
- `test_qwen3_query_format` - ChatML markers
- `test_qwen3_document_format` - Document markers
- `test_bge_gemma_query_format` - A: {query}
- `test_bge_gemma_document_format` - B: {doc}\n{prompt}

## Non-Breaking Guarantees

| Component | Status |
|-----------|--------|
| `CrossEncoderReranker` | ✅ Unchanged |
| `IdentityReranker` | ✅ Unchanged |
| `Reranker` Protocol | ✅ Same interface |
| `create_reranker()` | ✅ Extended |
| `InfinityReranker` | ⚠️ Deprecated |

## Files Changed

| File | Lines Changed |
|------|---------------|
| `rag_engine/retrieval/reranker.py` | +301 |
| `rag_engine/config/schemas.py` | +38 |
| `rag_engine/config/models.yaml` | +28 |
| `rag_engine/tests/test_reranker_contracts.py` | +340 (new) |
| `rag_engine/tests/fixtures/test_rerankers.yaml` | +188 (new) |
| `rag_engine/config/loader.py` | +14 |
| `rag_engine/tests/test_config_loader.py` | +14 |
| `rag_engine/tests/test_reranker_factory.py` | +16 |

## Integration Notes

### cmw-mosec Compatibility

- ✅ Works with DiTy cross-encoder (port 7998)
- ✅ Works with BGE-m3 cross-encoder
- ✅ Ready for Qwen3 LLM reranker

### Endpoint Derivation

```python
# Config endpoint: http://localhost:7998/v1/rerank
# RewankerAdapter derives base: http://localhost:7998
# Then uses /v1/score or /v1/rerank as needed
```

## Next Steps

1. Test with Qwen3 reranker model when available on cmw-mosec
2. Add integration tests for live Qwen3 server
3. Consider adding `/v1/score` method to `Reranker` Protocol for raw score access