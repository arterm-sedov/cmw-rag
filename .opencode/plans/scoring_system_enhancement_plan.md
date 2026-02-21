# RAG Scoring System Enhancement Plan

## Executive Summary

This plan addresses the scoring architecture issues discovered during code analysis. The system currently has three different scores at different pipeline stages, but one critical score (`rerank_score_raw`) is lost during aggregation. The XLS processing script attempts to display per-chunk raw scores but fails because chunks are never serialized to the trace.

## Problem Statement

### Current Issues

1. **`rerank_score_raw`** - Referenced in `trace_formatters.py:129` but **never set** anywhere in the pipeline
2. **Per-chunk scores lost** - Raw CrossEncoder outputs are overwritten with boosted scores during aggregation
3. **XLS output incomplete** - The `format_chunks_column_from_trace()` function tries to read `rerank_score_raw` from chunks, but chunks are dropped during JSON serialization in `retrieve_context.py`
4. **Confidence calculation unused** - `compute_retrieval_confidence()` exists but is rarely called in production

### Impact

- Users cannot see the raw CrossEncoder output (true relevance signal)
- Users cannot compare boosted vs raw scores to understand ranking decisions
- Debugging retrieval quality is difficult without raw scores
- The XLS processing script shows empty chunk scores

---

## Architecture Analysis

### Current Scoring Pipeline

```
Stage 1: Vector Search
├── Input: Query → Embedding
├── Output: Top-K documents by embedding similarity
└── Score: embedding_score (NOT STORED)

Stage 2: CrossEncoder Reranker (PER-CHUNK)
├── Input: (query, chunk_text) pairs
├── Raw Score: CrossEncoder.predict() output (e.g., 0.01 - 5.0)
├── Boosted Score: raw * (1.0 + metadata_boost)
│   ├── tag_match: +1.2
│   ├── code_presence: +1.15
│   └── section_match: +1.1
├── Output: (chunk, score) tuples sorted by score
└── Problem: RAW score is NOT stored - only boosted score kept

Stage 3: Article Aggregation (PER-ARTICLE)
├── Input: Multiple chunks per article
├── MAX Score: max(booste

d scores) → article.metadata["rerank_score"]
├── Output: Complete Article objects
└── Problem: Raw per-chunk scores are LOST here

Stage 4: Serialization (JSON)
├── Input: Article objects
├── Output: {
│   "articles": [{
│       "kb_id": ...,
│       "title": ...,
│       "url": ...,
│       "content": ...,
│       "metadata": {...}  # rerank_score stored, chunks DROPPED
│   }],
│   ...
}
└── Problem: matched_chunks are never included

Stage 5: UI Display
├── Input: Serialized articles
├── Output: Articles list with scores
└── Current: Only rerank_score (boosted) displayed
```

---

## Score Definitions

### 1. Raw CrossEncoder Score (`rerank_score_raw`)

**Definition:** Direct output from the CrossEncoder model before any metadata boosts.

**Calculation:**
```python
# From rag_engine/retrieval/reranker.py:117-121
pairs = [(query, doc.page_content) for doc, _ in candidates]
scores = self.model.predict(pairs, batch_size=self.batch_size)
# Returns unbounded float values (DiTy can output 0.01 to 5.0+)
```

**Current Status:** NEVER STORED - lost during reranking when boosted score overwrites it.

**Use Cases:**
- Debugging retrieval quality
- Understanding true model confidence
- Comparing different queries fairly (without boost bias)
- Training/evaluation of retrieval quality

### 2. Article Relevance Score (`rerank_score`)

**Definition:** Final relevance score used for ranking articles, includes metadata boosts.

**Calculation:**
```python
# From rag_engine/retrieval/reranker.py:123-136
boost = 0.0
if metadata_boost_weights and hasattr(doc, "metadata"):
    if meta.get("tags"):
        boost += metadata_boost_weights.get("tag_match")  # +1.2
    if meta.get("has_code"):
        boost += metadata_boost_weights.get("code_presence")  # +1.15
    if meta.get("section_heading"):
        boost += metadata_boost_weights.get("section_match")  # +1.1
final_score = float(score) * (1.0 + boost)

# From rag_engine/retrieval/retriever.py:254
articles_map[kb_id] = (chunks, max(best_score, score))  # MAX per article
```

**Example:**
- Raw score: 0.78
- Boost multiplier: 1.2 (has tags)
- Final article score: 0.78 * 1.2 = 0.936 → rounded to 0.96

**Current Status:** STORED in `article.metadata["rerank_score"]`

**Use Cases:**
- Primary ranking signal for articles
- Threshold filtering (`rerank_score_threshold`)
- UI display to users

### 3. Retrieval Confidence (`Уверенность`)

**Definition:** Query-level quality metric showing how confident the system is about retrieval results.

**Calculation:**
```python
# From rag_engine/retrieval/confidence.py:67-116
def compute_normalized_confidence_from_traces(query_traces):
    # Extract top_score from each query trace
    raw_scores = [conf["top_score"] for conf in traces]

    # Normalize to 0-1 range preserving relative differences
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    normalized = [(s - min) / (max - min) for s in raw_scores]

    return average(normalized)  # 0.0-1.0
```

**Current Status:** Computed on-demand but rarely called in production

**Use Cases:**
- Understanding retrieval quality at query level
- Identifying hard vs easy queries
- Debugging retrieval failures

---

## Current Code Issues

### Issue 1: Raw Score Never Stored

**Location:** `rag_engine/retrieval/reranker.py:117-139`

```python
# CURRENT CODE - overwrites raw score with boosted score
scores = self.model.predict(pairs, batch_size=self.batch_size)
for (doc, _), score in zip(candidates, scores):
    boost = calculate_boost(doc)
    final_score = float(score) * (1.0 + boost)
    scored.append((doc, final_score))  # RAW SCORE LOST HERE
```

**Fix Required:** Store raw score before applying boost.

### Issue 2: Chunks Never Serialized

**Location:** `rag_engine/tools/retrieve_context.py:142-176`

```python
# CURRENT CODE - chunks dropped during serialization
def _format_articles_to_json(articles, query, top_k):
    articles_data = []
    for article in articles:
        # ... article fields ...
        article_metadata = dict(article.metadata)  # rerank_score here
        articles_data.append({
            "kb_id": article.kb_id,
            "title": title,
            "url": url,
            "content": article.content,
            "metadata": article_metadata,  # NO matched_chunks!
        })
    return json.dumps({"articles": articles_data, ...})
```

**Fix Required:** Include `matched_chunks` in serialization.

### Issue 3: Query Traces Not Populated

**Location:** `rag_engine/utils/context_tracker.py:63-67`

```python
# query_traces structure exists but is never populated
query_traces: list[dict[str, Any]] = Field(
    default_factory=list,
    exclude=True,
    description="Actual executed retrieval calls trace",
)
```

**Trace Structure Expected:**
```python
{
    "query": "search query",
    "confidence": {
        "top_score": 0.96,
        "mean_top_k": 0.85,
        "score_gap": 0.12,
        "n_above_threshold": 5,
        "likely_relevant": True
    },
    "articles": [
        {
            "kb_id": "5000",
            "title": "...",
            "rerank_score": 0.96,
            "rerank_score_raw": 0.78,  # <-- MISSING
            "chunks": [  # <-- MISSING
                {"snippet": "...", "rerank_score_raw": 0.78},
                {"snippet": "...", "rerank_score_raw": 0.65}
            ]
        }
    ]
}
```

### Issue 4: XLS Processing Tries to Show Chunks But Fails

**Location:** `rag_engine/utils/trace_formatters.py:124-131`

```python
# Code tries to read rerank_score_raw from chunks
def _emit(idx, ch):
    score = ch.get("rerank_score_raw")  # Always None - not in data
    score_str = f" (score: {score:.3f})" if score is not None else ""
    lines.append(f"   - [chunk{idx}:{max_chars}]{score_str} {trimmed}{suffix}")
```

**Result:** Empty score strings in XLS output.

---

## Implementation Plan

### Phase 1: Store Raw Scores (High Priority)

#### 1.1 Modify CrossEncoderReranker to Store Raw Scores

**File:** `rag_engine/retrieval/reranker.py`

**Changes:**
1. Add `rerank_score_raw` to each document's metadata before applying boosts
2. Preserve raw score in scored candidates tuple structure

**Code Changes:**
```python
# Around line 117-139
def rerank(self, query, candidates, top_k, metadata_boost_weights=None, instruction=None):
    pairs = [
        (query, doc.page_content if hasattr(doc, "page_content") else str(doc))
        for doc, _ in candidates
    ]
    raw_scores = self.model.predict(pairs, batch_size=self.batch_size)

    scored = []
    for (doc, _), raw_score in zip(candidates, raw_scores, strict=False):
        # Store raw score in metadata BEFORE boost
        if hasattr(doc, "metadata"):
            doc.metadata["rerank_score_raw"] = float(raw_score)
        elif hasattr(doc, "__dict__"):
            doc.__dict__.setdefault("metadata", {})["rerank_score_raw"] = float(raw_score)

        # Apply metadata boosts (existing logic)
        boost = 0.0
        if metadata_boost_weights and hasattr(doc, "metadata"):
            meta = getattr(doc, "metadata", {})
            if meta.get("tags"):
                boost += metadata_boost_weights.get("tag_match", 0)
            # ... other boosts ...
        boosted_score = float(raw_score) * (1.0 + boost)
        scored.append((doc, boosted_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

#### 1.2 Preserve Raw Score Through Aggregation

**File:** `rag_engine/retrieval/retriever.py`

**Changes:**
1. When grouping chunks by article, track both max boosted score and max raw score
2. Store both in article metadata

**Code Changes:**
```python
# Around line 245-255
articles_map: dict[str, tuple[list[Any], float, float]] = defaultdict(
    lambda: ([], -float("inf"), -float("inf"))
)
for doc, boosted_score in scored_candidates:
    metadata = getattr(doc, "metadata", None) or {}
    raw_kb_id = metadata.get("kbId", "")
    raw_score = metadata.get("rerank_score_raw", -float("inf"))

    if raw_kb_id:
        kb_id = extract_numeric_kbid(raw_kb_id) or str(raw_kb_id)
        chunks, best_boosted, best_raw = articles_map[kb_id]
        chunks.append(doc)
        articles_map[kb_id] = (
            chunks,
            max(best_boosted, boosted_score),
            max(best_raw, raw_score)
        )
```

```python
# Around line 300-305
article.metadata["rerank_score"] = max_boosted_score
article.metadata["rerank_score_raw"] = max_raw_score  # ADD THIS LINE
articles.append(article)
```

---

### Phase 2: Serialize Chunks (Medium Priority)

#### 2.1 Include Chunks in Article Serialization

**File:** `rag_engine/tools/retrieve_context.py`

**Changes:**
1. Add `chunks` field to article serialization
2. Include raw score and snippet for each chunk

**Code Changes:**
```python
# Around line 142-176
def _format_articles_to_json(articles, query, top_k):
    articles_data = []
    for article in articles:
        # ... existing fields ...

        # Include chunks with raw scores
        chunks_data = []
        for chunk in getattr(article, "matched_chunks", []) or []:
            chunk_meta = getattr(chunk, "metadata", {}) or {}
            chunks_data.append({
                "kb_id": chunk_meta.get("kbId", article.kb_id),
                "snippet": getattr(chunk, "page_content", "")[:200],  # First 200 chars
                "rerank_score_raw": chunk_meta.get("rerank_score_raw"),
            })

        articles_data.append({
            "kb_id": article.kb_id,
            "title": title,
            "url": url,
            "content": article.content,
            "metadata": dict(article.metadata),
            "chunks": chunks_data,  # ADD THIS
        })

    return json.dumps({
        "articles": articles_data,
        "metadata": {
            "query": query,
            "top_k_requested": top_k,
            "articles_count": len(articles_data),
            "has_results": len(articles_data) > 0,
        },
    }, ensure_ascii=False)
```

---

### Phase 3: Build Query Traces (Medium Priority)

#### 3.1 Populate Query Traces with Confidence Data

**File:** `rag_engine/retrieval/retriever.py` or `rag_engine/tools/retrieve_context.py`

**Changes:**
1. Call `compute_retrieval_confidence()` after reranking
2. Build query trace structure with confidence metrics

**Code Changes:**
```python
# After reranking, before returning
from rag_engine.retrieval.confidence import compute_retrieval_confidence

# Prepare scored chunks for confidence calculation
scored_for_confidence = [(doc, score) for doc, score in scored_candidates]
confidence = compute_retrieval_confidence(
    scored_for_confidence,
    relevance_threshold=settings.rerank_score_threshold or 0.5,
    mean_top_k=5
)

# Build query trace
query_trace = {
    "query": query,
    "confidence": confidence,
    "articles": [
        {
            "kb_id": article.kb_id,
            "title": article.metadata.get("title", article.kb_id),
            "rerank_score": article.metadata.get("rerank_score"),
            "rerank_score_raw": article.metadata.get("rerank_score_raw"),
            "chunks": [
                {
                    "snippet": getattr(chunk, "page_content", "")[:200],
                    "rerank_score_raw": getattr(chunk, "metadata", {}).get("rerank_score_raw"),
                }
                for chunk in getattr(article, "matched_chunks", []) or []
            ]
        }
        for article in articles
    ]
}
```

---

### Phase 4: Display Both Scores (Low Priority)

#### 4.1 Update Article Display in UI

**File:** `rag_engine/api/app.py`

**Changes:**
1. Display both `rerank_score` and `rerank_score_raw` in article list

**Code Changes:**
```python
# Around line 130-145
# Current display
f"{meta.get('rerank_score', 0):.2f}"

# New display
rerank_score = meta.get("rerank_score", 0)
rerank_score_raw = meta.get("rerank_score_raw")
if rerank_score_raw is not None:
    score_display = f"{rerank_score:.2f} (raw: {rerank_score_raw:.3f})"
else:
    score_display = f"{rerank_score:.2f}"
```

#### 4.2 Update XLS Chunk Display

**File:** `rag_engine/utils/trace_formatters.py`

**Changes:**
1. The code at line 129 already tries to show `rerank_score_raw`
2. Once Phase 2 is complete, chunks will have the data

**Expected Output After Fix:**
```
Чанк 1: ...текст чанка... (score: 0.780)
Чанк 2: ...текст чанка... (score: 0.650)
```

---

## Files to Modify

| File | Phase | Change Type |
|------|-------|-------------|
| `rag_engine/retrieval/reranker.py` | Phase 1 | Store raw scores in chunk metadata |
| `rag_engine/retrieval/retriever.py` | Phase 1 | Preserve raw score through aggregation |
| `rag_engine/tools/retrieve_context.py` | Phase 2 | Serialize chunks with raw scores |
| `rag_engine/utils/trace_formatters.py` | Phase 2/4 | Display raw scores (code exists, needs data) |
| `rag_engine/api/app.py` | Phase 4 | Update UI to show both scores |
| `rag_engine/config/settings.py` | Optional | Add confidence threshold config |
| `.env-example` | Optional | Document new config options |

---

## Testing Plan

### Unit Tests

1. **test_reranker_raw_score.py**
   - Test that raw score is stored before boost
   - Test that raw score is preserved through aggregation

2. **test_chunk_serialization.py**
   - Test that chunks are included in JSON output
   - Test that raw scores are present in serialized chunks

3. **test_query_traces.py**
   - Test that confidence is computed per query
   - Test that trace structure matches expected format

### Integration Tests

1. **test_scoring_pipeline.py**
   - End-to-end test from query to final display
   - Verify all three scores are present and correct

2. **test_xls_output.py**
   - Run `process_requests_xlsx.py` on sample data
   - Verify chunks column shows raw scores

---

## Migration Strategy

### Backward Compatibility

- All changes are ADDITIVE - existing code continues to work
- `rerank_score` remains the primary ranking signal
- New `rerank_score_raw` is optional (None if not available)

### Gradual Rollout

1. **Phase 1:** Internal testing only, no user-facing changes
2. **Phase 2:** XLS output shows raw scores (behind feature flag)
3. **Phase 3:** Query traces fully populated
4. **Phase 4:** UI displays both scores

---

## Success Criteria

1. ✅ `rerank_score_raw` is stored in chunk metadata
2. ✅ `rerank_score_raw` is preserved through article aggregation
3. ✅ Chunks are serialized in `_format_articles_to_json()`
4. ✅ XLS "Чанки" column shows per-chunk raw scores
5. ✅ UI displays both boosted and raw scores
6. ✅ Query traces include confidence metrics

---

## Timeline Estimate

| Phase | Effort | Duration |
|-------|--------|----------|
| Phase 1: Store Raw Scores | 2-3 hours | 1 day |
| Phase 2: Serialize Chunks | 2-3 hours | 1 day |
| Phase 3: Build Query Traces | 3-4 hours | 2 days |
| Phase 4: Display Both Scores | 2-3 hours | 1 day |
| Testing | 4-6 hours | 2 days |
| **Total** | **13-19 hours** | **~1 week** |

---

## Open Questions

1. **Should raw scores be used for ranking instead of boosted scores?**
   - Currently: `rerank_score` (boosted) is used for ranking
   - Alternative: Use `rerank_score_raw` for pure semantic relevance
   - Recommendation: Keep boosted for ranking, show raw for transparency

2. **Should we expose raw scores in the public API?**
   - Internal tools already expect them
   - External API consumers might need documentation

3. **Should confidence threshold use raw or boosted scores?**
   - Recommendation: Use raw scores for consistency with training/evaluation

---

## References

- `rag_engine/retrieval/reranker.py` - CrossEncoder implementation
- `rag_engine/retrieval/retriever.py` - Aggregation logic
- `rag_engine/retrieval/confidence.py` - Confidence calculation
- `rag_engine/tools/retrieve_context.py` - JSON serialization
- `rag_engine/utils/trace_formatters.py` - XLS output formatting
- `rag_engine/scripts/process_requests_xlsx.py` - XLS batch processing
