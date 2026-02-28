# Plan: Reverse Normalized Rank

**Status:** NOT IMPLEMENTED (cancelled after analysis)
**Date:** 2026-02-18
**Reason:** "normalized_rank" is semantically a rank (0=1st place), not a relevancy score. Reversing would create confusion when displayed alongside rerank_score.

---

## Goal (Cancelled)

Make `normalized_rank` intuitive: **higher values = better relevance** (matches rerank_score pattern)

## Why Cancelled

The field is named `normalized_rank`, which semantically means:
- Rank 0 = 1st place (best)
- Rank 1 = last place (worst)

This is the standard interpretation for "rank" in competitive contexts (Olympics, leaderboards, etc.).

Displaying "normalized_rank=0.000" next to "Rel=0.54" would be confusing because:
- Users see "0.000" and think "zero relevance"
- But it actually means "1st place in ranking"

**Decision:** Keep current implementation (0=best, 1=worst) as it's semantically correct for "rank".

---

## Complete File Change List (NOT IMPLEMENTED)

### 1. rag_engine/retrieval/retriever.py

**Line 311:** Invert the calculation
```python
# BEFORE:
article.metadata["normalized_rank"] = idx / (len(articles) - 1)

# AFTER:
article.metadata["normalized_rank"] = 1.0 - (idx / (len(articles) - 1))
```

**Line 315:** Update edge case for single article
```python
# BEFORE:
articles[0].metadata["normalized_rank"] = 0.0

# AFTER:
articles[0].metadata["normalized_rank"] = 1.0
```

---

### 2. rag_engine/llm/compression.py

**Line 52:** Update default for None (was 1.0=worst, should be 0.0=worst)
```python
# BEFORE:
article.setdefault("metadata", {})["normalized_rank"] = 1.0

# AFTER:
article.setdefault("metadata", {})["normalized_rank"] = 0.0
```

**Line 55:** Update comment
```python
# BEFORE:
# "Higher rank (lower normalized_rank) = higher weight = more budget"

# AFTER:
# "Higher rank (higher normalized_rank) = higher weight = more budget"
```

**Line 56:** Update comment
```python
# BEFORE:
# "Formula: weight = 1.0 - (normalized_rank * 0.7) gives range [0.3, 1.0]"

# AFTER:
# "Formula: weight = 0.3 + (normalized_rank * 0.7) gives range [0.3, 1.0]"
```

**Line 57:** Update comment
```python
# BEFORE:
# "Best rank (0.0) gets weight 1.0, worst (1.0) gets weight 0.3"

# AFTER:
# "Best rank (1.0) gets weight 1.0, worst (0.0) gets weight 0.3"
```

**Line 62:** Update default for missing (was 1.0=worst, should be 0.0=worst)
```python
# BEFORE:
normalized_rank = article.get("metadata", {}).get("normalized_rank", 1.0)

# AFTER:
normalized_rank = article.get("metadata", {}).get("normalized_rank", 0.0)
```

**Line 63:** Update comment
```python
# BEFORE:
# "Weight: 1.0 for best rank (0.0), 0.3 for worst rank (1.0)"

# AFTER:
# "Weight: 1.0 for best rank (1.0), 0.3 for worst rank (0.0)"
```

**Line 64:** New formula
```python
# BEFORE:
weight = 1.0 - (normalized_rank * 0.7)

# AFTER:
weight = 0.3 + (normalized_rank * 0.7)
```

**Line 94:** Update default (was 1.0=worst, should be 0.0=worst)
```python
# BEFORE:
key=lambda x: x[1].get("metadata", {}).get("normalized_rank", 1.0)

# AFTER:
key=lambda x: x[1].get("metadata", {}).get("normalized_rank", 0.0)
```

**Line 93:** Add negation for sort stability
```python
# BEFORE:
key=lambda x: x[1].get("metadata", {}).get("normalized_rank", 0.0),

# AFTER:
key=lambda x: -x[1].get("metadata", {}).get("normalized_rank", 0.0),
```
*(Keep `reverse=True` unchanged line 95)*

---

### 3. rag_engine/tests/test_retriever.py

**Lines 259-260:** Update assertions
```python
# BEFORE:
assert articles[0].metadata["normalized_rank"] == 0.0
assert articles[1].metadata["normalized_rank"] == 1.0

# AFTER:
assert articles[0].metadata["normalized_rank"] == 1.0
assert articles[1].metadata["normalized_rank"] == 0.0
```

---

### 4. rag_engine/api/app.py

**Line 415:** Update docstring
```python
# BEFORE:
# "proportionally based on normalized_rank (0.0 = best, 1.0 = worst)."

# AFTER:
# "proportionally based on normalized_rank (1.0 = best, 0.0 = worst)."
```

---

## Summary Statistics

| File | Logic Changes | Comment Updates | Total |
|------|--------------|----------------|-------|
| `retriever.py` | 2 | 0 | 2 |
| `compression.py` | 4 | 5 | 9 |
| `test_retriever.py` | 2 | 0 | 2 |
| `app.py` | 0 | 1 | 1 |
| **TOTAL** | **8** | **6** | **14** |

---

## Robustness Verification

### Edge Cases Tested

| Test | Result | Status |
|------|--------|--------|
| Single article (normalized_rank=1.0) | Weight=1.0 (max budget) | PASS |
| Worst article (normalized_rank=0.0) | Weight=0.3 (min budget) | PASS |
| Missing normalized_rank → default 0.0 | Weight=0.3 (min) | PASS (FIXED) |
| None normalized_rank → default 0.0 | Weight=0.3 (min) | PASS (FIXED) |
| clamped value | Weight in [0.3, 1.0] | PASS |
| Negative clamped | Weight≥0.3 but min_token enforced | PASS |

### Mathematical Correctness

**Proof of formula equivalence:**
```
Old: weight = 1.0 - (n_old × 0.7)  where n_old = position (0=best)
New: weight = 0.3 + (n_new × 0.7)  where n_new = 1.0 - n_old

Substitute n_new = 1.0 - n_old:
weight = 0.3 + ((1.0 - n_old) × 0.7)
      = 0.3 + 0.7 - (n_old × 0.7)
      = 1.0 - (n_old × 0.7)
      = OLD formula ✓
```

### Sort Stability

**Test cases:**
- OLD order (0=best, 1=worst): Sort by -rank → descending → [worst, ..., best] ✓
- NEW order (1=best, 0=worst): Sort by -rank → descending → [worst, ..., best] ✓

**Result:** Both orderings yield correct sort order after negation.

---

## Alternative: Rename Field (Future Consideration)

If the confusion persists, consider renaming the field:
- `normalized_rank` → `rank_position` (emphasizes it's a position, not a score)
- Or add UI label: "Position" instead of "Normalized"

---

## Lessons Learned

1. **Naming matters:** "rank" vs "score" creates different user expectations
2. **Display context:** Adjacent columns (rerank_score=0.54, normalized_rank=0.000) can be confusing
3. **Semantic correctness:** Rank (0=1st) is correct, but may need better UI labeling
4. **User mental models:** Users expect "higher number = better" for scores
