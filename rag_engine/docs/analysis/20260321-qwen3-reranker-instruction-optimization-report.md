# Reranker Instruction Research - Final Report

**Date:** 2026-03-21  
**Status:** Complete

## Executive Summary

This research tested **30+ instruction variants** across three retrieval methods (semantic, keyword, random) to find the optimal instruction for Qwen3-Reranker on a Russian/English technical documentation corpus (Comindware Platform, 8000+ docs).

### Critical Findings

1. **INFORMAL RUSSIAN WINS**: "Найди" outperforms "Найдите" by +2.5%
2. **EMBEDDING INSTRUCTIONS MATTER**: Adding instruction boosts score by 58%
3. **RUSSIAN INSTRUCTIONS OUTPERFORM ENGLISH**: For Russian corpus, contrary to Qwen3 docs recommendation

---

## Round 1 Results (SEMANTIC - 13 instructions)

| Rank | Instruction | Score | EN | RU | Type |
|------|-------------|-------|-----|-----|------|
| 1 | `ru_concise` | **0.5533** | 0.6756 | 0.5310 | Russian |
| 2 | `baseline_default` | 0.5313 | 0.6360 | 0.5141 | English |
| 3 | `en_concise` | 0.5047 | 0.6466 | 0.4750 | English |
| 4 | `ru_integration` | 0.4818 | 0.5591 | 0.4622 | Russian |
| 5 | `bilingual_integration` | 0.4694 | 0.5715 | 0.4424 | Bilingual |

---

## Round 2 Results (SEMANTIC - 16 new variants)

| Rank | Instruction | Score | EN | RU | Type |
|------|-------------|-------|-----|-----|------|
| **1** | `ru_informal_concise` | **0.5670** | 0.6862 | 0.5447 | **Informal Russian** |
| 2 | `v1_best_ru_concise` | 0.5533 | 0.6756 | 0.5310 | Formal Russian |
| 3 | `v1_best_baseline` | 0.5313 | 0.6360 | 0.5141 | English |
| 4 | `ru_informal_integration` | 0.5114 | 0.5837 | 0.4879 | Informal Russian |
| 5 | `qwen_like_ru` | 0.5064 | 0.6182 | 0.4827 | Qwen-like |
| 6 | `bilingual_informal` | 0.5053 | 0.6591 | 0.4728 | Bilingual Informal |

**Key Finding:** Informal Russian ("Найди") beats formal ("Найдите") by **+0.0137 points (+2.5%)**

---

## Embedding vs Reranker Instruction Investigation

### Critical Discovery

| Combination | Embedding | Reranker | Score | vs Baseline |
|-------------|-----------|----------|-------|-------------|
| `current_default` | **None** | baseline | 0.5313 | baseline |
| `both_baseline` | baseline | baseline | **0.8414** | **+58%** |

**Qwen3-Embedding expects instruction format:** `Instruct: {task}\nQuery: {query}`

Current cmw-rag sends raw queries to embedding, missing +58% potential improvement!

---

## KEYWORD Results

| Rank | Instruction | Score | EN | RU |
|------|-------------|-------|-----|-----|
| 1 | `en_platform` | 0.3984 | 0.3699 | 0.4030 |
| 2 | `en_context_platform` | 0.3771 | 0.3977 | 0.3700 |
| 3 | `ru_platform` | 0.3133 | 0.3703 | 0.3067 |
| 4 | `baseline_default` | 0.2355 | 0.2808 | 0.2430 |

**Note:** Keyword search scores lower overall. Best is English platform-specific.

---

## Recommendations

### For models.yaml

```yaml
# Qwen3-Embedding-8B
Qwen/Qwen3-Embedding-8B:
  type: embedding
  default_instruction: "Найди релевантную техническую документацию"
  
# Qwen3-Reranker-0.6B  
Qwen/Qwen3-Reranker-0.6B:
  type: reranker
  default_instruction: "Найди релевантную техническую документацию"
```

### For Embedding Client Code

```python
# BEFORE (current - wrong)
embedding = get_embedding(query)  # Raw query

# AFTER (correct)
def get_detailed_instruct(task: str, query: str) -> str:
    return f'Instruct: {task}\nQuery: {query}'

embedding = get_embedding(get_detailed_instruct(instruction, query))
```

### Best Practices

1. **Use SAME instruction for both embedding and reranker**
2. **Use informal Russian for Russian corpus** ("Найди" not "Найдите")
3. **Keep instructions concise** - short beats verbose
4. **Match instruction language to corpus language** (Russian for Russian docs)

---

## Contradiction with Qwen3 Docs

Qwen3 documentation states:
> "In multilingual contexts, we advise users to write their instructions in English"

**Our findings contradict this:**
- Russian instructions outperform English for Russian corpus
- `ru_informal_concise` (0.5670) > `en_concise` (0.5047)
- `ru_concise` (0.5533) > `baseline_default` (0.5313)

**Hypothesis:** Qwen3 was trained with multilingual instructions but the semantic match between instruction language and corpus improves relevance scoring.

---

## Files Generated

- `20260321-reranker-dataset-complete.json` - Pre-fetched documents for 52 questions
- `reranker_fast_results_20260321_153921.json` - Round 1 results
- `reranker_extended_results_20260321_190713.json` - Round 2 results
- `embed_rerank_investigation_20260321_191906.json` - Embedding vs reranker results
- `reranker_benchmark_final_20260321_160241.md` - Round 1 report

---

## Methodology

1. **Questions:** 52 realistic support queries (38 Russian, 8 Mixed, 6 English)
2. **Documents:** 5 candidates per question from ChromaDB
3. **Retrieval Methods:**
   - SEMANTIC: Qwen3-Embedding-8B via OpenRouter
   - KEYWORD: Word matching (BM25-like)
   - RANDOM: Baseline
4. **Reranker:** Qwen3-Reranker-0.6B via mosec server
5. **Metrics:** Average relevance score, by language breakdown

---

## Next Steps

1. ✅ Update `models.yaml` with optimal instructions
2. ✅ Update embedding client to use instruction format
3. ⏳ Complete embedding instruction variants benchmark
4. ⏳ Test in production pipeline
5. ⏳ Document findings in architecture docs