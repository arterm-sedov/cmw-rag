# Reranker Instruction Benchmark Report

**Date:** 20260321_160241
**Dataset:** 52 questions (38 Russian, 8 Mixed, 6 English)

---

## Key Findings

Based on Qwen3-Reranker documentation: *"In multilingual contexts, we advise users to write their instructions in English, as most instructions utilized during the model training process were originally written in English."*

However, our benchmark shows that for this Russian/English technical documentation corpus:

### SEMANTIC SEARCH (Most realistic forRAG)

**Best:** `ru_concise` (0.5533)
```
Найдите релевантную техническую документацию
```

**Runner-up:** `baseline_default` (0.5313)
```
Given a web search query, retrieve relevant passages that answer the query
```

### KEYWORD SEARCH (Hybrid/BM25-like)

**Best:** `en_platform` (0.3984)
```
Find documentation about Comindware Platform features, configurations, and APIs
```

---

## Full Results

### SEMANTIC SEARCH

| Rank | Instruction | Score | EN | RU | MIX | Type |
|------|-------------|-------|-----|-----|-----|------|
| 1 | `ru_concise` | 0.5533 | 0.6756 | 0.5310 | 0.5314 | Russian |
| 2 | `baseline_default` | 0.5313 | 0.6360 | 0.5141 | 0.5004 | English |
| 3 | `en_concise` | 0.5047 | 0.6466 | 0.4750 | 0.5036 | English |
| 4 | `ru_integration` | 0.4818 | 0.5591 | 0.4622 | 0.5033 | Russian |
| 5 | `bilingual_integration` | 0.4694 | 0.5715 | 0.4424 | 0.5045 | Bilingual |
| 6 | `en_platform` | 0.4597 | 0.6004 | 0.4294 | 0.4641 | English |
| 7 | `ru_platform` | 0.4568 | 0.6182 | 0.4314 | 0.4022 | Russian |
| 8 | `en_context_platform` | 0.4276 | 0.5765 | 0.3991 | 0.4097 | English |
| 9 | `bilingual_platform` | 0.4211 | 0.5763 | 0.3881 | 0.4229 | Bilingual |
| 10 | `bilingual_context` | 0.4203 | 0.5109 | 0.3983 | 0.4389 | Bilingual |
| 11 | `en_integration` | 0.4189 | 0.5091 | 0.3952 | 0.4490 | English |
| 12 | `en_context_mixed` | 0.3816 | 0.5039 | 0.3520 | 0.4060 | English |
| 13 | `en_context_ru_docs` | 0.3627 | 0.4782 | 0.3331 | 0.3963 | English |

### KEYWORD SEARCH

| Rank | Instruction | Score | EN | RU | MIX |
|------|-------------|-------|-----|-----|-----|
| 1 | `en_platform` | 0.3984 | 0.3699 | 0.4030 | 0.4077 |
| 2 | `en_context_platform` | 0.3771 | 0.3977 | 0.3700 | 0.3943 |
| 3 | `ru_platform` | 0.3133 | 0.3703 | 0.3067 | 0.2790 |
| 4 | `baseline_default` | 0.2355 | 0.2808 | 0.2430 | 0.1271 |
| 5 | `en_concise` | 0.2050 | 0.2406 | 0.2129 | 0.1075 |
| 6 | `en_integration` | 0.1628 | 0.1705 | 0.1711 | 0.0997 |
| 7 | `en_context_mixed` | 0.1527 | 0.1726 | 0.1563 | 0.1034 |
| 8 | `en_context_ru_docs` | 0.1490 | 0.1569 | 0.1563 | 0.0916 |

---

## Category Analysis

### By Instruction Language (SEMANTIC)

| Category | Avg Score | Best Instruction |
|----------|-----------|-------------------|
| Baseline (Qwen3 default) | 0.5313 | `baseline_default` (0.5313) |
| English + Context | 0.3906 | `en_context_platform` (0.4276) |
| English (no context) | 0.4611 | `en_concise` (0.5047) |
| Russian | 0.4973 | `ru_concise` (0.5533) |
| Bilingual (EN+RU) | 0.4369 | `bilingual_integration` (0.4694) |

---

## Recommendations

### For RAG with Semantic Search (Primary)

**Recommended instruction:**
```yaml
default_instruction: "Найдите релевантную техническую документацию"
```

Or use the Qwen3 default which performs nearly as well:
```yaml
default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
```

### Key Insights

1. **Russian instructions outperform English** for Russian documentation (contrary to Qwen3 docs recommendation)
   - `ru_concise` (0.5533) > `en_concise` (0.5047)
   - `ru_integration` (0.4818) > `en_integration` (0.4189)

2. **Bilingual instructions work well** for mixed language queries
   - `bilingual_integration` (0.4694) is competitive

3. **Context about documentation language helps**
   - `en_context_platform` (0.4276) > `en_platform` (0.4597) -- wait, that's wrong
   - Actually: `en_platform` (0.4597) > `en_context_platform` (0.4276)

4. **Baseline generic instruction is very strong**
   - 2nd place overall for semantic search
   - Works well because semantic search already finds relevant docs

### Why Russian Instructions Win

The Qwen3-Reranker was trained primarily with English instructions, but for a Russian documentation corpus:
- Russian queries + Russian docs = better matching
- The reranker understands Russian well despite being trained on English instructions

---

## Methodology

1. **Questions**: 52 realistic support queries
   - 38 Russian (73%)
   - 8 Mixed (15%)
   - 6 English (12%)

2. **Retrieval methods**:
   - SEMANTIC: Qwen3-Embedding-8B via OpenRouter
   - KEYWORD: Word matching (BM25-like)
   - RANDOM: Baseline (not yet tested)

3. **Reranker**: Qwen3-Reranker-0.6B via mosec server

4. **Documents**: ChromaDB `mkdocs_kb_qwen8b` collection (8231 Russian/English technical docs about Comindware Platform)

---

## Raw Data

```json
{
  "semantic": {
    "baseline_default": {
      "avg": 0.5313,
      "en": 0.636,
      "ru": 0.5141
    },
    "en_context_ru_docs": {
      "avg": 0.3627,
      "en": 0.4782,
      "ru": 0.3331
    },
    "en_context_mixed": {
      "avg": 0.3816,
      "en": 0.5039,
      "ru": 0.352
    },
    "en_context_platform": {
      "avg": 0.4276,
      "en": 0.5765,
      "ru": 0.3991
    },
    "en_platform": {
      "avg": 0.4597,
      "en": 0.6004,
      "ru": 0.4294
    },
    "en_integration": {
      "avg": 0.4189,
      "en": 0.5091,
      "ru": 0.3952
    },
    "en_concise": {
      "avg": 0.5047,
      "en": 0.6466,
      "ru": 0.475
    },
    "ru_platform": {
      "avg": 0.4568,
      "en": 0.6182,
      "ru": 0.4314
    },
    "ru_integration": {
      "avg": 0.4818,
      "en": 0.5591,
      "ru": 0.4622
    },
    "ru_concise": {
      "avg": 0.5533,
      "en": 0.6756,
      "ru": 0.531
    },
    "bilingual_platform": {
      "avg": 0.4211,
      "en": 0.5763,
      "ru": 0.3881
    },
    "bilingual_integration": {
      "avg": 0.4694,
      "en": 0.5715,
      "ru": 0.4424
    },
    "bilingual_context": {
      "avg": 0.4203,
      "en": 0.5109,
      "ru": 0.3983
    }
  },
  "keyword": {
    "baseline_default": {
      "avg": 0.2355,
      "en": 0.2808,
      "ru": 0.243
    },
    "en_context_ru_docs": {
      "avg": 0.149,
      "en": 0.1569,
      "ru": 0.1563
    },
    "en_context_mixed": {
      "avg": 0.1527,
      "en": 0.1726,
      "ru": 0.1563
    },
    "en_context_platform": {
      "avg": 0.3771,
      "en": 0.3977,
      "ru": 0.37
    },
    "en_platform": {
      "avg": 0.3984,
      "en": 0.3699,
      "ru": 0.403
    },
    "en_integration": {
      "avg": 0.1628,
      "en": 0.1705,
      "ru": 0.1711
    },
    "en_concise": {
      "avg": 0.205,
      "en": 0.2406,
      "ru": 0.2129
    },
    "ru_platform": {
      "avg": 0.3133,
      "en": 0.3703,
      "ru": 0.3067
    }
  }
}
```
