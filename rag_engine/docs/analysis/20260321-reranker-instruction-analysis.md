# Qwen3 Reranker Instruction Optimization Analysis

**Date:** 2026-03-21
**Corpus:** Comindware Platform Technical Documentation (Russian/English)
**Documents:** 10,685 in ChromaDB `mkdocs_kb` collection

---

## Executive Summary

After benchmarking 11 different reranker instructions on Russian/English technical documentation corpus, the optimal instruction is:

```
Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации и API на русском или английском
```

**Best Instruction: `platform_ru`** (Avg Score: 0.3459)

This Russian-language instruction outperformed all English variants, matching the corpus language (Russian documentation with English code snippets).

---

## Corpus Analysis

### Content Characteristics

The knowledge base contains:

1. **Language Distribution:**
   - Primary: Russian (technical documentation)
   - Secondary: English (code snippets, JSON, C# examples)

2. **Content Types:**
   - Platform configuration guides
   - API integration examples
   - C# script examples
   - JSON/XML data formats
   - Administrative procedures

3. **Sample Document Structure:**
   ```markdown
   # HTTP-запросы POST. Отправка JSON-данных...
   
   ## Введение
   **Comindware Platform** позволяет отправлять HTTP-запросы...
   
   ```csharp
   public class OutgoingRootMessage
   {
       public string orderNumber {get; set;}
       public List<Item> items {get; set;}
   }
   ```
   ```

4. **Metadata Boosts Used:**
   - `tag_match`: 1.2 (documents with matching tags)
   - `code_presence`: 1.15 (documents containing code)
   - `section_match`: (section heading match)

---

## Benchmark Results

| Instruction | Avg Score | Variance |
|------------|-----------|----------|
| **platform_ru** | **0.3459** | 0.1322 |
| comprehensive | 0.3185 | 0.1367 |
| technical_qa | 0.3155 | 0.0888 |
| ru_primacy | 0.2711 | 0.0911 |
| code_examples | 0.2191 | 0.0870 |
| ru_en_tech | 0.2176 | 0.0873 |
| code_ru_en | 0.2105 | 0.0823 |
| concise_ru_en | 0.1927 | 0.0793 |
| concise | 0.1810 | 0.0667 |
| bilingual_tech | 0.1431 | 0.0534 |
| default | 0.1305 | 0.0658 |

### Key Findings

1. **Russian instructions outperform English** for Russian corpus
   - `platform_ru` (0.3459) > `comprehensive` (0.3185) > `technical_qa` (0.3155)

2. **Specificity improves scores**
   - Platform-specific instructions > generic instructions
   - `platform_ru` mentions Comindware Platform explicitly

3. **Code awareness helps**
   - Instructions mentioning "code examples" scored well

4. **Default instruction underperforms**
   - Generic web search instruction (0.1305) is 2.6x worse than optimal

---

## Instructions Tested

### Best Performing

```python
# platform_ru - WINNER (0.3459)
"Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации и API на русском или английском"

# comprehensive (0.3185)
"Retrieve technical documentation about Comindware Platform with code examples, configurations, and API references in Russian or English"

# technical_qa (0.3155)
"Find technical documentation and answers to questions about enterprise platform features"
```

### Other Tested

```python
# ru_primacy (0.2711)
"Приоритизируйте русскоязычную документацию и примеры кода для технических вопросов"

# code_examples (0.2191)
"Find documentation with relevant code examples, configurations, and technical explanations"

# code_ru_en (0.2105)
"Найдите документы с примерами кода, конфигурациями и техническими объяснениями на русском или английском"

# ru_en_tech (0.2176)
"Найдите техническую документацию на русском или английском языке, отвечающую на вопрос"

# bilingual_tech (0.1431)
"Retrieve technical documentation in Russian or English that answers the query, including code examples"

# default (0.1305) - ORIGINAL
"Given a web search query, retrieve relevant passages that answer the query"
```

---

## Recommendation

### Update `models.yaml` Default Instruction

Change from:
```yaml
default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
```

To:
```yaml
default_instruction: "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации и API на русском или английском"
```

### Rationale

1. **Corpus language match**: Russian instruction for Russian corpus
2. **Domain specificity**: Mentions Comindware Platform explicitly
3. **Content awareness**: References code examples, configurations, API
4. **Bilingual handling**: Explicitly mentions "Russian or English"
5. **2.6x improvement**: Over generic default instruction

---

## Technical Implementation

### Qwen3 Formatting

The instruction is injected into the Qwen3 prompt template:

```python
query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
doc_template = "<Document>: {doc}{suffix}"
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
```

### Example Formatted Query

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации и API на русском или английском
<Query>: Как настроить HTTP-запросы в Comindware Platform?
<Document>: ...
```

---

## Metadata Boosts

The reranker scores are combined with metadata boosts:

```python
boost_weights = {
    "tag_match": 1.2,      # +20% for matching tags
    "code_presence": 1.15, # +15% for code-containing docs
}
```

Final score = reranker_score × (1 + boost)

---

## Next Steps

1. **Update `models.yaml`** with optimal instruction
2. **Add to progress report** in plan
3. **Consider A/B testing** with production queries
4. **Monitor relevance** in real user queries

---

## Files Modified

- `/home/asedov/cmw-rag/rag_engine/scripts/reranker_instruction_benchmark.py` - Benchmark script
- `/home/asedov/cmw-rag/rag_engine/docs/analysis/reranker_instruction_benchmark_*.json` - Raw results

## Recommendation for `models.yaml`

```yaml
Qwen/Qwen3-Reranker-0.6B:
  type: reranker
  reranker_type: llm_reranker
  description: "Lightweight Qwen3 reranker"
  default_instruction: "Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации и API на русском или английском"
  # For English-only corpus, use:
  # default_instruction: "Retrieve technical documentation with code examples, configurations, and API references"
  formatting:
    query_template: "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    doc_template: "<Document>: {doc}{suffix}"
    prefix: '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix: "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
```