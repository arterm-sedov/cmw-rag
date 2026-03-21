# Model Sizing and API Parameters - Detailed Analysis

**Date:** 2026-03-21
**Repositories:** cmw-mosec, cmw-rag
**Status:** COMPLETED

## Executive Summary

**Dimensions parameter fix completed.** CMW-RAG now sends `dimensions` from config to all embedding endpoints (mosec, infinity, vLLM, OpenRouter) ensuring consistency between config and server response.

---

## Memory and Sizing Deep Dive

### Transformer Memory Model

**Model weights (fixed):**
```
VRAM_weights = params × bytes_per_param
Qwen3-0.6B fp16: ~0.6B × 2 = ~1.2GB
Qwen3-4B fp16: ~4B × 2 = ~8GB
FRIDA fp32: ~1.7B × 4 = ~6.8GB
```

**Inference memory (scales with input):**
```
VRAM_inference = input_embeddings + attention + activations

Input_embeddings = batch_size × seq_len × hidden_dim × bytes_per_param
Attention = batch_size × num_heads × seq_len² × head_dim
Activations = batch_size × seq_len × hidden_dim × layers × bytes_per_param
```

### Critical Insight: `max_length` is NOT Pre-allocation

**How tokenization works:**
```python
# Input: 500 tokens of text
# Config: max_length = 32768

tokens = tokenizer(text, truncation=True, max_length=32768)
# Result: tokens.shape = [batch, 500]  <-- ACTUAL length, not 32768

# Attention matrix
attention = [batch, heads, 500, 500]  # 500², not 32768²
```

**`max_length` is a ceiling, not pre-allocation:**
- Input < max_length → memory scales with actual input
- Input > max_length → truncated to max_length

| Actual Input | max_length | Memory Used |
|--------------|------------|--------------|
| 500 tokens | 32768 | ~500² = 250K elements |
| 2K tokens | 32768 | ~2K² = 4M elements |
| 35K tokens | 32768 | **truncated** → ~32K² = 1B elements |

### Real VRAM Savers

| Technique | Impact | How |
|-----------|--------|-----|
| Smaller model | High | 0.6B vs8B = 20x less weights |
| Smaller batch | High | Linear reduction |
| Shorter documents | High | Attention is O(n²) |
| Smaller `dimensions` (MRL) | Zero | Post-processing, not inference |
| `max_length` config | Low | Only truncates long inputs |

---

## Model-Specific Peculiarities

### Embedding Models

#### FRIDA (ai-forever/FRIDA)
- **Architecture**: T5 encoder-only (not decoder)
- **Context**: 512 tokens max (HF docs confirmed)
- **Pooling**: CLS token (first token)
- **Dtype**: float32 required (fp16 causes precision issues)
- **Prefixes**:
  - Queries: `search_query: {text}`
  - Documents: `search_document: {text}`
- **MRL**: Not supported
- **VRAM**: ~4GB (fp32 weights)

#### Qwen3-Embedding Series
- **Architecture**: Causal LM (GPT-like)
- **Context**: 32K tokens
- **Pooling**: Last token (requires `padding_side='left'`)
- **Dtype**: float16
- **MRL**: Supported (dimensions: 32 to native)
  - 0.6B: native=1024, MRL: [32-1024]
  - 4B: native=2560, MRL: [32-2560]
  - 8B: native=4096, MRL: [32-4096]
- **Instruction format**:
  ```python
  # Queries only (not documents)
  text = f"Instruct: {task}\nQuery: {query}"
  # Example: "Instruct: Retrieve relevant passages\nQuery: What is AI?"
  ```
- **VRAM**:
  - 0.6B: ~2GB
  - 4B: ~12GB
  - 8B: ~22GB

**Key difference**: FRIDA uses encoder-only (bidirectional), Qwen3 uses causal LM (unidirectional). This affects:
- Pooling: CLS vs last_token
- Context: 512 vs 32K
- Architecture: T5 vs GPT-style

### Reranker Models

#### Cross-Encoder (DiTy, BGE-M3)
- **Architecture**: BERT-style encoder
- **Input**: Query + document concatenated
- **Output**: Single relevance score
- **Context**: 512 (DiTy) to 8192 (BGE-M3)
- **No formatting needed**: Raw query/documents

#### LLM Reranker (Qwen3-Reranker, BGE-Gemma)
- **Architecture**: Causal LM
- **Input**: Formatted prompt
- **Output**: Logits for "yes"/"no" tokens
- **Context**: 32K (Qwen3), 1024 (Gemma)
- **Formatting**: Client MUST format with prefix/suffix/prompt
- **Scoring methods**:
  - `softmax`: P(true) = softmax([logit_false, logit_true])[1]
  - `raw_logit`: Score = logit for "true" token

**Qwen3-Reranker format:**
```python
# Query format
query = "<|im_start|>system\nJudge...<|im_end|>\n<|im_start|>user\n{query}<|im_end|>..."

# Document format  
doc = "<|im_start|>system\n...<|im_end|>\n...{document}<|im_end|>\n<|im_start|>assistant\n"
```

### Guard Models (Qwen3Guard)

- **Architecture**: Causal LM
- **Context**: 32K tokens
- **Output**: Generative classification
  ```
  Safety: Safe|Controversial|Unsafe
  Categories: Violent, PII, ...
  Refusal: Yes|No
  ```
- **Max new tokens**: 128 (generation length)
- **Input vs context**: max_length truncates input, max_new_tokens limits generation

---

## Parameter Reference

### `max_length` (All Models)

**Purpose**: Truncate input to prevent OOM on unexpectedly long inputs

**Behavior**:
- Tokenizer: `tokenizer(text, truncation=True, max_length=N)`
- If input >N tokens → truncated to N
- If input < N tokens → no change, memory scales with actual length

**When to set**:
- Config: Set to model's native max (safety ceiling)
- Client override: Use lower value if you want to force truncation

**Does NOT save VRAM** for short inputs (memory scales with actual input).
**DOES save VRAM** for long inputs (truncates before OOM).

**Config values**:
| Model | max_length | Notes |
|-------|------------|-------|
| FRIDA | 512 | Native limit |
| Qwen3-Embedding | 32768 | Native limit |
| DiTy Reranker | 512 | Native limit |
| BGE-M3 Reranker | 8192 | Native limit |
| Qwen3-Reranker | 32768 | Native limit |
| Qwen3Guard | 32768 | Native limit |

### `dimensions` (MRL - Matryoshka Representation Learning)

**Purpose**: Reduce embedding dimension while preserving quality

**Behavior**:
```python
# Server-side truncation
embedding = embedding[:, :dimensions]  # Simple slice
 embedding = F.normalize(embedding, p=2, dim=1)  # Re-normalize
```

**Does NOT save VRAM** (post-processing after inference)
**DOES save storage** (smaller vectors in DB)

**Valid only for Qwen3-Embedding**:
- Must be in range [32, native_dimension]
- Values outside range → 400 error

**OpenAI compatibility**:
```json
{
  "input": ["text"],
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "dimensions": 512
}
```

### `max_new_tokens` (Guard Models Only)

**Purpose**: Limit generation length

**Not client-controllable** - set in config only.

---

## Mosec API Parameters

### `/v1/embeddings`

| Parameter | Required | Client Override | Config Default |
|-----------|----------|-----------------|----------------|
| `input` | Yes | - | - |
| `model` | Yes | - | - |
| `dimensions` | No | Yes (MRL) | Native dimension |
| `max_length` | No | Yes | From YAML |
| `encoding_format` | No | Yes | "float" |

**Recommended**: Client should send `dimensions` from its config for consistency.

### `/v1/rerank` and `/v1/score`

| Parameter | Required | Client Override | Config Default |
|-----------|----------|-----------------|----------------|
| `query` | Yes | - | - |
| `documents` | Yes | - | - |
| `max_length` | No | Yes | From YAML |
| `top_n` | No | Yes | None |

### `/v1/moderate`

| Parameter | Required | Client Override | Config Default |
|-----------|----------|-----------------|----------------|
| `content` | Yes | - | - |
| `max_length` | No | Yes | From YAML |
| `moderation_type` | No | Yes | "prompt" |
| `context` | No | Yes | None |

---

## CMW-RAG Implementation Status

### Current Behavior

```python
# embedder.py - OpenAICompatibleEmbedder._embed_local()
payload = {"input": text, "model": self.config.model}
# dimensions: NOT sent (server uses native)
# max_length: NOT sent (server uses config default)
```

```python
# reranker.py - RerankerAdapter._get_scores()
payload = {"query": formatted_query, "documents": formatted_docs}
# max_length: NOT sent (server uses config default)
```

### Issue: Dimensions Inconsistency

**Problem**: cmw-rag has `dimensions` in its model config, but doesn't send it to mosec.

```python
# embedder.py line 494-496
dimensions_raw = model_data.get("dimensions")
assert dimensions_raw is not None, f"Missing dimensions for model: {model_slug}"
dimensions: int = int(dimensions_raw)

# config line 550-563
config = OpenAIEmbeddingConfig(
    dimensions=dimensions,  # Stored in config
    ...
)
```

**But never sent to server:**
```python
# _embed_local() doesn't include dimensions
resp = requests.post(
    self.config.endpoint,
    json={"input": text, "model": self.config.model},
    # Missing: "dimensions": self.config.dimensions
)
```

**Impact**:
- If mosec config has `dimensions: 1024` and cmw-rag has `dimensions: 512`
- mosec returns 1024-dim vectors
- cmw-rag expects 512-dim vectors
- **Mismatch or wasted storage**

### Fix Applied (Commit 2a675d0)

**All embedding paths now send dimensions:**

```python
# Local HTTP (mosec, infinity)
payload = {"input": text, "model": self.config.model}
if self.config.dimensions:
    payload["dimensions"] = self.config.dimensions

# Remote OpenAI SDK (OpenRouter, vLLM)
params = {"model": self.config.model, "input": text}
if self.config.dimensions:
    params["dimensions"] = self.config.dimensions
response = self.client.embeddings.create(**params)
```

**Updated methods:**
- `_embed_local()` - single query
- `_embed_documents_local()` - batch documents
- `_embed_remote()` - single query via SDK
- `_embed_documents_remote()` - batch via SDK
- All fallback paths within these methods

**This ensures**:
- Mosec truncates to expected dimension
- Config in both places matches
- MRL available if configured

---

## Action Items

### Completed

| Item | Repository | Status |
|------|------------|--------|
| Send `dimensions` from config | cmw-rag | DONE (commit 2a675d0) |

### Low Priority (Optional)

| Item | Repository | Benefit |
|------|------------|---------|
| Send `max_length` from config | cmw-rag | Consistency only |
| Document memory scaling | cmw-mosec | Better user understanding |

---

## Summary

1. **`max_length` does NOT pre-allocate memory** - it's a truncation ceiling
2. **VRAM scales with actual input length**, not max_length config
3. **`dimensions` (MRL) saves storage**, not VRAM (post-processing)
4. **cmw-rag NOW sends `dimensions` from config** - fix applied to all embedding paths:
   - Local HTTP (mosec, infinity)
   - Remote OpenAI SDK (OpenRouter, vLLM)
5. **`max_length` client override** only useful for forcing truncation of long inputs