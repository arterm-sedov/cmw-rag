# Deployment Architecture

## Host: 10.9.7.7

### Running Services

| Service | Port | Process | Status |
|---------|------|---------|--------|
| RAG Gradio UI | 7860 | `python rag_engine/api/app.py` | Active |
| CMW-Mosec server | 7998 | `cmw_mosec.v2.dynamic_server` | Active |
| ChromaDB | 8000 | `chroma run --host 0.0.0.0 --port 8000` | Active |

### Not Running on this Host

- vLLM — not running (no process found)
- CMW Platform Agent — not running (no process found)

---

## CMW-Mosec (port 7998)

Single Mosec process serving multiple model types via env-conditional route registration:

| Route | Worker | Model | Verified |
|-------|--------|-------|----------|
| `POST /v1/embeddings` | EmbeddingWorkerV2 | `Qwen/Qwen3-Embedding-0.6B` | ✅ Returns embeddings |
| `POST /v1/score` | ScoreWorkerV2 | `Qwen/Qwen3-Reranker-0.6B` | ✅ Returns scores |
| `POST /v1/rerank` | RerankWorkerV2 | `Qwen/Qwen3-Reranker-0.6B` | ❌ Returns error |
| `POST /v1/moderate` | GuardWorkerV2 | `Qwen/Qwen3Guard-Gen-0.6B` | ✅ Returns moderation |

Routes are registered dynamically based on env vars:
- `ACTIVE_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B` → enables `/v1/embeddings`
- `ACTIVE_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B` → enables `/v1/score`, `/v1/rerank`
- `ACTIVE_GUARD_MODEL=Qwen/Qwen3Guard-Gen-0.6B` → enables `/v1/moderate`

---

## ChromaDB (port 8000)

Vector store for RAG retrieval.

```
chroma run --host 0.0.0.0 --port 8000 --path ~/cmw-rag/data/chromadb_data
```

Active collection: `mkdocs_kb_qwen06b_linux_qwen_default_instruct_chunk_768` (1024 dim)

---

## RAG Gradio UI (port 7860)

Entry: `rag_engine/api/app.py`

### Service Dependencies

```
┌─────────────────────────────────────────────────────────┐
│                     RAG Gradio UI                        │
│                     10.9.7.7:7860                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Embeddings ──────────→ Mosec /v1/embeddings (:7998)     │
│  Reranker   ──────────→ Mosec /v1/score      (:7998)     │
│  Guard      ──────────→ Mosec /v1/moderate    (:7998)     │
│  Vector DB  ──────────→ ChromaDB              (:8000)     │
│  LLM        ──────────→ OpenRouter API (external)        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### LLM Configuration

| Setting | Value |
|---------|-------|
| Provider | openrouter |
| Model | `deepseek/deepseek-v4-pro` |
| Endpoint | `https://openrouter.ai/api/v1` |
| Reasoning | Enabled, max 1200 tokens |

### Embedding Configuration

| Setting | Value |
|---------|-------|
| Provider | mosec |
| Model | `Qwen/Qwen3-Embedding-0.6B` |
| Endpoint | `http://localhost:7998/v1/embeddings` |

### Reranker Configuration

| Setting | Value |
|---------|-------|
| Provider | mosec |
| Model | `Qwen/Qwen3-Reranker-0.6B` |
| Endpoint (score) | `http://localhost:7998/v1/score` |

### Guard Configuration

| Setting | Value |
|---------|-------|
| Provider | mosec |
| Threshold | unsafe |
| Endpoint | `http://localhost:7998/v1/moderate` |

### Retrieval Configuration

| Setting | Value |
|---------|-------|
| Top-K retrieve | 20 |
| Top-K rerank | 10 |
| Chunk size | 768 |
| Chunk overlap | 75 |
| Multi-query | Enabled (max 4 segments, 448 tokens) |
| Query decomposition | Enabled (max 4 subqueries) |

---

## Connection Map

```
User Browser
    │
    ├──→ 10.9.7.7:7860 (RAG UI)
    │       │
    │       ├──→ localhost:7998  (Mosec — embeddings, score, moderate)
    │       ├──→ localhost:8000  (ChromaDB)
    │       └──→ api.openrouter.ai (LLM)
    │
    └──→ 10.9.7.7:7860 (Platform Agent UI — if deployed)
                │
                ├──→ api.openrouter.ai (LLM)
                └──→ support.comindware.com (CMW Platform API)
```

---

## Notes

- Mosec `/v1/rerank` endpoint is registered but returns inference error at runtime — RAG uses `/v1/score` instead, which works correctly.
- vLLM configuration exists (`cmw-vllm/.env`, `VLLM_PORT=8000`, model `openai/gpt-oss-20b`) but the server is not currently running. Port 8000 on this host is occupied by ChromaDB.
- CMW Platform Agent (`cmw-platform-agent`) Gradio app is not running on this host. Its API is documented at `http://10.9.7.7:7860/gradio_api/call/ask_stream` — this refers to the RAG UI's Gradio API, which exposes compatible endpoints.
