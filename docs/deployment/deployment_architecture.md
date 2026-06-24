# Deployment Architecture

## Host: 10.9.7.7

### Running Services

| Service | Port | Process | Status |
|---------|------|---------|--------|
| RAG Gradio UI | 7860 | `python rag_engine/api/app.py` | Active |
| CMW-Mosec | 7998 | `cmw_mosec.v2.dynamic_server` | Active |
| ChromaDB | 8000 | `chroma run --host 0.0.0.0 --port 8000` | Active |

### Not Running on this Host

- vLLM — not running (no process found)
- CMW Platform Agent — not running (no process found)

---

## CMW-Mosec (port 7998)

Single Mosec process serving multiple model types via env-conditional route registration.

Source: `github.com/arterm-sedov/cmw-mosec` → added remote `cmw-team`

### Routes

| Route | Worker | Model | Verified |
|-------|--------|-------|----------|
| `POST /v1/embeddings` | EmbeddingWorkerV2 | `Qwen/Qwen3-Embedding-0.6B` | ✅ Returns embeddings |
| `POST /v1/score` | ScoreWorkerV2 | `Qwen/Qwen3-Reranker-0.6B` | ✅ Returns scores |
| `POST /v1/rerank` | RerankWorkerV2 | `Qwen/Qwen3-Reranker-0.6B` | ❌ Returns error |
| `POST /v1/moderate` | GuardWorkerV2 | `Qwen/Qwen3Guard-Gen-0.6B` | ✅ Returns moderation |

### Active Model Env

```
ACTIVE_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
ACTIVE_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
ACTIVE_GUARD_MODEL=Qwen/Qwen3Guard-Gen-0.6B
SERVER_PORT=7998
DEVICE=auto
DTYPE=float32
BATCH_SIZE=32
HF_TOKEN=<huggingface-token>
```

### Start Command

```bash
cd cmw-mosec
source .venv/bin/activate
cmw-mosec serve
```

### Available Model Alternatives

| Type | Models |
|------|--------|
| Embeddings | `ai-forever/FRIDA` (1536d, ~4GB), `Qwen/Qwen3-Embedding-0.6B` (1024d, ~2GB), `Qwen/Qwen3-Embedding-4B` (2560d, ~12GB), `Qwen/Qwen3-Embedding-8B` (4096d, ~22GB) |
| Rerankers | `DiTy/cross-encoder-russian-msmarco` (~2GB), `BAAI/bge-reranker-v2-m3` (~2GB), `Qwen/Qwen3-Reranker-0.6B` (~2GB), `Qwen/Qwen3-Reranker-4B` (~12GB), `Qwen/Qwen3-Reranker-8B` (~22GB) |
| Guards | `Qwen/Qwen3Guard-Gen-0.6B` (~4GB), `Qwen/Qwen3Guard-Gen-4B` (~10GB), `Qwen/Qwen3Guard-Gen-8B` (~20GB) |

---

## ChromaDB (port 8000)

Vector store for RAG retrieval.

### Start Command

```bash
cd cmw-rag
source .venv/bin/activate
python rag_engine/scripts/start_chroma_server.py
```

### Active Collection

`mkdocs_kb_qwen06b_linux_qwen_default_instruct_chunk_768` (1024 dim, matches Qwen3-Embedding-0.6B)

### Versioned Collections

| Collection | Content |
|------------|---------|
| `mkdocs_kb_qwen06b_linux_qwen_default_instruct_chunk_768_v5` | Platform v5 docs |
| `mkdocs_kb_qwen06b_linux_qwen_default_instruct_chunk_768_v6` | Platform v6 docs |

When switching embedding models with different dimensions, a new collection must be created and indexed.

---

## RAG Gradio UI (port 7860)

Entry: `rag_engine/api/app.py`

Source: `github.com/arterm-sedov/cmw-rag` → added remote `cmw-team`

### Two Agents

| Agent | Path | Feature |
|-------|------|---------|
| Support Assistant | `/` | Full UI with version selector, metadata panels, SRP (support resolution plan), chat export, MCP tools (`ask_comindware`, `get_knowledge_base_articles`) |
| KB Assist | `/kb_assist` | Same agent, minimal UI (chatbot + input only, no panels, no version selector, `skip_srp=True`) |

Both agents use the same LangChain agent handler (`chat_with_metadata`). The only difference is KB Assist suppresses metadata panel rendering and SRP generation.

### Start Command

```bash
cd cmw-rag
source .venv/bin/activate
python rag_engine/api/app.py
```

### Service Dependencies

```
User Browser
    │
    ├──→ 10.9.7.7:7860/        (Support Agent)
    │       └──→ uses same internal chain as KB Assist
    │
    ├──→ 10.9.7.7:7860/kb_assist (KB Assist Agent)
    │
    └── Internal (RAG app → backing services):
            │
            ├──→ localhost:7998  (Mosec — embeddings, score, moderate)
            ├──→ localhost:8000  (ChromaDB — vector search)
            └──→ api.openrouter.ai (LLM inference)
```

### Required Env Vars

```
# LLM
OPENROUTER_API_KEY=<key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_LLM_PROVIDER=openrouter
DEFAULT_MODEL=deepseek/deepseek-v4-pro

# Embedding
EMBEDDING_PROVIDER_TYPE=mosec
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
MOSEC_EMBEDDING_ENDPOINT=http://localhost:7998/v1/embeddings

# Reranker
RERANK_ENABLED=true
RERANKER_PROVIDER_TYPE=mosec
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
RERANKER_TIMEOUT=60.0
RERANKER_MAX_RETRIES=3
MOSEC_RERANKER_ENDPOINT=http://localhost:7998/v1/score

# Guard
GUARD_ENABLED=true
GUARD_BLOCK_THRESHOLD=unsafe
GUARD_PROVIDER_TYPE=mosec
GUARD_MOSEC_ENDPOINT=http://localhost:7998/v1/moderate

# ChromaDB
CHROMADB_PORT=8000
CHROMA_CLIENT_HOST=localhost
CHROMA_SERVER_BIND=0.0.0.0
CHROMADB_COLLECTION=mkdocs_kb_qwen06b_linux_qwen_default_instruct_chunk_768
CHROMADB_PERSIST_DIR=~/cmw-rag/data/chromadb_data

# Retrieval
TOP_K_RETRIEVE=20
TOP_K_RERANK=10
CHUNK_SIZE=768
CHUNK_OVERLAP=75
RETRIEVAL_MULTIQUERY_ENABLED=true
RETRIEVAL_MULTIQUERY_MAX_SEGMENTS=4
RETRIEVAL_MULTIQUERY_SEGMENT_TOKENS=448
RETRIEVAL_QUERY_DECOMP_ENABLED=true
RETRIEVAL_QUERY_DECOMP_MAX_SUBQUERIES=4

# Gradio
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_LOCALE=ru

# Optional: Google API key (for Gemini provider fallback)
# GOOGLE_API_KEY=<key>
# Optional: vLLM (if used instead of OpenRouter)
# VLLM_BASE_URL=http://localhost:8000/v1
# VLLM_API_KEY=EMPTY

# CMW Platform Integration (for support agent ticket operations)
CMW_BASE_URL=https://support.comindware.com
CMW_LOGIN=<login>
CMW_PASSWORD=<password>
CMW_TIMEOUT=30
CMW_API_KEY=<api-key>

# LLM Reasoning
LLM_REASONING_ENABLED=true
LLM_REASONING_MAX_TOKENS=1200
LLM_REASONING_EXCLUDE_FROM_RESPONSE=true
```

---

## Corpus Sync (MkDocs)

Syncs documentation content from an external MkDocs repo into ChromaDB for RAG indexing.

### Systemd Timer

| Component | Value |
|-----------|-------|
| Timer | `cmw-rag-corpus-sync.timer` — enabled and active |
| Schedule | Every 6h at 00:00, 06:00, 12:00, 18:00 UTC (+ 10min random delay) |
| Service | `cmw-rag-corpus-sync.service` — oneshot, runs on timer trigger |
| Command | `sync_mkdocs_corpus.py --index --corpus all` |

### Install Timer

```bash
cp systemd/cmw-rag-corpus-sync.{service,timer} ~/.config/systemd/user/
systemctl --user enable --now cmw-rag-corpus-sync.timer
```

### Upstream Repo

| Setting | Value |
|---------|-------|
| Remote | `github.com/arterm-sedov/cbap-mkdocs-ru` |
| Branch | `platform_v6` |
| Sparse path | `phpkb_content_rag` |
| Local clone | `.reference-repos/cbap-mkdocs-ru` |

### Corpus Versions

| Version | Source path | ChromaDB collection suffix |
|---------|-------------|---------------------------|
| v5 | `phpkb_content_rag/798-platform_v5` | `_v5` |
| v6 | `phpkb_content_rag/896-platform_v6` | `_v6` |

The sync script runs `git sparse-checkout` to fetch only `phpkb_content_rag`, then indexes new/changed files. If no changes since last sync, indexing is skipped.

### Manual Sync

```bash
python rag_engine/scripts/sync_mkdocs_corpus.py --index --corpus all
python rag_engine/scripts/sync_mkdocs_corpus.py --index --corpus v6
```

---

## Reconstruction Steps

To replicate this deployment on a new host:

1. **Install Python 3.11+** and create venvs for each project
2. **Install dependencies** — `pip install -e .` in cmw-mosec, `pip install -r rag_engine/requirements.txt` in cmw-rag
3. **Install ChromaDB** — `pip install chromadb`
4. **Configure .env** in cmw-mosec (models, port, HF token) and cmw-rag (as shown above)
5. **Start Mosec** → `cmw-mosec serve` (downloads models on first run, ~8GB total GPU mem)
6. **Start ChromaDB** → `python rag_engine/scripts/start_chroma_server.py`
7. **Build/verify index** → run corpus sync to populate ChromaDB collections
8. **Start RAG UI** → `python rag_engine/api/app.py`
9. **Verify**:
   - `curl http://localhost:7998/v1/embeddings -X POST -d '{"input":"test","model":"Qwen/Qwen3-Embedding-0.6B"}'`
   - `curl http://localhost:7998/v1/moderate -X POST -d '{"input":"test"}'`
   - `curl http://localhost:8000/api/v1/heartbeat`
   - Open `http://localhost:7860` and `http://localhost:7860/kb_assist` in browser

### GPU Requirements

Mosec with the three active models requires ~8GB GPU VRAM total (embedding 2GB, reranker 2GB, guard 4GB). Each worker runs on the same device via Mosec batching.

---

## Notes

- Mosec `/v1/rerank` endpoint is registered but returns inference error at runtime — RAG uses `/v1/score` instead, which works correctly.
- vLLM configuration exists (`cmw-vllm/.env`, `VLLM_PORT=8000`, model `openai/gpt-oss-20b`) but the server is not currently running. Port 8000 on this host is occupied by ChromaDB.
- CMW Platform Agent (`cmw-platform-agent`) Gradio app is not running on this host.
- When changing embedding model (different dimensions), create a new ChromaDB collection and reindex. Collection dimension must match embedding output dimension.
