# Deployment Architecture

## Host

### Running Services

| Service | Port | Process | Status |
|---------|------|---------|--------|
| RAG Gradio UI | 7860 | `python rag_engine/api/app.py` | Active |
| CMW-Mosec | 7998 | `cmw_mosec.v2.dynamic_server` | Active |
| ChromaDB | 8000 | `chroma run --host 0.0.0.0 --port 8000` | Active |

### Not Running on this Host

- None — all core services are active

---

## CMW-Mosec (port 7998)

Single Mosec process serving multiple model types via env-conditional route registration.

Source: `github.com/cmw-team/cmw-mosec`

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

Source: `github.com/cmw-team/cmw-rag` (pushurl: `arterm-sedov`)

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
    ├──→ https://ennoia.slickjump.org/  (HTTPS, nginx reverse proxy)
    │       │
    │       └──→ nginx (:443, Let's Encrypt TLS)
    │               │
    │               ├──→ localhost:7860/        (Support Agent, plain HTTP)
    │               │       └──→ uses same internal chain as KB Assist
    │               │
    │               └──→ localhost:7860/kb_assist (KB Assist Agent, plain HTTP)
    │
    ├── External integrations (from CMW Platform → RAG API):
    │       │
    │       ├──→ support.comindware.com
    │       │       POST /api/v1/cmw/process-support-request
    │       │       └── reads systemSolution.Requests
    │       │       └── writes systemSolution.agent_responses
    │       │
    │       └──→ lukoil.bau.cbap.ru
    │               POST /api/v1/cmw/summarize-document
    │               └── reads ArchitectureManagement.Zaprosinarazrabotky
    │               └── writes summary attribute
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

# CMW Platform Integration — Primary (Support Agent)
CMW_BASE_URL=https://support.comindware.com
CMW_LOGIN=<login>
CMW_PASSWORD=<password>
CMW_TIMEOUT=30
CMW_API_KEY=<api-key>

# CMW Platform Integration — Secondary / Lukoil (Document Summarization)
CMW2_BASE_URL=<secondary-platform-url>
CMW2_LOGIN=<login>
CMW2_PASSWORD=<password>
CMW2_TIMEOUT=30
CMW2_API_KEY=<api-key>

# LLM Reasoning
LLM_REASONING_ENABLED=true
LLM_REASONING_MAX_TOKENS=1200
LLM_REASONING_EXCLUDE_FROM_RESPONSE=true
```

---

## CMW Platform Integration

The RAG app integrates with two Comindware Platform instances via FastAPI endpoints and a YAML-driven pipeline. Both follow the same fire-and-forget pattern: fetch record → spawn background agent → write result back.

### Two Platform Instances

| Instance | Purpose | Config file | Env prefix |
|----------|---------|-------------|------------|
| Primary | Support desk: read support case → RAG answer → write response record | `cmw_platform.yaml` | `CMW_` |
| Secondary (Лукойл) | Document summarization: read attached doc → LLM summarize → write summary | `cmw_platform_secondary.yaml` | `CMW2_` |

### Primary: Support Request Pipeline

**Endpoint:** `POST /api/v1/cmw/process-support-request`

**Schema:**
```json
{"request_id": "string"}
```

**Flow:**
```
CMW Platform (support.comindware.com)
    │ POST /api/v1/cmw/process-support-request {request_id}
    ▼
RAG FastAPI endpoint
    │ 1. Auth via X-API-Key header (optional, if CMW_API_KEY set)
    │ 2. Fetch record from systemSolution.Requests
    │    Fields: name, Description, currentBuild, browserDetails
    │ 3. Build markdown request via template
    │ 4. Return 200 {success: true, message: "Request fetched, agent started at ..."}
    │ 5. Background thread → call LangChain agent (ask_comindware_structured)
    │ 6. Map agent response to output fields
    │ 7. Write response record to systemSolution.agent_responses
    │    (linked to original request via support_request attribute)
    ▼
Response written back to CMW Platform
```

**YAML pipeline config** (`cmw_platform.yaml`):

```yaml
pipeline:
  input:
    application: systemSolution
    template: Requests
    attributes:
      support_case_title: name
      support_case_question: Description
      product_version: currentBuild
      user_browser: browserDetails

  request_template: |
    ---
    - product version: {product_version}
    - user browser: {user_browser}
    ---
    # {support_case_title}
    {support_case_question}

  output:
    application: systemSolution
    template: agent_responses
    record_attribute: support_request
    linked_template: Requests
```

The agent response is mapped to output attributes via `mapping.py`:
- `answer` → agent's final answer text
- `question_for_agent` → original request with YAML frontmatter
- `agent_thinking` → SGR research report
- `issue_area` → classified category from `category_enum`
- `kb_articles` → cited documentation articles

### Secondary / Lukoil: Document Summarization

**Endpoint:** `POST /api/v1/cmw/summarize-document`

**Schema:**
```json
{"request_id": "string"}
```

**Flow:**
```
CMW Platform (lukoil.bau.cbap.ru, app: ArchitectureManagement)
    │ POST /api/v1/cmw/summarize-document {record_id}
    ▼
RAG FastAPI endpoint
    │ 1. Auth via X-API-Key header (optional, if CMW2_API_KEY set)
    │ 2. Read record from ArchitectureManagement.Zaprosinarazrabotky
    │    Fields: Commerpredloshenie (document), promt (user prompt)
    │ 3. Fetch document content → extract text
    │ 4. Return 200 {success: true, message: "Начата обработка данных"}
    │ 5. Background thread → create_summary_agent (LangChain)
    │    Agent has web_search tool, uses system prompt from YAML
    │ 6. Convert summary to HTML
    │ 7. Write summary back to record's summary attribute
    ▼
Summary written back to CMW Platform
```

**YAML pipeline config** (`cmw_platform_secondary.yaml`):

```yaml
pipeline:
  input:
    application: ArchitectureManagement
    template: Zaprosinarazrabotky
    attributes:
      document_file: Commerpredloshenie
      user_prompt: promt

  output:
    application: ArchitectureManagement
    template: Zaprosinarazrabotky
    summary_attribute: summary
    summary_as_html: true

  system_prompt: |
    Ты — профессиональный бизнес-ассистент. Твоя задача — составлять
    краткие и информативные резюме деловых документов.
    ...
```

### API Schemas (Pydantic)

```python
class ProcessSupportRequest(BaseModel):
    request_id: str

class SummarizeDocumentRequest(BaseModel):
    request_id: str

class ProcessResult:
    success: bool
    message: str | None = None
    error: str | None = None

class HTTPResponse(BaseModel):
    success: bool
    status_code: int
    raw_response: dict | str | None = None
    error: str | None = None
    base_url: str

class APIResponse(BaseModel):
    response: Any | None = None
    success: bool | None = None
    error: str | None = None

class RequestConfig(BaseModel):
    base_url: str
    login: str
    password: str
    timeout: int = 30
```

### Category Enum

Request issue areas are loaded from `cmw_platform.yaml` → `category_enum`. The list is fetched from CMW Platform via `scripts/fetch_issue_areas.py`. Example categories: `account`, `api`, `deployment`, `email`, `performance`, `integrations`, etc.

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
4. **Configure .env** in cmw-mosec (models, port, HF token) and cmw-rag (LLM keys, platform credentials, as shown above)
5. **Start Mosec** → `cmw-mosec serve` (downloads models on first run, ~8GB total GPU mem)
6. **Start ChromaDB** → `python rag_engine/scripts/start_chroma_server.py`
7. **Build/verify index** → run corpus sync to populate ChromaDB collections
8. **Start RAG UI** → `python rag_engine/api/app.py`
9. **Configure CMW Platform webhooks** to call:
   - `POST http://<host>:7860/api/v1/cmw/process-support-request` (support.comindware.com)
   - `POST http://<host>:7860/api/v1/cmw/summarize-document` (lukoil.bau.cbap.ru)
   - Both accept `{"request_id": "..."}` with optional `X-API-Key` header
10. **Verify**:
    - `curl http://localhost:7998/v1/embeddings -X POST -d '{"input":"test","model":"Qwen/Qwen3-Embedding-0.6B"}'`
    - `curl http://localhost:7998/v1/moderate -X POST -d '{"input":"test"}'`
    - `curl http://localhost:8000/api/v1/heartbeat`
    - `curl http://localhost:7860/api/v1/cmw/process-support-request -X POST -d '{"request_id":"test"}'`
    - Open `http://localhost:7860` and `http://localhost:7860/kb_assist` in browser

### GPU Requirements

Mosec with the three active models requires ~8GB GPU VRAM total (embedding 2GB, reranker 2GB, guard 4GB). Each worker runs on the same device via Mosec batching.

---

## Sibling Repos

### cmw-mosec

| Attribute | Value |
|-----------|-------|
| Source | `github.com/cmw-team/cmw-mosec` (pushurl: `arterm-sedov`) |
| Entry | `cmw_mosec.cli:cli` (Click CLI) |
| Server module | `cmw_mosec.v2.dynamic_server` |
| Model config | `config/models.yaml` |
| Active usage | Serving embeddings, scores, moderation on this host |

Command: `cmw-mosec serve` (reads `ACTIVE_*_MODEL` env vars, starts on `SERVER_PORT`).

### cmw-vllm

CLI tool to manage vLLM inference servers (OpenAI-compatible). Can serve LLM locally instead of OpenRouter.

| Attribute | Value |
|-----------|-------|
| Source | `github.com/cmw-team/cmw-vllm` (pushurl: `arterm-sedov`) |
| Entry | `cmw_vllm.cli:cli` (Click CLI) |
| Model registry | `cmw_vllm/model_registry.py` |
| Active model (if deployed) | `openai/gpt-oss-20b` |
| Default port | 8000 |
| Status | Not deployed. Port 8000 occupied by ChromaDB. |

```bash
# .env: VLLM_MODEL=openai/gpt-oss-20b  VLLM_PORT=8000  VLLM_HOST=0.0.0.0
cmw-vllm start openai/gpt-oss-20b
```

RAG uses it via `VLLM_BASE_URL=http://<host>:8000/v1` + `DEFAULT_LLM_PROVIDER=vllm`.

---

## Notes

- Mosec `/v1/rerank` endpoint is registered but returns inference error at runtime — RAG uses `/v1/score` instead, which works correctly.
- vLLM could serve as local LLM backend instead of OpenRouter if deployed on a separate host with free GPU and port.
- When changing embedding model (different dimensions), create a new ChromaDB collection and reindex.
