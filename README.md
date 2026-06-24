# CMW RAG Engine

Production-ready RAG (Retrieval-Augmented Generation) engine for document Q&A with support for MkDocs, markdown folders, and single-file ingestion. Features FRIDA embeddings (Russian/English), ChromaDB vector storage, reranking, and multi-LLM support with streaming responses.

**Original author:** [Arterm Sedov](https://github.com/arterm-sedov)  
**Co-author:** Marat Mutalimov

## AI-Enabled Repo

Chat with DeepWiki to get answers about this repo:

[Ask DeepWiki](https://deepwiki.com/cmw-team/cmw-rag)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/cmw-team/cmw-rag)

## Features

- **Multi-source ingestion**: Support for MkDocs export, markdown folders, and single combined files
- **Smart timestamp detection**: Three-tier fallback (frontmatter → Git → file modification) for accurate incremental reindexing
- **Bilingual support**: FRIDA embeddings for Russian and English content
- **Advanced retrieval**: Vector search with optional cross-encoder reranking
- **Multi-vector queries**: Split long queries into token-aware segments, retrieve per segment, union + rerank
- **Multi-LLM support**: Gemini (default) and OpenRouter with streaming responses
- **Reasoning tokens (optional)**: Unified reasoning-token control (OpenRouter-style `reasoning` config) for OpenAI-compatible chat models, with an optional UI bubble that shows the model's internal reasoning trace
- **Dynamic context budgeting**: Summarization-first trimming guided by the user question; falls back to lightweight stitching when needed
- **Immediate model fallback**: Optional auto-fallback to allowed larger-context models when estimated tokens exceed the current model
- **Consistent token accounting**: Shared `token_utils` for system + question + context + output budgeting
- **Web interface**: Gradio ChatInterface with citations and chat history
- **Embeddable widget**: Floating chat widget for embedding on external websites (kb.comindware.ru)
- **Content moderation**: Guardian support via MOSEC or VLLM with Qwen3Guard models for safety classification
- **REST API**: Programmatic access for integration
- **Context-aware**: Complete article context with citation support
  - **Per-session memory**: LangChain-backed conversation memory (scoped by Gradio session hash) with optional compression near context limits
  - **Copy button**: One-click copy on chat messages

## Architecture

The codebase follows a clean separation of concerns:

### Indexing Pipeline (`rag_engine/core/`)

- **`indexer.py` - RAGIndexer**: Handles document indexing operations
  - Chunks documents, generates embeddings, writes to vector store
  - Implements incremental reindexing with timestamp-based deduplication
  - Normalizes `kbId` to numeric format for consistent document identification
  - Requires `kbId` in document metadata (from frontmatter); no fallback to `source_file`
  - Located in `core/` alongside other document processing components

- **`document_processor.py`**: Processes markdown files from various sources (folder, file, mkdocs)
- **`chunker.py`**: Token-aware text chunking with overlap
- **`metadata_enricher.py`**: Enriches chunk metadata with code detection, section info, etc.

### Retrieval Pipeline (`rag_engine/retrieval/`)

- **`retriever.py` - RAGRetriever**: Handles query retrieval operations
  - Vector search, reranking, article reconstruction
  - Context budgeting and article summarization
  - Query segmentation and multi-vector retrieval
  - Normalizes `kbId` values during retrieval for robustness (handles edge cases)
  - Groups chunks by normalized `kbId` to ensure consistent article identification
  - **Note**: Indexing has been separated into `RAGIndexer` for better separation of concerns

- **`embedder.py`**: FRIDA embedding model wrapper
- **`vector_search.py`**: ChromaDB vector search utilities
- **`reranker.py`**: Cross-encoder reranking models

### Scripts (`rag_engine/scripts/`)

- **`build_index.py`**: Main indexing script that uses `RAGIndexer`
  - Supports `--dry-run` mode to analyze timestamps without indexing
  - Incremental reindexing with timestamp-based deduplication
  - `--max-files` limits files processed during scanning (before indexing)
- **`maintain_chroma.py`**: ChromaDB maintenance and diagnostics
  - `--action diagnose`: Comprehensive database health check
    - Lists collections and their chunk counts
    - Checks consistency between SQLite metadata and vector data directories
    - Detects orphaned vector directories and UUID mismatches
    - Shows WAL (Write-Ahead Log) status
    - Counts unique articles vs total chunks
  - `--action commit-wal`: Commits pending transactions from WAL
- **`migrate_normalize_kbids.py`**: One-time migration script to normalize kbId values
  - Finds documents with suffixed kbIds (e.g., `"4578-toc"`) and normalizes to numeric format
  - Handles both suffixed kbIds and path-like kbIds (from old fallback logic)
  - Dry-run mode by default; use `--apply` to perform migration
- **`inspect_db_schema.py`**: Inspect ChromaDB data model and sample records
  - Shows all metadata fields and their types
  - Displays random sample records for validation
  - Provides statistics and validation checks
- **`start_app.sh` / `start_app.ps1`**: Application startup scripts

### LangChain Tool Integration (`rag_engine/tools/`)

The RAG engine provides a LangChain 1.0-compatible `retrieve_context` tool for agent integration. The tool is self-sufficient (handles RAGRetriever initialization internally), returns JSON with articles and metadata, and supports multiple calls per conversation.

#### Agent Mode (Recommended)

The application includes a built-in agent mode using LangChain's `create_agent`. Enable it via environment variable:

```bash
# .env
USE_AGENT_MODE=true
```

**Features:**
- Automatic tool calling via LangChain ReAct agent
- Forced tool execution via `tool_choice` parameter (per [LangChain docs](https://docs.langchain.com/oss/python/langchain/models#tool-calling))
- Uses standard Comindware Platform system prompt
- **Real-time token streaming** - see text appear as it's generated
- **Visual tool feedback** - collapsible metadata messages show search progress
- Session-based conversation memory
- Citations automatically added

**How it works:**
1. User asks question → Agent analyzes it
2. Agent shows "🔍 Searching information in the knowledge base" (collapsible metadata message)
3. Agent calls `retrieve_context` tool (forced via `tool_choice="retrieve_context"`)
4. Agent shows "✅ Found X articles" (completion metadata message)
5. Agent generates answer based on retrieved articles
6. Citations automatically added

Metadata messages follow the [Gradio agents pattern](https://www.gradio.app/4.44.1/guides/agents-and-tool-usage#the-metadata-key) for displaying tool execution status.

The agent mode is production-ready and can be toggled without code changes.

#### Tool Usage (For Custom Agents)

**Direct tool invocation:**

```python
from rag_engine.tools import retrieve_context

# Direct invocation
result = retrieve_context.invoke({"query": "How to configure authentication?", "top_k": 5})

# Agent integration with forced tool execution
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4")
model_with_tools = model.bind_tools([retrieve_context], tool_choice="retrieve_context")
```

**Multiple tool calls:**

When an agent makes multiple `retrieve_context` calls (iterative search refinement), accumulate articles for comprehensive citations:

```python
from rag_engine.tools import accumulate_articles_from_tool_results
from rag_engine.utils.formatters import format_with_citations

# Collect tool results from agent
tool_results = [result1, result2, result3]  # From multiple tool calls

# Accumulate and generate answer
all_articles = accumulate_articles_from_tool_results(tool_results)
answer = llm_manager.stream_response(question, all_articles, ...)
final_answer = format_with_citations(answer, all_articles)  # Auto-deduplicates by kbId/URL
```

**Available utilities**: `parse_tool_result_to_articles()`, `accumulate_articles_from_tool_results()`, `extract_metadata_from_tool_result()`

See `rag_engine/tests/test_tools_utils.py` and `rag_engine/tests/test_agent_handler.py` for detailed examples.

## Prerequisites

- Python 3.12+
- Virtual environment (`.venv`)
- API keys: Google (Gemini) or OpenRouter
- Source documents (markdown files) for indexing

## Quick Start

### 1. Setup Environment

**Linux:**
```bash
source .venv/bin/activate
pip install -r rag_engine/requirements.txt
```

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
pip install -r rag_engine\requirements.txt
```

### 2. Configure Environment Variables

Copy `.env-example` to `.env` and configure your API keys:

```bash
# WSL/Linux
cp .env-example .env

# Windows PowerShell
Copy-Item .env-example .env
```

Minimum required: `GOOGLE_API_KEY` (for Gemini) or `OPENROUTER_API_KEY`

### 3. Build the Index

Choose one of three ingestion modes:

**Sync MkDocs RAG Corpora (Recommended for CMW KB)**
```bash
# Fetch/update both V5 and V6 PHPKB RAG corpora via sparse Git checkout
python rag_engine/scripts/sync_mkdocs_corpus.py

# Fetch/update and index both corpora
python rag_engine/scripts/sync_mkdocs_corpus.py --index

# Index only one corpus if needed
python rag_engine/scripts/sync_mkdocs_corpus.py --index --corpus v6
```

The sync script keeps the managed MkDocs clone under `.reference-repos/cbap-mkdocs-ru`,
sparse-checks out `phpkb_content_rag`, and then delegates indexing to `build_index.py`.

Each corpus version indexes into a **separate ChromaDB collection** (e.g. `mkdocs_kb_v5`
and `mkdocs_kb_v6`). The collection names are derived from `CHROMADB_COLLECTION` and can
be overridden via `CHROMADB_COLLECTION_V5` / `CHROMADB_COLLECTION_V6` env vars. The Gradio
UI exposes a session-scoped dropdown to select the active version for retrieval.

**Automated (systemd timer)** — Runs every 6h. View logs:
```bash
journalctl --user -u cmw-rag-corpus-sync.service --no-pager -n 50
```
Install: `cp systemd/cmw-rag-corpus-sync.{service,timer} ~/.config/systemd/user/ && systemctl --user enable --now cmw-rag-corpus-sync.timer`

**Mode: Folder (Recommended)**
```bash
python rag_engine/scripts/build_index.py \
  --source "path/to/your/markdown/folder/" \
  --mode folder \
  # Optional: limit number of actually indexed docs (skips don't count)
  --max-files 100
```

**Mode: Single File**
```bash
python rag_engine/scripts/build_index.py \
  --source "kb.comindware.ru.platform_v5_for_llm_ingestion.md" \
  --mode file
```

**Mode: MkDocs Export**
```bash
# First export from MkDocs project
python rag_engine/scripts/run_mkdocs_export.py \
  --project-dir ../CBAP_MKDOCS_RU \
  --inherit-config mkdocs_guide_complete_ru.yml \
  --output-dir ./data/compiled_md_for_rag

# Then index
python rag_engine/scripts/build_index.py \
  --source ./data/compiled_md_for_rag \
  --mode mkdocs \
  --max-files 100
```

Optional flags and behavior:
- `--max-files N`: Limit the number of files processed during indexing. The limit is applied during file scanning (before indexing), so only the first N files from the source are processed. Useful for quick test runs or incremental indexing.
- `--dry-run`: Analyze timestamps without indexing. Shows which timestamp source (frontmatter/git/file) is used for each file and whether it would be indexed, skipped, or reindexed.
- **Three-tier timestamp fallback**: The system uses timestamps in priority order:
  1. **Frontmatter `updated` field** - Parsed from YAML frontmatter (e.g., `updated: '2024-06-14 12:33:36'`)
  2. **Git commit timestamp** - Last commit time for the file from its Git repository (automatically detected per file)
  3. **File modification date** - Filesystem `stat().st_mtime` as fallback
- Incremental reindexing: If a file's timestamp is unchanged since the last run, it is skipped. If it changed, all previous chunks for that document are replaced with fresh ones.
- Safe to re-run: Stable IDs and per-document replacement make repeated indexing idempotent.

### 4. Run the Application

**Using helper scripts (recommended):**

```bash
# WSL/Linux
bash rag_engine/scripts/start_app.sh

# Windows PowerShell
.\rag_engine\scripts\start_app.ps1
```

**Production (systemd user service):**

```bash
# Install (one-time)
ln -sf "$(pwd)/systemd/cmw-rag-app.service" ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable cmw-rag-app.service

# Manage
systemctl --user start/stop/restart/status cmw-rag-app

# Logs
journalctl --user -u cmw-rag-app -f
```

The app starts after ChromaDB and Mosec (`After=cmw-rag-chroma.service cmw-rag-mosec.service`) and restarts automatically on failure and after reboot.

**Development (manual):**

```bash
# Linux
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python rag_engine/api/app.py

# Windows PowerShell
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$(Get-Location);$env:PYTHONPATH"
python rag_engine\api\app.py
```

### 5. Access the Application

- **Web UI**: [http://localhost:7860](http://localhost:7860)
- **KB Assist demo**: http://localhost:7860/kb_assist
- **Production widget**: embedded via `ai-widget.php` in the [kb.comindware.ru](https://kb.comindware.ru) repo
- **REST API**: http://localhost:7860/api/query_rag

## Usage

### Web Interface

The Gradio ChatInterface provides:
- Bilingual queries (Russian/English)
- Streaming responses
- Automatic citations
- Chat history
- Graceful handling of empty results (see Troubleshooting section)

### Embeddable Widget

The KB copilot widget is embedded on [kb.comindware.ru](https://kb.comindware.ru) via `ai-widget.php` in the sibling `kb.comindware.ru` repo. Gradio styling lives in this repo under `rag_engine/resources/css/cmw_copilot_theme.css`.

**Features:**
- Floating toggle button (bottom-right corner)
- Collapsible chat panel with full Gradio functionality
- Resizable from top-left corner (custom resize handle)
- Position and size persistence (localStorage)
- Preloading for instant widget opening
- Responsive design (mobile/tablet support)
- KB site theme integration (Open Sans font, KB colors)
- Keyboard accessibility (Escape to close)
- Dark theme support

**Configuration:**
The widget connects to the Gradio app URL configured in `ai-widget.php` (default points at the deployed RAG engine). Override via the `GRADIO_URL` JavaScript variable when testing locally.

**Local testing:**
Run `python rag_engine/api/app.py`, then load a KB page that includes `ai-widget.php` against your local Gradio endpoint.

### REST API

Query the RAG engine programmatically:

```python
import requests

response = requests.post(
    "http://localhost:7860/api/query_rag",
    json={
        "question": "How to use N3?",
        "provider": "gemini",
        "top_k": 5
    }
)
print(response.json())
```

**API Endpoint**: `POST /api/query_rag`

**Request Body**:
- `question` (string, required): Query text
- `provider` (string, optional): LLM provider (`gemini` or `openrouter`)
- `top_k` (int, optional): Number of results to retrieve
  
Environment-driven behavior:
- If `LLM_FALLBACK_ENABLED=true`, the engine will estimate total tokens and immediately select a larger allowed model when necessary.
- Summarization-first budgeting compresses overflow articles (guided by the question) before falling back to lightweight stitching.
- Token counting uses exact tiktoken encoding for accurate counting across all languages.

**Response**:
```json
{
  "answer": "Response text with citations",
  "sources": [
    {
      "title": "Article Title",
      "url": "section-anchor",
      "score": 0.95
    }
  ]
}
```

### CMW Platform Integration

Process support requests from CMW Platform:

```bash
curl -X POST http://localhost:7862/api/v1/cmw/process-support-request \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"request_id": "322919"}'
```

**Endpoint**: `POST /api/v1/cmw/process-support-request`

**Authentication**: Set `CMW_API_KEY` in `.env` (empty = no auth, present = require `X-API-Key` header)

**Request Body**:
- `request_id` (string, required): TPAIModel record ID

**Response**:
```json
{
  "success": true,
  "message": "Request fetched, agent started",
  "error": null
}
```

## Configuration

All configuration is managed through environment variables. Copy `.env-example` to `.env` and configure your settings:

```bash
# WSL/Linux
cp .env-example .env

# Windows PowerShell
Copy-Item .env-example .env
```

**Important:** Never commit `.env` to version control. Use `.env-example` as a template with placeholder values only.

### Quick Reference

| Category | Key Settings |
|----------|-------------|
| LLM | `DEFAULT_LLM_PROVIDER`, `DEFAULT_MODEL` |
| Embeddings | `EMBEDDING_PROVIDER_TYPE`, `EMBEDDING_MODEL` |
| Vector Store | `CHROMADB_HOST`, `CHROMADB_PORT` |
| Retrieval | `TOP_K_RETRIEVE`, `TOP_K_RERANK`, `RERANK_SCORE_THRESHOLD` |
| Web UI | `GRADIO_SERVER_NAME`, `GRADIO_SERVER_PORT` |

See `.env-example` for complete documentation of all environment variables.

### Functional Configuration

#### Retrieval – Multi-vector Queries and Query Decomposition

Multi-vector queries split long queries into token-aware segments, retrieve per segment, and union + rerank results. Configure in `.env`:

```
RETRIEVAL_MULTIQUERY_ENABLED=true           # Enable multi-vector query retrieval
RETRIEVAL_MULTIQUERY_MAX_SEGMENTS=4        # Maximum query segments
RETRIEVAL_MULTIQUERY_SEGMENT_TOKENS=448    # Target tokens per segment (≤512)
RETRIEVAL_MULTIQUERY_SEGMENT_OVERLAP=64    # Overlap tokens between segments
RETRIEVAL_MULTIQUERY_PRE_RERANK_LIMIT=60   # Cap merged candidates before rerank
```

Query decomposition uses LLM to generate sub-queries:

```
RETRIEVAL_QUERY_DECOMP_ENABLED=false       # Enable LLM-based query decomposition
RETRIEVAL_QUERY_DECOMP_MAX_SUBQUERIES=4    # Maximum sub-queries to generate
```

#### Retrieval – Reranking and Score Threshold

Reranking filters and reorders retrieved chunks before loading complete articles. Configure:

```
RERANK_ENABLED=true                         # Enable cross-encoder reranking
RERANK_SCORE_THRESHOLD=0.5                  # Minimum rerank score (articles below excluded)
```

- `RERANK_SCORE_THRESHOLD`: Articles with rerank score below this threshold are excluded before loading from disk. Set to `0.0` to disable filtering (default).

**Supported Reranker Types:**

| Type | Models | Formatting |
|------|--------|------------|
| `cross_encoder` | DiTy, BGE-m3 | None (raw query/documents) |
| `llm_reranker` | Qwen3-Reranker | ChatML prefix/suffix (see `models.yaml`) |

**Endpoint:** Uses `/v1/score` (vLLM format). The RAG engine has documents locally and applies metadata boosts client-side, so the lightweight score endpoint is sufficient.

**API Format:**
```json
// Request: {query, documents}
// Response: {data: [{index, score}, ...]}
```

**Configuration:**
```bash
MOSEC_RERANKER_ENDPOINT=http://localhost:7998/v1/score
```

#### LLM Context Budgeting

The engine uses summarization-first budgeting guided by the user question:

```
LLM_SUMMARIZATION_ENABLED=true              # Enable article summarization
LLM_SUMMARIZATION_TARGET_TOKENS_PER_ARTICLE=1200  # Target tokens after summarization
```

Compression thresholds control when to compress accumulated tool results:

```
LLM_COMPRESSION_THRESHOLD_PCT=0.80          # Trigger compression at 80% of context
LLM_COMPRESSION_TARGET_PCT=0.80              # Target 80% of context after compression
LLM_COMPRESSION_MIN_TOKENS=300              # Minimum tokens per article
```

#### ChromaDB Vector Store

HTTP client settings for the vector database connection:

```
CHROMADB_HOST=localhost                     # ChromaDB server host
CHROMADB_PORT=8000                         # ChromaDB server port
CHROMADB_HTTP_KEEPALIVE_SECS=60.0           # Keep HTTP connections alive (seconds)
CHROMADB_MAX_CONNECTIONS=100                # Maximum connection pool size
```

#### Gradio Web Interface

Web UI configuration for the chat interface:

```
GRADIO_SERVER_NAME=0.0.0.0                  # Bind address
GRADIO_SERVER_PORT=7860                    # HTTP port
GRADIO_SHARE=false                         # Create public share link
GRADIO_DEFAULT_CONCURRENCY_LIMIT=3         # Max concurrent requests
GRADIO_EMBEDDED_WIDGET=false               # Use compact widget layout
GRADIO_LOCALE=ru                           # UI language (ru/en)
```

#### Guardian (Content Moderation)

Optional content moderation layer for classifying user queries:

```
GUARD_ENABLED=true                          # Enable/disable guardian
GUARD_MODE=enforce                          # enforce (block unsafe) | report (log only)
GUARD_PROVIDER_TYPE=mosec                   # mosec | vllm
```

**Provider Options:**

**MOSEC** (returns JSON directly):
```
GUARD_MOSEC_ENDPOINT=http://localhost:7999/api/v1/guard  # Complete endpoint URL
```

**VLLM** (OpenAI-compatible API, parses raw text):
```
GUARD_VLLM_URL=http://localhost:8000/v1     # VLLM base URL
GUARD_VLLM_MODEL=Qwen/Qwen3Guard-Gen-0.6B   # Guard model name
```

Both providers use [Qwen3Guard](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) models and return identical JSON structure for seamless interchangeability.

## Project Structure

```
rag_engine/
├── api/              # Gradio UI and REST API
├── config/           # Configuration and settings
├── core/             # Document processing and chunking
├── llm/              # LLM manager and prompts
├── mkdocs/           # MkDocs integration hooks
├── retrieval/        # Embedding, vector search, and reranking
├── scripts/          # CLI tools for indexing and startup
├── storage/          # ChromaDB vector store
├── cmw_platform/     # CMW Platform integration connectors
└── utils/            # Logging and formatting utilities
```

## Sibling Repos

This project depends on the following sibling repositories:

| Repo | Purpose | Active |
|------|---------|--------|
| [cmw-mosec](https://github.com/cmw-team/cmw-mosec) | Embedding, reranker, and guard inference server | ✅ Deployed on :7998 |
| [cmw-vllm](https://github.com/cmw-team/cmw-vllm) | vLLM inference server management (local LLM alternative) | ⬜ Not deployed (needs separate GPU) |

### Running ChromaDB Server

The RAG engine requires a running ChromaDB instance for vector storage.

#### Production (systemd user service)

```bash
# Install (one-time)
ln -sf "$(pwd)/systemd/cmw-rag-chroma.service" ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable cmw-rag-chroma.service

# Manage
systemctl --user start/stop/restart/status cmw-rag-chroma

# Logs
journalctl --user -u cmw-rag-chroma -f
```

The service restarts automatically after reboot (`loginctl enable-linger asedov` ensures the user systemd instance starts at boot).

Configure in `.env`:
```bash
CHROMADB_PORT=8000
CHROMA_CLIENT_HOST=localhost
CHROMADB_COLLECTION=mkdocs_kb_qwen06b_linux_qwen_default_instruct_chunk_768
CHROMADB_PERSIST_DIR=~/cmw-rag/data/chromadb_data
```

#### Development (manual)

```bash
# Create data directory
mkdir -p ~/cmw-rag/data/chromadb_data

# From the project root
source .venv/bin/activate

# Via helper script (reads .env)
python rag_engine/scripts/start_chroma_server.py --foreground

# Or directly
chroma run --host 0.0.0.0 --port 8000 --path ~/cmw-rag/data/chromadb_data
```

### Running Mosec Server

Embedding, reranker, and guard inference server on port 7998.

#### Production (systemd user service)

```bash
# Install (one-time)
ln -sf "$(pwd)/systemd/cmw-rag-mosec.service" ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable cmw-rag-mosec.service

# Manage
systemctl --user start/stop/restart/status cmw-rag-mosec

# Logs
journalctl --user -u cmw-rag-mosec -f
```

The service restarts automatically after reboot. Configured via `~/.config/systemd/user/cmw-rag-mosec.service`.

Config in `~/cmw-mosec/.env`:
```bash
ACTIVE_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
ACTIVE_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
ACTIVE_GUARD_MODEL=Qwen/Qwen3Guard-Gen-0.6B
SERVER_PORT=7998
```

#### Development (manual)

```bash
cd ~/cmw-mosec
source .venv/bin/activate
cmw-mosec serve --foreground
```

### HTTPS Reverse Proxy (Production)

In production, the Gradio app runs behind an **nginx** reverse proxy that handles TLS termination:

```
https://ennoia.slickjump.org  →  nginx (:443)  →  localhost:7860 (Gradio HTTP)
```

Key nginx config (`/etc/nginx/sites-enabled/ennoia.conf`):
- SSL certificate from Let's Encrypt (Certbot-managed)
- WebSocket proxy headers (`Upgrade`, `Connection`) for Gradio streaming
- `proxy_buffering off` — required for real-time token streaming
- `proxy_read_timeout 300s` — long-running requests
- HTTP → HTTPS redirect (301)

The RAG app itself is not TLS-aware — it serves plain HTTP on `GRADIO_SERVER_PORT`.

## Troubleshooting

### No Virtual Environment Found

Create and activate a virtual environment:
- Linux: `python3 -m venv .venv`
- Windows: `python -m venv .venv`

### Settings Validation Error

Ensure `.env` file contains all required variables. Check `.env-example` for reference.

### No Relevant Results Found

- Verify index was built (Step 3)
- Check ChromaDB collection: `./data/chromadb_data/`
- Run diagnostics: `python rag_engine/scripts/maintain_chroma.py --action diagnose`
- Inspect schema: `python rag_engine/scripts/inspect_db_schema.py`
- **Behavior**: When no documents are retrieved, the system injects a "no results" message into the LLM context. The LLM will respond appropriately, acknowledging that no relevant information was found in the knowledge base, rather than blocking completely.

### ChromaDB Maintenance

Diagnose database health:
```bash
python rag_engine/scripts/maintain_chroma.py --action diagnose
```

Commit pending transactions:
```bash
python rag_engine/scripts/maintain_chroma.py --action commit-wal
```

Inspect data model and sample records:
```bash
python rag_engine/scripts/inspect_db_schema.py --samples 10
```

### Normalizing kbId Values

If you have documents with suffixed kbIds (e.g., `"4578-toc"`) from older indexing:
```bash
# Check what would be normalized (dry-run)
python rag_engine/scripts/migrate_normalize_kbids.py

# Apply normalization
python rag_engine/scripts/migrate_normalize_kbids.py --apply
```

After migration, reindex documents with `build_index.py` to restore them with normalized kbIds.

### FRIDA Model Download Fails / Disk Space

The FRIDA model requires ~4 GB. Move HuggingFace cache to `/mnt/d`:

```bash
export HF_HOME=/mnt/d/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/d/.cache/huggingface/hub

# Make persistent
echo 'export HF_HOME=/mnt/d/.cache/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/mnt/d/.cache/huggingface/hub' >> ~/.bashrc
source ~/.bashrc
```

For complete WSL migration guide, see `docs/troubleshooting/wsl-disk-space.md`.

### Port Already in Use

Change port in `.env`:
```
GRADIO_SERVER_PORT=7861
```

### Gemini API Rate Limits

Switch to OpenRouter in `.env`:
```
DEFAULT_LLM_PROVIDER=openrouter
```

## Development

### Running Tests

```bash
# Activate venv first
pytest rag_engine/tests/
```

Notes on tests:

- Retriever tests expect summarization-first budgeting; assertions check token budgets rather than hard-coding article counts.
- Token accounting is centralized in `rag_engine/llm/token_utils.py` and used consistently in tests and implementation.
- Widget testing: Test the production widget via `ai-widget.php` on kb.comindware.ru (or a local KB checkout) with browser dev tools. Automated browser tests are not included in the test suite. Key areas to verify:
  - Widget appears/disappears correctly
  - Resize functionality works
  - Gradio app loads and functions properly
  - localStorage persistence works
  - Responsive design on mobile devices

### Code Style

Uses Ruff for linting:
```bash
ruff check rag_engine/
```

Configuration follows LangChain and Gradio best practices (see `pyproject.toml`).

## License

See [LICENSE](LICENSE) file for details.

## Additional Resources

- Testing Documentation: [docs/TESTING.md](docs/TESTING.md)
- Progress Reports: [docs/progress_reports/](docs/progress_reports/)
