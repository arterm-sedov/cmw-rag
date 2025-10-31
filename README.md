# CMW RAG Engine

Production-ready RAG (Retrieval-Augmented Generation) engine for document Q&A with support for MkDocs, markdown folders, and single-file ingestion. Features FRIDA embeddings (Russian/English), ChromaDB vector storage, reranking, and multi-LLM support with streaming responses.

## AI-Enabled Repo

Chat with DeepWiki to get answers about this repo:

[Ask DeepWiki](https://deepwiki.com/arterm-sedov/cmw-rag)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/arterm-sedov/cmw-rag)

## Features

- **Multi-source ingestion**: Support for MkDocs export, markdown folders, and single combined files
- **Smart timestamp detection**: Three-tier fallback (frontmatter → Git → file modification) for accurate incremental reindexing
- **Bilingual support**: FRIDA embeddings for Russian and English content
- **Advanced retrieval**: Vector search with optional cross-encoder reranking
- **Multi-vector queries**: Split long queries into token-aware segments, retrieve per segment, union + rerank
- **Multi-LLM support**: Gemini (default) and OpenRouter with streaming responses
- **Dynamic context budgeting**: Summarization-first trimming guided by the user question; falls back to lightweight stitching when needed
- **Immediate model fallback**: Optional auto-fallback to allowed larger-context models when estimated tokens exceed the current model
- **Consistent token accounting**: Shared `token_utils` for system + question + context + output budgeting
- **Web interface**: Gradio ChatInterface with citations and chat history
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

## Prerequisites

- Python 3.13+
- Virtual environment (`.venv` for Windows, `.venv-wsl` for WSL/Linux)
- API keys: Google (Gemini) or OpenRouter
- Source documents (markdown files) for indexing

## Quick Start

### 1. Setup Environment

**WSL/Linux:**
```bash
source .venv-wsl/bin/activate
pip install -r rag_engine/requirements.txt
```

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
pip install -r rag_engine\requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and configure your API keys:

```bash
# WSL/Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Minimum required: `GOOGLE_API_KEY` (for Gemini) or `OPENROUTER_API_KEY`

### 3. Build the Index

Choose one of three ingestion modes:

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

**Manual start:**

```bash
# WSL/Linux
source .venv-wsl/bin/activate
python rag_engine/api/app.py

# Windows PowerShell
.venv\Scripts\Activate.ps1
python rag_engine\api\app.py
```

### 5. Access the Application

- **Web UI**: http://localhost:7860
- **REST API**: http://localhost:7860/api/query_rag

## Usage

### Web Interface

The Gradio ChatInterface provides:
- Bilingual queries (Russian/English)
- Streaming responses
- Automatic citations
- Chat history
- Graceful handling of empty results (see Troubleshooting section)

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
- Token counting uses fast approximation (chars // 4) for strings exceeding `RETRIEVAL_FAST_TOKEN_CHAR_THRESHOLD` to avoid slow encodes.

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

## Configuration

Environment variables (configure in `.env`):

- `GOOGLE_API_KEY`: Google Gemini API key (required if using Gemini)
- `OPENROUTER_API_KEY`: OpenRouter API key (required if using OpenRouter)
- `DEFAULT_LLM_PROVIDER`: Default provider (`gemini` or `openrouter`)
- `LLM_FALLBACK_ENABLED`: Enable immediate model fallback (`true`/`false`)
- `LLM_FALLBACK_PROVIDER`: Provider for fallback (`gemini`/`openrouter`), otherwise inferred per model
- `LLM_ALLOWED_FALLBACK_MODELS`: Comma-separated list of allowed fallback models
- `LLM_SUMMARIZATION_ENABLED`: Enable summarization-first budgeting (`true`/`false`)
- `LLM_SUMMARIZATION_TARGET_TOKENS_PER_ARTICLE`: Optional override for per-article summary target
- `TOP_K_RETRIEVE`: Initial retrieval count (default: 20)
- `TOP_K_RERANK`: Final results after reranking (default: 10)
- `GRADIO_SERVER_PORT`: Web UI port (default: 7860)
- `EMBEDDING_MODEL`: Embedding model name (default: `ai-forever/FRIDA`)
- `EMBEDDING_DEVICE`: Device for embeddings (`cpu` or `cuda`)
- `MEMORY_COMPRESSION_THRESHOLD_PCT`: Trigger compression when estimated request exceeds this percent of the model window (default: `85`)
- `MEMORY_COMPRESSION_TARGET_TOKENS`: Target tokens for the compressed history turn (default: `1000`)
- `RETRIEVAL_FAST_TOKEN_CHAR_THRESHOLD`: For strings exceeding this length, approximate tokens as chars // 4 instead of full encode (default: `200000`)

See `.env.example` for full configuration options.

### Retrieval – Multi-vector and Query Decomposition

Environment flags controlling long-query behavior:

- `RETRIEVAL_MULTIQUERY_ENABLED` (default: `true`): Enable multi-vector query retrieval
- `RETRIEVAL_MULTIQUERY_MAX_SEGMENTS` (default: `4`): Max query segments
- `RETRIEVAL_MULTIQUERY_SEGMENT_TOKENS` (default: `448`): Target tokens per segment (≤ 512)
- `RETRIEVAL_MULTIQUERY_SEGMENT_OVERLAP` (default: `64`): Overlap tokens between segments
- `RETRIEVAL_MULTIQUERY_PRE_RERANK_LIMIT` (default: `60`): Cap merged candidates before rerank
- `RETRIEVAL_QUERY_DECOMP_ENABLED` (default: `false`): Enable LLM-based query decomposition
- `RETRIEVAL_QUERY_DECOMP_MAX_SUBQUERIES` (default: `4`): Max sub-queries to generate

Recommended ranges:
- `SEGMENT_TOKENS`: 384–512; `OVERLAP`: 32–96; `MAX_SEGMENTS`: ≤ 4; `PRE_RERANK_LIMIT`: ≈ 3×`TOP_K_RETRIEVE`.

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
└── utils/            # Logging and formatting utilities
```

## Troubleshooting

### No Virtual Environment Found

Create and activate a virtual environment:
- WSL: `python3 -m venv .venv-wsl`
- Windows: `python -m venv .venv`

### Settings Validation Error

Ensure `.env` file contains all required variables. Check `.env.example` for reference.

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
