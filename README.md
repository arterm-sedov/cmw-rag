# CMW RAG Engine

Production-ready RAG (Retrieval-Augmented Generation) engine for document Q&A with support for MkDocs, markdown folders, and single-file ingestion. Features FRIDA embeddings (Russian/English), ChromaDB vector storage, reranking, and multi-LLM support with streaming responses.

[Ask DeepWiki](https://deepwiki.com/arterm-sedov/cmw-rag)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/arterm-sedov/cmw-rag)

## Features

- **Multi-source ingestion**: Support for MkDocs export, markdown folders, and single combined files
- **Bilingual support**: FRIDA embeddings for Russian and English content
- **Advanced retrieval**: Vector search with optional cross-encoder reranking
- **Multi-LLM support**: Gemini (default) and OpenRouter with streaming responses
- **Dynamic context budgeting**: Summarization-first trimming guided by the user question; falls back to lightweight stitching when needed
- **Immediate model fallback**: Optional auto-fallback to allowed larger-context models when estimated tokens exceed the current model
- **Consistent token accounting**: Shared `token_utils` for system + question + context + output budgeting
- **Web interface**: Gradio ChatInterface with citations and chat history
- **REST API**: Programmatic access for integration
- **Context-aware**: Complete article context with citation support

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

Optional flags and behavior
- `--max-files N`: Stop after indexing N documents that actually changed or are new. Unchanged files (by timestamp) are skipped and do not count against this limit.
- Incremental reindexing: If a file’s modification time (mtime) is unchanged since the last run, it is skipped. If it changed, all previous chunks for that document are replaced with fresh ones.
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

See `.env.example` for full configuration options.

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
- Reindex with `--reindex` flag

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
