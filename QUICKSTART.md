# RAG Engine MVP - Quick Start Guide

## Prerequisites

- Python 3.13+
- Virtual environment (`.venv` for Windows, `.venv-wsl` for WSL/Linux)
- API keys: Google (Gemini) or OpenRouter
- Source documents (markdown files) for indexing

## Step 1: Setup Environment

### WSL/Linux

```bash
# Activate venv
source .venv-wsl/bin/activate

# Install dependencies (if not already done)
pip install -r rag_engine/requirements.txt
```

### Windows PowerShell

```powershell
# Activate venv
.venv\Scripts\Activate.ps1

# Install dependencies (if not already done)
pip install -r rag_engine\requirements.txt
```

## Step 2: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

   Or on Windows:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Edit `.env` and add your API keys:
   - At minimum, set `GOOGLE_API_KEY` (for Gemini) or `OPENROUTER_API_KEY`
   - Other settings have defaults but can be customized

## Step 3: Build the Index

Before running the app, you need to index your documents. Choose one of three modes:

### Mode 3: Compiled MD Folder (Default, Recommended)

If you have a folder with markdown files:

```bash
python rag_engine/scripts/build_index.py \
  --source "path/to/your/markdown/folder/" \
  --mode folder
```

Example:
```bash
python rag_engine/scripts/build_index.py \
  --source "phpkb_content/798. Версия 5.0. Текущая рекомендованная/" \
  --mode folder
```

### Mode 2: Single Combined MD File

If you have a single large markdown file:

```bash
python rag_engine/scripts/build_index.py \
  --source "kb.comindware.ru.platform_v5_for_llm_ingestion.md" \
  --mode file
```

### Mode 1: MkDocs Export (Optional)

First export compiled MD from an external MkDocs project:

```bash
python rag_engine/scripts/run_mkdocs_export.py \
  --project-dir ../CBAP_MKDOCS_RU \
  --inherit-config mkdocs_guide_complete_ru.yml \
  --output-dir ./data/compiled_md_for_rag
```

Then index it:

```bash
python rag_engine/scripts/build_index.py \
  --source ./data/compiled_md_for_rag \
  --mode mkdocs
```

## Step 4: Run the RAG Agent

### Using Helper Scripts (Recommended)

**WSL/Linux:**
```bash
bash rag_engine/scripts/start_app.sh
```

**Windows PowerShell:**
```powershell
.\rag_engine\scripts\start_app.ps1
```

### Manual Start

**WSL/Linux:**
```bash
source .venv-wsl/bin/activate
python rag_engine/api/app.py
```

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
python rag_engine\api\app.py
```

## Step 5: Access the Application

Once running, access:

- **UI**: http://localhost:7860
- **REST API**: http://localhost:7860/api/query_rag

### Chat Interface

The Gradio ChatInterface supports:
- Bilingual queries (Russian/English)
- Streaming responses
- Automatic citations
- Chat history

### REST API

You can also call the API programmatically:

```python
import requests

response = requests.post(
    "http://localhost:7860/api/query_rag",
    json={"question": "How to use N3?", "provider": "gemini", "top_k": 5}
)
print(response.json())
```

## Troubleshooting

### "No virtual environment found"

Make sure you've created and activated a venv:
- WSL: `python3 -m venv .venv-wsl`
- Windows: `python -m venv .venv`

### "Settings validation error"

Check that your `.env` file has all required variables. See `.env.example` for the full list.

### "No relevant results found"

- Make sure you've built the index first (Step 3)
- Verify your ChromaDB collection has documents: check `./data/chromadb_data/`
- Try reindexing with `--reindex` flag

### "FRIDA model download fails" / "No space left on device"

The FRIDA model requires ~4 GB of disk space. If you get an `OSError: [Errno 28] No space left on device`:

**Move HuggingFace cache to /mnt/d** (quick solution):

```bash
# Set cache to D: drive
export HF_HOME=/mnt/d/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/d/.cache/huggingface/hub

# Make it persistent (add to ~/.bashrc)
echo 'export HF_HOME=/mnt/d/.cache/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/mnt/d/.cache/huggingface/hub' >> ~/.bashrc
source ~/.bashrc
```

**Other options:**

- **Clean up existing cache**:
  ```bash
  python -c "from huggingface_hub import scan_cache_dir; scan_cache_dir().delete_revisions([], min_size=1024**3*2)"
  ```

- **Pre-download the model** (to catch errors early):
  ```bash
  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('ai-forever/FRIDA')"
  ```

- **For complete WSL migration**: See `docs/troubleshooting/wsl-disk-space.md`

### Port already in use

Change the port in `.env`:
```
GRADIO_SERVER_PORT=7861
```

### Gemini API rate limits

Switch to OpenRouter by updating `.env`:
```
DEFAULT_LLM_PROVIDER=openrouter
```

## Full Workflow Example

```bash
# 1. Setup (one-time)
source .venv-wsl/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
cp .env.example .env
# Edit .env with your keys

# 2. Index documents
python rag_engine/scripts/build_index.py \
  --source "path/to/markdown/files/" \
  --mode folder

# 3. Run app
python rag_engine/api/app.py

# 4. Open browser to http://localhost:7860
```

## What's Next?

- Test queries in the ChatInterface
- Try the REST API endpoint
- Customize prompts in `rag_engine/llm/prompts.py`
- Adjust retrieval settings in `.env` (TOP_K_RETRIEVE, TOP_K_RERANK, etc.)

For more details, see the plan documents in `.cursor/plans/`.

