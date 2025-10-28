# Testing Guide for RAG Engine

## Quick Start

### WSL/Linux
```bash
./setup_and_test.sh
```

### Windows PowerShell
```powershell
.\setup_and_test.ps1
```

This will:
1. Create/activate virtual environment
2. Install all dependencies
3. Run linter
4. Run all tests

## Manual Testing

### Setup Environment

**WSL/Linux**:
```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install -r rag_engine/requirements.txt
```

**Windows**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r rag_engine\requirements.txt
```

### Run Tests

**All tests**:
```bash
python -m pytest rag_engine/tests/ -v
```

**Specific test file**:
```bash
python -m pytest rag_engine/tests/test_llm_manager.py -v
python -m pytest rag_engine/tests/test_retriever.py -v
```

**With coverage**:
```bash
pip install pytest-cov
python -m pytest rag_engine/tests/ --cov=rag_engine --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Test Coverage

### Tests Written (27 total)

#### LLM Manager (15 tests)
- ✅ Model configuration validation
- ✅ Dynamic token limit retrieval
- ✅ Model-specific context windows
- ✅ Fallback to default config
- ✅ Partial model name matching
- ✅ Context budgeting scaling

#### Retriever (12 tests)
- ✅ Article loading from files
- ✅ Frontmatter removal
- ✅ Chunk grouping by kbId
- ✅ Complete article retrieval (not chunks!)
- ✅ Context budgeting with token limits
- ✅ Multiple article handling
- ✅ Error handling

#### Smoke Tests (1 test)
- ✅ Basic chunking functionality

## Critical Tests to Verify

### 1. Dynamic Token Limits
```bash
pytest -k "test_get_current_llm_context_window" -v
```
**Expected**: Returns 1,048,576 tokens for gemini-1.5-flash

### 2. Article Loading
```bash
pytest -k "test_retrieve_returns_articles" -v
```
**Expected**: Returns Article objects, not chunks

### 3. Context Budgeting
```bash
pytest -k "test_context_budgeting" -v
```
**Expected**: Respects 75% of context window

## Linting

```bash
ruff check rag_engine/ --fix
```

## Success Criteria

- ✅ All 27 tests pass
- ✅ No linter errors
- ✅ Token limits are model-specific (not hardcoded 8K)
- ✅ Retriever returns Articles (not chunks)
- ✅ Context budgeting uses dynamic limits

## Troubleshooting

### "No module named pytest"
```bash
pip install pytest
```

### "No module named rag_engine"
Run tests from project root:
```bash
cd /path/to/cmw-rag
python -m pytest rag_engine/tests/
```

### Import errors
Install all dependencies:
```bash
pip install -r rag_engine/requirements.txt
```

## Next Steps

1. ✅ Run setup script
2. ✅ Verify all tests pass
3. 🔲 Test with real data (requires FRIDA model download)
4. 🔲 E2E integration test (optional, slow)

See `docs/progress_reports/2025-10-28-testing-and-verification.md` for detailed report.

