# Agent Guide for CMW RAG Engine

**Philosophy:** Lean, dry, minimal, abstract, modular, pythonic, deduplicated, non-breaking code.

---

## 🛠️ Environment & Commands

### Virtual Environment (ALWAYS activate first)
```bash
source .venv/bin/activate        # Linux/WSL
.venv\Scripts\Activate.ps1        # PowerShell
pip install -r rag_engine/requirements.txt
```

### Testing
```bash
pytest                                           # All tests
pytest rag_engine/tests/test_tools_utils.py      # Single file
pytest rag_engine/tests/test_tools_utils.py::test_accumulate_articles  # Single function
pytest --cov=rag_engine --cov-report=term-missing  # With coverage
pytest -m "not external"                          # Skip external tests
pytest -m external                                # Only external tests
```

### Linting & Type Checking
- **Lint:** `ruff check <file_path>` (configured in pyproject.toml)
- **Line length:** 100 characters | **Target Python:** 3.11
- **Coverage:** Minimum 95% required (fails CI below)

### Running the App
```bash
python rag_engine/api/app.py
bash rag_engine/scripts/start_app.sh
python rag_engine/scripts/build_index.py --source "path/to/docs" --mode folder
```

---

## 📐 Code Style & Conventions

### Before Coding
1. Search internet for framework/library references
2. Scan official documentation hierarchy for best practices
3. Gather ground truth before planning
4. **PLAN** your course of action after gathering reference info

### General
- **Style:** LangChain-pure for LangChain code, Gradio-pure for Gradio code
- **Architecture:** Separation of concerns. Group code by function in different files
- **Purity:** Prefer purity for respective frameworks
- Produce flawless code. Reanalyze changes twice for issues introduced

### Imports & Formatting
- Place imports always at top of file
- Ruff handles sorting (isort compatible)
- `known-first-party = ["rag_engine"]` in pyproject.toml
- **Line Length:** 100 characters | **Whitespace:** No orphan spaces on empty lines
- Follow PEP 8 (https://peps.python.org/pep-0008/)

### Naming Conventions
- **Variables/Functions:** `snake_case` (e.g., `process_document`)
- **Classes:** `PascalCase` (e.g., `RAGIndexer`)
- **Constants:** `UPPER_CASE` (e.g., `DEFAULT_TIMEOUT`)
- **Private:** Prefix with `_` (e.g., `_internal_helper`)

### Typing
- Standard Python type hints (`str`, `int`, `list[str]`, `dict[str, Any]`)
- Use Pydantic for data validation/settings (`BaseSettings`)
- Follow PEP 257 for docstrings (Google style preferred)

### Error Handling
- Avoid unnecessary try-catches
- Prefer robust, explicit logic
- Catch exceptions only when necessary and meaningful
- Avoid hardcoded fallbacks - code should fail gracefully or handle edge cases explicitly
- "Code should work correctly without hidden fallbacks"

### Logging & Comments
- Do not delete logging; update if needed
- Add comments for *why*, not *what*
- Update existing comments; don't delete them

### Secrets & Config
- NEVER hardcode secrets - use environment variables
- NEVER commit `.env` files - use `.env-example` template
- Store all env-specific config in env vars (12-factor)

---

## 🧪 Test Practices

Test **behavior**, not implementation:
1. **Test Outcomes, Not Mechanisms** - Don't test internal function calls
2. **Avoid Hardcoded Values** - Assert functional requirements, not specific ports/paths
3. **Test Behavior Contracts** - Define inputs → outputs, test the contract
4. **Use Mocks Judiciously** - Mock external deps (databases, APIs), not internal details
5. **Test Real Scenarios** - User-facing behavior, edge cases, error handling
6. Always run test suite after any changes

### Verification Checklist
- [ ] Run `ruff check <modified_file>` after any changes
- [ ] Run relevant tests after changes
- [ ] Ensure test coverage stays above 95%

---

## 🤖 Agent Instructions

### Commit Discipline (per .cursor/rules/cmw_rag_commit.mdc)
- Do NOT create/push commits unless explicitly asked
- Generate commit message text only (do not stage or push)
- Keep messages concise, structured, and strictly relevant
- Keep length to necessary minimum. Avoid blabber
- Focus on the "why" not the "what"

### No Breakage
- Never break existing code
- When refactoring, change only relevant parts
- Do not duplicate code - encapsulate reused code in methods/functions

---

## 📁 Project Structure

```
rag_engine/
├── tests/              # Test files
├── api/                # API/app entry points
├── scripts/            # Build/maintenance scripts
├── llm/                # LLM provider configs
├── core/                # Indexing pipeline (indexer, chunker, document_processor)
├── retrieval/           # Retrieval pipeline
├── storage/             # Vector store (ChromaDB)
├── tools/               # LangChain tools
├── utils/               # Utilities (token counting, etc.)
└── ...                  # Other modules
```

**Config:** `pyproject.toml` (linting), `.env-example`

---

## 📚 Key References

### LangChain
- https://docs.langchain.com/oss/python/langchain/overview
- https://docs.langchain.com/oss/python/langchain/agents
- https://reference.langchain.com/python/langchain/
- https://langchain-ai.github.io/langgraph/concepts/

### Gradio
- https://www.gradio.app/docs
- https://www.gradio.app/guides/the-interface-class/
- https://www.gradio.app/guides/blocks-and-event-listeners/

### Python
- https://peps.python.org/pep-0008/ (Style)
- https://peps.python.org/pep-257/ (Docstrings)
- https://google.github.io/styleguide/pyguide.html

---

**IMPORTANT:** Code must be clean, lean, brilliant, dry, minimalistic, abstract, non-duplicating, non-breaking, perfect, and pythonic.
