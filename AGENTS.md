# Agent Guide for CMW RAG Engine

This document provides essential commands, code style guidelines, and development rules for agents.

**Philosophy:** Lean, dry, minimal, abstract, modular, pythonic, beautiful code. Deduplicated, reusable, non-breaking.

---

## 🛠️ Environment & Commands

### Activation
Always activate venv before running commands:
- Linux/WSL: `source .venv/bin/activate`
- PowerShell: `.venv\Scripts\Activate.ps1`
- Install deps: `pip install -r rag_engine/requirements.txt`

### Testing (pytest - config in pyproject.toml)
```bash
pytest                                    # Run all tests
pytest rag_engine/tests/test_tools_utils.py  # Specific file
pytest rag_engine/tests/test_tools_utils.py::test_accumulate_articles  # Specific function
pytest --cov=rag_engine --cov-report=term-missing  # With coverage
pytest -m "not external"                  # Skip external tests (requires model downloads)
```

### Linting & Running
- **Lint:** `ruff check <file_path>` (configured in pyproject.toml, 100 char lines)
- **Lint all:** `ruff check .`
- **Start App:** `python rag_engine/api/app.py` or `bash rag_engine/scripts/start_app.sh`
- **Build Index:** `python rag_engine/scripts/build_index.py --source "path/to/docs" --mode folder`

---

## 📐 Code Style & Conventions

### General
- **Style:** LangChain-pure for LangChain code, Gradio-pure for Gradio code
- **Architecture:** Separation of concerns. Group code by function in different files.
- **Extensibility:** Ensure testability and extensibility
- **Purity:** Prefer purity for respective frameworks

### Formatting & Imports
- **Imports:** Always at top of file
- **Sorting:** Ruff handles (isort compatible)
- **Line Length:** 100 characters
- **Whitespace:** No orphan spaces on empty lines

### Naming
- **Variables/Functions:** `snake_case` (e.g., `process_document`)
- **Classes:** `PascalCase` (e.g., `RAGIndexer`)
- **Constants:** `UPPER_CASE` (e.g., `DEFAULT_TIMEOUT`)
- **Private:** Prefix with `_` (e.g., `_internal_helper`)

### Typing
- Standard Python type hints (`str`, `int`, `list[str]`, `dict[str, Any]`)
- Use Pydantic for data validation/settings (`BaseSettings`)

### Error Handling
- Avoid unnecessary try-catches
- Prefer robust, explicit logic
- Catch exceptions only when necessary and meaningful
- Avoid hardcoded fallbacks - code should fail gracefully or handle edge cases explicitly

### Logging & Comments
- Do not delete logging; update if needed
- Add comments for *why*, not *what*
- Update existing comments; don't delete them
- Follow PEP 257 (Google style preferred)

---

## 🧪 Test Practices

Test **behavior**, not implementation (Google Test Primer, IBM Unit Testing Guidelines):

1. **Test Outcomes, Not Mechanisms** - Don't test internal function calls
2. **Avoid Hardcoded Values** - Assert functional requirements, not specific ports/paths
3. **Test Behavior Contracts** - Define inputs → outputs, test the contract
4. **Use Mocks Judiciously** - Mock external deps (databases, APIs), not internal details
5. **Test Real Scenarios** - User-facing behavior, edge cases, error handling
6. Always run testsuite after any changes

---

## 🤖 Agent Instructions

### Before Coding
1. Search docs/internet for framework/library references
2. Scan official documentation hierarchy for best practices
3. Gather ground truth before planning

### Planning
- PLAN your course of action after gathering reference info
- Reanalyze changes twice for introduced issues

### Verification
- Run `ruff check <modified_file>` after any changes
- Run relevant tests after changes

### Commit Discipline
- Do NOT create/push commits unless explicitly asked
- Keep messages concise, structured, strictly relevant
- Avoid fluff
- Focus on the "why" not the "what"

### Secrets & Config
- NEVER hardcode secrets - use environment variables
- NEVER commit `.env` files - use `.env-example` template with placeholders
- Store all env-specific config in env vars (12-factor)

### No Breakage
- Never break existing code
- When refactoring, change only relevant parts

---

## 📁 Project Structure

- **Tests:** `rag_engine/tests`
- **Docs:** `docs/progress_reports/`
- **Scripts:** `rag_engine/scripts/`
- **Config:** `pyproject.toml` (linting), `.env-example`

---

## 📚 References

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
