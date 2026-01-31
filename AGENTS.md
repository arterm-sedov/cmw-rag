# Agent Guide for CMW RAG Engine

This document provides essential commands, code style guidelines, and development rules for agents working in this repository.

## 🛠️ Build, Lint & Test Commands

### 1. Environment Setup
Always ensure the virtual environment is activated before running commands.
- **Linux (native):** `source .venv/bin/activate`
- **Windows (WSL):** `source .venv-wsl/bin/activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Install Dependencies:** `pip install -r rag_engine/requirements.txt`

### 2. Testing
Use `pytest` for testing. Configuration is in `pyproject.toml`.
- **Run all tests:**
  ```bash
  pytest
  ```
- **Run a specific test file:**
  ```bash
  pytest rag_engine/tests/test_tools_utils.py
  ```
- **Run a specific test function:**
  ```bash
  pytest rag_engine/tests/test_tools_utils.py::test_accumulate_articles
  ```
- **Run with coverage:**
  ```bash
  pytest --cov=rag_engine --cov-report=term-missing
  ```

### 3. Linting & Formatting
The project uses `ruff` for linting and import sorting. Configuration is in `pyproject.toml`.
- **Check all files:**
  ```bash
  ruff check rag_engine/
  ```
- **Check a specific file:**
  ```bash
  ruff check path/to/file.py
  ```
- **Fix issues (safe fixes):**
  ```bash
  ruff check --fix path/to/file.py
  ```
- **Note:** Only lint the files that were modified, not the entire codebase. Be critical about Ruff reports; implement only necessary changes.

### 4. Running the Application
- **Start App (Bash):** `bash rag_engine/scripts/start_app.sh`
- **Start App (Python):** `python rag_engine/api/app.py`
- **Build Index:** `python rag_engine/scripts/build_index.py --source "path/to/docs" --mode folder`

---

## 📐 Code Style & Conventions

### General Philosophy
- **Style:** LangChain-pure, dry, lean, modular, pythonic.
- **Architecture:** Separation of concerns. Isolate code by function. Group code by function in different files to avoid clutter.
- **Extensibility:** Ensure testability and extensibility.
- **Purity:** Prefer LangChain purity for LangChain code and Gradio purity for Gradio code.
- **Libraries:** Use `sentence_transformers` when needed and practical.

### Formatting & Imports
- **Imports:** Always place imports at the top of the file.
- **Sorting:** `ruff` handles import sorting (isort compatible).
- **Line Length:** 100 characters (as per `pyproject.toml`).
- **Whitespace:** Do not add orphan spaces on empty lines.

### Naming Conventions
- **Variables/Functions:** `snake_case` (e.g., `process_document`, `user_id`)
- **Classes:** `PascalCase` (e.g., `RAGIndexer`, `Settings`)
- **Constants:** `UPPER_CASE` (e.g., `DEFAULT_TIMEOUT`)
- **Private Members:** Prefix with `_` (e.g., `_internal_helper`)

### Typing
- **Type Hints:** Use standard Python type hints (`str`, `int`, `list[str]`, `dict[str, Any]`).
- **Pydantic:** Use Pydantic models for data validation and settings (e.g., `class Settings(BaseSettings)`).

### Error Handling
- **Philosophy:** Avoid unnecessary try-catches. Prefer robust, explicit logic.
- **Catching:** Catch exceptions only when necessary and meaningful.
- **Fallbacks:** Avoid hardcoded fallbacks. Code should fail gracefully or handle expected edge cases explicitly.

### Logging & Comments
- **Logging:** Do not delete logging; update it if necessary.
- **Comments:** Add comments for *why*, not *what*. Update existing comments; do not delete them.
- **Docs Strings:** Follow PEP 257 docstring conventions (Google style is preferred).

---

## 🤖 Agent Instructions (OpenCode, Cursor, Copilot)

### Commit Messages
- **Format:** Concise, structured, and strictly relevant to the changes.
- **Content:** Keep length to the necessary minimum. Avoid fluff.
- **Tool:** Use the commit message generation tool but verify content.

### Agent Behavior
- **Information Gathering:**
    - Search docs/internet before coding. Digest best practices from official sources.
    - Do not read just one page; scan the docs, links, and hierarchy to ground actions in truth.
- **Planning:** PLAN your course of action before implementing.
- **Verification:**
    - Run `ruff check <modified_file>` after making changes.
    - Run relevant tests after changes.
    - Reanalyze changes twice for introduced issues.
- **Secrets:** NEVER hardcode secrets. Use environment variables.
- **No Breakage:** Never break existing code.
- **Git Commits:** Do NOT create or push commits unless explicitly asked by the user.

### Project Specifics
- **Docs:** Put reports in `docs/progress_reports/`.
- **Tests:** Put tests in `rag_engine/tests`.
- **Updates:** Always update `README.md` if changes affect it.
- **Refactoring:** When refactoring, change only the relevant parts.
