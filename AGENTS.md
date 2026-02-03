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

### 5. Test Practices

Following industry best practices from Google Test Primer and IBM Unit Testing Guidelines:

#### Test Behavior, Not Implementation

**Core Principle:** Tests should validate what code **does**, not **how** it does it.

**❌ BAD - Testing implementation details:**
```python
# Testing hardcoded ports or specific paths
def test_config_loads_infinity():
    config = load_embedding_config("infinity_qwen3_8b")
    assert config.endpoint == "http://localhost:8000/v1"  # Fragile!
    mock_post.assert_called_once_with("/v1/embeddings")  # Implementation detail!
```

**✅ GOOD - Testing behavior:**
```python
# Testing valid configuration and functionality
def test_config_loads_infinity():
    config = load_embedding_config("infinity_qwen3_8b")
    assert config.endpoint.startswith("http://localhost:")  # Valid URL pattern
    assert "/v1" in config.endpoint  # Functional requirement
    assert config.type == "server"
```

**Key Guidelines:**

1. **Test Outcomes, Not Mechanisms**
   - Test that a feature works correctly
   - Don't test internal function calls or implementation paths
   - Example: Test that config returns a valid server endpoint, not which port is used

2. **Avoid Hardcoded Values**
   - Don't assert on specific ports, paths, or internal states
   - Assert on functional requirements and valid patterns
   - Example: Assert endpoint is a valid URL, not "http://localhost:8000"

3. **Test Behavior Contracts**
   - Define what the function should do (inputs → outputs)
   - Test the contract, not the implementation
   - Example: "Given a query, return relevant articles" not "call vector_search()"

4. **Use Mocks Judiciously**
   - Mock external dependencies (databases, APIs)
   - Don't mock internal implementation details
   - Example: Mock ChromaDB client, not internal collection.get()

5. **Test Real Scenarios**
   - Test user-facing behavior
   - Test edge cases and error handling
   - Example: Test "no results found" scenario, not "empty list returned"

**Benefits of Testing Behavior:**
- ✅ Tests remain valid when implementation changes
- ✅ Tests document intended functionality
- ✅ Easier to refactor code without breaking tests
- ✅ Tests serve as specifications

**References:**
- [Google Test Primer](https://google.github.io/googletest/primer.html)
- [IBM Unit Testing Best Practices](https://www.ibm.com/think/insights/unit-testing-best-practices)

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

### 12-Factor App Principles
Following twelve-factor methodology for SaaS apps:

- **Codebase:** One codebase tracked in revision control, many deploys.
- **Dependencies:** Declare all dependencies explicitly in `requirements.txt` (and `pyproject.toml` for build metadata). See Environment Setup section for isolation commands.
- **Config:** Store all environment-specific config in env vars (never in code). Use `.env` files for local development, ensure the codebase could be open-sourced without compromising credentials.
- **Backing Services:** Treat vector stores, databases, caches as attached resources accessed via URLs/credentials in config. No distinction between local and third-party services in code.
- **Build, Release, Run:** Strictly separate build and run stages. See scripts in `rag_engine/scripts/` for build/release automation.
- **Processes:** Execute the app as one or more stateless processes. Session data in backing services. Note: Local Chroma SQLite is acceptable for dev; production should use external vector store.
- **Port Binding:** Export services via port binding. App should be self-contained and bind to a port specified via env var.
- **Concurrency:** Scale out via the process model. Use Gradio's concurrency limits (`gradio_default_concurrency_limit`) and thread pools for non-blocking operation under multi-user load.
- **Disposability:** Maximize robustness with fast startup and graceful shutdown. Processes should start quickly and shut down gracefully on SIGTERM, finishing current requests before exiting.
- **Dev/Prod Parity:** Keep development, staging, and production as similar as possible. Use the same backing services (types and versions) across all environments.
- **Logs:** Treat logs as event streams. Prefer stdout for containerized deployments; file logging optional via `LOG_FILE_ENABLED` env var. Default to both for local dev.
- **Admin Processes:** Run admin/management tasks as one-off processes using the same codebase and config as the main app.

### Project Specifics
- **Docs:** Put reports in `docs/progress_reports/`.
- **Tests:** Put tests in `rag_engine/tests`.
- **Updates:** Always update `README.md` if changes affect it.
- **Refactoring:** When refactoring, change only the relevant parts.
