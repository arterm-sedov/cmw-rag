# Agent Guide for CMW RAG Engine

This document provides essential commands, code style guidelines, and development rules for agents working in this repository.

## Research & Planning

Before any coding, changes or implementation:

- Do a deep codebase research.
- Do a deep web research in the internet for reference documentation on frameworks, libraries, and best practices.
- Gather all information before planning course of action.
- Plan after gathering reference information, not before.
- Write a concise plan **file**:
    - actionable
    - detailed
    - TDD
    - step-by-step tasks
    - checkpoints
    - expected verification commands
    - follows best practices in TDD, SDD, Python, 12-factor agents and software

## Common Engineering Baseline and Design Principles

- Follow SDD for scope/contract clarity and TDD for behavior-first implementation.
- Keep code: lean, DRY, modular, and non-breaking, brilliant, abstract, minimal, genius.
- Follow best practices:
    - **TDD:** Write tests first, define behavior contracts. Test behavior, not implementation details.
    - **SDD:** Plan with specs, understand requirements before coding.
    - **Non-breaking:** Never break existing functionality.
    - **Lean:** Minimal code, no overengineering.
    - **Pythonic:** Follow Python idioms, prefer clarity over cleverness, explicit data contracts, strong typing.
    - **Modular:** Single responsibility, group related functionality.
    - **Open/Closed:** Design for extension without modification.
    - **DRY:** 2+ uses -> extract to helper, super dry, super lean.
- For LangChain see LangChain docs, repo and source code for reference, prefer LCEL/runnables, typed tool schemas, and streaming-safe patterns.
- For Gradio see Gradio's docs, repo and source code for reference, follow best practices, keep state/event flow explicit and UI logic separated from domain logic.
- Validate external data and avoid silent exception handling (`except: pass` is forbidden).
- Run lint and relevant tests for modified areas before completion.
- Never hardcode secrets; use environment variables and `.env.example` placeholders only.

## Dev Commands

```bash
# Activate venv first (required)
# Use the active/default Cursor terminal session first
source .venv/bin/activate        # Linux (native)
source .venv-wsl/bin/activate    # Windows (WSL)
.venv\Scripts\Activate.ps1       # Windows (PowerShell)

# Install dependencies
pip install -r rag_engine/requirements.txt

# Start app
bash rag_engine/scripts/start_app.sh
python rag_engine/api/app.py

# Build index
python rag_engine/scripts/build_index.py --source "path/to/docs" --mode folder

# Test
pytest
pytest rag_engine/tests/test_tools_utils.py
pytest rag_engine/tests/test_tools_utils.py::test_accumulate_articles
pytest --cov=rag_engine --cov-report=term-missing
pytest -m "not slow"
pytest -m integration
```

## Project Structure

- **App Entry:** `rag_engine/api/app.py`
- **Scripts:** `rag_engine/scripts/`
- **Tests:** `rag_engine/tests/`
- **Dependencies:** `rag_engine/requirements.txt`

## Code Style & Conventions

### General Philosophy

- Style: dry, lean, minimal, abstract, modular, pythonic, clean, testable.
- Prefer LangChain purity for LangChain code and Gradio purity for Gradio code.
- Use `sentence_transformers` when needed and practical.

### Formatting, Naming, and Typing

- Imports at the top of file; `ruff` handles import sorting.
- Line length: 100 (per `pyproject.toml`).
- Do not add orphan spaces on empty lines.
- Variables/functions: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_CASE`.
- Private members: `_prefixed`.
- Use Python type hints.
- Use Pydantic models for data validation and settings.
- Docstrings: PEP 257 (Google style preferred).
- Comments explain *why*, not *what*.

## Framework Conventions

- **LangChain:** Prefer LCEL/runnables, typed tool schemas, and streaming-safe patterns.
- **Gradio:** Keep state/event flow explicit and UI logic separated from domain logic.

## Error Handling

- Avoid unnecessary try-catches; use robust explicit logic.
- Catch exceptions only when meaningful.
- Avoid hardcoded fallbacks.
- Do not delete logging; update it when needed.
- No silent exception handling (`except: pass` is forbidden).

## Testing Guidelines

- Test behavior, not implementation details.
- Start with test suite before implementing features.
- Avoid hardcoding internal details in tests.
- Mock external dependencies judiciously; avoid mocking internal implementation details.
- Test real scenarios, edge cases, and user-facing outcomes.
- Use parametrization for cross-model/config testing.
- Verify shared-logic equivalence when multiple endpoints compute the same outcome.
- Prefer integration tests for endpoint-level behavior; use pytest markers:
  - `pytest -m "not slow"` for fast unit checks.
  - `pytest -m integration` for integration checks.

### Example: Behavior Over Implementation

Bad:

```python
def test_infinity_embeddings_implementation_details(mocker):
    mock_post = mocker.patch("rag_engine.embedding_client.post")
    client = InfinityEmbeddingClient(config=load_embedding_config("infinity_qwen3_8b"))
    client.embed("hello world")
    mock_post.assert_called_once_with(
        "http://localhost:8000/v1/embeddings",
        json={"input": "hello world"},
    )
```

Good:

```python
def test_infinity_embeddings_returns_vector_of_expected_size(infinity_client):
    vector = infinity_client.embed("hello world")
    assert isinstance(vector, list)
    assert len(vector) == infinity_client.dim
    assert all(isinstance(x, float) for x in vector)
```

References:

- https://google.github.io/googletest/primer.html
- https://www.ibm.com/think/insights/unit-testing-best-practices

## Verification Checklist

Before considering work complete:

1. Run `ruff check` on modified files.
2. Run relevant tests (unit/integration as applicable).
3. Confirm no user-facing regressions (non-breaking behavior).
4. Ensure shared logic remains DRY (extract helpers for repeated blocks).
5. Update docs/README when behavior, workflows, or commands changed.

## Documentation Guidelines

- Use clear heading hierarchy (single H1 per file).
- Front-load conclusions and recommendations.
- Use actionable, chunked sections (avoid walls of text).
- Keep source traceability for claims (inline citation where relevant).
- Add a blank line after headings and before lists in Markdown.
- Keep sections action-oriented: each section should clearly imply a decision or next step.
- Documentation files should be placed under `docs/` where applicable.
- Progress reports should use `docs/**/progress_reports/` with `YYYYMMDD_` prefix.
- Generate `YYYYMMDD` timestamps with native commands:
  - PowerShell: `Get-Date -Format "yyyyMMdd"`
  - Bash/WSL: `date +%Y%m%d`
  - Python (with active venv): `python -c "from datetime import datetime; print(datetime.now().strftime('%Y%m%d'))"`

## Security & Secrets

- NEVER hardcode secrets.
- Use `.env` files for local config, never commit secrets.
- NEVER commit `.env`; use `.env.example` placeholders only.
- Never include real passwords, keys, tokens, personal names, or business/entity names in code/tests/docs.
- Use synthetic neutral placeholders.
- Load secrets via dotenv in tests and code. Never hardcode sensitive information.

## Commit Guidelines

- Do NOT create or push commits unless explicitly asked by the user.
- If asked to only draft commit text, generate the message but do not add files, stage, push, or commit.
- Generate commit message text, but do NOT add files, stage, or push unless requested.
- Keep commit messages concise and strictly relevant.
- Keep commit message length to the necessary minimum.
- Verify generated commit text before use.
- Avoid blabber.

## Work Tracking & Reports

- Plans: `.opencode/plans/YYYYMMDD_<topic>/plan.md`
- Research: `.opencode/research/YYYYMMDD_<topic>/research.md`
- Progress: `.opencode/progress_reports/YYYYMMDD_<topic>/progress_YYYYMMDD.md`
- Use GitHub Markdown format; keep dated parent folders.
- Update `.opencode/README.md` and related docs when affected.

## UI/UX Principles

- **Clarity over clutter:** Remove redundant elements.
- **Maximize data-ink ratio:** Every element adds value.
- **Visual hierarchy:** Group related information consistently.
- **Progressive disclosure:** Essential info prominent, details on demand.
- **Data integrity:** Display zero values when meaningful, not just absence.

## 12-Factor App Principles

Based on https://12factor.net/ and https://github.com/humanlayer/12-factor-agents:

- One codebase, many deploys.
- Declare dependencies explicitly.
- Keep config in env vars.
- Treat backing services as attached resources.
- Separate build, release, and run stages.
- Prefer stateless processes with session state in backing services.
- Export services via port binding.
- Scale via process model.
- Optimize disposability (fast startup, graceful shutdown).
- Keep dev/prod parity.
- Treat logs as event streams.
- Run admin tasks as one-off processes.

## Repo-Specific Details

### Agent Behavior

- Reanalyze changes twice for introduced issues.
- Compact memory proactively to avoid context overflow.

### Key Dependencies

- Source of truth: `rag_engine/requirements.txt` (versions may change over time).
- Key packages to be aware of: `langchain`, `gradio`, `pydantic`, `ruff`, `pytest`.

### Project Specifics

- Docs: put reports in `docs/progress_reports/`.
- Tests: put tests in `rag_engine/tests`.
- Updates: always update `README.md` if changes affect it.
- Refactoring: change only relevant parts.

## Related Instruction Files

- `.cursor/rules/cmw-rag-agent.mdc`
- `.cursor/rules/terminal.mdc`
- `.cursor/rules/cmw_rag_commit.mdc`

---

**Remember:** Make sure code is clean, lean, brilliant, dry, minimalistic, abstract, non-duplicating, non-breaking, perfect, genius, and pythonic.
