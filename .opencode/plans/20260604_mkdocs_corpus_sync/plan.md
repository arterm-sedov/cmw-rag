# MkDocs Corpus Sync Plan

## Objective

Add a small, testable utility script and local skill so agents can fetch or update the MkDocs RAG corpus and index it through the existing `build_index.py` workflow.

## Scope

- Add `rag_engine/scripts/sync_mkdocs_corpus.py`.
- Add focused tests for clone/update/index command construction and safety behavior.
- Add a local skill under `.agents/skills/` describing when and how to use the sync/index workflow.
- Do not change indexing internals.
- Do not commit or push.

## Behavior Contract

The sync script should:

1. Default to remote `https://github.com/arterm-sedov/cbap-mkdocs-ru.git`.
2. Default to branch `platform_v6`.
3. Default to sparse path `phpkb_content_rag` so both V5 and V6 corpora are fetched.
4. Default to target `.reference-repos/cbap-mkdocs-ru`.
5. Clone with `--filter=blob:none --sparse --branch <branch>` when target is missing.
6. Configure sparse checkout after clone and update.
7. Update existing Git repositories with fast-forward-only pull.
8. Refuse non-Git existing directories by default.
9. Support `--dry-run`.
10. Support optional `--index` to invoke `build_index.py --mode folder`.
11. Support `--corpus v5|v6|all`, defaulting to `all` for indexing.

## TDD Tasks

1. Add tests for argument defaults and corpus path derivation.
2. Add tests for missing-target clone command sequence.
3. Add tests for existing Git target update sequence.
4. Add tests that existing non-Git target raises a clear error.
5. Add tests that indexing builds the expected Python commands for V5, V6, and `all`.
6. Implement the script to satisfy tests.
7. Add the local skill and keep it concise.

## Verification Commands

```powershell
.venv\Scripts\Activate.ps1
pytest rag_engine/tests/test_scripts_sync_mkdocs_corpus.py
ruff check rag_engine/scripts/sync_mkdocs_corpus.py rag_engine/tests/test_scripts_sync_mkdocs_corpus.py
```

Optional manual dry run:

```powershell
python rag_engine/scripts/sync_mkdocs_corpus.py --dry-run
```

## Completion Checkpoints

- Script commands are deterministic and non-destructive by default.
- Tests pass without network access by mocking subprocess calls.
- Skill points users and agents to the script instead of ad hoc Git commands.
- Final response includes exact files changed and verification status.
