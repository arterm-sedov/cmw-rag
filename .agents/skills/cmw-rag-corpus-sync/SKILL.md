---
name: cmw-rag-corpus-sync
description: Use this skill whenever the user wants to fetch, sync, update, refresh, or index the Comindware MkDocs/PHPKB RAG corpora for the D:\Repo\cmw-rag repository. This skill is specifically for the managed sparse clone workflow that keeps both V5 and V6 corpora under .reference-repos/cbap-mkdocs-ru and indexes them with rag_engine/scripts/build_index.py.
---

# CMW RAG Corpus Sync

Use this skill in `D:\Repo\cmw-rag` when the task is about preparing or indexing the external MkDocs RAG corpus.

## Workflow

Platform-specific venv paths:

| OS | Python |
|----|--------|
| Windows | `.venv\Scripts\python.exe` |
| Linux/WSL | `.venv/bin/python` |

All commands below use PowerShell; substitute the Python path for Linux/WSL.

1. Work from the repository root:

   ```powershell
   Set-Location D:\Repo\cmw-rag
   ```

2. Sync both tracked corpora from the MkDocs repo:

   ```powershell
   .venv\Scripts\python.exe rag_engine\scripts\sync_mkdocs_corpus.py
   ```

3. Sync and index both corpora:

   ```powershell
   .venv\Scripts\python.exe rag_engine\scripts\sync_mkdocs_corpus.py --index
   ```

4. Index only one corpus when the user asks for a specific version:

   ```powershell
   .venv\Scripts\python.exe rag_engine\scripts\sync_mkdocs_corpus.py --index --corpus v5
   .venv\Scripts\python.exe rag_engine\scripts\sync_mkdocs_corpus.py --index --corpus v6
   ```

## Live PHPKB Refresh Flow

Use this only when the user explicitly asks to update or refresh the corpora from the live PHPKB source, not merely to fetch the latest tracked corpus from Git.

1. First sync or open the managed MkDocs clone:

   ```powershell
   .venv\Scripts\python.exe rag_engine\scripts\sync_mkdocs_corpus.py
   Set-Location .reference-repos\cbap-mkdocs-ru
   ```

2. Run the MkDocs repo's PHPKB importer for the requested corpus version.
   Use the MkDocs repo's Python environment if available; otherwise use a Python environment with the MkDocs repo dependencies installed.

   V6:

   ```powershell
   python phpkb_import_for_rag.py --category-id 896 --kb-dir phpkb_content_rag --include-private
   ```

   V5:

   ```powershell
   python phpkb_import_for_rag.py --category-id 798 --kb-dir phpkb_content_rag
   ```

3. Inspect changes inside the MkDocs clone:

   ```powershell
   git status --short
   ```

4. Return to `cmw-rag` and index the refreshed local corpus when requested:

   ```powershell
   Set-Location D:\Repo\cmw-rag
   .venv\Scripts\python.exe rag_engine\scripts\sync_mkdocs_corpus.py --index --corpus all
   ```

5. Do not commit or push MkDocs corpus changes unless the user explicitly asks.

## Publish Refreshed Corpora Upstream

Use this only when the user explicitly asks to push the refreshed corpus changes to the upstream MkDocs repository.

1. Complete the live PHPKB refresh flow first.

2. Stay inside the managed MkDocs clone:

   ```powershell
   Set-Location D:\Repo\cmw-rag\.reference-repos\cbap-mkdocs-ru
   ```

3. Inspect the changed corpus files before staging:

   ```powershell
   git status --short
   git diff --stat
   ```

4. Spot-check representative changed Markdown files for valid frontmatter:

   ```yaml
   ---
   title: ...
   kbId: ...
   url: ...
   updated: ...
   ---
   ```

5. Ask the user for explicit confirmation before staging and committing. The confirmation should include the target branch and remote.

6. Stage only intended corpus changes:

   ```powershell
   git add phpkb_content_rag
   ```

7. Commit with a concise message:

   ```powershell
   git commit -m "docs: refresh PHPKB RAG corpora"
   ```

8. Push the current MkDocs branch only after the commit succeeds:

   ```powershell
   git push origin HEAD
   ```

9. Return to `cmw-rag`, sync the managed clone if needed, and index:

   ```powershell
   Set-Location D:\Repo\cmw-rag
   .venv\Scripts\python.exe rag_engine\scripts\sync_mkdocs_corpus.py --index --corpus all
   ```

10. Never commit or push from the `cmw-rag` repository as part of publishing MkDocs corpus changes unless the user separately asks for it.

## Defaults

- Remote: `https://github.com/arterm-sedov/cbap-mkdocs-ru.git`
- Branch: `platform_v6`
- Managed clone: `.reference-repos/cbap-mkdocs-ru`
- Sparse checkout path: `phpkb_content_rag`
- Indexed corpora:
  - V5: `phpkb_content_rag/798-platform_v5`
  - V6: `phpkb_content_rag/896-platform_v6`
- ChromaDB collections (auto-derived from CHROMADB_COLLECTION):
  - V5: `{CHROMADB_COLLECTION}_v5` (e.g. `mkdocs_kb_v5`)
  - V6: `{CHROMADB_COLLECTION}_v6` (e.g. `mkdocs_kb_v6`)
  - Override via `CHROMADB_COLLECTION_V5` / `CHROMADB_COLLECTION_V6` env vars
- Corpus root: `CORPUS_ROOT` env var or `.reference-repos/cbap-mkdocs-ru` default

## Retrieval Tools

Three LangChain tools are available for agent knowledge base access:

| Tool | Purpose | Key params |
|------|---------|------------|
| `retrieve_context` | Semantic vector search | `query`, `top_k`, `product_version` (default v6) |
| `fetch_kb_articles` | Direct article fetch by kbId | `kb_ids`, `product_version` |
| `grep_kb_articles` | Regex full-text corpus search (ripgrep) | `pattern`, `product_version`, `max_matches` |

All tools return the same JSON article format. `product_version` selects the Chroma
collection (semantic + fetch) or corpus folder (grep). The Gradio UI provides a
session-scoped version selector dropdown that acts as the default when no explicit
version is passed by the agent tool call.

## Safety Notes

- Do not copy corpus files into `cmw-rag`; `.reference-repos/**` is ignored and is the intended local data area.
- If `.reference-repos/cbap-mkdocs-ru` is a symlink to an existing full MkDocs checkout, the sync script leaves sparse-checkout settings unchanged and only performs Git fetch/checkout/pull.
- Do not reset or delete an existing checkout unless the user explicitly asks for destructive recovery.
- Use `--dry-run` to show the Git and indexing commands without executing them.
- Use `--reindex` only when the user wants to force replacement of existing chunks.
- Use `--prune-missing` only when the current corpus should become the source of truth for deleting missing `kbId`s from ChromaDB.

## Useful Commands

Dry run:

```powershell
.venv\Scripts\python.exe rag_engine\scripts\sync_mkdocs_corpus.py --dry-run --index
```

Force reindex both corpora:

```powershell
.venv\Scripts\python.exe rag_engine\scripts\sync_mkdocs_corpus.py --index --reindex
```

Limit indexing for a smoke test:

```powershell
.venv\Scripts\python.exe rag_engine\scripts\sync_mkdocs_corpus.py --index --max-files 10
```
