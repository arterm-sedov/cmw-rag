# Plan: Direct PHPKB Import for Corpus Sync

**Date:** 2026-06-08  
**Status:** planned (not implemented)

## Goal

Remove the human-in-the-loop dependency from the RAG corpus refresh pipeline. Instead of waiting for someone to import PHPKB articles into the upstream git repo, the sync script should import them directly from the PHPKB MySQL database via SSH tunnel.

## Current Pipeline

```
human runs phpkb_import_for_rag.py → commits → pushes to git remote
    ↓
systemd timer: git pull → build_index.py
```

**Problem:** Data freshness depends on manual action. Stale if upstream is forgotten.

## Target Pipeline

```
systemd timer: phpkb_import_for_rag.py (SSH tunnel → MySQL) → build_index.py
```

**Result:** Full autonomy. Max 6h staleness. Git sync remains as fallback.

## Existing Assets (already built, already accessible)

All in `.reference-repos/cbap-mkdocs-ru/` (the managed sparse clone):

| Asset | What it does |
|---|---|
| `ssh_kb_ru.py` | SSH tunnel (`sshtunnel`+Paramiko) to `31.135.15.59:8223`, MySQL on `phpkbv9` |
| `phpkb_import_for_rag.py` | Walks PHPKB category tree, HTML→Markdown, writes to `phpkb_content_rag/` |
| `.env` | Tunnel credentials. Passwords in keyring (`~/.local/share/python_keyring/`) |
| `.venv/` | Has `sshtunnel`, `mysql-connector-python`, `paramiko`, `bs4`, `markdownify` |

Both SSH and SQL passwords **confirmed** in keyring (PlaintextKeyring, headless-safe).

`phpkb_import_for_rag.py` already supports non-interactive mode:
```bash
.venv/bin/python phpkb_import_for_rag.py --category-id 896 --kb-dir phpkb_content_rag --include-private  # v6
.venv/bin/python phpkb_import_for_rag.py --category-id 798 --kb-dir phpkb_content_rag                   # v5
```

## Missing Pieces (to implement)

1. **Sparse checkout expansion** — `git sparse-checkout add tools phpkb_import_for_rag.py .env` so source files materialize.

2. **Tunnel connectivity test** — Verify `establish_connection_interactive()` works non-interactively with `SSH_USE_STORED_CREDENTIALS=1`.

3. **Smoke test import** — Run `phpkb_import_for_rag.py` against a small category, verify Markdown output.

4. **`sync_mkdocs_corpus.py` — add `--phpkb` flag**. When set:
   - Skip git fetch/pull
   - Shell out to mkdocs venv: `{mkdocs_repo}/.venv/bin/python phpkb_import_for_rag.py` with appropriate `--category-id` per corpus version
   - Proceed to `build_index.py` as usual
   - When not set: behave exactly as today (git pull + index)

5. **Update SKILL.md** — document `--phpkb` workflow and systemd timer configuration for full autonomy.

6. **Update systemd service** (optional) — switch from plain sync to `--phpkb --index --corpus all` for direct PHPKB import.

## Non-Goals

- No new dependencies in `cmw-rag` venv (tunnel libs stay in mkdocs venv)
- No removal of git sync path (it remains as fallback)
- No changes to `ssh_kb_ru.py` or `phpkb_import_for_rag.py` (they're owned by the mkdocs repo)
- No credential handling changes (keyring already works)

## Risks

- **Keyring unavailability in systemd context** — PlaintextKeyring uses files at `~/.local/share/python_keyring/`. Should work since it's just files, but verify during testing.
- **Tunnel drop during large import** — `phpkb_import_for_rag.py` has no automatic retry. Acceptable for now (timer will retry next cycle).
- **Article ID drift** — If PHPKB articles are deleted upstream, the Git-based `.article_id_filename_map_v6.json` may be stale. The import script handles this via its own mapping.
