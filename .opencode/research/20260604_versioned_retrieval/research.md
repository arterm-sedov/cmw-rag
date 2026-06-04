# Research: Versioned Retrieval & Direct kbId Fetch

## Domains

- **ChromaDB** multi-collection vs metadata-filtering isolation
- **LangChain** StructuredTool Pydantic schemas and `@tool` decorator patterns
- **Ripgrep** subprocess invocation for corpus-level file-first grepping
- **Pydantic v2** `Literal[a, b]`, env-driven `BaseSettings` expansion

## Key Findings

### 1. Chroma multi-collection is the natural V5/V6 boundary

ChromaStore already accepts `collection_name` at init; no refactor needed to point to
different collections. The repo already splits by embedding model operationally. V5/V6
are the same class of isolation.

Today both V5 and V6 index into `settings.chromadb_collection` (`mkdocs_kb`).
`doc_stable_id = sha1(numeric_kb_id)` — IDs do not overlap across versions (user
domain fact), but semantic search still mixes cross-version near-duplicates.
Incremental reindex and prune are per-collection: `--corpus v5` with `--prune-missing`
on a shared collection removes V6 entries.

**Decision: one collection per version.** Semantically clean, operationally safe.

### 2. Collection naming conventions (benchmark precedent)

From `.opencode/plans/chunk_size_model_cap_rewire_1e8b497d.plan.md` and inline docs:
`mkdocs_kb` (FRIDA), `mkdocs_kb_qwen8b` (Qwen3-Embedding-8B).

Recommended:
```
{CHROMADB_COLLECTION}_v5  →  mkdocs_kb_v5
{CHROMADB_COLLECTION}_v6  →  mkdocs_kb_v6
```

Env-driven:
- `CHROMADB_COLLECTION` stays as default/legacy.
- `CHROMADB_COLLECTION_V5` / `CHROMADB_COLLECTION_V6` overrides.

### 3. Retriever singleton → registry

Today `_get_or_create_retriever()` in `retrieve_context.py` creates **one** global
retriever. `set_app_retriever` injects one from the Gradio app.

Both paths hard-instantiate `ChromaStore(collection_name=settings.chromadb_collection)`.

Plan: keep singleton pattern, parameterize by `product_version`. Either:
- (A) Multi-retriever registry: `_retrievers: dict[str, RAGRetriever]` lazy-loaded.
- (B) Single retriever with version-aware store that re-points `self.store`.
- (A) is simpler for tests and doesn't need retriever-level surgery.

### 4. Direct kbId fetch: Chroma metadata lookup is enough

`RAGRetriever.retrieve_async` is overkill for known kbId: embed → ANN → rerank → group.

`ChromaStore` already exposes `get_collection()` on which you can call `.get(where=...)`.
`search_kbid.py` proves the pattern (sync `.collection` though; async counterpart needed).

Minimal path:
1. `collection.get(where={"kbId": normalized}, limit=1, include=["metadatas"])`
2. Read `source_file` from metadata.
3. `_read_article(source_file)` — already exists in retriever.
4. Build `Article` with `kb_id + content + metadata`, return `_format_articles_to_json`.

No embedder, no reranker, no threshold.

### 5. Corpus grep: ripgrep subprocess over `corpus_root`

Chroma has no full-text search. `ripgrep` (`rg`) is fast, pre-installed on dev machines,
handles regex, encoding, and binary filtering natively.

Plan:
- Separate tool: `grep_kb_articles`.
- Inputs: `pattern: str`, `product_version: Literal["v5","v6"] = "v6"`.
- Resolve `corpus_root` from env or convention → `*.md` tree per version.
- `subprocess.run(["rg", "--files-with-matches", pattern, ...])` → list of paths.
- For each matched file: read, strip frontmatter, wrap into Article JSON.
- Return via `_format_articles_to_json` sharing same output shape.
- `exclude_kb_ids` and `max_matches` cap for safety.
- Regex escaping: validate pattern compiles in Python `re` first.

### 6. build_index.py collection injection

No `--collection` flag exists. Need: `parser.add_argument("--collection", ...)`, fallback
to `settings.chromadb_collection`.

`sync_mkdocs_corpus.index_corpus` maps corpus→collection:
- `v5` → `settings.chromadb_collection_v5` or `f"{settings.chromadb_collection}_v5"`
- `v6` → `settings.chromadb_collection_v6` or `f"{settings.chromadb_collection}_v6"`
- Passes `--collection <name>` to each `build_index.py` call.

### 7. Plugin compatibility (& git_safety)

- `sync_mkdocs_corpus.query_git` at the end of scripts tries to load a
  `plugin_dir.query_kb_git_safety` function. If present it finalizes checks; if not
  present it fails over to a warning.
- Unknown impact.

## Sources

- ChromaDB multi-collection: https://docs.trychroma.com/guides
- LangChain `@tool` / Pydantic schemas: https://python.langchain.com/docs/how_to/custom_tools/
- Ripgrep: https://github.com/BurntSushi/ripgrep
- Internal: `retrieve_context.py`, `retriever.py`, `settings.py`, `search_kbid.py`,
  `sync_mkdocs_corpus.py`, `build_index.py`, `.env-example`
