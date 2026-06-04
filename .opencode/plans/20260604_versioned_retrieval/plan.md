# Versioned Retrieval, Direct kbId Fetch, and Corpus Grep

## Objective

Add multi-version KB support to the RAG pipeline without breaking existing
single-collection workflows. Three new capabilities:

1. **Versioned Chroma collections** — V5 and V6 index into separate collections
2. **Direct kbId article fetch** — fetch by known article ID, no semantic search
3. **Corpus grep tool** — ripgrep-powered full-text search over the corpus filesystem

All gated by a `product_version` enum (`v5` | `v6`, default `v6`) on
retrieval tools, API, and indexing paths. No retrieval `"all"` mode.

## Non-Goals

- Do not modify chunking, embedding, reranking, or LLM wiring.
- Do not change the agent conversation loop or Gradio UI flow.
- Do not add multi-version semantic retrieval in a single call.
- Do not rename existing `CHROMADB_COLLECTION` — it stays the default/legacy name.
- Do not remove the shared `get_or_create_retriever` singleton; extend it.

## SDD Contracts

### Settings / env

```python
# .env / settings.py additions
chromadb_collection: str  # unchanged, default/legacy
chromadb_collection_v5: str = ""  # override, falls back to f"{chromadb_collection}_v5"
chromadb_collection_v6: str = ""  # override, falls back to f"{chromadb_collection}_v6"
corpora_root: str = ""  # filesystem corpus root for grep and fallback file reads
```

Helper: `def get_collection_name(version: str, /) -> str` — resolves env vars.

### Product version enum

```python
ProductVersion = Literal["v5", "v6"]
DEFAULT_PRODUCT_VERSION: ProductVersion = "v6"
```

### RetrieveContextSchema addition

```python
product_version: ProductVersion = DEFAULT_PRODUCT_VERSION
```

### Direct kbId fetch tool (new)

```python
class FetchArticleSchema(BaseModel):
    kb_ids: list[str]
    product_version: ProductVersion = DEFAULT_PRODUCT_VERSION

# Tool: fetch_kb_articles
# Core: _fetch_articles_by_kb_ids_core(kb_ids, version) -> str
```

### Corpus grep tool (new)

```python
class GrepKbArticlesSchema(BaseModel):
    pattern: str
    product_version: ProductVersion = DEFAULT_PRODUCT_VERSION
    max_matches: int = 20
    exclude_kb_ids: list[str] | None = None

# Tool: grep_kb_articles
# Implementation: ripgrep subprocess, file read, frontmatter strip, format as JSON
```

**Return contract:** same `_format_articles_to_json` as `retrieve_context` — identical
`kb_id`, `title`, `url`, `content`, `metadata` envelope. Differences:
- `metadata.rerank_score` absent (set to `None` or omit); add `metadata.match_source: "grep"`.
- `metadata.matched_lines`: list of `{line_number, text}` for each hit in the article (capped).
- `metadata.match_count`: total grep hits in the file.
- `"query"` in result metadata becomes the grep pattern.
- Same JSON output shape; downstream consumers (agent, MCP) parse identically.

### Gradio version selector (UI)

A Gradio `gr.Dropdown` (`choices=["v6", "v5"]`, default `"v6"`) in the app sidebar or
header, backed by `gr.State` per user session. The selected version flows into the
agent's `Runtime.context` (e.g. a `product_version` field on `AgentContext` or a
dedicated session state model). Tool cores read the session version as fallback when
`product_version` is not explicitly passed in the tool schema.

Agent override rule: if the agent provides `product_version` in the tool schema, that
wins over the session default. The session default only applies when the tool call omits
the param. Different concurrent users get independent session defaults.<｜end▁of▁thinking｜>1. Do not change the agent conversation loop or Gradio UI flow.
2. Do not add multi-version semantic retrieval in a single call.
3. Do not rename existing `CHROMADB_COLLECTION` — it stays the default/legacy name.
4. Do not remove the shared `get_or_create_retriever` singleton; extend it.

### API / MCP mirror

```python
def get_knowledge_base_articles(
    query: str, ...,
    product_version: str = "v6",
    kb_ids: list[str] | None = None,
) -> str:
    # kb_ids set → direct fetch, else semantic
```

### Indexing

- `build_index.py`: `--collection <name>` override (falls back to `settings.chromadb_collection`).
- `sync_mkdocs_corpus.py`: `index_corpus` maps `v5`/`v6` → collection name, passes
  `--collection` to `build_index.py`. Skill updated with collection names.

### ChromaStore

- `get_by_kb_id_async(kb_id: str) -> dict | None` — metadata fetch by kbId.

## TDD Tasks

### Phase 1 — Settings & collection resolution

1. Add `chromadb_collection_v5`/`_v6` + `corpora_root` to `Settings`.
2. Add `get_collection_name(version)` helper. Test: v5→v5 name, v6→v6 name, fallback.
3. Add entry in `.env-example`.

### Phase 2 — build_index and sync_mkdocs_corpus collection wiring

4. Add `--collection` flag to `build_index.py`.
5. Map `v5`/`v6`→collection in `sync_mkdocs_corpus.index_corpus`.
6. Test: sync `--corpus v5` passes correct `--collection`; `--corpus all` passes two distinct names.
7. Test: `build_index --collection custom_name` overrides env default.

### Phase 3 — ChromaStore get_by_kb_id

8. Add `get_by_kb_id_async` to `ChromaStore`.
9. Test: returns metadata+source_file for known kbId; `None` for missing.

### Phase 4 — Direct kbId fetch

10. Add `_fetch_articles_by_kb_ids_core(kb_ids, version)` to `retrieve_context.py`.
11. Add `fetch_kb_articles` LangChain tool with `FetchArticleSchema`.
12. Test: single kbId → article JSON; multiple kbIds → array; missing kbId handled gracefully.
13. Test: v5 fetch opens v5 collection, v6 opens v6.

### Phase 5 — Retriever product_version support

14. Add `product_version` to `RetrieveContextSchema`.
15. `_get_or_create_retriever(version)` → lazy registry.
16. `_retrieve_context_core` maps version → retriever → store.
17. Test: v5 semantic search opens v5 collection; v6 opens v6.
18. Test: default (no version arg) → v6; non-breaking for existing callers.

### Phase 6 — Corpus grep tool

19. Add `grep_kb_articles` LangChain tool with `GrepKbArticlesSchema`.
20. Implement: validate regex, `rg --files-with-matches` in corpus root, read matched files.
21. Format output via shared `_format_articles_to_json`; inject `match_source: "grep"`,
    `matched_lines`, `match_count` into article metadata.
22. Test: valid pattern returns articles in standard JSON shape.
23. Test: grep metadata fields present; no `rerank_score` set.
24. Test: invalid regex raises clean `ValueError`.
25. Test: version selector scopes to correct corpus folder.
26. Test: `max_matches` caps output; `exclude_kb_ids` filters.

### Phase 7 — API & MCP updates

27. Add `product_version` and `kb_ids` params to `get_knowledge_base_articles`.
28. Dispatch: `kb_ids` set → `_fetch_articles_by_kb_ids_core`, else semantic.
29. Update MCP tool schemas to mirror new params.

### Phase 8 — Gradio version selector

30. Add `gr.Dropdown` for product version in the Gradio app header/sidebar.
31. Wire to `gr.State` per user session; flow version into `Runtime.context`.
32. Agent tool schema's `product_version` param wins over session default when explicitly set.
33. Test: UI selector changes session default; agent override per-call works; concurrent
    sessions isolated; no UI → v6.

### Phase 9 — Skill & docs

34. Update `.agents/skills/cmw-rag-corpus-sync/SKILL.md` with collection names, grep tool,
    and version selector usage.
35. Update `README.md` with new env vars, Gradio version selector, and tool usage examples.

## Checkpoints

### CP1 — After Phase 2 (settings + build_index)
- `get_collection_name("v5")` returns correct name.
- `.env-example` has all new vars with inline comments.
- `build_index.py --collection mkdocs_kb_v5` runs without error.
- `sync_mkdocs_corpus --corpus v5 --index --dry-run` prints collection-specific commands.

### CP2 — After Phase 5 (semantic + direct fetch both version-aware)
- Mocked retriever tests: V5 search never touches V6 store.
- Direct fetch: `kb_id=123, version=v5` returns V5 article.
- Existing callers without `product_version` still work (default v6).

### CP3 — After Phase 6 (grep tool)
- `grep_kb_articles` returns same JSON shape as `retrieve_context`.
- Grep metadata fields (`match_source`, `matched_lines`, `match_count`) present.
- Grep articles have no `rerank_score`.

### CP4 — After Phase 8 (end-to-end)
- `get_knowledge_base_articles(query="...", product_version="v5")` works.
- `get_knowledge_base_articles(kb_ids=["123"], product_version="v6")` direct fetches.
- Gradio version dropdown sets per-session default; agent override per-call works;
  concurrent sessions isolated.

## Verification Commands

```powershell
.venv\Scripts\Activate.ps1

# Unit tests (fast, mocked)
pytest rag_engine/tests/test_retriever.py --no-cov -q
pytest rag_engine/tests/test_tools_retrieve_context.py --no-cov -q
# New test files
pytest rag_engine/tests/test_tools_fetch_articles.py --no-cov -q
pytest rag_engine/tests/test_tools_grep_kb.py --no-cov -q
pytest rag_engine/tests/test_storage_vector_store.py --no-cov -q
pytest rag_engine/tests/test_scripts_sync_mkdocs_corpus.py --no-cov -q
pytest rag_engine/tests/test_scripts_build_index.py --no-cov -q

# Lint
ruff check rag_engine/

# Dry-run sanity (requires corpus checkout)
python rag_engine/scripts/sync_mkdocs_corpus.py --dry-run --index --corpus all
```

## Non-Breaking Guarantees

- Existing `CHROMADB_COLLECTION` env var unchanged; defaults still resolved.
- `retrieve_context` tool schema backward-compatible: `product_version` has default `"v6"`.
- `get_knowledge_base_articles` new params have defaults; old callers unaffected.
- `build_index.py --source ... --mode folder` still works without `--collection`.
- Gradio version dropdown is additive, session-scoped via `gr.State`; existing UI layout unchanged; default is v6.
- Concurrent user sessions independently isolated.
- Test suite for existing retriever/indexer paths passes unmodified.

## 12-Factor Notes

- Collection names driven by env vars, not code constants.
- `corpora_root` env-overridable; falls back to convention path.
- Stateless processes: version is a request parameter, not session state.
- Ripgrep is a backing service (pre-installed binary); checked at tool init.
