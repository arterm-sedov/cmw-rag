# MkDocs Corpus Sync Research

## Goal

Create a repeatable way for `cmw-rag` to fetch the tracked RAG corpus from the MkDocs repository and index it without copying corpus files into this repository.

## Local Findings

- `cmw-rag` currently indexes local Markdown sources through `rag_engine/scripts/build_index.py`.
- `DocumentProcessor` supports `folder`, `file`, and `mkdocs` modes, and useful corpus indexing requires article-level `kbId` frontmatter.
- `.reference-repos/**` is already ignored in `.gitignore`, so a managed clone can live there without polluting this repository.
- `.reference-repos/cbap-mkdocs-ru` currently exists as a local symlink to `D:\Repo\CBAP_MKDOCS_RU`; it is not a submodule.
- `git submodule status --recursive` is empty for `cmw-rag`.
- The MkDocs repo tracks `phpkb_content_rag`; local `git ls-files phpkb_content_rag` reports 1203 tracked files.
- The current MkDocs repo branch is `platform_v6`.
- The target V5 corpus folder is `phpkb_content_rag/798. Версия 5.0. Текущая рекомендованная`.
- The target V6 corpus folder is `phpkb_content_rag/896-platform_v6`.

## Web Reference Notes

- Git sparse checkout supports selecting directory subsets in cone mode. The official Git sparse-checkout docs describe cone mode as directory-oriented, which fits `phpkb_content_rag/896-platform_v6`.
- Git clone supports `--filter=blob:none`; official Git clone docs describe it as filtering blob contents until needed, reducing clone size for partial clones.
- Combining `git clone --filter=blob:none --sparse` with `git sparse-checkout set <directory>` is the right pattern for a managed external corpus clone.

## Design Conclusion

Use a managed sparse clone under `.reference-repos/cbap-mkdocs-ru`.

Fetch the whole `phpkb_content_rag` root with sparse checkout so both V5 and V6 RAG corpora are available while unrelated MkDocs/PDF/export assets remain outside the working tree.

Preferred default flow:

1. Clone if missing:
   `git clone --filter=blob:none --sparse --branch platform_v6 <remote> .reference-repos/cbap-mkdocs-ru`
2. Configure sparse path:
   `git sparse-checkout set phpkb_content_rag`
3. On later runs, fetch and fast-forward only:
   `git fetch origin platform_v6`, `git checkout platform_v6`, `git pull --ff-only origin platform_v6`
4. Index with:
   `python rag_engine/scripts/build_index.py --source .reference-repos/cbap-mkdocs-ru/phpkb_content_rag/<corpus-folder> --mode folder`

## Risks

- Existing local symlink should not be overwritten. The script should detect it and either use it as-is or fail with a clear message if it is not a Git repository.
- Existing managed clone with local changes should not be destructively reset by default.
- The sync script should keep indexing optional so corpus fetch and vector indexing remain separate operational steps.
