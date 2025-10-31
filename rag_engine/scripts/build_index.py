from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rag_engine.config.settings import settings
from rag_engine.core.document_processor import DocumentProcessor
from rag_engine.core.indexer import RAGIndexer
from rag_engine.retrieval.embedder import FRIDAEmbedder
from hashlib import sha1

from rag_engine.storage.vector_store import ChromaStore
from rag_engine.utils.git_utils import get_file_timestamp
from rag_engine.utils.metadata_utils import extract_numeric_kbid
from rag_engine.utils.logging_manager import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG index from markdown sources")
    parser.add_argument("--source", required=True, help="Path to folder or file")
    parser.add_argument("--mode", choices=["folder", "file", "mkdocs"], default="folder")
    parser.add_argument("--reindex", action="store_true", help="Force reindex (ignored in MVP)")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of source files to scan/process (for quick runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze timestamps without indexing (shows which source is used for each file)",
    )
    args = parser.parse_args()

    setup_logging()

    dp = DocumentProcessor(mode=args.mode)
    # Pass max_files to limit files processed (before indexing)
    docs = dp.process(args.source, max_files=args.max_files)

    # Dry-run mode: analyze timestamps without indexing
    if args.dry_run:
        print("=" * 80)
        print("DRY-RUN MODE: Timestamp Analysis")
        print("=" * 80)
        print(f"{'File':<60} {'Source':<12} {'Epoch':<12} {'ISO String':<25}")
        print("-" * 80)

        store = ChromaStore(
            persist_dir=settings.chromadb_persist_dir, collection_name=settings.chromadb_collection
        )

        for doc in docs:
            base_meta = getattr(doc, "metadata", {})
            kb_id = base_meta.get("kbId", "unknown")
            source_file = base_meta.get("source_file", "")

            # Get timestamp using three-tier fallback
            epoch, iso_string, source = get_file_timestamp(source_file, base_meta)

            # Check if would be skipped (compare with existing)
            # Normalize kbId to numeric for consistent lookup
            numeric_kb_id = extract_numeric_kbid(kb_id) or str(kb_id)
            doc_stable_id = sha1(numeric_kb_id.encode("utf-8")).hexdigest()[:12]
            existing = store.get_any_doc_meta({"doc_stable_id": doc_stable_id}) if store else None
            existing_epoch = existing.get("file_mtime_epoch") if existing else None

            status = ""
            if existing_epoch is not None:
                if isinstance(existing_epoch, int) and epoch is not None and existing_epoch >= epoch:
                    status = " [SKIP]"
                elif epoch is not None:
                    status = " [REINDEX]"
                else:
                    status = " [NO_TS]"
            else:
                status = " [NEW]"

            file_display = str(source_file)[-60:] if source_file else "N/A"
            epoch_str = str(epoch) if epoch else "None"
            iso_str = iso_string if iso_string else "None"

            print(f"{file_display:<60} {source:<12} {epoch_str:<12} {iso_str:<25}{status}")

        print("=" * 80)
        print("Dry-run complete. No files were indexed.")
        return

    embedder = FRIDAEmbedder(model_name=settings.embedding_model, device=settings.embedding_device)
    store = ChromaStore(persist_dir=settings.chromadb_persist_dir, collection_name=settings.chromadb_collection)
    
    indexer = RAGIndexer(embedder=embedder, vector_store=store)
    indexer.index_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_files=args.max_files,
    )
    print("Index build complete.")


if __name__ == "__main__":
    main()


