"""Migration script to normalize kbId values in ChromaDB.

This script finds all documents with kbId values containing suffixes (e.g., "4578-toc")
and normalizes them to numeric-only format (e.g., "4578"). It updates both the metadata
and recalculates doc_stable_id to match the normalized format.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rag_engine.config.settings import settings
from rag_engine.storage.vector_store import ChromaStore
from rag_engine.utils.metadata_utils import extract_numeric_kbid


def get_all_metadata_paginated(store: ChromaStore, limit_per_batch: int = 1000) -> list[dict]:
    """Get all document metadatas with pagination (local utility for migration).

    Returns metadata with ChromaDB document IDs attached as '_chroma_id' field.
    """
    all_metadata = []
    offset = 0

    while True:
        # IDs are always returned by ChromaDB, don't include in 'include' parameter
        res = store.collection.get(limit=limit_per_batch, offset=offset, include=["metadatas"])
        batch_metas = res.get("metadatas", [])
        batch_ids = res.get("ids", [])  # IDs are always returned, even if not in include
        if not batch_metas:
            break

        # Include ids with metadata for deletion/update
        for meta, doc_id in zip(batch_metas, batch_ids):
            meta_with_id = meta.copy()
            meta_with_id["_chroma_id"] = doc_id
            all_metadata.append(meta_with_id)

        if len(batch_metas) < limit_per_batch:
            break
        offset += limit_per_batch

    return all_metadata


def normalize_kbids(store: ChromaStore, dry_run: bool = True) -> dict:
    """Normalize all kbId values to numeric-only format.

    Args:
        store: ChromaStore instance
        dry_run: If True, only report changes without applying them

    Returns:
        Dictionary with migration statistics
    """
    print("=" * 80)
    print("Normalizing kbId Values in ChromaDB")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will update)'}\n")

    # Get all metadata
    print("Fetching all documents...")
    all_metadata = get_all_metadata_paginated(store)
    print(f"Total documents: {len(all_metadata)}\n")

    # Find documents that need normalization
    # Only normalize kbIds that match pattern: number followed by dash (e.g., "4578-toc")
    # Don't normalize paths or other strings that happen to start with numbers
    import re

    needs_normalization = []
    for meta in all_metadata:
        kb_id = meta.get("kbId")
        if kb_id is None:
            continue

        kb_id_str = str(kb_id)
        source_file = meta.get("source_file", "")

        # Detect if kbId is actually a file path (common indicators):
        # - Contains path separators (\ or /)
        # - Contains spaces (paths often have spaces)
        # - Starts with number followed by dot and space (e.g., "801. Руководства")
        # - Is very long (paths are typically longer than simple kbIds)
        is_path_like = (
            "\\" in kb_id_str
            or "/" in kb_id_str
            or " " in kb_id_str
            or (len(kb_id_str) > 50)
            or re.match(r"^\d+\.\s", kb_id_str) is not None  # "801. Руководства" pattern
        )

        # If kbId is path-like, extract the filename (last component)
        # This handles cases where source_file was stored as kbId due to fallback
        if is_path_like:
            # Extract filename from path (handle both Windows and Unix separators)
            path_parts = kb_id_str.replace("\\", "/").split("/")
            filename = path_parts[-1] if path_parts else kb_id_str
            # Remove .md extension if present
            if filename.endswith(".md"):
                filename = filename[:-3]
            # Use filename for pattern matching
            kb_id_to_check = filename
        else:
            # Not a path, use as-is
            kb_id_to_check = kb_id_str

        # Check if the extracted kbId (or original if not path) matches suffix pattern
        matches_suffix_pattern = re.match(r"^\d+-", kb_id_to_check) is not None

        if matches_suffix_pattern:
            numeric_kb_id = extract_numeric_kbid(kb_id_to_check)
            if numeric_kb_id and numeric_kb_id != kb_id_to_check:
                needs_normalization.append(
                    {
                        "chroma_id": meta["_chroma_id"],
                        "old_kb_id": kb_id,
                        "extracted_from_path": kb_id_to_check if is_path_like else None,
                        "new_kb_id": numeric_kb_id,
                        "doc_stable_id": meta.get("doc_stable_id"),
                        "metadata": meta,
                        "source_file": source_file,
                    }
                )

    print(f"Documents needing normalization: {len(needs_normalization)}")

    if not needs_normalization:
        print("\n✅ All kbId values are already normalized!")
        return {"total": len(all_metadata), "normalized": 0, "errors": 0}

    # Group by old doc_stable_id to find conflicts
    stats = {
        "total": len(all_metadata),
        "needs_normalization": len(needs_normalization),
        "normalized": 0,
        "deleted": 0,
        "errors": 0,
    }

    print("\nDocuments to normalize:")
    print("-" * 80)
    for item in needs_normalization[:10]:  # Show first 10
        old_kbid = item["old_kb_id"]
        extracted = item.get("extracted_from_path")
        if extracted:
            print(f"  kbId (path): '{old_kbid}'")
            print(f"    Extracted filename: '{extracted}' → '{item['new_kb_id']}'")
        else:
            print(f"  kbId: '{old_kbid}' → '{item['new_kb_id']}'")
        source = item.get("source_file", "N/A")
        source_display = source[-60:] if len(source) > 60 else source
        print(f"    Source file: {source_display}")
    if len(needs_normalization) > 10:
        print(f"  ... and {len(needs_normalization) - 10} more")

    if dry_run:
        print("\n💡 This was a dry run. Re-run with --apply to perform the migration.")
        return stats

    # Perform migration
    print("\n" + "=" * 80)
    print("Applying normalization...")
    print("-" * 80)

    # Group by new numeric kbId to detect duplicates
    from collections import defaultdict

    by_new_kbid = defaultdict(list)
    for item in needs_normalization:
        by_new_kbid[item["new_kb_id"]].append(item)

    # For each normalized kbId, check if there's already a document with that kbId
    for new_kb_id, items in by_new_kbid.items():
        # Check if document with normalized kbId already exists
        new_doc_stable_id = hashlib.sha1(new_kb_id.encode("utf-8")).hexdigest()[:12]
        existing = store.get_any_doc_meta({"doc_stable_id": new_doc_stable_id})

        if existing:
            # Document with normalized kbId already exists - delete the suffixed version
            print(f"\nkbId '{new_kb_id}' already exists. Deleting suffixed versions:")
            for item in items:
                try:
                    store.collection.delete(ids=[item["chroma_id"]])
                    stats["deleted"] += 1
                    print(f"  ✓ Deleted: '{item['old_kb_id']}' (ID: {item['chroma_id'][:20]}...)")
                except Exception as e:  # noqa: BLE001
                    stats["errors"] += 1
                    print(f"  ✗ Error deleting '{item['old_kb_id']}': {e}")
        else:
            # No conflict - delete suffixed version (will be reindexed with normalized kbId)
            for item in items:
                try:
                    # ChromaDB doesn't support direct metadata updates. Since source files
                    # still exist with normalized kbId in frontmatter, we delete the old
                    # suffixed version and let build_index reindex it with the correct kbId.
                    print(f"  Deleting: '{item['old_kb_id']}' → will be reindexed as '{new_kb_id}'")
                    store.collection.delete(ids=[item["chroma_id"]])
                    stats["deleted"] += 1
                    stats["normalized"] += 1
                except Exception as e:  # noqa: BLE001
                    stats["errors"] += 1
                    print(f"  ✗ Error deleting '{item['old_kb_id']}': {e}")

    print("\n" + "=" * 80)
    print("Migration Summary:")
    print("-" * 80)
    print(f"Total documents: {stats['total']}")
    print(f"Needed normalization: {stats['needs_normalization']}")
    print(f"Deleted (will be reindexed): {stats['deleted']}")
    print(f"Errors: {stats['errors']}")
    print(f"\n💡 Note: Deleted documents will be reindexed on next 'build_index' run")
    print(f"   with normalized kbId values.")

    return stats


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Normalize kbId values in ChromaDB (remove suffixes like '-toc')"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the migration (default is dry-run)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name (default: from settings)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="ChromaDB persist directory (default: from settings)",
    )
    args = parser.parse_args()

    load_dotenv()

    persist_dir = args.persist_dir or settings.chromadb_persist_dir
    collection_name = args.collection or settings.chromadb_collection

    store = ChromaStore(collection_name=collection_name)

    normalize_kbids(store, dry_run=not args.apply)


if __name__ == "__main__":
    main()
