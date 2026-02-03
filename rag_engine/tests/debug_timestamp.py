"""Debug script to analyze timestamp comparison for a single file."""

from __future__ import annotations

import sys
from pathlib import Path
from hashlib import sha1

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rag_engine.config.settings import settings
from rag_engine.core.document_processor import DocumentProcessor
from rag_engine.storage.vector_store import ChromaStore
from rag_engine.utils.git_utils import get_file_timestamp
from rag_engine.utils.metadata_utils import extract_numeric_kbid
import yaml


def _get_all_metadata_paginated(store: ChromaStore, limit_per_batch: int = 1000) -> list[dict]:
    """Get all document metadatas with pagination (local utility for debug script)."""
    all_metadata = []
    offset = 0

    while True:
        res = store.collection.get(limit=limit_per_batch, offset=offset, include=["metadatas"])
        batch_metas = res.get("metadatas", [])
        if not batch_metas:
            break
        all_metadata.extend(batch_metas)
        if len(batch_metas) < limit_per_batch:
            break
        offset += limit_per_batch

    return all_metadata


def debug_single_file(file_path: str) -> None:
    """Debug timestamp comparison for a single file."""
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    print("=" * 80)
    print("SINGLE FILE TIMESTAMP DEBUG TEST")
    print("=" * 80)
    print(f"File: {file_path}")
    print()

    # Step 1: Process the file to get metadata
    print("Step 1: Processing file to extract metadata...")
    dp = DocumentProcessor(mode="folder")

    # Process just the parent directory but filter to our file
    parent_dir = file_path_obj.parent
    docs = dp.process(str(parent_dir))

    # Find our specific file
    target_doc = None
    for doc in docs:
        doc_source = Path(doc.metadata.get("source_file", ""))
        if doc_source.resolve() == file_path_obj.resolve():
            target_doc = doc
            break

    if not target_doc:
        print(
            f"ERROR: Could not process file. Make sure it's a valid markdown file with kbId in frontmatter."
        )
        return

    base_meta = target_doc.metadata
    kb_id = base_meta.get("kbId", "unknown")
    source_file = base_meta.get("source_file", "")

    print(f"  ✓ kbId from metadata: {kb_id} (type: {type(kb_id).__name__})")
    print(f"  ✓ source_file: {source_file}")
    print(f"  ✓ Frontmatter 'updated': {base_meta.get('updated', 'NOT FOUND')}")

    # Check raw frontmatter
    try:
        content_raw = file_path_obj.read_text(encoding="utf-8")
        if content_raw.startswith("---"):
            parts = content_raw.split("---", 2)
            if len(parts) >= 3:
                raw_fm = yaml.safe_load(parts[1])
                raw_kb_id = raw_fm.get("kbId") if raw_fm else None
                print(
                    f"  ✓ Raw frontmatter kbId: {raw_kb_id} (type: {type(raw_kb_id).__name__ if raw_kb_id else 'None'})"
                )
                if str(raw_kb_id) != str(kb_id):
                    print(
                        f"  ⚠ WARNING: Raw frontmatter kbId ({raw_kb_id}) != processed kbId ({kb_id})"
                    )
    except Exception as e:
        print(f"  ⚠ Could not read raw frontmatter: {e}")
    print()

    # Step 2: Calculate doc_stable_id (same as retriever does - using normalized kbId)
    print("Step 2: Calculating doc_stable_id...")
    numeric_kb_id = extract_numeric_kbid(kb_id) or str(kb_id)
    doc_stable_id = sha1(numeric_kb_id.encode("utf-8")).hexdigest()[:12]
    print(f"  ✓ Numeric kbId extracted: {numeric_kb_id}")
    print(f"  ✓ doc_stable_id (from normalized kbId): {doc_stable_id}")
    print()

    # Step 3: Get current timestamp using three-tier fallback
    print("Step 3: Getting current timestamp (three-tier fallback)...")
    epoch, iso_string, source = get_file_timestamp(source_file, base_meta)
    print(f"  ✓ Timestamp source: {source}")
    print(f"  ✓ Epoch: {epoch}")
    print(f"  ✓ ISO string: {iso_string}")
    print()

    # Step 4: Query ChromaDB for existing metadata
    print("Step 4: Querying ChromaDB for existing document...")
    store = ChromaStore(collection_name=settings.chromadb_collection)

    existing = store.get_any_doc_meta({"doc_stable_id": doc_stable_id})

    if existing:
        existing_epoch = existing.get("file_mtime_epoch")
        existing_iso = existing.get("file_modified_at")
        existing_kb_id = existing.get("kbId")

        print(f"  ✓ Found existing document in ChromaDB")
        print(f"  ✓ Existing kbId: {existing_kb_id}")
        print(f"  ✓ Existing epoch: {existing_epoch} (type: {type(existing_epoch)})")
        print(f"  ✓ Existing ISO: {existing_iso}")
        print()

        # Step 5: Compare timestamps
        print("Step 5: Comparing timestamps...")
        if existing_epoch is None:
            print("  ⚠ Existing document has no file_mtime_epoch → Would be [NEW] or [REINDEX]")
        elif epoch is None:
            print("  ⚠ Current file has no timestamp → Would be [NO_TS]")
        elif isinstance(existing_epoch, int):
            if existing_epoch >= epoch:
                status = "[SKIP]"
                print(f"  ✓ existing_epoch ({existing_epoch}) >= current ({epoch})")
                print(f"  ✓ Status: {status} - File would be skipped")
            else:
                status = "[REINDEX]"
                print(f"  ✓ existing_epoch ({existing_epoch}) < current ({epoch})")
                print(f"  ✓ Status: {status} - File would be reindexed")
                print(
                    f"  ✓ Difference: {epoch - existing_epoch} seconds ({(epoch - existing_epoch) / 86400:.1f} days)"
                )
        else:
            print(f"  ⚠ Existing epoch is not an int: {type(existing_epoch)}")
    else:
        print(f"  ✗ No existing document found in ChromaDB with doc_stable_id={doc_stable_id}")
        print(f"  ✓ Status: [NEW] - File would be indexed")

        # Try searching by kbId to see if it exists under different doc_stable_id
        print()
        print("  Checking if document exists with different doc_stable_id (searching by kbId)...")
        try:
            # Try exact match first
            found_exact = False
            for kb_id_variant in [str(kb_id), int(kb_id) if str(kb_id).isdigit() else None]:
                if kb_id_variant is None:
                    continue
                try:
                    all_docs = store.collection.get(
                        where={"kbId": kb_id_variant}, limit=1, include=["metadatas"]
                    )
                    if all_docs.get("metadatas"):
                        found_meta = all_docs["metadatas"][0]
                        found_doc_stable_id = found_meta.get("doc_stable_id")
                        found_kb_id = found_meta.get("kbId")
                        print(
                            f"  ✓ Found document with exact kbId={found_kb_id} (queried as {kb_id_variant}, type: {type(kb_id_variant).__name__})"
                        )
                        print(f"    Found doc_stable_id: {found_doc_stable_id}")
                        print(f"    Expected doc_stable_id: {doc_stable_id}")
                        if found_doc_stable_id != doc_stable_id:
                            print(f"    ⚠ MISMATCH: doc_stable_id doesn't match!")
                        print(
                            f"    Found metadata file_mtime_epoch: {found_meta.get('file_mtime_epoch')}"
                        )
                        found_exact = True
                        break
                except Exception as e:
                    continue

            # If no exact match, try searching by numeric part
            if not found_exact and numeric_kb_id:
                print(f"  No exact match found. Searching by numeric part ({numeric_kb_id})...")
                # Get all documents and filter by numeric kbId
                try:
                    all_metadata = _get_all_metadata_paginated(store)
                    matching_docs = []
                    for meta in all_metadata:
                        stored_kb_id = meta.get("kbId")
                        stored_numeric = extract_numeric_kbid(stored_kb_id)
                        if stored_numeric == numeric_kb_id:
                            matching_docs.append(meta)

                    if matching_docs:
                        print(
                            f"  ⚠ Found {len(matching_docs)} document(s) with matching numeric kbId={numeric_kb_id}:"
                        )
                        for i, found_meta in enumerate(matching_docs[:3], 1):
                            found_kb_id = found_meta.get("kbId")
                            found_doc_stable_id = found_meta.get("doc_stable_id")
                            found_numeric = extract_numeric_kbid(found_kb_id)
                            print(
                                f"    Match {i}: kbId={found_kb_id} (numeric={found_numeric}), doc_stable_id={found_doc_stable_id}"
                            )
                            print(f"      Expected doc_stable_id: {doc_stable_id}")
                            if found_doc_stable_id != doc_stable_id:
                                print(
                                    f"      ⚠ MISMATCH: doc_stable_id doesn't match! (file has suffix: '{found_kb_id}' vs expected '{kb_id}')"
                                )
                            print(
                                f"      Found metadata file_mtime_epoch: {found_meta.get('file_mtime_epoch')}"
                            )
                    else:
                        print(f"  ✓ No documents found with numeric kbId={numeric_kb_id}")
                except Exception as e:
                    print(f"  ⚠ Could not search by numeric kbId: {e}")
            elif not found_exact:
                print(f"  ✓ No documents found with kbId={kb_id} (tried both string and int)")
        except Exception as e:
            print(f"  ⚠ Could not search by kbId: {e}")

        # Also check what kbId values exist in the collection (sample)
        print()
        print("  Sampling existing documents in ChromaDB to check kbId types...")
        try:
            sample = store.collection.get(limit=5, include=["metadatas"])
            sample_metas = sample.get("metadatas", [])
            if sample_metas:
                print(f"  Found {len(sample_metas)} sample documents:")
                for i, meta in enumerate(sample_metas[:3], 1):
                    sample_kb_id = meta.get("kbId")
                    sample_doc_stable_id = meta.get("doc_stable_id")
                    print(
                        f"    Sample {i}: kbId={sample_kb_id} (type: {type(sample_kb_id).__name__}), doc_stable_id={sample_doc_stable_id}"
                    )
            else:
                print(f"  ⚠ ChromaDB appears to be empty")
        except Exception as e:
            print(f"  ⚠ Could not sample ChromaDB: {e}")

    print()
    print("=" * 80)
    print("DEBUG TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Default file path, can be overridden via command line
    default_file = r".reference-repos\.cbap-mkdocs\phpkb_content\798. Версия 5.0. Текущая рекомендованная\887. Решение проблем\5065-process_error_monitor.md"

    file_path = sys.argv[1] if len(sys.argv) > 1 else default_file
    debug_single_file(file_path)
