"""ChromaDB maintenance script for diagnostics and maintenance operations."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rag_engine.config.settings import settings
from rag_engine.storage.vector_store import ChromaStore


def get_all_metadata_paginated(store: ChromaStore, limit_per_batch: int = 1000) -> list[dict]:
    """Get all document metadatas with pagination (utility for maintenance scripts).

    Args:
        store: ChromaStore instance
        limit_per_batch: Number of documents to fetch per batch (default: 1000)

    Returns:
        List of all metadata dictionaries
    """
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


def get_db_info(persist_dir: str) -> dict:
    """Get database information from SQLite metadata and ChromaDB API."""
    sqlite_path = Path(persist_dir) / "chroma.sqlite3"
    if not sqlite_path.exists():
        return {"error": "chroma.sqlite3 not found"}

    info = {}
    try:
        # Use ChromaDB API to get collection info (more reliable than direct SQLite)
        import chromadb

        client = chromadb.PersistentClient(path=persist_dir)
        collections_list = client.list_collections()

        info["collections"] = []
        for col in collections_list:
            col_name = getattr(col, "name", None) or str(col)
            col_id = getattr(col, "id", None)
            col_metadata = getattr(col, "metadata", {})

            # Get actual collection to count items
            try:
                collection = client.get_collection(name=col_name)
                count = collection.count()
            except Exception:  # noqa: BLE001
                count = 0

            info["collections"].append(
                {
                    "id": col_id or "unknown",
                    "name": col_name,
                    "metadata": col_metadata,
                    "embedding_count": count,
                }
            )

        # Check for WAL files
        wal_path = sqlite_path.with_suffix(".sqlite3-wal")
        info["wal_exists"] = wal_path.exists()
        if info["wal_exists"]:
            info["wal_size"] = wal_path.stat().st_size

        # Database file size
        info["sqlite_size"] = sqlite_path.stat().st_size
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)

    return info


def list_vector_dirs(persist_dir: str) -> list[dict]:
    """List UUID directories containing vector data."""
    persist_path = Path(persist_dir)
    vector_dirs = []

    for item in persist_path.iterdir():
        if item.is_dir() and len(item.name) == 36 and item.name.count("-") == 4:
            # Looks like a UUID
            data_file = item / "data_level0.bin"
            if data_file.exists():
                vector_dirs.append(
                    {
                        "uuid": item.name,
                        "path": str(item),
                        "data_level0_size": data_file.stat().st_size,
                        "has_header": (item / "header.bin").exists(),
                        "has_index_metadata": (item / "index_metadata.pickle").exists(),
                    }
                )

    return vector_dirs


def check_consistency(persist_dir: str, collection_name: str | None = None) -> dict:
    """Check consistency between ChromaDB collections and vector data directories."""
    db_info = get_db_info(persist_dir)
    vector_dirs = list_vector_dirs(persist_dir)

    result = {
        "metadata_collections": [],
        "vector_dirs": vector_dirs,
        "issues": [],
    }

    if "error" in db_info:
        result["issues"].append(f"Failed to read collections: {db_info['error']}")
        return result

    # Match collections with vector directories
    for col in db_info.get("collections", []):
        col_uuid = col.get("id", "")
        matching_vector = next((v for v in vector_dirs if v["uuid"] == col_uuid), None)

        col_info = {
            "name": col.get("name"),
            "uuid": col_uuid,
            "embedding_count": col.get("embedding_count", 0),
            "has_vector_data": matching_vector is not None,
        }

        if matching_vector:
            col_info["vector_data_size"] = matching_vector["data_level0_size"]
        else:
            result["issues"].append(
                f"Collection '{col.get('name')}' (UUID: {col_uuid}) has no vector data directory"
            )

        result["metadata_collections"].append(col_info)

    # Check for orphaned vector directories
    metadata_uuids = {col.get("id") for col in db_info.get("collections", [])}
    for vec_dir in vector_dirs:
        if vec_dir["uuid"] not in metadata_uuids:
            result["issues"].append(
                f"Orphaned vector directory: {vec_dir['uuid']} (no matching collection)"
            )

    return result


def diagnose(persist_dir: str, collection_name: str | None = None) -> None:
    """Run comprehensive diagnostics."""
    print("=" * 80)
    print("ChromaDB Diagnostics")
    print("=" * 80)
    print(f"Persist directory: {persist_dir}\n")

    # Database info
    print("ChromaDB Collections:")
    print("-" * 80)
    db_info = get_db_info(persist_dir)
    if "error" in db_info:
        print(f"❌ Error: {db_info['error']}")
    else:
        sqlite_path = Path(persist_dir) / "chroma.sqlite3"
        if sqlite_path.exists():
            print(f"SQLite file: {sqlite_path}")
            print(f"SQLite size: {db_info.get('sqlite_size', 0) / 1024 / 1024:.2f} MB")
            print(f"WAL exists: {db_info.get('wal_exists', False)}")
            if db_info.get("wal_exists"):
                print(f"WAL size: {db_info.get('wal_size', 0) / 1024:.2f} KB")
                print("⚠️  WARNING: WAL file exists - uncommitted transactions may be present")

        print(f"\nCollections: {len(db_info.get('collections', []))}")
        for col in db_info.get("collections", []):
            print(f"  - {col.get('name')} (UUID: {col.get('id')})")
            print(f"    Chunks: {col.get('embedding_count', 0)}")
            if collection_name and col.get("name") == collection_name:
                print(f"    ⭐ Target collection")

    # Vector directories
    print("\n" + "=" * 80)
    print("Vector Data Directories (HNSW Index):")
    print("-" * 80)
    vector_dirs = list_vector_dirs(persist_dir)
    if not vector_dirs:
        print("No vector data directories found")
    else:
        for vec_dir in vector_dirs:
            print(f"UUID: {vec_dir['uuid']}")
            print(f"  data_level0.bin: {vec_dir['data_level0_size'] / 1024 / 1024:.2f} MB")
            print(f"  Has header.bin: {vec_dir['has_header']}")
            print(f"  Has index_metadata.pickle: {vec_dir['has_index_metadata']}")

    # Consistency check
    print("\n" + "=" * 80)
    print("Consistency Check:")
    print("-" * 80)
    consistency = check_consistency(persist_dir, collection_name)
    if not consistency["issues"]:
        print("✅ All collections have matching vector data")
        print("✅ No orphaned vector directories found")
    else:
        print("⚠️  Issues found:")
        for issue in consistency["issues"]:
            print(f"  - {issue}")

        # If UUID mismatch but data is accessible, it might be okay
        if consistency["issues"] and db_info.get("collections"):
            print("\n📝 Note: UUID mismatch detected, but if ChromaDB can query data")
            print("   (verified by matching chunk counts), this may be benign.")
            print("   The old directory might be from a previous collection version.")
            print("   ChromaDB may handle this internally via SQLite metadata mappings.")

    # Count actual chunks and articles in ChromaDB
    print("\n" + "=" * 80)
    print("ChromaDB Collection Statistics:")
    print("-" * 80)
    try:
        store = ChromaStore(collection_name=collection_name or settings.chromadb_collection)
        # Use ChromaDB's built-in count() method (most efficient)
        api_count = store.collection.count()
        print(f"Total chunks: {api_count}")

        # Optionally count unique articles (requires loading all metadata - can be slow)
        try:
            all_metadata = get_all_metadata_paginated(store)
            unique_doc_ids = {
                meta.get("doc_stable_id") for meta in all_metadata if meta.get("doc_stable_id")
            }
            print(f"Unique articles (by doc_stable_id): {len(unique_doc_ids)}")
            if len(all_metadata) != api_count:
                print(
                    f"⚠️  Note: Metadata pagination count ({len(all_metadata)}) != API count ({api_count})"
                )
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Could not count unique articles (may be slow for large collections): {e}")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Could not get collection statistics: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("-" * 80)
    print(f"Metadata collections: {len(consistency['metadata_collections'])}")
    print(f"Vector directories: {len(consistency['vector_dirs'])}")
    print(f"Issues: {len(consistency['issues'])}")

    if db_info.get("wal_exists"):
        print("\n💡 Recommendation: Run with --commit-wal to commit pending transactions")


def commit_wal(persist_dir: str) -> None:
    """Commit Write-Ahead Log by connecting and disconnecting cleanly."""
    print("=" * 80)
    print("Committing WAL")
    print("=" * 80)

    try:
        import chromadb

        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()

        print(f"Found {len(collections)} collection(s)")

        # Force a commit by accessing each collection
        for col in collections:
            name = getattr(col, "name", None) or str(col)
            collection = client.get_collection(name=name)
            # Access collection to trigger any pending operations
            count = collection.count()
            print(f"  - {name}: {count} items")

        # Close client to ensure all operations are flushed
        del client

        # Check if WAL still exists
        sqlite_path = Path(persist_dir) / "chroma.sqlite3"
        wal_path = sqlite_path.with_suffix(".sqlite3-wal")
        if wal_path.exists():
            print(f"\n⚠️  WAL file still exists: {wal_path}")
            print("This is normal if ChromaDB is actively being used.")
            print("The WAL will be committed when the database is closed.")
        else:
            print("\n✅ WAL committed successfully")

    except Exception as e:  # noqa: BLE001
        print(f"❌ Error committing WAL: {e}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ChromaDB maintenance and diagnostics")
    parser.add_argument(
        "--action",
        choices=["diagnose", "commit-wal"],
        default="diagnose",
        help="Action to perform (default: diagnose)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name to focus on (optional)",
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

    if args.action == "diagnose":
        diagnose(persist_dir, args.collection)
    elif args.action == "commit-wal":
        commit_wal(persist_dir)


if __name__ == "__main__":
    main()
