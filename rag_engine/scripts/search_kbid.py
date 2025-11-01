"""Search for a specific kbid in ChromaDB.

This script queries ChromaDB for all documents matching a given kbid and displays
their metadata, content preview, and statistics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rag_engine.config.settings import settings
from rag_engine.storage.vector_store import ChromaStore


def search_kbid(kbid: str, show_content: bool = False, limit: int | None = None) -> None:
    """Search for documents with a specific kbid in ChromaDB.
    
    Args:
        kbid: The kbid to search for
        show_content: Whether to display document content
        limit: Maximum number of results to display (None for all)
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize store
    store = ChromaStore(
        persist_dir=settings.chromadb_persist_dir,
        collection_name=settings.chromadb_collection
    )
    
    # Search for documents with the specified kbid
    where_filter = {"kbId": kbid}
    
    print(f"Searching for kbid: '{kbid}'")
    print("=" * 80)
    
    # Get all matching documents
    # Note: IDs are always returned by ChromaDB, don't include in 'include' parameter
    include_params = ["documents", "metadatas"]
    res = store.collection.get(
        where=where_filter,
        include=include_params,
        limit=limit if limit is not None else 10000
    )
    
    documents = res.get("documents", [])
    metadatas = res.get("metadatas", [])
    ids = res.get("ids", [])
    
    total_count = len(documents)
    
    if total_count == 0:
        print(f"\nNo documents found with kbid: '{kbid}'")
        print("\nTip: Use 'check_kbids_in_db.py' to see available kbids in the database.")
        return
    
    print(f"\nFound {total_count} document(s) with kbid '{kbid}':")
    print("-" * 80)
    
    # Group by source file to show statistics
    source_files = {}
    for meta in metadatas:
        source = meta.get("source_file", "N/A")
        if source not in source_files:
            source_files[source] = 0
        source_files[source] += 1
    
    print("\nStatistics:")
    print(f"  Total documents: {total_count}")
    print(f"  Unique source files: {len(source_files)}")
    print("\nSource file distribution:")
    for source, count in sorted(source_files.items()):
        print(f"  - {source}: {count} chunk(s)")
    
    # Display detailed information for each document
    print("\n" + "=" * 80)
    print("Detailed Results:")
    print("-" * 80)
    
    for i, (doc_id, doc_content, meta) in enumerate(zip(ids, documents, metadatas), 1):
        print(f"\n[{i}/{total_count}] Document ID: {doc_id[:50]}{'...' if len(doc_id) > 50 else ''}")
        print(f"  Source file: {meta.get('source_file', 'N/A')}")
        print(f"  kbid: {meta.get('kbId', 'N/A')}")
        print(f"  Chunk index: {meta.get('chunk_index', 'N/A')}")
        print(f"  doc_stable_id: {meta.get('doc_stable_id', 'N/A')}")
        
        # Show additional metadata fields
        other_meta = {k: v for k, v in meta.items() if k not in ["kbId", "source_file", "chunk_index", "doc_stable_id"]}
        if other_meta:
            print(f"  Other metadata: {other_meta}")
        
        if show_content:
            content_preview = doc_content[:500] if len(doc_content) > 500 else doc_content
            print(f"  Content preview: {content_preview}")
            if len(doc_content) > 500:
                print(f"  ... (truncated, total length: {len(doc_content)} characters)")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Search for a specific kbid in ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python search_kbid.py 4578
  python search_kbid.py 4578 --show-content
  python search_kbid.py 4578 --limit 10
        """
    )
    parser.add_argument(
        "kbid",
        type=str,
        help="The kbid to search for"
    )
    parser.add_argument(
        "--show-content",
        action="store_true",
        help="Display document content previews"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of results to display (default: all)"
    )
    
    args = parser.parse_args()
    
    search_kbid(
        kbid=args.kbid,
        show_content=args.show_content,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
