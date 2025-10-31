"""Inspect ChromaDB schema and show sample records for validation."""
from __future__ import annotations

import argparse
import random
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
    """Get all document metadatas with pagination (utility for inspection)."""
    all_metadata = []
    offset = 0
    
    while True:
        res = store.collection.get(limit=limit_per_batch, offset=offset, include=["metadatas", "documents", "ids"])
        batch_metas = res.get("metadatas", [])
        batch_docs = res.get("documents", [])
        batch_ids = res.get("ids", [])
        if not batch_metas:
            break
        
        # Combine metadata with documents and ids
        for meta, doc, doc_id in zip(batch_metas, batch_docs, batch_ids):
            combined = meta.copy()
            combined["_document"] = doc
            combined["_chroma_id"] = doc_id
            all_metadata.append(combined)
        
        if len(batch_metas) < limit_per_batch:
            break
        offset += limit_per_batch
    
    return all_metadata


def inspect_schema(store: ChromaStore, num_samples: int = 5) -> None:
    """Inspect ChromaDB schema and show sample records.
    
    Args:
        store: ChromaStore instance
        num_samples: Number of random records to show
    """
    print("=" * 80)
    print("ChromaDB Schema and Data Inspection")
    print("=" * 80)
    
    # Get collection info
    print("\nCollection Information:")
    print("-" * 80)
    print(f"Collection name: {store.collection_name}")
    print(f"Persist directory: {store.persist_dir}")
    total_count = store.collection.count()
    print(f"Total chunks: {total_count}")
    
    # Get sample records
    if total_count == 0:
        print("\n⚠️  Collection is empty")
        return
    
    print(f"\nFetching all metadata (for schema analysis and sampling)...")
    all_records = get_all_metadata_paginated(store)
    print(f"Retrieved {len(all_records)} records\n")
    
    # Analyze schema (all unique metadata keys)
    print("=" * 80)
    print("Data Model (Metadata Schema):")
    print("-" * 80)
    all_keys = set()
    key_types = {}
    key_examples = {}
    
    for record in all_records:
        for key, value in record.items():
            if key.startswith("_"):  # Skip internal fields
                continue
            all_keys.add(key)
            value_type = type(value).__name__
            if key not in key_types:
                key_types[key] = set()
            key_types[key].add(value_type)
            
            # Store example value (first non-empty we see)
            if key not in key_examples and value is not None:
                if isinstance(value, str) and len(value) > 100:
                    key_examples[key] = value[:100] + "..."
                else:
                    key_examples[key] = value
    
    # Sort keys logically (put common ones first)
    common_keys = ["kbId", "doc_stable_id", "stable_id", "title", "source_file", 
                   "source_type", "chunk_index", "file_mtime_epoch", "file_modified_at"]
    other_keys = sorted(all_keys - set(common_keys))
    ordered_keys = [k for k in common_keys if k in all_keys] + other_keys
    
    print(f"\nFound {len(ordered_keys)} metadata fields:\n")
    for key in ordered_keys:
        types_str = ", ".join(sorted(key_types[key]))
        example = key_examples.get(key, "N/A")
        example_str = str(example)[:60] if example != "N/A" else "N/A"
        print(f"  {key:<25} Type: {types_str:<15} Example: {example_str}")
    
    # Show sample records
    print("\n" + "=" * 80)
    print(f"Sample Records (random {num_samples} of {len(all_records)}):")
    print("-" * 80)
    
    if len(all_records) < num_samples:
        samples = all_records
    else:
        samples = random.sample(all_records, num_samples)
    
    for i, record in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"ChromaDB ID: {record.get('_chroma_id', 'N/A')}")
        print(f"Document preview: {str(record.get('_document', ''))[:100]}...")
        print("\nMetadata:")
        for key in ordered_keys:
            if key in record:
                value = record[key]
                # Truncate long values
                if isinstance(value, str) and len(value) > 80:
                    display_value = value[:80] + "..."
                else:
                    display_value = value
                print(f"  {key}: {display_value}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("Statistics:")
    print("-" * 80)
    
    # Count unique values for key fields
    unique_kbids = {r.get("kbId") for r in all_records if r.get("kbId")}
    unique_doc_stable_ids = {r.get("doc_stable_id") for r in all_records if r.get("doc_stable_id")}
    unique_sources = {r.get("source_file") for r in all_records if r.get("source_file")}
    
    print(f"Unique kbIds: {len(unique_kbids)}")
    print(f"Unique doc_stable_ids (articles): {len(unique_doc_stable_ids)}")
    print(f"Unique source files: {len(unique_sources)}")
    print(f"Chunks per article (avg): {len(all_records) / len(unique_doc_stable_ids) if unique_doc_stable_ids else 0:.1f}")
    
    # Check for common issues
    print("\n" + "=" * 80)
    print("Validation Checks:")
    print("-" * 80)
    
    issues = []
    records_without_kbid = [r for r in all_records if not r.get("kbId")]
    records_without_doc_stable_id = [r for r in all_records if not r.get("doc_stable_id")]
    records_without_source = [r for r in all_records if not r.get("source_file")]
    
    if records_without_kbid:
        issues.append(f"⚠️  {len(records_without_kbid)} records missing kbId")
    if records_without_doc_stable_id:
        issues.append(f"⚠️  {len(records_without_doc_stable_id)} records missing doc_stable_id")
    if records_without_source:
        issues.append(f"⚠️  {len(records_without_source)} records missing source_file")
    
    # Check for path-like kbIds (shouldn't exist after migration)
    path_like_kbids = [r for r in all_records if r.get("kbId") and ("\\" in str(r.get("kbId")) or "/" in str(r.get("kbId")) or " " in str(r.get("kbId")))]
    if path_like_kbids:
        issues.append(f"⚠️  {len(path_like_kbids)} records have path-like kbIds (should be numeric or normalized)")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ No validation issues found")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Inspect ChromaDB schema and show sample records")
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of random records to show (default: 5)",
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
    
    store = ChromaStore(persist_dir=persist_dir, collection_name=collection_name)
    inspect_schema(store, num_samples=args.samples)


if __name__ == "__main__":
    main()

