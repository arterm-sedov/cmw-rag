from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


def main() -> None:
    # Load .env the same way as the rest of the project
    load_dotenv()
    try:
        import chromadb
    except Exception as e:  # noqa: BLE001
        print("Chromadb import failed:", e)
        sys.exit(1)

    # Use HTTP client with settings from .env
    from rag_engine.config.settings import settings

    client = chromadb.HttpClient(
        host=settings.chromadb_host,
        port=settings.chromadb_port,
    )

    # Detect collection
    collection_name = os.getenv("CHROMADB_COLLECTION")
    collection = None
    if collection_name:
        try:
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:  # noqa: BLE001
            print(f"Failed to open collection {collection_name}: {e}")

    if collection is None:
        cols = client.list_collections()
        if not cols:
            print("No collections found on server")
            return
        # Pick the first collection
        col0 = cols[0]
        name = getattr(col0, "name", None) or str(col0)
        collection = client.get_or_create_collection(name=name)

    # Optional filters and limits via env
    sample_limit = int(os.getenv("CHROMA_SAMPLE_LIMIT", "5"))
    filter_kbid = os.getenv("CHROMA_SAMPLE_KBID")

    # Count items (paged)
    limit = 1000
    offset = 0
    count = 0
    while True:
        res = collection.get(include=[], limit=limit, offset=offset)
        ids = res.get("ids", [])
        batch = len(ids)
        count += batch
        offset += batch
        if batch < limit:
            break
    print("ChromaDB Server:", f"{settings.chromadb_host}:{settings.chromadb_port}")
    print("Collection:", getattr(collection, "name", "<unknown>"))
    print("Total items:", count)

    # Sample items (optionally filtered by kbId)
    where = {"kbId": filter_kbid} if filter_kbid else None
    # Note: ids are always returned by collection.get; don't request via include
    sample = collection.get(include=["metadatas", "documents"], where=where, limit=sample_limit)
    metas = sample.get("metadatas", [])
    docs = sample.get("documents", [])
    ids = sample.get("ids", [])

    if not ids:
        print("No sample items to show." + (f" Filter kbId={filter_kbid}" if filter_kbid else ""))
        return

    for i, (id_, m, d) in enumerate(zip(ids, metas, docs), 1):
        print(
            f"\nSample {i} id={id_} kbId={m.get('kbId')} "
            f"chunk_index={m.get('chunk_index')} has_code={m.get('has_code')}"
        )
        print("source_file:", m.get("source_file"))
        print("doc[:120]:", (d or "")[:120].replace("\n", " "))


if __name__ == "__main__":
    main()
