"""Quick script to inspect kbId and source_file values in ChromaDB."""

from rag_engine.config.settings import settings
from rag_engine.storage.vector_store import ChromaStore

store = ChromaStore(collection_name=settings.chromadb_collection)
res = store.collection.get(limit=20, include=["metadatas"])

print("Sample of kbId and source_file values:")
print("=" * 80)
for i, meta in enumerate(res.get("metadatas", [])[:20], 1):
    kb_id = meta.get("kbId")
    source_file = meta.get("source_file")
    print(f"{i}. kbId: {kb_id}")
    print(f"   source_file: {source_file}")

    # Check if kbId looks like a path
    if source_file and kb_id and kb_id in str(source_file):
        print(f"   ⚠️  WARNING: kbId appears to be a path fragment!")
    print()
