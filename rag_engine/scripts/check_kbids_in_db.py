"""Quick script to check what kbId values are actually in the database."""

import re
from rag_engine.config.settings import settings
from rag_engine.storage.vector_store import ChromaStore

store = ChromaStore(collection_name=settings.chromadb_collection)

# Get sample of documents
res = store.collection.get(limit=100, include=["metadatas"])
metadatas = res.get("metadatas", [])

print("Sample of kbId values in database:")
print("=" * 80)
kbids_seen = {}
for meta in metadatas:
    kb_id = str(meta.get("kbId", ""))
    source_file = meta.get("source_file", "")

    # Count occurrences
    if kb_id not in kbids_seen:
        kbids_seen[kb_id] = []
    kbids_seen[kb_id].append(source_file[-50:] if source_file else "N/A")

# Show unique kbIds
print(f"\nUnique kbIds found in sample (first 30):")
for i, (kb_id, sources) in enumerate(sorted(kbids_seen.items())[:30], 1):
    print(f"{i}. kbId: '{kb_id}' (appears {len(sources)} times)")
    if re.match(r"^\d+-", kb_id):
        # Check if it's path-like
        is_path = "\\" in kb_id or "/" in kb_id or " " in kb_id or len(kb_id) > 50
        if not is_path:
            numeric = kb_id.split("-")[0]
            print(f"   → Would normalize to: '{numeric}'")
        else:
            print(f"   → Path-like, skipping")

# Check for suffixed pattern
print("\n" + "=" * 80)
print("Checking for suffixed kbIds (pattern: ^\\d+-):")
suffixed = []
for kb_id in kbids_seen.keys():
    kb_str = str(kb_id)
    is_path_like = (
        "\\" in kb_str
        or "/" in kb_str
        or " " in kb_str
        or (len(kb_str) > 50)
        or re.match(r"^\d+\.\s", kb_str) is not None
    )
    matches_pattern = re.match(r"^\d+-", kb_str) is not None

    if matches_pattern and not is_path_like:
        suffixed.append(kb_id)

if suffixed:
    print(f"Found {len(suffixed)} suffixed kbIds in sample:")
    for kb in suffixed[:10]:
        print(f"  - '{kb}'")
else:
    print("No suffixed kbIds found in sample (all are already normalized or path-like)")
