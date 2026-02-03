"""Simple ChromaDB connection test using HTTP client."""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv


def check_chroma_sync():
    """Synchronous check for backwards compatibility."""
    load_dotenv()
    persist_dir = os.getenv("CHROMADB_PERSIST_DIR", "data/chromadb_data")

    try:
        import chromadb

        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()

        if collections:
            print(f"✓ ChromaDB connected (embedded mode)")
            print(f"Found {len(collections)} collection(s):")
            for col in collections:
                name = getattr(col, "name", str(col))
                try:
                    count = client.get_collection(name).count()
                    print(f"  - {name}: {count} items")
                except Exception as e:
                    print(f"  - {name}: error counting ({e})")
            return True
        else:
            print("No collections found in:", persist_dir)
            return False
    except Exception as e:
        print(f"✗ Failed to connect to ChromaDB: {e}")
        return False


async def check_chroma_async():
    """Async check using HTTP client."""
    load_dotenv()

    try:
        import chromadb

        host = os.getenv("CHROMADB_HOST", "localhost")
        port = int(os.getenv("CHROMADB_PORT", "8000"))

        client = await chromadb.AsyncHttpClient(host=host, port=port)
        collections = await client.list_collections()

        if collections:
            print(f"✓ ChromaDB connected (HTTP: {host}:{port})")
            print(f"Found {len(collections)} collection(s):")
            for col in collections:
                name = getattr(col, "name", str(col))
                try:
                    collection = await client.get_collection(name)
                    count = await collection.count()
                    print(f"  - {name}: {count} items")
                except Exception as e:
                    print(f"  - {name}: error counting ({e})")
            return True
        else:
            print(f"No collections found on {host}:{port}")
            return False
    except Exception as e:
        print(f"✗ Failed to connect to ChromaDB HTTP: {e}")
        return False


def main():
    """Main entry point - supports both sync and async modes."""
    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        result = asyncio.run(check_chroma_async())
        return 0 if result else 1
    else:
        result = check_chroma_sync()
        return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
