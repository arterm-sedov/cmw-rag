"""Simple script to test connection to a remote ChromaDB server."""
from __future__ import annotations

import argparse
import sys

try:
    import chromadb
except ImportError as e:
    print(f"❌ ChromaDB import failed: {e}")
    print("Please ensure chromadb is installed: pip install chromadb")
    sys.exit(1)


def test_connection(url: str) -> bool:
    """Test connection to remote ChromaDB server.
    
    Args:
        url: Server URL (e.g., http://10.9.7.7:8000/)
        
    Returns:
        True if connection successful, False otherwise
    """
    print(f"🔗 Testing connection to: {url}")
    
    try:
        # Parse URL to extract host and port
        # Remove trailing slash and protocol
        clean_url = url.rstrip("/")
        if clean_url.startswith("http://"):
            clean_url = clean_url[7:]
        elif clean_url.startswith("https://"):
            clean_url = clean_url[8:]
            print("⚠️  Warning: HTTPS detected, but SSL not configured in this script")
        
        # Split host and port
        if ":" in clean_url:
            host, port_str = clean_url.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                print(f"❌ Invalid port number: {port_str}")
                return False
        else:
            host = clean_url
            port = 8000  # Default ChromaDB port
            print(f"ℹ️  No port specified, using default: {port}")
        
        # Create HTTP client
        print(f"📡 Connecting to {host}:{port}...")
        client = chromadb.HttpClient(host=host, port=port)
        
        # Test connection by listing collections
        print("🔍 Listing collections...")
        collections = client.list_collections()
        
        print(f"✅ Connection successful!")
        print(f"📊 Found {len(collections)} collection(s):")
        for collection in collections:
            col_name = getattr(collection, "name", str(collection))
            print(f"   - {col_name}")
        
        return True
        
    except Exception as e:  # noqa: BLE001
        print(f"❌ Connection failed: {e}")
        return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test connection to a remote ChromaDB server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_chroma_connection.py
  python test_chroma_connection.py --url http://10.9.7.7:8000/
  python test_chroma_connection.py --url http://localhost:8000
        """,
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://10.9.7.7:8000/",
        help="ChromaDB server URL (default: http://10.9.7.7:8000/)",
    )
    
    args = parser.parse_args()
    
    success = test_connection(args.url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

