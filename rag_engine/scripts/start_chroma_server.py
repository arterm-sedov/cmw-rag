#!/usr/bin/env python3
"""Helper script to start ChromaDB server with configuration from .env file.

This script reads ChromaDB configuration from .env and starts the ChromaDB
HTTP server with the correct parameters, ensuring consistency between
the app configuration and the server startup.

Usage:
    python scripts/start_chroma_server.py
    python scripts/start_chroma_server.py --verbose
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def get_env_file_path() -> Path:
    """Find .env file in project root."""
    # Try current directory first
    env_path = Path(".env")
    if env_path.exists():
        return env_path.absolute()

    # Try project root (2 levels up from this script)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        return env_path

    return Path(".env")  # Return default even if not found


def read_env_file(env_path: Path) -> dict[str, str]:
    """Read environment variables from .env file."""
    env_vars: dict[str, str] = {}

    if not env_path.exists():
        print(f"⚠️  Warning: {env_path} not found. Using defaults.")
        return env_vars

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Remove inline comments
                if "#" in value:
                    value = value.split("#")[0].strip()
                env_vars[key] = value

    return env_vars


def get_chroma_config(env_vars: dict[str, str]) -> dict[str, str | int]:
    """Extract ChromaDB configuration from env vars."""
    config = {
        "host": env_vars.get("CHROMADB_HOST", "localhost"),
        "port": int(env_vars.get("CHROMADB_PORT", "8000")),
        "persist_dir": env_vars.get("CHROMADB_PERSIST_DIR", "./data/chromadb_data"),
    }
    return config


def build_chroma_command(config: dict[str, str | int]) -> list[str]:
    """Build chroma run command with configuration."""
    cmd = [
        "chroma",
        "run",
        "--host",
        str(config["host"]),
        "--port",
        str(config["port"]),
        "--path",
        str(config["persist_dir"]),
    ]
    return cmd


def start_chroma_server(verbose: bool = False) -> None:
    """Start ChromaDB server with configuration from .env."""
    # Find and read .env file
    env_path = get_env_file_path()
    if verbose:
        print(f"📄 Reading configuration from: {env_path}")

    env_vars = read_env_file(env_path)

    # Check if required vars are present
    required_vars = ["CHROMADB_HOST", "CHROMADB_PORT", "CHROMADB_PERSIST_DIR"]
    missing = [var for var in required_vars if var not in env_vars]
    if missing:
        print(f"⚠️  Warning: Missing required variables in .env: {', '.join(missing)}")
        print("   Using default values for missing variables.")

    # Get ChromaDB configuration
    config = get_chroma_config(env_vars)

    if verbose:
        print(f"🔧 Configuration:")
        print(f"   Host: {config['host']}")
        print(f"   Port: {config['port']}")
        print(f"   Data path: {config['persist_dir']}")

    # Build and run command
    cmd = build_chroma_command(config)

    if verbose:
        print(f"🚀 Starting: {' '.join(cmd)}")
        print()

    try:
        # Run chroma server
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("❌ Error: 'chroma' command not found.")
        print("   Please install ChromaDB: pip install chromadb")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: ChromaDB server exited with code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n👋 ChromaDB server stopped.")
        sys.exit(0)


def main() -> None:
    """Main entry point."""
    # Check for verbose flag
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print("🎯 ChromaDB Server Starter")
    print("   Reading configuration from .env file...")
    print()

    start_chroma_server(verbose=verbose)


if __name__ == "__main__":
    main()
