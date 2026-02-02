#!/usr/bin/env python3
"""Helper script to start ChromaDB server with configuration from .env file.

This script reads ChromaDB configuration from .env and starts the ChromaDB
HTTP server with the correct parameters, ensuring consistency between
the app configuration and the server startup.

By default, runs in daemon (background) mode. Use --foreground to see logs.

Usage:
    # Run in background (default, daemon mode)
    python scripts/start_chroma_server.py

    # Run in foreground (blocking, shows logs)
    python scripts/start_chroma_server.py --foreground

    # Stop background server
    python scripts/start_chroma_server.py --stop

    # Check if running
    python scripts/start_chroma_server.py --status

    # Verbose output
    python scripts/start_chroma_server.py --verbose
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
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


def get_pid_file() -> Path:
    """Get path to PID file for background process tracking."""
    script_dir = Path(__file__).parent
    return script_dir / ".chroma_server.pid"


def is_server_running(port: int) -> bool:
    """Check if ChromaDB server is already running on the specified port."""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_running_pid() -> int | None:
    """Get PID of running ChromaDB server from PID file."""
    pid_file = get_pid_file()
    if not pid_file.exists():
        return None

    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        # Check if process actually exists
        import psutil

        if psutil.pid_exists(pid):
            return pid
        else:
            # Stale PID file
            pid_file.unlink()
            return None
    except (ValueError, FileNotFoundError):
        return None


def stop_server(verbose: bool = False) -> bool:
    """Stop the background ChromaDB server."""
    pid = get_running_pid()

    if not pid:
        print("ℹ️  No background ChromaDB server is running.")
        return False

    try:
        import psutil

        process = psutil.Process(pid)
        process.terminate()

        # Wait for process to terminate
        try:
            process.wait(timeout=5)
            print(f"✅ Stopped ChromaDB server (PID: {pid})")
        except psutil.TimeoutExpired:
            process.kill()
            print(f"⚠️  Force-killed ChromaDB server (PID: {pid})")

        # Clean up PID file
        pid_file = get_pid_file()
        if pid_file.exists():
            pid_file.unlink()

        return True
    except Exception as e:
        print(f"❌ Error stopping server: {e}")
        return False


def check_status() -> None:
    """Check if ChromaDB server is running."""
    pid = get_running_pid()

    if pid:
        print(f"✅ ChromaDB server is running (PID: {pid})")
        print(f"   Connect at: http://localhost:8000")
    else:
        # Check if something else is using the port
        if is_server_running(8000):
            print("⚠️  Something is running on port 8000, but it's not our tracked process.")
            print("   You may need to stop it manually.")
        else:
            print("ℹ️  ChromaDB server is not running.")


def start_chroma_server(foreground: bool = True, verbose: bool = False) -> None:
    """Start ChromaDB server with configuration from .env."""
    # Check if already running
    if is_server_running(8000):
        pid = get_running_pid()
        if pid:
            print(f"⚠️  ChromaDB server is already running (PID: {pid})")
            print(f"   Connect at: http://localhost:8000")
            return
        else:
            print("⚠️  Port 8000 is already in use by another process.")
            sys.exit(1)

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

    if verbose or foreground:
        print(f"🔧 Configuration:")
        print(f"   Host: {config['host']}")
        print(f"   Port: {config['port']}")
        print(f"   Data path: {config['persist_dir']}")

    # Build command
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

    if foreground:
        # Run in foreground (blocking)
        if verbose:
            print(f"🚀 Starting: {' '.join(cmd)}")
            print()

        try:
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
    else:
        # Run in background (detached)
        try:
            if sys.platform == "win32":
                # Windows: Use CREATE_NEW_PROCESS_GROUP and DETACHED_PROCESS
                process = subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
            else:
                # Unix/Linux/Mac: Use start_new_session
                process = subprocess.Popen(
                    cmd,
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )

            # Write PID to file
            pid_file = get_pid_file()
            with open(pid_file, "w") as f:
                f.write(str(process.pid))

            print(f"✅ ChromaDB server started in background (PID: {process.pid})")
            print(f"   Connect at: http://localhost:8000")
            print(f"   Data path: {config['persist_dir']}")
            print(f"   PID file: {pid_file}")
            print()
            print("Commands:")
            print(f"   Stop:   python {__file__} --stop")
            print(f"   Status: python {__file__} --status")
            print(f"   Logs:   Check terminal or system process logs")
            print()
            print("Tracing:")
            print(
                f"   Process: chroma run --host {config['host']} --port {config['port']} --path {config['persist_dir']}"
            )
            print(f"   Working directory: {Path.cwd()}")
            print(f"   Python: {sys.executable}")
            if sys.platform == "win32":
                print(f"   Platform: Windows (PID: {process.pid})")
            else:
                print(f"   Platform: Unix/Linux (PID: {process.pid})")

        except Exception as e:
            print(f"❌ Error starting background server: {e}")
            sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start ChromaDB HTTP server with configuration from .env",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in background (default, daemon mode)
  python start_chroma_server.py
  
  # Run in foreground (blocking, shows logs)
  python start_chroma_server.py --foreground
  
  # Stop background server
  python start_chroma_server.py --stop
  
  # Check if running
  python start_chroma_server.py --status
  
  # Verbose output
  python start_chroma_server.py --verbose
        """,
    )

    parser.add_argument(
        "--foreground",
        "-f",
        action="store_true",
        help="Run server in foreground (blocking, shows logs)",
    )
    parser.add_argument("--stop", "-s", action="store_true", help="Stop the background server")
    parser.add_argument("--status", action="store_true", help="Check if server is running")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")

    args = parser.parse_args()

    if args.stop:
        stop_server(args.verbose)
    elif args.status:
        check_status()
    else:
        print("🎯 ChromaDB Server Starter")
        print("   Reading configuration from .env file...")
        print()
        # Default to daemon mode (background)
        start_chroma_server(foreground=args.foreground, verbose=args.verbose)


if __name__ == "__main__":
    main()
