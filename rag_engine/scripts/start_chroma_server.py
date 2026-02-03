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

    # Restart background server
    python scripts/start_chroma_server.py --restart

    # Show recent log output
    python scripts/start_chroma_server.py --log

    # Follow log output in real-time
    python scripts/start_chroma_server.py --tail

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
    """Get path to PID file for background process tracking (project root)."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    return project_root / ".chroma_server.pid"


def get_log_file() -> Path:
    """Get path to log file for background process output (project root)."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    return project_root / "chroma_server.log"


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


def tail_logs(lines: int = 50, follow: bool = False) -> None:
    """Show the tail of the ChromaDB server log.

    Args:
        lines: Number of lines to show from the end of the log
        follow: If True, continuously follow log updates (like tail -f)
    """
    log_file = get_log_file()

    if not log_file.exists():
        print(f"ℹ️  Log file not found: {log_file}")
        print(
            "   The server may not have been started in background mode, or no logs have been written yet."
        )
        return

    try:
        # Read the file and show last N lines
        with open(log_file, "r", encoding="utf-8") as f:
            if follow:
                # Follow mode: show existing content then follow
                import time

                # Go to end of file
                f.seek(0, 2)
                print(f"📋 Following log file: {log_file}")
                print("   Press Ctrl+C to stop\n")
                try:
                    while True:
                        line = f.readline()
                        if line:
                            print(line, end="")
                        else:
                            time.sleep(0.1)
                except KeyboardInterrupt:
                    print("\n👋 Stopped following logs.")
            else:
                # Tail mode: show last N lines
                all_lines = f.readlines()
                tail_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                print(f"📋 Last {len(tail_lines)} lines from {log_file}:\n")
                print("".join(tail_lines))
    except Exception as e:
        print(f"❌ Error reading log file: {e}")


def check_status() -> None:
    """Check if ChromaDB server is running."""
    env_path = get_env_file_path()
    env_vars = read_env_file(env_path)
    config = get_chroma_config(env_vars)
    port = config["port"]
    host = config["host"]

    pid = get_running_pid()

    if pid:
        print(f"✅ ChromaDB server is running (PID: {pid})")
        print(f"   Connect at: http://{host}:{port}")
    else:
        if is_server_running(port):
            print(f"⚠️  Something is running on port {port}, but it's not our tracked process.")
            print("   You may need to stop it manually.")
        else:
            print("ℹ️  ChromaDB server is not running.")


def start_chroma_server(foreground: bool = True, verbose: bool = False) -> None:
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
    port = config["port"]
    host = config["host"]

    # Check if already running on configured port
    if is_server_running(port):
        pid = get_running_pid()
        if pid:
            print(f"⚠️  ChromaDB server is already running (PID: {pid})")
            print(f"   Connect at: http://{host}:{port}")
            return
        print(f"⚠️  Port {port} is already in use by another process.")
        sys.exit(1)

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
            log_file_path = get_log_file()
            # Open log file for appending stdout/stderr
            log_handle = open(log_file_path, "a", encoding="utf-8")
            log_handle.write(f"\n{'=' * 60}\n")
            log_handle.write(f"ChromaDB server started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_handle.write(f"Command: {' '.join(cmd)}\n")
            log_handle.write(f"{'=' * 60}\n\n")
            log_handle.flush()

            if sys.platform == "win32":
                # Windows: no console window (invisible), detached process
                creation_flags = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    | subprocess.DETACHED_PROCESS
                    | subprocess.CREATE_NO_WINDOW
                )
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                process = subprocess.Popen(
                    cmd,
                    creationflags=creation_flags,
                    startupinfo=startupinfo,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            else:
                # Unix/Linux/Mac: Use start_new_session
                process = subprocess.Popen(
                    cmd,
                    start_new_session=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )

            # Write PID to file
            pid_file = get_pid_file()
            with open(pid_file, "w") as f:
                f.write(str(process.pid))

            print(f"✅ ChromaDB server started in background (PID: {process.pid})")
            print(f"   Connect at: http://{host}:{port}")
            print(f"   Data path: {config['persist_dir']}")
            print(f"   PID file: {pid_file}")
            print(f"   Log file: {log_file_path}")
            print()
            print("Commands:")
            print(f"   Stop:   python {__file__} --stop")
            print(f"   Status: python {__file__} --status")
            print(f"   Logs:   python {__file__} --log")
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

  # Restart background server
  python start_chroma_server.py --restart

  # Show recent log output
  python start_chroma_server.py --log

  # Follow log output in real-time
  python start_chroma_server.py --tail

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
    parser.add_argument(
        "--restart",
        "-r",
        action="store_true",
        help="Restart the background server (stop if running, then start)",
    )
    parser.add_argument("--status", action="store_true", help="Check if server is running")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument(
        "--log",
        action="store_true",
        help="Show the last 50 lines of the server log file",
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help="Follow the server log in real-time (like tail -f)",
    )

    args = parser.parse_args()

    if args.stop:
        stop_server(args.verbose)
    elif args.restart:
        # Restart: stop if running, then start
        stop_server(args.verbose)
        print("🔄 Restarting ChromaDB server...")
        print()
        start_chroma_server(foreground=args.foreground, verbose=args.verbose)
    elif args.status:
        check_status()
    elif args.log:
        tail_logs(lines=50, follow=False)
    elif args.tail:
        tail_logs(lines=50, follow=True)
    else:
        print("🎯 ChromaDB Server Starter")
        print("   Reading configuration from .env file...")
        print()
        # Default to daemon mode (background)
        start_chroma_server(foreground=args.foreground, verbose=args.verbose)


if __name__ == "__main__":
    main()
