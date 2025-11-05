import contextlib
import os
import signal
import socket
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Tuple

import kb_proxy


class StaticHandler(SimpleHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (http.server API)
        # Map /favicon.ico -> /images/favicon.png
        if self.path == "/favicon.ico":
            self.path = "/images/favicon.png"
            return super().do_GET()
        
        
        return super().do_GET()


def _serve_http(directory: str, host: str, port: int) -> ThreadingHTTPServer:
    os.chdir(directory)
    httpd = ThreadingHTTPServer((host, port), StaticHandler)
    thread = threading.Thread(target=httpd.serve_forever, name="static-http", daemon=True)
    thread.start()
    print(f"Static server running at http://{host}:{port}")
    return httpd


def _serve_proxy(host: str, port: int) -> kb_proxy.socketserver.TCPServer:
    httpd = kb_proxy.socketserver.TCPServer((host, port), kb_proxy.ProxyHandler)
    thread = threading.Thread(target=httpd.serve_forever, name="kb-proxy", daemon=True)
    thread.start()
    print(f"Proxy server running at http://{host}:{port}")
    return httpd


def _get_default_host() -> str:
    # Try to choose a non-loopback address if possible
    try:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def run(static_host: str = "0.0.0.0", static_port: int = 8000, proxy_host: str = "0.0.0.0", proxy_port: int = 8010) -> Tuple[ThreadingHTTPServer, kb_proxy.socketserver.TCPServer]:
    directory = os.path.dirname(os.path.abspath(__file__))
    static_srv = _serve_http(directory, static_host, static_port)
    proxy_srv = _serve_proxy(proxy_host, proxy_port)

    host_hint = _get_default_host()
    print("\nOpen the widget at:")
    print(f"  http://{host_hint}:{static_port}/gradio-embedded.html")
    print("")
    return static_srv, proxy_srv


if __name__ == "__main__":
    _srv_static = None
    _srv_proxy = None

    def shutdown_servers(signum=None, frame=None):
        """Shutdown both servers gracefully."""
        print("\nShutting down...")
        if _srv_static:
            _srv_static.shutdown()
        if _srv_proxy:
            _srv_proxy.shutdown()
        print("Servers stopped.")
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_servers)
    signal.signal(signal.SIGTERM, shutdown_servers)

    try:
        _srv_static, _srv_proxy = run()
        # Keep main thread alive - use a loop with timeout to allow signal handling
        stop_event = threading.Event()
        try:
            while not stop_event.is_set():
                stop_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            shutdown_servers()
    except Exception as e:
        print(f"\nError: {e}")
        shutdown_servers()

