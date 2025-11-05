import http.server
import socketserver
import urllib.parse
import urllib.request
import re


TARGET_DEFAULT = "https://kb.comindware.ru/category/comindware-platform/versiya-5-0-tekushaya-rekomendovannaya/798/"
PROXY_PORT = 8010


def _is_html(content_type: str | None) -> bool:
    return bool(content_type and "text/html" in content_type.lower())


def _rewrite_html(html: bytes, base_url: str, proxy_origin: str) -> bytes:
    text = html.decode("utf-8", errors="ignore")
    # Ensure <base> for relative URL resolution in browser
    if "<head" in text and "<base" not in text:
        text = re.sub(r"(<head[^>]*>)", rf"\1<base href=\"{base_url}\">", text, count=1, flags=re.IGNORECASE)

    # Rewrite absolute and root-relative URLs to pass back through proxy
    def to_proxy(url: str) -> str:
        url = url.strip()
        if not url or url.startswith("data:") or url.startswith("javascript:"):
            return url
        # Already proxied
        if url.startswith(f"{proxy_origin}/proxy"):
            return url
        # Absolute URL
        if re.match(r"^https?://", url):
            return f"{proxy_origin}/proxy?url={urllib.parse.quote(url, safe='') }"
        # Protocol-relative
        if url.startswith("//"):
            return f"{proxy_origin}/proxy?url={urllib.parse.quote('https:' + url, safe='')}"
        # Anchor-only
        if url.startswith("#"):
            return url
        # Relative or root-relative
        resolved = urllib.parse.urljoin(base_url, url)
        return f"{proxy_origin}/proxy?url={urllib.parse.quote(resolved, safe='')}"

    # Attributes to rewrite
    attrs = ["href", "src", "action"]
    for attr in attrs:
        text = re.sub(
            rf"({attr}\s*=\s*\")([^\"]+)(\")",
            lambda m: f"{m.group(1)}{to_proxy(m.group(2))}{m.group(3)}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"({attr}\s*=\s*')([^']+)(')",
            lambda m: f"{m.group(1)}{to_proxy(m.group(2))}{m.group(3)}",
            text,
            flags=re.IGNORECASE,
        )

    return text.encode("utf-8")


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path.rstrip("/") == "" or parsed.path == "/":
            # Redirect to default target via proxy endpoint
            target = TARGET_DEFAULT
            location = f"/proxy?url={urllib.parse.quote(target, safe='')}"
            self.send_response(302)
            self.send_header("Location", location)
            self.end_headers()
            return

        if parsed.path.startswith("/proxy"):
            target = qs.get("url", [TARGET_DEFAULT])[0]
            self._proxy_to(target)
            return

        # Fallback: serve 404 for other paths
        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Not Found")

    def _proxy_to(self, target_url: str) -> None:
        try:
            req = urllib.request.Request(
                target_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "ru,en;q=0.9",
                },
            )
            with urllib.request.urlopen(req, timeout=20) as resp:  # nosec - local dev tool
                status = getattr(resp, "status", 200)
                content = resp.read()
                content_type = resp.headers.get("Content-Type", "")

                # Prepare response
                self.send_response(status)

                # Copy headers excluding frame and CSP blockers
                hop_by_hop = {
                    "connection",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailers",
                    "transfer-encoding",
                    "upgrade",
                }
                blocked = {
                    "x-frame-options",
                    "content-security-policy",
                    "frame-ancestors",
                    "strict-transport-security",
                }
                for k, v in resp.headers.items():
                    lk = k.lower()
                    if lk in hop_by_hop or lk in blocked:
                        continue
                    # Prevent cross-origin cookies leakage to proxy origin
                    if lk == "set-cookie":
                        continue
                    if lk == "content-length":
                        # We'll set after potential rewrite
                        continue
                    self.send_header(k, v)

                proxy_origin = f"http://{self.headers.get('Host')}"
                if _is_html(content_type):
                    rewritten = _rewrite_html(content, target_url, proxy_origin)
                    content = rewritten
                    self.send_header("Content-Type", "text/html; charset=utf-8")

                # CORS to make iframe happier
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
        except Exception as exc:  # noqa: BLE001 - simple dev server
            msg = f"Proxy error: {exc}".encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)


def run() -> None:
    with socketserver.TCPServer(("127.0.0.1", PROXY_PORT), ProxyHandler) as httpd:
        print(f"Proxy server running at http://127.0.0.1:{PROXY_PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    run()


