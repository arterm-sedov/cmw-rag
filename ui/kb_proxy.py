import http.server
import socketserver
import urllib.error
import urllib.parse
import urllib.request
import re


TARGET_DEFAULT = "https://kb.comindware.ru/category/comindware-platform/versiya-5-0-tekushaya-rekomendovannaya/798/"
PROXY_PORT = 8010


def _is_html(content_type: str | None) -> bool:
    return bool(content_type and "text/html" in content_type.lower())


def _is_css(content_type: str | None) -> bool:
    return bool(content_type and "text/css" in content_type.lower())


def _is_json(content_type: str | None) -> bool:
    return bool(content_type and ("application/json" in content_type.lower() or "text/json" in content_type.lower()))


def _rewrite_css(css: bytes, base_url: str, proxy_origin: str) -> bytes:
    """Rewrite url() references in CSS to go through proxy."""
    text = css.decode("utf-8", errors="ignore")

    def to_proxy(url: str) -> str:
        url = url.strip().strip('"\'')
        if not url or url.startswith("data:") or url.startswith("javascript:"):
            return url
        if url.startswith(f"{proxy_origin}/proxy"):
            return url
        if re.match(r"^https?://", url):
            return f"{proxy_origin}/proxy?url={urllib.parse.quote(url, safe='')}"
        if url.startswith("//"):
            return f"{proxy_origin}/proxy?url={urllib.parse.quote('https:' + url, safe='')}"
        if url.startswith("#"):
            return url
        # Root-relative URLs (start with /) - resolve to KB site root
        if url.startswith("/"):
            parsed_base = urllib.parse.urlparse(base_url)
            kb_root = f"{parsed_base.scheme}://{parsed_base.netloc}"
            resolved = urllib.parse.urljoin(kb_root + "/", url.lstrip("/"))
        else:
            # Relative URLs - resolve relative to CSS file directory
            resolved = urllib.parse.urljoin(base_url, url)
        return f"{proxy_origin}/proxy?url={urllib.parse.quote(resolved, safe='')}"

    # Rewrite url() references in CSS
    def replace_url(match: re.Match) -> str:
        full = match.group(0)
        url_part = match.group(1).strip()
        # Preserve quotes if present
        has_quote_start = url_part.startswith('"') or url_part.startswith("'")
        has_quote_end = url_part.endswith('"') or url_part.endswith("'")
        if has_quote_start and has_quote_end:
            quote_char = url_part[0]
            url_clean = url_part[1:-1]
            proxied = to_proxy(url_clean)
            return f"url({quote_char}{proxied}{quote_char})"
        proxied = to_proxy(url_part)
        return f"url({proxied})"

    # Match url(...) with optional quotes
    text = re.sub(
        r"url\s*\(\s*([^)]+)\s*\)",
        replace_url,
        text,
        flags=re.IGNORECASE,
    )

    # Also handle @import statements
    text = re.sub(
        r"@import\s+['\"]([^'\"]+)['\"]",
        lambda m: f'@import "{to_proxy(m.group(1))}"',
        text,
        flags=re.IGNORECASE,
    )

    return text.encode("utf-8")


def _rewrite_html(html: bytes, base_url: str, proxy_origin: str) -> bytes:
    """Rewrite HTML - only rewrite resource URLs (CSS, JS, images), keep everything else intact."""
    text = html.decode("utf-8", errors="ignore")
    
    # Helper to resolve and proxy URLs
    def to_proxy(url: str) -> str:
        url = url.strip()
        if not url or url.startswith("data:") or url.startswith("javascript:") or url.startswith("mailto:"):
            return url
        if url.startswith(f"{proxy_origin}/proxy"):
            return url
        if re.match(r"^https?://", url):
            return f"{proxy_origin}/proxy?url={urllib.parse.quote(url, safe='')}"
        if url.startswith("//"):
            return f"{proxy_origin}/proxy?url={urllib.parse.quote('https:' + url, safe='')}"
        if url.startswith("#"):
            return url
        # Root-relative URLs - resolve to KB site root
        if url.startswith("/"):
            parsed_base = urllib.parse.urlparse(base_url)
            kb_root = f"{parsed_base.scheme}://{parsed_base.netloc}"
            resolved = urllib.parse.urljoin(kb_root + "/", url.lstrip("/"))
        else:
            # Relative URLs - resolve relative to current page
            resolved = urllib.parse.urljoin(base_url, url)
        return f"{proxy_origin}/proxy?url={urllib.parse.quote(resolved, safe='')}"

    # Only rewrite href/src for link, script, img, and form action
    # This is more conservative and preserves inline JavaScript
    
    # Rewrite resource URLs - use a single, robust function for all attribute rewriting
    def rewrite_attribute_in_tag(tag_pattern: str, attr_name: str, skip_protocols: tuple = ()) -> None:
        """Helper to rewrite specific attribute in specific tags."""
        def rewrite_tag(match: re.Match) -> str:
            full_tag = match.group(0)
            # Match attribute with double quotes
            attr_pattern_double = rf'{attr_name}\s*=\s*"([^"]+)"'
            attr_match = re.search(attr_pattern_double, full_tag, re.IGNORECASE)
            if attr_match:
                original_url = attr_match.group(1)
                if any(original_url.startswith(proto) for proto in skip_protocols):
                    return full_tag
                proxied_url = to_proxy(original_url)
                return full_tag.replace(f'{attr_name}="{original_url}"', f'{attr_name}="{proxied_url}"')
            # Match attribute with single quotes
            attr_pattern_single = rf"{attr_name}\s*=\s*'([^']+)'"
            attr_match = re.search(attr_pattern_single, full_tag, re.IGNORECASE)
            if attr_match:
                original_url = attr_match.group(1)
                if any(original_url.startswith(proto) for proto in skip_protocols):
                    return full_tag
                proxied_url = to_proxy(original_url)
                return full_tag.replace(f"{attr_name}='{original_url}'", f"{attr_name}='{proxied_url}'")
            return full_tag
        
        nonlocal text
        text = re.sub(tag_pattern, rewrite_tag, text, flags=re.IGNORECASE)
    
    # Rewrite <link href=...> tags (CSS, fonts, etc.)
    rewrite_attribute_in_tag(r'<link[^>]*>', 'href')
    
    # Rewrite <script src=...> tags (external scripts only, not inline)
    rewrite_attribute_in_tag(r'<script[^>]+src\s*=[^>]*>', 'src')
    
    # Rewrite <img src=...> tags
    rewrite_attribute_in_tag(r'<img[^>]*>', 'src')
    
    # Rewrite <form action=...> tags
    rewrite_attribute_in_tag(r'<form[^>]*>', 'action')
    
    # Rewrite <a href=...> tags (skip javascript: and mailto:)
    rewrite_attribute_in_tag(r'<a[^>]*href\s*=[^>]*>', 'href', skip_protocols=('javascript:', 'mailto:'))
    
    # Rewrite inline <style> tags
    def rewrite_style_tag(match: re.Match) -> str:
        open_tag = match.group(1)
        style_content = match.group(2)
        close_tag = match.group(3)
        css_bytes = style_content.encode("utf-8")
        rewritten = _rewrite_css(css_bytes, base_url, proxy_origin)
        return f"{open_tag}{rewritten.decode('utf-8', errors='ignore')}{close_tag}"
    
    text = re.sub(
        r"(<style[^>]*>)(.*?)(</style>)",
        rewrite_style_tag,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    
    # Rewrite style attributes
    def rewrite_style_attr(match: re.Match) -> str:
        before = match.group(1)
        style_value = match.group(2)
        after = match.group(3)
        css_bytes = style_value.encode("utf-8")
        rewritten = _rewrite_css(css_bytes, base_url, proxy_origin)
        return f"{before}{rewritten.decode('utf-8', errors='ignore')}{after}"
    
    text = re.sub(
        r'(style\s*=\s*["\'])([^"\']*?)(["\'])',
        rewrite_style_attr,
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
                    "Accept": "text/html,application/xhtml+xml,application/xml,application/json,text/json,*/*;q=0.9",
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
                # Only rewrite HTML and CSS - pass through everything else unchanged (JSON, images, JS, etc.)
                if _is_html(content_type):
                    rewritten = _rewrite_html(content, target_url, proxy_origin)
                    content = rewritten
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                elif _is_css(content_type):
                    # For CSS, use the CSS file's directory as base URL
                    # This ensures relative URLs in CSS (like ../fonts/file.woff2) resolve correctly
                    parsed = urllib.parse.urlparse(target_url)
                    css_base_url = urllib.parse.urlunparse((
                        parsed.scheme, parsed.netloc,
                        urllib.parse.urljoin(parsed.path, '.'),  # Get directory
                        '', '', ''
                    ))
                    rewritten = _rewrite_css(content, css_base_url, proxy_origin)
                    content = rewritten
                    self.send_header("Content-Type", "text/css; charset=utf-8")
                elif _is_json(content_type):
                    # JSON files pass through unchanged - preserve original Content-Type
                    # Content-Type header already set in the loop above
                    pass
                # For all other content types (images, JS, fonts, etc.), pass through unchanged
                # Content-Type and other headers already set in the loop above

                # CORS to make iframe happier
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
        except urllib.error.HTTPError as e:
            # HTTP errors (404, 500, etc.) - pass through the status and error body
            error_msg = f"HTTP {e.code}: {e.reason}"
            if e.code == 404:
                error_msg += f"\n\nURL not found: {target_url}"
            else:
                error_msg += f"\n\nTarget URL: {target_url}"
            print(f"Proxy HTTP error: {error_msg}", flush=True)
            try:
                error_body = e.read()
            except Exception:
                error_body = error_msg.encode("utf-8")
            self.send_response(e.code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(error_body)))
            self.end_headers()
            self.wfile.write(error_body)
        except urllib.error.URLError as e:
            # Network errors - show the actual error
            error_msg = f"Network error: {e.reason}\n\nTarget URL: {target_url}"
            print(f"Proxy URL error: {error_msg}", flush=True)
            msg = error_msg.encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
        except Exception as exc:  # noqa: BLE001 - simple dev server
            # Other errors - show full traceback
            import traceback
            error_msg = f"Proxy error: {exc}\n\nTarget URL: {target_url}\n\n{traceback.format_exc()}"
            print(f"Proxy error: {error_msg}", flush=True)
            msg = f"Proxy error: {exc}\n\nTarget URL: {target_url}".encode("utf-8")
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


