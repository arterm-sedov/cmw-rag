from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _normalize_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(str(url).strip())
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = (parsed.path or "").rstrip("/")
        # Keep query (contains kbId), drop only fragment for dedup
        query = f"?{parsed.query}" if parsed.query else ""
        return f"{scheme}://{netloc}{path}{query}"
    except Exception:
        return str(url or "").strip()


def format_with_citations(answer: str, docs: Iterable[Any]) -> str:
    """Format answer with citations using proper article URLs; never use file paths.

    Deduplicates by kbId to ensure each article is cited only once.

    Assumptions:
    - metadata contains numeric kbId (enforced by indexer)
    - metadata may already contain a canonical 'url' or precomputed 'article_url'
    """
    # Deduplicate by kbId to avoid citing the same article multiple times
    seen_kbids: Dict[str, Dict[str, Any]] = {}
    seen_urls: set[str] = set()
    for d in docs:
        meta: Dict[str, Any] = getattr(d, "metadata", {}) or {}
        kbid = meta.get("kbId") or getattr(d, "kb_id", None)
        url = meta.get("url") or meta.get("article_url")
        norm_url = _normalize_url(url)

        if kbid:
            # Use first occurrence of each kbId
            if kbid not in seen_kbids:
                seen_kbids[kbid] = meta
        elif norm_url:
            # Fallback dedup by normalized URL when kbId is missing
            if norm_url in seen_urls:
                continue
            seen_urls.add(norm_url)
            # Use a synthetic key to preserve ordering
            seen_kbids[norm_url] = meta

    citations: List[str] = []
    for i, (kbid, meta) in enumerate(seen_kbids.items(), 1):
        title = meta.get("title") or kbid or f"Source {i}"

        # 1) explicit URL from frontmatter/metadata
        url = meta.get("url")
        # 2) or precomputed article_url
        if not url:
            url = meta.get("article_url")
        # 3) or construct from numeric kbId (guaranteed by indexer)
        if not url and kbid and str(kbid).isdigit():
            url = f"https://kb.comindware.ru/article.php?id={kbid}"

        # Append anchor only if we have a base URL and the base doesn't already include one
        anchor = meta.get("section_anchor") or ""
        full_url = f"{url}{anchor}" if (url and ("#" not in str(url))) else (url or "")

        link = f"[{title}]({full_url})" if full_url else title
        citations.append(f"{i}. {link}")
    if citations:
        return answer + "\n\n## Источники:\n\n" + "\n".join(citations)
    return answer


