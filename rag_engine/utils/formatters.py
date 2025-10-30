from __future__ import annotations

from typing import Any, Dict, Iterable, List


def format_with_citations(answer: str, docs: Iterable[Any]) -> str:
    """Format answer with citations using proper article URLs; never use file paths.

    Deduplicates by kbId to ensure each article is cited only once.

    Assumptions:
    - metadata contains numeric kbId (enforced by indexer)
    - metadata may already contain a canonical 'url' or precomputed 'article_url'
    """
    # Deduplicate by kbId to avoid citing the same article multiple times
    seen_kbids: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        meta: Dict[str, Any] = getattr(d, "metadata", {}) or {}
        kbid = meta.get("kbId") or getattr(d, "kb_id", None)
        if not kbid:
            continue  # Skip docs without kbId
        
        # Use first occurrence of each kbId
        if kbid not in seen_kbids:
            seen_kbids[kbid] = meta

    citations: List[str] = []
    for i, (kbid, meta) in enumerate(seen_kbids.items(), 1):
        title = meta.get("title") or kbid or f"Source {i}"

        # 1) explicit URL from frontmatter/metadata
        url = meta.get("url")
        # 2) or precomputed article_url
        if not url:
            url = meta.get("article_url")
        # 3) or construct from numeric kbId (guaranteed by indexer)
        if not url and kbid:
            url = f"https://kb.comindware.ru/article.php?id={kbid}"

        # Append anchor only if we have a base URL and the base doesn't already include one
        anchor = meta.get("section_anchor") or ""
        full_url = f"{url}{anchor}" if (url and ("#" not in str(url))) else (url or "")

        link = f"[{title}]({full_url})" if full_url else title
        citations.append(f"{i}. {link}")
    if citations:
        return answer + "\n\n## Источники:\n\n" + "\n".join(citations)
    return answer


