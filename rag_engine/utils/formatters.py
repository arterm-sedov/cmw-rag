from __future__ import annotations

from typing import Any, Dict, Iterable, List


def format_with_citations(answer: str, docs: Iterable[Any]) -> str:
    citations: List[str] = []
    for i, d in enumerate(docs, 1):
        meta: Dict[str, Any] = getattr(d, "metadata", {})
        title = meta.get("title") or meta.get("kbId") or f"Source {i}"
        url = meta.get("url") or meta.get("source_file", "")
        anchor = meta.get("section_anchor", "")
        link = f"[{title}]({url}{anchor})" if url else title
        citations.append(f"{i}. {link}")
    if citations:
        return answer + "\n\n## Sources:\n\n" + "\n".join(citations)
    return answer


