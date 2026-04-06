"""Rebuild unique_sources_complete.json from report-pack appendices A and F."""

from __future__ import annotations

import json
import re
from pathlib import Path

# legacy-files/ → executive-research-technology-transfer/ → research/ → docs/ → repo root
ROOT = Path(__file__).resolve().parents[4]
REPORT_PACK = ROOT / "docs/research/executive-research-technology-transfer/report-pack"
OUT = Path(__file__).resolve().parent / "unique_sources_complete.json"

SOURCES = [
    REPORT_PACK / "20260325-research-appendix-a-index-ru.md",
    REPORT_PACK / "20260325-research-appendix-f-extended-reading-ru.md",
]


def _strip_section_anchor(raw: str) -> str:
    return re.sub(r"\s*\{:\s*#[^}]+\}\s*$", "", raw).strip()


def parse_markdown_links(md: str, file_label: str) -> list[dict[str, str]]:
    section = file_label
    out: list[dict[str, str]] = []
    for line in md.splitlines():
        if line.startswith("## "):
            section = _strip_section_anchor(line.removeprefix("## ").strip())
            continue
        m = re.match(r"^\s*-\s*\[([^\]]+)\]\((https?://[^)]+)\)\s*$", line)
        if not m:
            continue
        title, url = m.group(1).strip(), m.group(2).strip()
        out.append({"title": title, "url": url, "context": section})
    return out


def main() -> None:
    seen: set[str] = set()
    merged: list[dict[str, str]] = []
    for path in SOURCES:
        if not path.is_file():
            raise FileNotFoundError(path)
        label = path.stem
        for row in parse_markdown_links(path.read_text(encoding="utf-8"), label):
            u = row["url"]
            if u in seen:
                continue
            seen.add(u)
            merged.append(row)

    OUT.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(merged)} unique http(s) sources to {OUT}")


if __name__ == "__main__":
    main()
