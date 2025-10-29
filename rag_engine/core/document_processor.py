"""Unified document processor for all 3 input modes."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class Document:
    """Parsed document with metadata."""

    def __init__(self, content: str, metadata: dict[str, Any]):
        self.content = content
        self.metadata = metadata


class DocumentProcessor:
    """Process documents from folder, file, or mkdocs export."""

    def __init__(self, mode: str = "folder"):
        """Initialize processor.

        Args:
            mode: 'folder', 'file', or 'mkdocs'
        """
        self.mode = mode
        logger.info("DocumentProcessor initialized in %s mode", mode)

    def process(self, source: str) -> list[Document]:
        """Process documents from source.

        Args:
            source: Path to folder, file, or mkdocs export

        Returns:
            List of parsed documents with metadata
        """
        if self.mode == "folder":
            return self._process_folder(source)
        if self.mode == "file":
            return self._process_file(source)
        if self.mode == "mkdocs":
            return self._process_mkdocs(source)
        raise ValueError(f"Unknown mode: {self.mode}")

    def _process_folder(self, folder_path: str) -> list[Document]:
        """Mode 3: Scan folder for MD files."""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        documents: list[Document] = []
        md_files = list(folder.rglob("*.md"))
        md_files.sort()
        logger.info("Found %d markdown files in %s", len(md_files), folder_path)

        for md_file in md_files:
            try:
                content, metadata = self._parse_md_with_frontmatter(md_file)
                rel_path = md_file.relative_to(folder)
                kb_id = str(rel_path.with_suffix(""))
                metadata.setdefault("kbId", kb_id)
                metadata.setdefault("title", md_file.stem)
                metadata.setdefault("source_file", str(md_file))
                documents.append(Document(content, metadata))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to process %s: %s", md_file, exc)
                continue

        logger.info("Successfully processed %d documents", len(documents))
        return documents

    def _process_file(self, file_path: str) -> list[Document]:
        """Mode 2: Parse single large MD file."""
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("Processing single file: %s", file_path)
        content = file.read_text(encoding="utf-8")
        sections = self._split_by_headings(content)

        documents: list[Document] = []
        for i, (title, section_content) in enumerate(sections):
            metadata: dict[str, Any] = {
                "kbId": f"{file.stem}_{i}",
                "title": title or f"Section {i}",
                "source_file": str(file),
                "section_index": i,
            }
            documents.append(Document(section_content, metadata))

        logger.info("Split file into %d sections", len(documents))
        return documents

    def _process_mkdocs(self, export_dir: str) -> list[Document]:
        """Mode 1: Process MkDocs export with manifest."""
        export_path = Path(export_dir)
        manifest_file = export_path / "rag_manifest.json"

        if not manifest_file.exists():
            logger.warning("No manifest found, falling back to folder mode")
            return self._process_folder(export_dir)

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        logger.info("Processing MkDocs export: %s files", manifest.get("total_files"))

        documents: list[Document] = []
        for file_path in manifest.get("files", []):
            md_file = export_path / file_path
            if md_file.exists():
                content, metadata = self._parse_md_with_frontmatter(md_file)
                metadata.setdefault("kbId", str(Path(file_path).with_suffix("")))
                metadata.setdefault("title", md_file.stem)
                metadata["source_type"] = "mkdocs_export"
                documents.append(Document(content, metadata))

        logger.info("Processed %d MkDocs documents", len(documents))
        return documents

    def _parse_md_with_frontmatter(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Parse markdown file with optional YAML frontmatter."""
        content = file_path.read_text(encoding="utf-8")

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1])
                    return parts[2].strip(), frontmatter or {}
                except yaml.YAMLError as exc:  # noqa:TRY003
                    logger.warning("Failed to parse frontmatter in %s: %s", file_path, exc)

        return content, {}

    def _split_by_headings(self, content: str) -> list[tuple[str | None, str]]:
        """Split content by H1 headings."""
        lines = content.split("\n")
        sections: list[tuple[str | None, str]] = []
        current_title: str | None = None
        current_content: list[str] = []

        for line in lines:
            if line.startswith("# "):
                if current_content:
                    sections.append((current_title, "\n".join(current_content)))
                current_title = line[2:].strip()
                current_content = [line]
            else:
                current_content.append(line)

        if current_content:
            sections.append((current_title, "\n".join(current_content)))

        return sections


