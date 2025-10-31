from __future__ import annotations

from pathlib import Path

from rag_engine.core.document_processor import DocumentProcessor


def test_process_folder_returns_documents():
    assets_dir = Path(__file__).parent / "_assets" / "document_processor" / "folder"
    processor = DocumentProcessor(mode="folder")
    docs = processor.process(str(assets_dir))

    assert len(docs) == 2
    kb_ids = {d.metadata["kbId"] for d in docs}
    assert "101" in kb_ids and "102" in kb_ids
    # Validate intro doc metadata regardless of ordering
    intro_doc = next(d for d in docs if d.metadata["kbId"] == "101")
    assert intro_doc.metadata["title"] == "intro"
    assert Path(intro_doc.metadata["source_file"]).exists()
    assert intro_doc.metadata["source_type"] == "folder"
    assert intro_doc.metadata["section_index"] == 0


def test_process_file_splits_by_heading():
    combined = Path(__file__).parent / "_assets" / "document_processor" / "file" / "combined_kb.md"
    processor = DocumentProcessor(mode="file")
    docs = processor.process(str(combined))

    assert len(docs) == 2  # Two H1 sections
    titles = {doc.metadata["title"] for doc in docs}
    assert {"Getting Started", "Administration Guide"} <= titles
    for i, doc in enumerate(docs):
        assert doc.metadata["kbId"] == "201"
        assert doc.metadata["source_type"] == "file"
        assert doc.metadata["section_index"] == i


def test_process_mkdocs_with_manifest():
    export_dir = Path(__file__).parent / "_assets" / "document_processor" / "mkdocs"
    processor = DocumentProcessor(mode="mkdocs")
    docs = processor.process(str(export_dir))

    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["kbId"] == "301"
    assert meta["title"] == "intro"
    assert meta["source_type"] == "mkdocs"
    assert meta["section_index"] == 0


def test_process_mkdocs_without_manifest(tmp_path: Path):
    # Create directory without manifest; should fall back to folder mode
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    f1 = doc_dir / "file.md"
    f1.write_text("---\nkbId: 401\ntitle: file\n---\n\n# Section\n\nContent", encoding="utf-8")

    processor = DocumentProcessor(mode="mkdocs")
    docs = processor.process(str(doc_dir))

    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["kbId"] == "401"
    assert meta["title"] == "file"
    assert meta["source_type"] == "folder"  # fallback uses folder mode
    assert meta["section_index"] == 0

