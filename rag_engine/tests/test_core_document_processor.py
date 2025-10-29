from __future__ import annotations

from pathlib import Path

from rag_engine.core.document_processor import DocumentProcessor


def test_process_folder_returns_documents(docs_fixture_path: Path):
    processor = DocumentProcessor(mode="folder")
    docs = processor.process(str(docs_fixture_path))

    assert len(docs) == 2
    kb_ids = {d.metadata["kbId"] for d in docs}
    assert any(k.endswith("intro") for k in kb_ids)
    assert any(k.endswith("advanced") for k in kb_ids)
    # Validate intro doc metadata regardless of ordering
    intro_doc = next(d for d in docs if d.metadata["kbId"].endswith("intro"))
    assert intro_doc.metadata["title"] == "intro"
    assert Path(intro_doc.metadata["source_file"]).exists()
    assert intro_doc.metadata["source_type"] == "folder"
    assert intro_doc.metadata["section_index"] == 0


def test_process_file_splits_by_heading(tmp_path, fixtures_path: Path):
    combined = fixtures_path / "combined_kb.md"
    processor = DocumentProcessor(mode="file")
    docs = processor.process(str(combined))

    assert len(docs) == 2  # Two H1 sections
    titles = {doc.metadata["title"] for doc in docs}
    assert {"Getting Started", "Administration Guide"} <= titles
    for i, doc in enumerate(docs):
        assert doc.metadata["source_type"] == "file"
        assert doc.metadata["section_index"] == i


def test_process_mkdocs_with_manifest(mkdocs_export_path: Path):
    processor = DocumentProcessor(mode="mkdocs")
    docs = processor.process(str(mkdocs_export_path))

    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["kbId"] == "intro"
    assert meta["source_type"] == "mkdocs"
    assert meta["section_index"] == 0


def test_process_mkdocs_without_manifest(tmp_path):
    # Create directory without manifest and reuse docs
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "file.md").write_text("# Section\n\nContent", encoding="utf-8")

    processor = DocumentProcessor(mode="mkdocs")
    docs = processor.process(str(doc_dir))

    assert len(docs) == 1
    assert docs[0].metadata["kbId"] == "file"
    assert docs[0].metadata["source_type"] == "folder"
    assert docs[0].metadata["section_index"] == 0

