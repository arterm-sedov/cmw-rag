from __future__ import annotations

import importlib
import sys


def test_run_mkdocs_export_creates_files(monkeypatch, tmp_path):
    module = importlib.reload(importlib.import_module("rag_engine.scripts.run_mkdocs_export"))

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    output_dir = tmp_path / "output"

    copied = {}

    def fake_copyfile(src, dst):  # noqa: ANN001
        copied["src"] = src
        copied["dst"] = dst

    monkeypatch.setattr(module, "shutil", type("S", (), {"copyfile": staticmethod(fake_copyfile)}))

    called = {}

    def fake_check_call(cmd, cwd=None):  # noqa: ANN001
        called["cmd"] = cmd
        called["cwd"] = cwd

    monkeypatch.setattr(module, "subprocess", type("P", (), {"check_call": staticmethod(fake_check_call)}))

    argv = sys.argv
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mkdocs_export.py",
            "--project-dir",
            str(project_dir),
            "--inherit-config",
            "base.yml",
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()

    assert called["cmd"][0] == "mkdocs"
    assert called["cwd"] == str(project_dir)
    yaml_file = project_dir / ".rag_export" / "mkdocs_for_rag_indexing.yml"
    assert yaml_file.exists()
    assert "INHERIT: base.yml" in yaml_file.read_text(encoding="utf-8")
    assert (project_dir / ".rag_export" / "rag_indexing_hook.py").exists()
    sys.argv = argv

