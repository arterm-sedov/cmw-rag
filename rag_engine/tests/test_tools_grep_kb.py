"""Tests for corpus grep tool."""

from __future__ import annotations

import json
import sys

import pytest

MODULE = "rag_engine.tools.retrieve_context"
from rag_engine.tools import retrieve_context as _trigger  # noqa: F401, E402


def _tc():
    return sys.modules[MODULE]


def asyncio_run(coro):
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)


class TestGrepKbArticles:
    def test_grep_returns_articles_in_standard_json(self, monkeypatch, tmp_path):
        """grep_kb_articles returns articles in JSON shape matching retrieve_context."""
        tc = _tc()
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc.md").write_text(
            "---\nkbId: 100\ntitle: Test Doc\n---\nHello world\n", encoding="utf-8"
        )

        def fake_run(args, **kwargs):  # noqa: ANN001, ARG001
            return type("R", (), {"returncode": 0, "stdout": str(corpus / "doc.md") + "\n"})

        monkeypatch.setattr(tc.subprocess, "run", fake_run)
        monkeypatch.setattr(tc, "get_corpus_dir", lambda v: str(corpus))
        monkeypatch.setattr(tc, "_read_article", lambda p: "Hello world")

        result = tc._grep_kb_articles_core("test", product_version="v6")
        data = json.loads(result)

        assert len(data["articles"]) == 1
        assert data["articles"][0]["kb_id"] == "100"
        assert data["articles"][0]["title"] == "Test Doc"
        assert data["articles"][0]["content"] == "Hello world"
        assert data["metadata"]["has_results"] is True

    def test_grep_respects_max_matches(self, monkeypatch, tmp_path):
        """max_matches caps the number of returned articles."""
        tc = _tc()
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        files = []
        for i in range(5):
            path = corpus / f"{i}.md"
            path.write_text(f"---\nkbId: {i}\ntitle: T{i}\n---\nBody\n", encoding="utf-8")
            files.append(str(path))

        def fake_run(args, **kwargs):  # noqa: ANN001, ARG001
            return type("R", (), {"returncode": 0, "stdout": "\n".join(files)})

        monkeypatch.setattr(tc.subprocess, "run", fake_run)
        monkeypatch.setattr(tc, "get_corpus_dir", lambda v: str(corpus))
        monkeypatch.setattr(tc, "_read_article", lambda p: "Body")

        result = tc._grep_kb_articles_core("test", product_version="v6", max_matches=2)
        data = json.loads(result)

        assert len(data["articles"]) == 2

    def test_grep_invalid_regex_raises_error(self, monkeypatch, tmp_path):
        """Invalid regex pattern raises a clean ValueError."""
        tc = _tc()
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        monkeypatch.setattr(tc, "get_corpus_dir", lambda v: str(corpus))

        with pytest.raises(ValueError, match="regex"):
            tc._grep_kb_articles_core("[invalid", product_version="v6")
