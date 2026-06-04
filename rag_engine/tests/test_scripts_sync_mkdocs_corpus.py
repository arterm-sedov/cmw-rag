from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rag_engine.scripts import sync_mkdocs_corpus as sync


def test_default_config_derives_corpus_path() -> None:
    config = sync.CorpusSyncConfig()

    assert config.remote_url == "https://github.com/arterm-sedov/cbap-mkdocs-ru.git"
    assert config.branch == "platform_v6"
    assert config.sparse_path == "phpkb_content_rag"
    assert config.target_dir == Path(".reference-repos/cbap-mkdocs-ru")
    assert config.corpus_dirs == [
        Path(
            ".reference-repos/cbap-mkdocs-ru/phpkb_content_rag/"
            "798. Версия 5.0. Текущая рекомендованная"
        ),
        Path(".reference-repos/cbap-mkdocs-ru/phpkb_content_rag/896-platform_v6"),
    ]


def test_sync_clones_missing_target_and_sets_sparse_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "cbap-mkdocs-ru"
    commands: list[tuple[list[str], Path | None]] = []

    def fake_run_command(command, *, cwd=None, dry_run=False):  # noqa: ANN001
        commands.append((list(command), cwd))

    monkeypatch.setattr(sync, "run_command", fake_run_command)
    config = sync.CorpusSyncConfig(target_dir=target)

    sync.sync_corpus(config)

    assert commands == [
        (
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                "platform_v6",
                "https://github.com/arterm-sedov/cbap-mkdocs-ru.git",
                str(target),
            ],
            None,
        ),
        (["git", "sparse-checkout", "set", "phpkb_content_rag"], target),
    ]


def test_sync_updates_existing_git_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "cbap-mkdocs-ru"
    (target / ".git").mkdir(parents=True)
    commands: list[tuple[list[str], Path | None]] = []

    def fake_run_command(command, *, cwd=None, dry_run=False):  # noqa: ANN001
        commands.append((list(command), cwd))

    monkeypatch.setattr(sync, "run_command", fake_run_command)
    config = sync.CorpusSyncConfig(target_dir=target)

    sync.sync_corpus(config)

    assert commands == [
        (["git", "fetch", "origin", "platform_v6"], target),
        (["git", "checkout", "platform_v6"], target),
        (["git", "sparse-checkout", "set", "phpkb_content_rag"], target),
        (["git", "pull", "--ff-only", "origin", "platform_v6"], target),
    ]


def test_sync_does_not_apply_sparse_checkout_to_symlink_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "cbap-mkdocs-ru"
    (target / ".git").mkdir(parents=True)
    commands: list[tuple[list[str], Path | None]] = []

    def fake_run_command(command, *, cwd=None, dry_run=False):  # noqa: ANN001
        commands.append((list(command), cwd))

    monkeypatch.setattr(sync, "run_command", fake_run_command)
    monkeypatch.setattr(type(target), "is_symlink", lambda self: self == target)
    config = sync.CorpusSyncConfig(target_dir=target)

    sync.sync_corpus(config)

    assert commands == [
        (["git", "fetch", "origin", "platform_v6"], target),
        (["git", "checkout", "platform_v6"], target),
        (["git", "pull", "--ff-only", "origin", "platform_v6"], target),
    ]


def test_sync_rejects_existing_non_git_target(tmp_path: Path) -> None:
    target = tmp_path / "cbap-mkdocs-ru"
    target.mkdir()

    config = sync.CorpusSyncConfig(target_dir=target)

    with pytest.raises(sync.CorpusSyncError, match="not a Git repository"):
        sync.sync_corpus(config)


def test_index_corpus_builds_expected_commands_for_all_corpora(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "cbap-mkdocs-ru"
    v5 = target / "phpkb_content_rag" / "798. Версия 5.0. Текущая рекомендованная"
    v6 = target / "phpkb_content_rag" / "896-platform_v6"
    v5.mkdir(parents=True)
    v6.mkdir(parents=True)
    commands: list[tuple[list[str], Path | None]] = []

    def fake_run_command(command, *, cwd=None, dry_run=False):  # noqa: ANN001
        commands.append((list(command), cwd))

    monkeypatch.setattr(sync, "run_command", fake_run_command)
    config = sync.CorpusSyncConfig(target_dir=target)

    sync.index_corpus(config)

    assert commands == [
        (
            [
                sys.executable,
                "rag_engine/scripts/build_index.py",
                "--source",
                str(v5),
                "--mode",
                "folder",
            ],
            sync.PROJECT_ROOT,
        ),
        (
            [
                sys.executable,
                "rag_engine/scripts/build_index.py",
                "--source",
                str(v6),
                "--mode",
                "folder",
            ],
            sync.PROJECT_ROOT,
        )
    ]


def test_index_corpus_adds_reindex_and_prune_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "cbap-mkdocs-ru"
    corpus = target / "phpkb_content_rag" / "896-platform_v6"
    corpus.mkdir(parents=True)
    recorded: dict[str, list[str]] = {}

    def fake_run_command(command, *, cwd=None, dry_run=False):  # noqa: ANN001, ARG001
        recorded["command"] = list(command)

    monkeypatch.setattr(sync, "run_command", fake_run_command)
    config = sync.CorpusSyncConfig(
        target_dir=target,
        corpus="v6",
        reindex=True,
        prune_missing=True,
    )

    sync.index_corpus(config)

    assert "--reindex" in recorded["command"]
    assert "--prune-missing" in recorded["command"]


def test_run_command_raises_readable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        raise subprocess.CalledProcessError(7, ["git", "fetch"])

    monkeypatch.setattr(sync.subprocess, "run", fake_run)

    with pytest.raises(sync.CorpusSyncError, match="Command failed"):
        sync.run_command(["git", "fetch"])
