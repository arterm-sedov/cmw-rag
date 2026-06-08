from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Add project root to path so imports work without PYTHONPATH
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rag_engine.config.settings import get_collection_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REMOTE_URL = "https://github.com/arterm-sedov/cbap-mkdocs-ru.git"
DEFAULT_BRANCH = "platform_v6"
DEFAULT_TARGET_DIR = Path(".reference-repos/cbap-mkdocs-ru")
DEFAULT_SPARSE_PATH = "phpkb_content_rag"

V5_CORPUS_PATH = Path("phpkb_content_rag") / "798-platform_v5"
V6_CORPUS_PATH = Path("phpkb_content_rag") / "896-platform_v6"

CorpusChoice = Literal["v5", "v6", "all"]


class CorpusSyncError(RuntimeError):
    """Raised when corpus sync or indexing cannot proceed safely."""


@dataclass(frozen=True)
class CorpusSyncConfig:
    remote_url: str = DEFAULT_REMOTE_URL
    branch: str = DEFAULT_BRANCH
    sparse_path: str = DEFAULT_SPARSE_PATH
    target_dir: Path = DEFAULT_TARGET_DIR
    corpus: CorpusChoice = "all"
    dry_run: bool = False
    run_index: bool = False
    reindex: bool = False
    prune_missing: bool = False
    max_files: int | None = None

    @property
    def resolved_target_dir(self) -> Path:
        return (PROJECT_ROOT / self.target_dir).resolve() if not self.target_dir.is_absolute() else self.target_dir

    @property
    def corpus_dirs(self) -> list[Path]:
        if self.corpus == "v5":
            return [self.target_dir / V5_CORPUS_PATH]
        if self.corpus == "v6":
            return [self.target_dir / V6_CORPUS_PATH]
        return [self.target_dir / V5_CORPUS_PATH, self.target_dir / V6_CORPUS_PATH]

    @property
    def corpus_version_pairs(self) -> list[tuple[Path, str]]:
        if self.corpus == "v5":
            return [(self.target_dir / V5_CORPUS_PATH, "v5")]
        if self.corpus == "v6":
            return [(self.target_dir / V6_CORPUS_PATH, "v6")]
        return [
            (self.target_dir / V5_CORPUS_PATH, "v5"),
            (self.target_dir / V6_CORPUS_PATH, "v6"),
        ]

    @property
    def resolved_corpus_dirs(self) -> list[Path]:
        return [
            self.resolved_target_dir / path.relative_to(self.target_dir)
            for path in self.corpus_dirs
        ]


def run_command(command: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    """Run a subprocess command with readable failure reporting."""
    if dry_run:
        cwd_text = f" (cwd={cwd})" if cwd else ""
        print(f"DRY RUN: {' '.join(command)}{cwd_text}")
        return
    try:
        subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)
    except subprocess.CalledProcessError as exc:
        raise CorpusSyncError(f"Command failed with exit code {exc.returncode}: {command}") from exc


def _is_git_repo(path: Path) -> bool:
    return path.exists() and (path / ".git").exists()


def _set_sparse_checkout(config: CorpusSyncConfig, target: Path) -> None:
    if config.target_dir.is_symlink():
        print(
            "Existing symlink target detected; leaving sparse-checkout settings unchanged. "
            f"Target: {target}"
        )
        return
    run_command(
        ["git", "sparse-checkout", "set", config.sparse_path],
        cwd=target,
        dry_run=config.dry_run,
    )


def sync_corpus(config: CorpusSyncConfig) -> None:
    """Clone or fast-forward the managed sparse MkDocs corpus checkout."""
    target = config.resolved_target_dir

    if target.exists() and not _is_git_repo(target):
        raise CorpusSyncError(
            f"Target exists but is not a Git repository: {target}. "
            "Move it aside or choose a different --target-dir."
        )

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                config.branch,
                config.remote_url,
                str(target),
            ],
            dry_run=config.dry_run,
        )
        _set_sparse_checkout(config, target)
        return

    run_command(["git", "fetch", "origin", config.branch], cwd=target, dry_run=config.dry_run)
    run_command(["git", "checkout", config.branch], cwd=target, dry_run=config.dry_run)
    _set_sparse_checkout(config, target)
    run_command(
        ["git", "pull", "--ff-only", "origin", config.branch],
        cwd=target,
        dry_run=config.dry_run,
    )


def index_corpus(config: CorpusSyncConfig) -> None:
    """Index selected corpus folders through the existing build_index script."""
    for corpus_dir, version in config.corpus_version_pairs:
        resolved = config.resolved_target_dir / corpus_dir.relative_to(config.target_dir)
        if not config.dry_run and not resolved.exists():
            raise CorpusSyncError(f"Corpus folder not found: {resolved}")

        collection = get_collection_name(version)
        command = [
            sys.executable,
            "rag_engine/scripts/build_index.py",
            "--source",
            str(resolved),
            "--mode",
            "folder",
            "--collection",
            collection,
        ]
        if config.reindex:
            command.append("--reindex")
        if config.prune_missing:
            command.append("--prune-missing")
        if config.max_files is not None:
            command.extend(["--max-files", str(config.max_files)])

        run_command(command, cwd=PROJECT_ROOT, dry_run=config.dry_run)


def parse_args(argv: list[str] | None = None) -> CorpusSyncConfig:
    parser = argparse.ArgumentParser(
        description="Fetch/update the MkDocs RAG corpora and optionally index them."
    )
    parser.add_argument("--remote", default=DEFAULT_REMOTE_URL, help="MkDocs Git remote URL")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="MkDocs branch to sync")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Managed clone directory",
    )
    parser.add_argument(
        "--sparse-path",
        default=DEFAULT_SPARSE_PATH,
        help="Sparse checkout path to materialize",
    )
    parser.add_argument(
        "--corpus",
        choices=["v5", "v6", "all"],
        default="all",
        help="Corpus folder(s) to index when --index is used",
    )
    parser.add_argument("--index", action="store_true", help="Run build_index.py after sync")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--reindex", action="store_true", help="Pass --reindex to build_index.py")
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="Pass --prune-missing to build_index.py",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Pass --max-files to build_index.py",
    )

    args = parser.parse_args(argv)
    config = CorpusSyncConfig(
        remote_url=args.remote,
        branch=args.branch,
        sparse_path=args.sparse_path,
        target_dir=args.target_dir,
        corpus=args.corpus,
        dry_run=args.dry_run,
        run_index=args.index,
        reindex=args.reindex,
        prune_missing=args.prune_missing,
        max_files=args.max_files,
    )
    return config


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        sync_corpus(config)
        if config.run_index:
            index_corpus(config)
    except CorpusSyncError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
