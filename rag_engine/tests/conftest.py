from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_path() -> Path:
    return Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="session")
def docs_fixture_path(fixtures_path: Path) -> Path:
    return fixtures_path / "docs"


@pytest.fixture(scope="session")
def mkdocs_export_path(fixtures_path: Path) -> Path:
    return fixtures_path / "mkdocs_export"


