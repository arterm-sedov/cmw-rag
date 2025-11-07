"""Tests for path normalization utilities."""
from __future__ import annotations

from pathlib import Path

from rag_engine.utils.path_utils import normalize_path


def test_normalize_path_windows_backslashes():
    """Test that Windows-style backslashes are converted to forward slashes."""
    windows_path = ".reference-repos\\.cbap-mkdocs\\phpkb_content_rag\\file.md"
    normalized = normalize_path(windows_path)

    # Should convert backslashes to forward slashes
    assert "\\" not in str(normalized)
    assert "/" in str(normalized) or str(normalized) == "file.md"


def test_normalize_path_posix_slashes():
    """Test that POSIX-style paths remain unchanged."""
    posix_path = ".reference-repos/.cbap-mkdocs/phpkb_content_rag/file.md"
    normalized = normalize_path(posix_path)

    # Should remain the same
    assert str(normalized) == posix_path


def test_normalize_path_path_object():
    """Test that Path objects are handled correctly."""
    path_obj = Path(".reference-repos/.cbap-mkdocs/file.md")
    normalized = normalize_path(path_obj)

    # Should return a Path object
    assert isinstance(normalized, Path)
    assert str(normalized) == str(path_obj)


def test_normalize_path_mixed_separators():
    """Test paths with mixed separators."""
    mixed_path = ".reference-repos\\.cbap-mkdocs/phpkb_content_rag\\file.md"
    normalized = normalize_path(mixed_path)

    # All backslashes should be converted
    assert "\\" not in str(normalized)
    assert "/" in str(normalized)
