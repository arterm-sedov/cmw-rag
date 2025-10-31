"""Tests for Git utilities."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_engine.utils.git_utils import get_file_timestamp, get_git_timestamp
from rag_engine.utils.metadata_utils import _epoch_to_iso, parse_frontmatter_timestamp


class TestEpochToIso:
    """Tests for epoch to ISO conversion."""

    def test_epoch_to_iso(self):
        """Test epoch timestamp conversion to ISO string."""
        # Calculate correct epoch for 2024-06-14 12:33:36 UTC
        from datetime import datetime, timezone
        expected_dt = datetime(2024, 6, 14, 12, 33, 36, tzinfo=timezone.utc)
        epoch = int(expected_dt.timestamp())
        iso = _epoch_to_iso(epoch)
        assert iso == "2024-06-14T12:33:36+00:00"


class TestParseFrontmatterTimestamp:
    """Tests for frontmatter timestamp parsing."""

    def test_parse_datetime_format(self):
        """Test parsing 'YYYY-MM-DD HH:MM:SS' format."""
        epoch, iso = parse_frontmatter_timestamp("2024-06-14 12:33:36")
        assert epoch is not None
        assert iso is not None
        assert iso.startswith("2024-06-14T12:33:36")

    def test_parse_date_only_format(self):
        """Test parsing 'YYYY-MM-DD' format."""
        epoch, iso = parse_frontmatter_timestamp("2024-06-14")
        assert epoch is not None
        assert iso is not None
        assert iso.startswith("2024-06-14")

    def test_parse_iso_like_format(self):
        """Test parsing ISO-like format without timezone."""
        epoch, iso = parse_frontmatter_timestamp("2024-06-14T12:33:36")
        assert epoch is not None
        assert iso is not None

    def test_parse_invalid_format(self):
        """Test parsing invalid format returns None."""
        epoch, iso = parse_frontmatter_timestamp("invalid-date")
        assert epoch is None
        assert iso is None

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        epoch, iso = parse_frontmatter_timestamp("")
        assert epoch is None
        assert iso is None

    def test_parse_with_whitespace(self):
        """Test parsing with whitespace is handled."""
        epoch, iso = parse_frontmatter_timestamp("  2024-06-14 12:33:36  ")
        assert epoch is not None
        assert iso is not None


class TestGetGitTimestamp:
    """Tests for Git timestamp retrieval."""

    @patch("rag_engine.utils.git_utils.subprocess.run")
    def test_get_git_timestamp_success(self, mock_run, tmp_path):
        """Test successful Git timestamp retrieval."""
        file_path = tmp_path / "test.md"
        file_path.write_text("content")

        # Mock git rev-parse
        mock_rev_parse = MagicMock()
        mock_rev_parse.returncode = 0
        mock_rev_parse.stdout = str(tmp_path) + "\n"

        # Mock git log
        mock_log = MagicMock()
        mock_log.returncode = 0
        mock_log.stdout = "1718367216\n"

        mock_run.side_effect = [mock_rev_parse, mock_log]

        epoch, iso = get_git_timestamp(file_path)
        assert epoch == 1718367216
        assert iso is not None
        assert iso.startswith("2024-06-14")

    @patch("rag_engine.utils.git_utils.subprocess.run")
    def test_get_git_timestamp_not_in_repo(self, mock_run, tmp_path):
        """Test when file is not in a Git repository."""
        file_path = tmp_path / "test.md"
        file_path.write_text("content")

        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""

        epoch, iso = get_git_timestamp(file_path)
        assert epoch is None
        assert iso is None

    def test_get_git_timestamp_nonexistent_file(self, tmp_path):
        """Test with non-existent file."""
        file_path = tmp_path / "nonexistent.md"
        epoch, iso = get_git_timestamp(file_path)
        assert epoch is None
        assert iso is None

    @patch("rag_engine.utils.git_utils.subprocess.run")
    def test_get_git_timestamp_timeout(self, mock_run, tmp_path):
        """Test Git command timeout."""
        file_path = tmp_path / "test.md"
        file_path.write_text("content")

        mock_run.side_effect = subprocess.TimeoutExpired("git", 5)

        epoch, iso = get_git_timestamp(file_path)
        assert epoch is None
        assert iso is None


class TestGetFileTimestamp:
    """Tests for three-tier timestamp fallback."""

    def test_tier1_frontmatter(self, tmp_path):
        """Test Tier 1: Frontmatter updated field."""
        file_path = tmp_path / "test.md"
        file_path.write_text("content")

        base_meta = {"updated": "2024-06-14 12:33:36"}
        epoch, iso, source = get_file_timestamp(file_path, base_meta)

        assert epoch is not None
        assert iso is not None
        assert source == "frontmatter"

    @patch("rag_engine.utils.git_utils.get_git_timestamp")
    def test_tier2_git(self, mock_git, tmp_path):
        """Test Tier 2: Git commit timestamp (when frontmatter not present)."""
        file_path = tmp_path / "test.md"
        file_path.write_text("content")

        mock_git.return_value = (1718367216, "2024-06-14T12:33:36+00:00")

        base_meta = {}  # No frontmatter updated field
        epoch, iso, source = get_file_timestamp(file_path, base_meta)

        assert epoch == 1718367216
        assert iso is not None
        assert source == "git"

    @patch("rag_engine.utils.git_utils.get_git_timestamp")
    def test_tier3_file_stat(self, mock_git, tmp_path):
        """Test Tier 3: File modification date (when Git fails)."""
        file_path = tmp_path / "test.md"
        file_path.write_text("content")
        # Set file mtime to known value
        os.utime(file_path, (1718367216, 1718367216))

        mock_git.return_value = (None, None)  # Git fails

        base_meta = {}  # No frontmatter updated field
        epoch, iso, source = get_file_timestamp(file_path, base_meta)

        assert epoch == 1718367216
        assert iso is not None
        assert source == "file"

    @patch("rag_engine.utils.git_utils.get_git_timestamp")
    def test_fallback_chain(self, mock_git, tmp_path):
        """Test complete fallback chain when tiers fail."""
        file_path = tmp_path / "test.md"
        file_path.write_text("content")
        os.utime(file_path, (1718367216, 1718367216))

        # Test: frontmatter invalid → Git fails → file stat succeeds
        base_meta = {"updated": "invalid-date"}
        mock_git.return_value = (None, None)

        epoch, iso, source = get_file_timestamp(file_path, base_meta)

        assert epoch == 1718367216
        assert source == "file"

        # Test: no frontmatter → Git fails → file stat succeeds
        base_meta = {}
        epoch, iso, source = get_file_timestamp(file_path, base_meta)

        assert epoch == 1718367216
        assert source == "file"

    def test_no_source_file(self):
        """Test when source_file is None."""
        base_meta = {}
        epoch, iso, source = get_file_timestamp(None, base_meta)

        assert epoch is None
        assert iso is None
        assert source == "none"

    @patch("rag_engine.utils.git_utils.get_git_timestamp")
    def test_frontmatter_takes_priority(self, mock_git, tmp_path):
        """Test that frontmatter takes priority even if Git would succeed."""
        file_path = tmp_path / "test.md"
        file_path.write_text("content")

        mock_git.return_value = (9999999999, "future-date")

        base_meta = {"updated": "2024-06-14 12:33:36"}
        epoch, iso, source = get_file_timestamp(file_path, base_meta)

        assert source == "frontmatter"
        assert epoch != 9999999999  # Should use frontmatter, not Git

