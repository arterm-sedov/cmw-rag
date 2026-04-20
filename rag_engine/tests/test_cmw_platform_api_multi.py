"""Tests for multi-platform API (Phase 2)."""

import os
from unittest.mock import patch

import pytest


class TestLoadEnvFile:
    """Test _load_env_file loads platform-specific .env files."""

    def test_load_env_file_loads_primary_from_default_env(self):
        """Primary platform uses default .env file."""
        from rag_engine.cmw_platform.api import _load_env_file

        with patch("rag_engine.cmw_platform.api.load_dotenv") as mock_load:
            _load_env_file("primary")
            mock_load.assert_called()
            call_args = mock_load.call_args[0]
            assert call_args[0].name == ".env"

    def test_load_env_file_loads_secondary_from_env_secondary(self):
        """Secondary platform uses .env.secondary if present."""
        from rag_engine.cmw_platform.api import _load_env_file

        with patch("rag_engine.cmw_platform.api.Path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("rag_engine.cmw_platform.api.load_dotenv") as mock_load:
                _load_env_file("secondary")
                mock_load.assert_called_once()
                call_args = mock_load.call_args[0]
                assert ".env.secondary" in str(call_args[0])

    def test_load_env_file_falls_back_to_default_for_secondary_if_no_file(self):
        """Secondary falls back to default .env if .env.secondary doesn't exist."""
        from rag_engine.cmw_platform.api import _load_env_file

        with patch("rag_engine.cmw_platform.api.Path.exists") as mock_exists:
            mock_exists.return_value = False
            with patch("rag_engine.cmw_platform.api.load_dotenv") as mock_load:
                _load_env_file("secondary")
                mock_load.assert_called()
                call_args = mock_load.call_args[0]
                assert call_args[0].name == ".env"


class TestLoadServerConfig:
    """Test _load_server_config loads platform-specific credentials."""

    def test_load_server_config_primary_uses_cmw_vars(self):
        """Primary platform uses CMW_* env vars."""
        from rag_engine.cmw_platform.api import _load_server_config

        with patch.dict(
            os.environ,
            {"CMW_BASE_URL": "http://primary.local/", "CMW_LOGIN": "user1", "CMW_PASSWORD": "pass1"},
        ):
            config = _load_server_config("primary")
            assert config.base_url == "http://primary.local"
            assert config.login == "user1"
            assert config.password == "pass1"

    def test_load_server_config_secondary_uses_cmw2_vars(self):
        """Secondary platform uses CMW2_* env vars."""
        from rag_engine.cmw_platform.api import _load_server_config

        with patch.dict(
            os.environ,
            {"CMW2_BASE_URL": "http://secondary.local/", "CMW2_LOGIN": "user2", "CMW2_PASSWORD": "pass2"},
        ):
            config = _load_server_config("secondary")
            assert config.base_url == "http://secondary.local"
            assert config.login == "user2"
            assert config.password == "pass2"

    def test_load_server_config_default_is_primary(self):
        """Default platform (None) is primary."""
        from rag_engine.cmw_platform.api import _load_server_config

        with patch.dict(
            os.environ,
            {"CMW_BASE_URL": "http://default.local/", "CMW_LOGIN": "default", "CMW_PASSWORD": "default"},
        ):
            config = _load_server_config()
            assert config.base_url == "http://default.local"

    def test_load_server_config_timeout_from_env(self):
        """Timeout is loaded from CMW_TIMEOUT / CMW2_TIMEOUT."""
        from rag_engine.cmw_platform.api import _load_server_config

        with patch.dict(os.environ, {"CMW_TIMEOUT": "60"}):
            config = _load_server_config("primary")
            assert config.timeout == 60


class TestBasicHeaders:
    """Test _basic_headers returns platform-specific auth headers."""

    def test_basic_headers_differs_between_platforms(self):
        """Primary and secondary produce different headers (different credentials)."""
        from rag_engine.cmw_platform.api import _basic_headers

        with patch.dict(
            os.environ,
            {
                "CMW_BASE_URL": "http://primary.local/",
                "CMW_LOGIN": "user1",
                "CMW_PASSWORD": "pass1",
                "CMW2_BASE_URL": "http://secondary.local/",
                "CMW2_LOGIN": "user2",
                "CMW2_PASSWORD": "pass2",
            },
        ):
            primary_headers = _basic_headers("primary")
            secondary_headers = _basic_headers("secondary")
            assert primary_headers != secondary_headers
