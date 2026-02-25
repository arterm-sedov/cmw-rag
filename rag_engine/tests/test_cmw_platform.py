from __future__ import annotations

from unittest.mock import Mock, patch

import pytest


class TestLoadServerConfig:
    """Test server configuration loading from environment."""

    def test_load_server_config_returns_config(self, monkeypatch):
        """Test returns valid config from environment variables."""
        monkeypatch.setenv("CMW_BASE_URL", "https://test.comindware.com")
        monkeypatch.setenv("CMW_LOGIN", "test_user")
        monkeypatch.setenv("CMW_PASSWORD", "test_pass")
        monkeypatch.setenv("CMW_TIMEOUT", "45")

        from rag_engine.cmw_platform.api import _load_server_config

        config = _load_server_config()

        assert config.base_url == "https://test.comindware.com"
        assert config.login == "test_user"
        assert config.password == "test_pass"
        assert config.timeout == 45

    def test_load_server_config_default_timeout(self, monkeypatch):
        """Test default timeout is applied."""
        monkeypatch.setenv("CMW_BASE_URL", "https://test.comindware.com")
        monkeypatch.setenv("CMW_LOGIN", "test_user")
        monkeypatch.setenv("CMW_PASSWORD", "test_pass")
        monkeypatch.delenv("CMW_TIMEOUT", raising=False)

        from rag_engine.cmw_platform.api import _load_server_config

        config = _load_server_config()

        assert config.timeout == 30


class TestBasicHeaders:
    """Test Basic Authentication header generation."""

    def test_basic_headers_creates_valid_auth(self, monkeypatch):
        """Test header contains valid base64 encoded credentials."""
        import base64

        monkeypatch.setenv("CMW_BASE_URL", "https://test.comindware.com")
        monkeypatch.setenv("CMW_LOGIN", "test_user")
        monkeypatch.setenv("CMW_PASSWORD", "test_pass")

        from rag_engine.cmw_platform.api import _basic_headers

        headers = _basic_headers()
        auth_header = headers.get("Authorization", "")

        assert auth_header.startswith("Basic ")
        encoded = auth_header[6:]
        decoded = base64.b64decode(encoded).decode("utf-8")
        assert decoded == "test_user:test_pass"


class TestAPIResponse:
    """Test API response structures."""

    def test_api_response_success_structure(self):
        """Test success response has expected keys."""
        from rag_engine.cmw_platform.models import APIResponse

        response = APIResponse(response={"data": "test"}, success=True)

        assert response.success is True
        assert response.response == {"data": "test"}
        assert response.error is None

    def test_api_response_error_structure(self):
        """Test error response has expected keys."""
        from rag_engine.cmw_platform.models import APIResponse

        response = APIResponse(response=None, success=False, error="Not found")

        assert response.success is False
        assert response.response is None
        assert response.error == "Not found"


class TestRequestConfig:
    """Test RequestConfig model."""

    def test_request_config_strips_trailing_slash(self):
        """Test base_url trailing slash is stripped."""
        from rag_engine.cmw_platform.models import RequestConfig

        config = RequestConfig(
            base_url="https://test.comindware.com/",
            login="user",
            password="pass",
            timeout=30,
        )

        assert config.base_url == "https://test.comindware.com"

    def test_request_config_default_timeout(self):
        """Test default timeout is applied."""
        from rag_engine.cmw_platform.models import RequestConfig

        config = RequestConfig(
            base_url="https://test.comindware.com",
            login="user",
            password="pass",
        )

        assert config.timeout == 30

    def test_request_config_empty_base_url_fails(self):
        """Test empty base_url raises validation error."""
        from rag_engine.cmw_platform.models import RequestConfig

        with pytest.raises(ValueError, match="base_url"):
            RequestConfig(base_url="", login="user", password="pass")


class TestReadRecord:
    """Test read_record function."""

    @patch("rag_engine.cmw_platform.api.requests.post")
    def test_get_property_values_request_format(self, mock_post, monkeypatch):
        """Test request body has correct structure."""
        monkeypatch.setenv("CMW_BASE_URL", "https://test.comindware.com")
        monkeypatch.setenv("CMW_LOGIN", "test_user")
        monkeypatch.setenv("CMW_PASSWORD", "test_pass")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "record-uuid-123": {"user_question": "test?", "title": "Test"}
        }
        mock_response.content = b"{}"
        mock_post.return_value = mock_response

        from rag_engine.cmw_platform.records import read_record

        read_record("record-uuid-123", fields=["user_question", "title"])

        call_args = mock_post.call_args
        json_body = call_args.kwargs.get("json") or call_args[1].get("json")

        assert "objects" in json_body
        assert "record-uuid-123" in json_body["objects"]
        assert "propertiesByAlias" in json_body
        assert "user_question" in json_body["propertiesByAlias"]
        assert "title" in json_body["propertiesByAlias"]

    @patch("rag_engine.cmw_platform.api.requests.post")
    def test_read_record_filters_fields(self, mock_post, monkeypatch):
        """Test correctly filters response to requested fields."""
        monkeypatch.setenv("CMW_BASE_URL", "https://test.comindware.com")
        monkeypatch.setenv("CMW_LOGIN", "test_user")
        monkeypatch.setenv("CMW_PASSWORD", "test_pass")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "record-uuid-123": {"user_question": "test?", "title": "Test"}
        }
        mock_response.content = b"{}"
        mock_post.return_value = mock_response

        from rag_engine.cmw_platform.records import read_record

        result = read_record("record-uuid-123", fields=["user_question", "title"])

        assert result["success"] is True
        assert "record-uuid-123" in result["data"]
        assert "user_question" in result["data"]["record-uuid-123"]
        assert "title" in result["data"]["record-uuid-123"]


class TestCreateRecord:
    """Test create_record function."""

    @patch("rag_engine.cmw_platform.api.requests.post")
    def test_create_record_request_format(self, mock_post, monkeypatch):
        """Test request body contains values with correct structure."""
        monkeypatch.setenv("CMW_BASE_URL", "https://test.comindware.com")
        monkeypatch.setenv("CMW_LOGIN", "test_user")
        monkeypatch.setenv("CMW_PASSWORD", "test_pass")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": "new-record-id"}
        mock_response.content = b"{}"
        mock_post.return_value = mock_response

        from rag_engine.cmw_platform.records import create_record

        create_record(
            application_alias="support_app",
            template_alias="resolution_output",
            values={"summary": "AI summary", "confidence": 0.95},
        )

        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url")
        json_body = call_args.kwargs.get("json") or call_args[1].get("json")

        assert "resolution_output" in url
        assert "Template@support_app.resolution_output" in url
        assert "summary" in json_body
        assert json_body["summary"] == "AI summary"

    @patch("rag_engine.cmw_platform.api.requests.post")
    def test_create_record_success(self, mock_post, monkeypatch):
        """Test successful record creation returns record_id."""
        monkeypatch.setenv("CMW_BASE_URL", "https://test.comindware.com")
        monkeypatch.setenv("CMW_LOGIN", "test_user")
        monkeypatch.setenv("CMW_PASSWORD", "test_pass")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": "new-record-id"}
        mock_response.content = b"{}"
        mock_post.return_value = mock_response

        from rag_engine.cmw_platform.records import create_record

        result = create_record(
            application_alias="support_app",
            template_alias="resolution_output",
            values={"summary": "AI summary"},
        )

        assert result["success"] is True
        assert result["record_id"] == "new-record-id"

    @patch("rag_engine.cmw_platform.api.requests.post")
    def test_create_record_api_error(self, mock_post, monkeypatch):
        """Test API error handling."""
        monkeypatch.setenv("CMW_BASE_URL", "https://test.comindware.com")
        monkeypatch.setenv("CMW_LOGIN", "test_user")
        monkeypatch.setenv("CMW_PASSWORD", "test_pass")

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"success": False, "error": "Validation failed"}
        mock_response.content = b"{}"
        mock_post.return_value = mock_response

        from rag_engine.cmw_platform.records import create_record

        result = create_record(
            application_alias="support_app",
            template_alias="resolution_output",
            values={"summary": "AI summary"},
        )

        assert result["success"] is False
        assert result["error"] == "Validation failed"

    @patch("rag_engine.cmw_platform.api.requests.post")
    def test_create_record_timeout_error(self, mock_post, monkeypatch):
        """Test timeout error handling."""
        import requests

        monkeypatch.setenv("CMW_BASE_URL", "https://test.comindware.com")
        monkeypatch.setenv("CMW_LOGIN", "test_user")
        monkeypatch.setenv("CMW_PASSWORD", "test_pass")
        mock_post.side_effect = requests.Timeout()

        from rag_engine.cmw_platform.records import create_record

        result = create_record(
            application_alias="support_app",
            template_alias="resolution_output",
            values={"summary": "AI summary"},
        )

        assert result["success"] is False
        assert result["status_code"] == 408
        assert "timeout" in result["error"].lower()
