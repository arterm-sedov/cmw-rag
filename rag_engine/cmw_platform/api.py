import base64
import logging
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from rag_engine.cmw_platform.models import HTTPResponse, RequestConfig

logger = logging.getLogger(__name__)

DEFAULT_PLATFORM = os.getenv("CMW_PLATFORM_NAME", "primary")


def _load_env_file(platform: str | None = None) -> None:
    """Load .env file, optionally platform-specific.

    Args:
        platform: Platform name (e.g., "primary", "secondary").
                 If "secondary" and .env.secondary exists, loads it.
                 Otherwise loads default .env.
    """
    env_path = Path(__file__).parent.parent.parent / ".env"
    if platform and platform != "primary":
        platform_env_path = Path(__file__).parent.parent.parent / f".env.{platform}"
        if platform_env_path.exists():
            load_dotenv(platform_env_path)
            return
    load_dotenv(env_path)


def _load_server_config(platform: str | None = None) -> RequestConfig:
    """Load server configuration from .env file.

    Args:
        platform: Platform name (e.g., "primary", "secondary").
                 Defaults to "primary".

    Returns:
        RequestConfig with URL, login, password, timeout.
    """
    _load_env_file(platform)
    platform = platform or DEFAULT_PLATFORM

    # Build env var suffix: "" for primary, "2" for secondary
    suffix = "" if platform == "primary" else "2"
    base = f"CMW{suffix}_BASE_URL"
    login = f"CMW{suffix}_LOGIN"
    pw = f"CMW{suffix}_PASSWORD"
    timeout_var = f"CMW{suffix}_TIMEOUT"

    return RequestConfig(
        base_url=os.getenv(base, ""),
        login=os.getenv(login, ""),
        password=os.getenv(pw, ""),
        timeout=int(os.getenv(timeout_var, "30")),
    )


def _basic_headers(platform: str | None = None) -> dict[str, str]:
    """Create Basic Auth header with base64-encoded credentials."""
    config = _load_server_config(platform)
    credentials = f"{config.login}:{config.password}"
    encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {encoded}"}


def _get_request(endpoint: str, platform: str | None = None) -> dict[str, Any]:
    """Make GET request with Basic Auth header."""
    config = _load_server_config(platform)
    url = f"{config.base_url}{endpoint}"
    headers = _basic_headers(platform)

    try:
        response = requests.get(url, headers=headers, timeout=config.timeout)
        http_response = HTTPResponse(
            success=response.status_code == 200,
            status_code=response.status_code,
            raw_response=response.json() if response.content else None,
            base_url=url,
        )

        return {
            "success": http_response.success,
            "status_code": http_response.status_code,
            "data": http_response.raw_response,
            "error": http_response.error,
        }
    except requests.Timeout:
        logger.error(f"Request timeout: {url}")
        return {"success": False, "status_code": 408, "error": "Request timeout", "data": None}
    except requests.ConnectionError:
        logger.error(f"Connection error: {url}")
        return {"success": False, "status_code": 503, "error": "Connection error", "data": None}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"success": False, "status_code": 500, "error": str(e), "data": None}


def _post_request(body: dict[str, Any], endpoint: str, platform: str | None = None) -> dict[str, Any]:
    """Make POST request with Basic Auth header."""
    config = _load_server_config(platform)
    url = f"{config.base_url}{endpoint}"
    headers = _basic_headers(platform)

    try:
        response = requests.post(url, json=body, headers=headers, timeout=config.timeout)
        http_response = HTTPResponse(
            success=response.status_code == 200,
            status_code=response.status_code,
            raw_response=response.json() if response.content else None,
            base_url=url,
        )

        raw = http_response.raw_response
        extracted_error = None
        api_success = http_response.success

        if isinstance(raw, dict):
            api_success = raw.get("success", http_response.success)
            if not api_success:
                error_data = raw.get("error", {})
                if isinstance(error_data, dict):
                    extracted_error = error_data.get("message") or error_data.get("inner", {}).get("message")
                else:
                    extracted_error = str(error_data) if error_data else None

        return {
            "success": api_success,
            "status_code": http_response.status_code,
            "data": raw,
            "error": extracted_error,
        }
    except requests.Timeout:
        logger.error(f"Request timeout: {url}")
        return {"success": False, "status_code": 408, "error": "Request timeout", "data": None}
    except requests.ConnectionError:
        logger.error(f"Connection error: {url}")
        return {"success": False, "status_code": 503, "error": "Connection error", "data": None}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"success": False, "status_code": 500, "error": str(e), "data": None}


def _put_request(body: dict[str, Any], endpoint: str, platform: str | None = None) -> dict[str, Any]:
    """Make PUT request with Basic Auth header."""
    config = _load_server_config(platform)
    url = f"{config.base_url}{endpoint}"
    headers = _basic_headers(platform)

    try:
        response = requests.put(url, json=body, headers=headers, timeout=config.timeout)
        http_response = HTTPResponse(
            success=response.status_code == 200,
            status_code=response.status_code,
            raw_response=response.json() if response.content else None,
            base_url=url,
        )

        raw = http_response.raw_response
        extracted_error = None
        api_success = http_response.success

        if isinstance(raw, dict):
            api_success = raw.get("success", http_response.success)
            if not api_success:
                error_data = raw.get("error", {})
                if isinstance(error_data, dict):
                    extracted_error = error_data.get("message") or error_data.get("inner", {}).get("message")
                else:
                    extracted_error = str(error_data) if error_data else None

        return {
            "success": api_success,
            "status_code": http_response.status_code,
            "data": raw,
            "error": extracted_error,
        }
    except requests.Timeout:
        logger.error(f"Request timeout: {url}")
        return {"success": False, "status_code": 408, "error": "Request timeout", "data": None}
    except requests.ConnectionError:
        logger.error(f"Connection error: {url}")
        return {"success": False, "status_code": 503, "error": "Connection error", "data": None}
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"success": False, "status_code": 500, "error": str(e), "data": None}
