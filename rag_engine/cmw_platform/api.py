import base64
import logging
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from rag_engine.cmw_platform.models import HTTPResponse, RequestConfig

logger = logging.getLogger(__name__)


def _load_server_config() -> RequestConfig:
    """Load server configuration from .env file."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)

    return RequestConfig(
        base_url=os.getenv("CMW_BASE_URL", ""),
        login=os.getenv("CMW_LOGIN", ""),
        password=os.getenv("CMW_PASSWORD", ""),
        timeout=int(os.getenv("CMW_TIMEOUT", "30")),
    )


def _basic_headers() -> dict[str, str]:
    """Create Basic Auth header with base64-encoded credentials."""
    config = _load_server_config()
    credentials = f"{config.login}:{config.password}"
    encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {encoded}"}


def _get_request(endpoint: str) -> dict[str, Any]:
    """Make GET request with Basic Auth header."""
    config = _load_server_config()
    url = f"{config.base_url}{endpoint}"
    headers = _basic_headers()

    try:
        response = requests.get(
            url, headers=headers, timeout=config.timeout
        )
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


def _post_request(body: dict[str, Any], endpoint: str) -> dict[str, Any]:
    """Make POST request with Basic Auth header."""
    config = _load_server_config()
    url = f"{config.base_url}{endpoint}"
    headers = _basic_headers()

    try:
        response = requests.post(
            url, json=body, headers=headers, timeout=config.timeout
        )
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


def _put_request(body: dict[str, Any], endpoint: str) -> dict[str, Any]:
    """Make PUT request with Basic Auth header."""
    config = _load_server_config()
    url = f"{config.base_url}{endpoint}"
    headers = _basic_headers()

    try:
        response = requests.put(
            url, json=body, headers=headers, timeout=config.timeout
        )
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
