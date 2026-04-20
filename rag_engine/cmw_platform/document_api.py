"""CMW Platform Document API.

Provides functions to fetch document content from CMW Platform.
"""

import base64
import logging
from typing import Any

from rag_engine.cmw_platform import api

logger = logging.getLogger(__name__)
DEFAULT_PLATFORM = "primary"


def _get_document_raw(document_id: str, platform: str | None = None) -> dict[str, Any]:
    """Fetch document as raw bytes (for Lukoil-style binary responses)."""
    import requests

    config = api._load_server_config(platform)
    base = config.base_url.rstrip("/")
    url = f"{base}/webapi/Document/{document_id}/Content"

    try:
        response = requests.get(
            url,
            headers=api._basic_headers(platform),
            timeout=config.timeout,
        )
        if response.status_code == 200:
            return {
                "success": True,
                "content": response.content,
                "status_code": 200,
            }
        return {
            "success": False,
            "error": f"HTTP {response.status_code}",
            "status_code": response.status_code,
        }
    except requests.Timeout:
        return {"success": False, "error": "Request timeout", "status_code": 408}
    except Exception as e:
        logger.error(f"Document fetch failed: {e}")
        return {"success": False, "error": str(e), "status_code": 500}


def get_document_content(document_id: str, platform: str | None = None) -> dict[str, Any]:
    """Fetch document content from CMW Platform.

    Step 1: GET /webapi/Document/{documentId}/Content
    Step 2: Returns {"success": bool, "content": base64_string, "mime_type": str, "filename": str}

    Args:
        document_id: Document ID from document attribute value
        platform: Platform name (default: "primary")

    Returns:
        Dict with keys:
            - success: bool
            - content: base64-encoded content (if success)
            - mime_type: MIME type string (if success)
            - filename: original filename (if success)
            - error: error message (if not success)
    """
    platform = platform or DEFAULT_PLATFORM

    # Try JSON response first (standard CMW)
    response = api._get_request(f"/webapi/Document/{document_id}/Content", platform=platform)

    if response.get("success"):
        raw = response.get("data", {})
        if isinstance(raw, dict) and raw.get("content"):
            return {
                "success": True,
                "content": raw.get("content"),
                "mime_type": raw.get("mimeType") or raw.get("contentType"),
                "filename": raw.get("fileName"),
            }

    # Fall back to raw binary (Lukoil-style)
    raw_response = _get_document_raw(document_id, platform)

    if not raw_response.get("success"):
        return {
            "success": False,
            "error": raw_response.get("error", "Failed to fetch document"),
        }

    content = base64.b64encode(raw_response["content"]).decode("utf-8")
    return {
        "success": True,
        "content": content,
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "filename": f"{document_id}.docx",
    }
