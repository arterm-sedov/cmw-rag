"""CMW Platform Document API.

Provides functions to fetch document content from CMW Platform.
"""

from typing import Any

from rag_engine.cmw_platform import api

DEFAULT_PLATFORM = "primary"


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
    endpoint = f"/webapi/Document/{document_id}/Content"

    response = api._get_request(endpoint, platform=platform)

    if not response.get("success"):
        return {
            "success": False,
            "error": response.get("error", "Failed to fetch document"),
        }

    raw = response.get("data", {})
    if not raw:
        return {
            "success": False,
            "error": "Empty response from document API",
        }

    return {
        "success": True,
        "content": raw.get("content"),
        "mime_type": raw.get("mimeType") or raw.get("contentType"),
        "filename": raw.get("fileName"),
    }
