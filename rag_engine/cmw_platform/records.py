from typing import Any

from rag_engine.cmw_platform import api
from rag_engine.cmw_platform.attribute_types import to_api_alias
from rag_engine.cmw_platform.config import coerce_attribute_value, get_attribute_metadata


def create_record(
    application_alias: str,
    template_alias: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Create a new record in the CMW Platform.

    Args:
        application_alias: The application system name (e.g., "dima")
        template_alias: The template system name (e.g., "TPAIModel", "response")
        values: Dictionary of field names to values

    Returns:
        Dictionary with keys: success (bool), status_code (int),
        record_id (str|None), data (dict|list|None), error (str|None)
    """
    attr_metadata = get_attribute_metadata(application_alias, template_alias)

    coerced_values: dict[str, Any] = {}
    for key, val in values.items():
        if val is None:
            continue
        attr = attr_metadata.get(key)
        if attr and attr.is_system and key != "_color":
            continue
        coerced = coerce_attribute_value(application_alias, template_alias, key, val)
        if coerced is not None and coerced != "":
            # Convert key to API alias format (e.g. PascalCase -> camelCase)
            api_key = to_api_alias(key)
            coerced_values[api_key] = coerced

    # Build template global alias - handle both raw template name and full alias
    if template_alias.startswith("Template@"):
        template_global_alias = template_alias
    else:
        template_global_alias = f"Template@{application_alias}.{template_alias}"
    
    endpoint = f"/webapi/Record/{template_global_alias}"

    result = api._post_request(coerced_values, endpoint)

    record_id = None
    if result.get("success") and result.get("data"):
        data = result["data"]
        if isinstance(data, dict):
            record_id = data.get("response") or data.get("data") or data.get("recordId") or data.get("id")
        elif isinstance(data, str):
            record_id = data

    return {
        "success": result.get("success", False),
        "status_code": result.get("status_code", 0),
        "record_id": record_id,
        "data": result.get("data"),
        "error": result.get("error"),
    }


def update_record(
    record_id: str,
    values: dict[str, Any],
    application_alias: str = "",
    template_alias: str = "",
) -> dict[str, Any]:
    """Update an existing record in the CMW Platform using PUT.

    Args:
        record_id: The record UUID
        values: Dictionary of field names to values
        application_alias: Optional application system name for type coercion
        template_alias: Optional template system name for type coercion

    Returns:
        Dictionary with keys: success (bool), status_code (int), data, error
    """
    processed_values: dict[str, Any] = {}
    
    for key, val in values.items():
        if val is None:
            continue
            
        # Use coercion if metadata is available
        if application_alias and template_alias:
            coerced = coerce_attribute_value(application_alias, template_alias, key, val)
        else:
            coerced = val
            
        if coerced is not None:
            # Convert key to API alias format (e.g. PascalCase -> camelCase)
            api_key = to_api_alias(key)
            processed_values[api_key] = coerced

    endpoint = f"/webapi/Record/{record_id}"
    result = api._put_request(processed_values, endpoint)

    return result


def read_record(
    record_id: str,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Read a record from the CMW Platform with server-side field filtering.

    Uses GetPropertyValues endpoint for efficient field filtering.

    Args:
        record_id: The record UUID
        fields: List of field aliases to retrieve. If None, returns all fields.

    Returns:
        Dictionary with keys: success (bool), status_code (int),
        data (dict|list|None), error (str|None)
    """
    if fields is None:
        fields = []

    endpoint = "/api/public/system/TeamNetwork/ObjectService/GetPropertyValues"
    body = {
        "objects": [record_id],
        "propertiesByAlias": fields,
    }

    result = api._post_request(body, endpoint)

    if result.get("success") and result.get("data"):
        data = result["data"]
        if isinstance(data, dict):
            filtered_data = {record_id: data.get(record_id, {})}
            return {
                "success": True,
                "status_code": result.get("status_code", 200),
                "data": filtered_data,
                "error": None,
            }

    return {
        "success": result.get("success", False),
        "status_code": result.get("status_code", 0),
        "data": result.get("data"),
        "error": result.get("error"),
    }
