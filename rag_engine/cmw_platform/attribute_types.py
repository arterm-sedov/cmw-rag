import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel


def to_api_alias(alias: str) -> str:
    """Convert attribute alias to API format.
    
    CMW Platform API expects FirstCapital to be firstLowerCase.
    Everything else (snake_case, already lowerCase) passes through as-is.
    
    Examples:
        RequestsIssueArea -> requestsIssueArea
        requests_issue_area -> requests_issue_area
        topic -> topic
    """
    if not alias:
        return alias
    # If first letter is uppercase and rest is lowercase, convert to lowerFirst
    if alias[0].isupper() and len(alias) > 1 and alias[1].islower():
        return alias[0].lower() + alias[1:]
    return alias


class AttributeMetadata(BaseModel):
    """Metadata for a template attribute."""

    alias: str
    type: str = "string"
    is_system: bool = False
    is_multivalue: bool = False


class CoercionResult(BaseModel):
    """Result of value coercion."""

    value: Any
    success: bool = True
    error: str | None = None


def coerce_string(value: Any) -> CoercionResult:
    """Coerce to string."""
    if value is None or value == "":
        return CoercionResult(value=value)
    return CoercionResult(value=str(value))


def coerce_enum(value: Any, attribute_alias: str = "") -> CoercionResult:
    """Coerce to enum value.
    
    CMW Platform expects enum values in the format:
    {
        "alias": {
            "type": "Variant",
            "owner": "<attribute_alias>",
            "alias": "<enum_value_system_name>"
        }
    }
    """
    if value is None or value == "":
        return CoercionResult(value=None)
    
    # Convert to string if not already
    enum_value = str(value) if value is not None else ""
    
    # Build the enum alias structure
    enum_alias = {
        "type": "Variant",
        "owner": attribute_alias,
        "alias": enum_value,
    }
    
    return CoercionResult(value={"alias": enum_alias})


def coerce_boolean(value: Any) -> CoercionResult:
    """Coerce to boolean."""
    if isinstance(value, bool):
        return CoercionResult(value=value)
    if value is None or value == "":
        return CoercionResult(value="")
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return CoercionResult(value=True)
    if s in ("false", "0", "no", "n", "off"):
        return CoercionResult(value=False)
    return CoercionResult(value="", success=False, error=f"Cannot coerce '{value}' to boolean")


def coerce_datetime(value: Any) -> CoercionResult:
    """Coerce to datetime - pass through ISO string."""
    if value is None or value == "":
        return CoercionResult(value=value)
    if isinstance(value, datetime):
        return CoercionResult(value=value.isoformat())
    return CoercionResult(value=str(value))


def coerce_decimal(value: Any) -> CoercionResult:
    """Coerce to decimal."""
    if value is None or value == "":
        return CoercionResult(value=value)
    if isinstance(value, (int, float, Decimal)):
        return CoercionResult(value=value)
    try:
        return CoercionResult(value=Decimal(str(value)))
    except (InvalidOperation, ValueError):
        return CoercionResult(value="", success=False, error=f"Cannot coerce '{value}' to decimal")


def coerce_integer(value: Any) -> CoercionResult:
    """Coerce to integer."""
    if value is None or value == "":
        return CoercionResult(value=value)
    if isinstance(value, int):
        return CoercionResult(value=value)
    if isinstance(value, float):
        if value.is_integer():
            return CoercionResult(value=int(value))
        return CoercionResult(value="", success=False, error=f"Cannot coerce '{value}' to integer")
    try:
        return CoercionResult(value=int(float(value)))
    except (ValueError, TypeError):
        return CoercionResult(value="", success=False, error=f"Cannot coerce '{value}' to integer")


def coerce_record(value: Any) -> CoercionResult:
    """Coerce record reference - expects record ID."""
    if value is None or value == "":
        return CoercionResult(value=value)
    if isinstance(value, dict):
        # If dict with id field, extract it
        if "id" in value:
            return CoercionResult(value=str(value["id"]))
        if "value" in value:
            return CoercionResult(value=str(value["value"]))
        return CoercionResult(value="", success=False, error="Record reference dict must have 'id' or 'value'")
    return CoercionResult(value=str(value))


# Platform attribute type coercion map
# These are immutable platform types - defined in code, not config
ATTRIBUTE_TYPE_COERCERS: dict[str, callable] = {
    "string": coerce_string,
    "text": coerce_string,
    "document": coerce_string,
    "image": coerce_string,
    "drawing": coerce_string,
    "record": coerce_record,
    "role": coerce_string,
    "account": coerce_string,
    "enum": coerce_enum,
    "boolean": coerce_boolean,
    "datetime": coerce_datetime,
    "decimal": coerce_decimal,
    "integer": coerce_integer,
}


def coerce_value(attr_type: str, value: Any, is_multivalue: bool = False, attribute_alias: str = "") -> CoercionResult:
    """Coerce value to the correct type based on attribute type.

    Args:
        attr_type: The platform attribute type (string, boolean, record, etc.)
        value: The value to coerce
        is_multivalue: Whether this is a multi-value attribute
        attribute_alias: The attribute alias (needed for enum coercion)

    Returns:
        CoercionResult with coerced value
    """
    if value is None:
        return CoercionResult(value=None)

    # Get the coercer for this type, default to string
    coercer = ATTRIBUTE_TYPE_COERCERS.get(attr_type.lower(), coerce_string)

    if is_multivalue:
        # Multi-value: coerce each item in the list
        value_list = value if isinstance(value, list) else [value]
        coerced_list = []
        for item in value_list:
            # Pass attribute_alias to coerce_enum
            if attr_type.lower() == "enum":
                result = coercer(item, attribute_alias)
            else:
                result = coercer(item)
            if not result.success:
                return result
            coerced_list.append(result.value)
        return CoercionResult(value=coerced_list)

    # Single value
    # Pass attribute_alias to coerce_enum
    if attr_type.lower() == "enum":
        return coercer(value, attribute_alias)
    return coercer(value)
