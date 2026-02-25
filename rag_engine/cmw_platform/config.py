from pathlib import Path
from typing import Any

import yaml

from rag_engine.cmw_platform.attribute_types import AttributeMetadata, coerce_value


def load_cmw_config() -> dict[str, Any]:
    """Load CMW Platform configuration from YAML."""
    config_path = Path(__file__).parent.parent / "config" / "cmw_platform.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_template_config(app: str, template: str) -> dict[str, Any] | None:
    """Get configuration for a specific template."""
    config = load_cmw_config()
    return config.get("templates", {}).get(app, {}).get(template)


def get_attribute_metadata(app: str, template: str) -> dict[str, AttributeMetadata]:
    """Get full attribute metadata for a template.

    Returns a dictionary mapping attribute aliases to AttributeMetadata.
    """
    template_config = get_template_config(app, template)
    if not template_config:
        return {}

    attrs = template_config.get("attributes", {})
    result = {}
    for alias, config in attrs.items():
        result[alias] = AttributeMetadata(
            alias=alias,
            type=config.get("type", "string"),
            is_system=config.get("system", False),
            is_multivalue=config.get("multivalue", False),
        )
    return result


def get_attribute_type(app: str, template: str, attribute: str) -> str:
    """Get the type of an attribute."""
    metadata = get_attribute_metadata(app, template)
    attr = metadata.get(attribute)
    return attr.type if attr else "string"


def coerce_attribute_value(
    app: str, template: str, attribute: str, value: Any
) -> Any:
    """Coerce a value based on attribute metadata from config.

    Args:
        app: Application system name
        template: Template system name
        attribute: Attribute alias
        value: Value to coerce

    Returns:
        Coerced value
    """
    metadata = get_attribute_metadata(app, template)
    attr = metadata.get(attribute)

    if not attr:
        return str(value) if value is not None else None

    result = coerce_value(attr.type, value, attr.is_multivalue)
    return result.value if result.success else value
