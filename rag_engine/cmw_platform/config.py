from pathlib import Path
from typing import Any

import yaml

from rag_engine.cmw_platform.attribute_types import AttributeMetadata, coerce_value


def load_cmw_config() -> dict[str, Any]:
    """Load CMW Platform configuration from YAML."""
    config_path = Path(__file__).parent.parent / "config" / "cmw_platform.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pipeline_config() -> dict[str, Any]:
    """Load pipeline configuration section from YAML."""
    config = load_cmw_config()
    return config.get("pipeline", {})


def get_input_config() -> dict[str, Any]:
    """Get input configuration (template to fetch from)."""
    pipeline = load_pipeline_config()
    return pipeline.get("input", {})


def get_output_config() -> dict[str, Any]:
    """Get output configuration (template to create in)."""
    pipeline = load_pipeline_config()
    return pipeline.get("output", {})


def get_request_template() -> str:
    """Get the markdown request template."""
    pipeline = load_pipeline_config()
    return pipeline.get("request_template", "")


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
    for alias, cfg in attrs.items():
        # Handle both formats: 'title: string' or 'title: {type: string, from_agent: ...}'
        if isinstance(cfg, str):
            attr_type = cfg
            from_agent = None
        else:
            attr_type = cfg.get("type", "string") if cfg else "string"
            from_agent = cfg.get("from_agent")

        result[alias] = AttributeMetadata(
            alias=alias,
            type=attr_type,
            is_system=False,
            is_multivalue=False,
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

    result = coerce_value(attr.type, value, attr.is_multivalue, attribute)
    return result.value if result.success else value
