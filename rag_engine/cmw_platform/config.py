import os
from pathlib import Path
from typing import Any

import yaml

from rag_engine.cmw_platform.attribute_types import AttributeMetadata, coerce_value

DEFAULT_PLATFORM = os.getenv("CMW_PLATFORM_NAME", "primary")

_config_cache: dict[str, dict[str, Any]] = {}


def _get_config_path(platform: str | None = None) -> Path:
    """Get path to platform config YAML."""
    platform = platform or DEFAULT_PLATFORM
    config_dir = Path(__file__).parent.parent / "config"

    if platform == "primary":
        return config_dir / "cmw_platform.yaml"

    return config_dir / f"cmw_platform_{platform}.yaml"


def load_cmw_config(platform: str | None = None) -> dict[str, Any]:
    """Load CMW Platform configuration from YAML.

    Args:
        platform: Platform name (e.g., "primary", "secondary").
                 Defaults to CMW_PLATFORM_NAME env var or "primary".

    Returns:
        Full config dict.
    """
    platform = platform or DEFAULT_PLATFORM

    if platform not in _config_cache:
        config_path = _get_config_path(platform)

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            _config_cache[platform] = yaml.safe_load(f)

    return _config_cache[platform]


def load_pipeline_config(platform: str | None = None) -> dict[str, Any]:
    """Load pipeline configuration section from YAML."""
    cfg = load_cmw_config(platform)
    return cfg.get("pipeline", {})


def get_input_config(platform: str | None = None) -> dict[str, Any]:
    """Get input configuration (template to fetch from)."""
    pipeline = load_pipeline_config(platform)
    return pipeline.get("input", {})


def get_output_config(platform: str | None = None) -> dict[str, Any]:
    """Get output configuration (template to create in)."""
    pipeline = load_pipeline_config(platform)
    return pipeline.get("output", {})


def get_input_attributes(platform: str | None = None) -> dict[str, str]:
    """Get Python -> Platform attribute mapping from input config."""
    return get_input_config(platform).get("attributes", {})


def get_platform_attribute(python_name: str, platform: str | None = None) -> str | None:
    """Map Python name to platform attribute name."""
    attrs = get_input_attributes(platform)
    return attrs.get(python_name)


def get_python_attribute(platform_name: str, platform: str | None = None) -> str | None:
    """Map platform attribute name to Python name."""
    attrs = get_input_attributes(platform)
    for python_name, platform_name_val in attrs.items():
        if platform_name_val == platform_name:
            return python_name
    return None


def get_request_template(platform: str | None = None) -> str:
    """Get the markdown request template."""
    pipeline = load_pipeline_config(platform)
    return pipeline.get("request_template", "")


def get_template_config(
    app: str, template: str, platform: str | None = None
) -> dict[str, Any] | None:
    """Get configuration for a specific template."""
    cfg = load_cmw_config(platform)
    return cfg.get("templates", {}).get(app, {}).get(template)


def get_attribute_metadata(
    app: str, template: str, platform: str | None = None
) -> dict[str, AttributeMetadata]:
    """Get full attribute metadata for a template."""
    template_config = get_template_config(app, template, platform)
    if not template_config:
        return {}

    attrs = template_config.get("attributes", {})
    result = {}
    for alias, cfg in attrs.items():
        if isinstance(cfg, str):
            attr_type = cfg
        else:
            attr_type = cfg.get("type", "string") if cfg else "string"

        result[alias] = AttributeMetadata(
            alias=alias,
            type=attr_type,
            is_system=False,
            is_multivalue=False,
        )
    return result


def get_attribute_type(app: str, template: str, attribute: str, platform: str | None = None) -> str:
    """Get the type of an attribute."""
    metadata = get_attribute_metadata(app, template, platform)
    attr = metadata.get(attribute)
    return attr.type if attr else "string"


def coerce_attribute_value(
    app: str, template: str, attribute: str, value: Any, platform: str | None = None
) -> Any:
    """Coerce a value based on attribute metadata from config."""
    metadata = get_attribute_metadata(app, template, platform)
    attr = metadata.get(attribute)

    if not attr:
        return str(value) if value is not None else None

    result = coerce_value(attr.type, value, attr.is_multivalue, attribute)
    return result.value if result.success else value
