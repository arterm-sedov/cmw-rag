from typing import Any

from markdownify import markdownify as html_to_markdown

from rag_engine.cmw_platform import config
from rag_engine.cmw_platform.attribute_types import to_api_alias


def build_request(record_data: dict[str, Any], platform: str | None = None) -> str:
    """Build markdown request from record data using config template.

    Args:
        record_data: Dictionary of field values from CMW Platform record
        platform: Platform name (e.g., "primary", "secondary")

    Returns:
        Markdown-formatted request string
    """
    template = config.get_request_template(platform)
    attr_mapping = config.get_input_attributes(platform)
    if not attr_mapping:
        return ""

    input_cfg = config.get_input_config(platform)
    attr_metadata = config.get_attribute_metadata(
        input_cfg.get("application", ""), input_cfg.get("template", ""), platform,
    )

    formatted_data = {}
    for python_name, platform_name in attr_mapping.items():
        value = record_data.get(to_api_alias(platform_name), "")
        if value is None:
            value = ""
        attr = attr_metadata.get(python_name)
        if attr and attr.type == "text" and value:
            value = html_to_markdown(value)
        formatted_data[python_name] = value

    return template.format(**formatted_data)
