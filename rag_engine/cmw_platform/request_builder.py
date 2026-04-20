from typing import Any

from markdownify import markdownify as html_to_markdown

from rag_engine.cmw_platform import config


def build_request(record_data: dict[str, Any]) -> str:
    """Build markdown request from record data using config template.

    Args:
        record_data: Dictionary of field values from CMW Platform record

    Returns:
        Markdown-formatted request string
    """
    template = config.get_request_template()

    # Get attribute mapping from config
    attr_mapping = config.get_input_attributes()
    if not attr_mapping:
        return ""

    # Get attribute metadata for type checking
    attr_metadata = config.get_attribute_metadata("systemSolution", "Requests")

    # Format template with record data
    formatted_data = {}
    for python_name, platform_name in attr_mapping.items():
        value = record_data.get(platform_name, "")
        if value is None:
            value = ""
        # Convert HTML to markdown for 'text' type attributes
        attr = attr_metadata.get(python_name)
        if attr and attr.type == "text" and value:
            value = html_to_markdown(value)
        formatted_data[python_name] = value

    return template.format(**formatted_data)
