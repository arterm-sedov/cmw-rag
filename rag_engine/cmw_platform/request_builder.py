from typing import Any

from rag_engine.cmw_platform import config


def build_request(record_data: dict[str, Any]) -> str:
    """Build markdown request from record data using config template.

    Args:
        record_data: Dictionary of field values from CMW Platform record

    Returns:
        Markdown-formatted request string
    """
    template = config.get_request_template()

    if not template:
        # Fallback: simple concatenation
        title = record_data.get("title", "")
        question = record_data.get("question", "")
        return f"# {title}\n\n{question}"

    # Format template with record data
    # Handle missing fields gracefully
    formatted_data = {}
    for key in ["title", "question", "version", "browser"]:
        value = record_data.get(key, "")
        if value is None:
            value = ""
        formatted_data[key] = value

    return template.format(**formatted_data)
