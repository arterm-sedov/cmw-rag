import json
from typing import Any

from rag_engine.cmw_platform import config

import markdown2


def convert_markdown_to_html(md_text: str) -> str:
    """Convert Markdown to clean HTML for CMW Platform.

    CMW Platform ignores most HTML attributes and only renders basic tags.
    """
    if not md_text:
        return ""

    extras = [
        "tables",
        "fenced-code-blocks",
        "break-on-newline",
        "cuddled-lists",
        "strike",
        "code-friendly",
    ]

    return markdown2.markdown(md_text, extras=extras)


def convert_array_to_html_list(items: list | str, ordered: bool = False) -> str:
    """Convert array (list or JSON string) to HTML list.

    Args:
        items: List of items or JSON string
        ordered: If True, use <ol> (numbered), else <ul> (bullets)

    Returns:
        HTML list string, or plain text for empty/single item
    """
    # Handle None or empty
    if items is None:
        return ""

    if isinstance(items, str):
        # If it's a JSON array string like '["a", "b"]', parse it
        if items.startswith("["):
            try:
                items = json.loads(items)
            except (json.JSONDecodeError, TypeError):
                # Not valid JSON - check if it's "None" string
                if items.strip() in ("", "None", "none", "NONE"):
                    return ""
                return str(items)
        # If it's just a plain string like "None", handle it
        elif items.strip() in ("", "None", "none", "NONE"):
            return ""

    # After parsing, should be a list
    if not isinstance(items, list):
        return str(items) if items else ""

    if not items:
        return ""

    # Single item - return as paragraph (only if not "None")
    if len(items) == 1:
        item_str = str(items[0]).strip()
        if item_str in ("", "None", "none", "NONE"):
            return ""
        return f"<p>{item_str.replace('<', '&lt;').replace('>', '&gt;')}</p>"

    # Multiple items - return as HTML list
    # Filter out None or "None" string items
    valid_items = [str(item) for item in items if str(item).strip() not in ("", "None", "none", "NONE")]
    
    if not valid_items:
        return ""

    li_items = []
    for item in valid_items:
        # Escape HTML in item content
        item_str = item.replace("<", "&lt;").replace(">", "&gt;")
        li_items.append(f"<li>{item_str}</li>")

    if ordered:
        return f"<ol>{''.join(li_items)}</ol>"
    return f"<ul>{''.join(li_items)}</ul>"


def _format_articles_html(agent_result: Any) -> str:
    """Format final_articles as HTML table.

    Args:
        agent_result: The StructuredAgentResult from the agent

    Returns:
        HTML table string with columns: Rank, Title, Relevance, Normalized, URL
    """
    final_articles = getattr(agent_result, 'final_articles', None)
    if not final_articles:
        return ""

    # Header row using td with bold (th may be stripped by CMW)
    header = "<tr><td><b>Ранг</b></td><td><b>Название</b></td><td><b>Релевантность</b></td><td><b>Нормализованная</b></td><td><b>URL</b></td></tr>"

    rows = []
    for i, article in enumerate(final_articles):
        metadata = article.get('metadata', {})
        title = article.get('title', 'N/A')
        url = article.get('url', '')
        # Handle both 'rerank_score' and 'score' for compatibility
        relevance = metadata.get('rerank_score') or metadata.get('score', 0)
        normalized = metadata.get('normalized_rank', 0)

        # Format values
        relevance_str = f"{relevance:.2f}" if relevance else "0.00"
        normalized_str = f"{normalized:.3f}" if normalized else "0.000"

        rows.append(f"<tr><td>{i + 1}</td><td>{title}</td><td>{relevance_str}</td><td>{normalized_str}</td><td>{url}</td></tr>")

    html = f"<table>{header}{''.join(rows)}</table>"
    return html


def get_nested_value(obj: Any, path: str) -> Any:
    """Get a nested value from an object using dot notation.

    Args:
        obj: The object to traverse (dict, object, etc.)
        path: Dot-separated path (e.g., "plan.user_intent")

    Returns:
        The value at the path, or None if not found
    """
    if path is None:
        return None

    # Handle function calls like "len(plan.knowledge_base_search_queries)"
    if path.startswith("len(") and path.endswith(")"):
        inner_path = path[4:-1]
        value = get_nested_value(obj, inner_path)
        if value is None:
            return 0
        if isinstance(value, (list, dict, str)):
            return len(value)
        return 0

    # Handle special function mappings
    if path == "_format_articles_html":
        from rag_engine.cmw_platform.mapping import _format_articles_html
        return _format_articles_html(obj)

    if path == "_convert_markdown_to_html_answer":
        from rag_engine.cmw_platform.mapping import convert_markdown_to_html
        answer_text = getattr(obj, 'answer_text', '')
        return convert_markdown_to_html(answer_text)

    # Handle array to HTML conversion - use field name suffix
    # Must check for _ordered_html BEFORE _as_html since it's longer
    if path.endswith("_ordered_html"):
        # Extract field name - e.g., "plan.action_plan_ordered_html" -> "plan.action_plan"
        # Path is like "X.Y.Z_ordered_html" -> "X.Y.Z"
        field_name = path[:-len("_ordered_html")]
        from rag_engine.cmw_platform.mapping import convert_array_to_html_list
        value = get_nested_value(obj, field_name)
        return convert_array_to_html_list(value, ordered=True)

    if path.endswith("_as_html"):
        # Extract field name and get value from obj
        field_name = path[:-len("_as_html")]
        from rag_engine.cmw_platform.mapping import convert_array_to_html_list
        value = get_nested_value(obj, field_name)
        return convert_array_to_html_list(value, ordered=False)

    # Handle special cases
    if path == "_input_record_id":
        return obj

    # Regular dot notation traversal
    parts = path.split(".")
    current = obj

    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            # Try attribute access for objects
            current = getattr(current, part, None)

    return current


def extract_value(agent_result: Any, from_agent: str) -> Any:
    """Extract a value from agent result using the from_agent path.

    Args:
        agent_result: The StructuredAgentResult from the agent
        from_agent: The path to extract (e.g., "plan.user_intent", "len(final_articles)")

    Returns:
        The extracted value
    """
    return get_nested_value(agent_result, from_agent)


def serialize_value(value: Any, attr_type: str) -> Any:
    """Serialize a value based on attribute type.

    Args:
        value: The value to serialize
        attr_type: The target attribute type (json, string, boolean, etc.)

    Returns:
        Serialized value suitable for CMW Platform
    """
    if value is None:
        return None

    # Handle Enum types - extract the value
    if hasattr(value, 'value'):
        value = value.value

    # Handle empty strings - convert to None for certain types
    if isinstance(value, str) and not value.strip():
        return None

    # JSON types - serialize to JSON string
    if attr_type in ("json", "text"):
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    # Boolean
    if attr_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "y", "on")
        return bool(value)

    # Numeric types
    if attr_type in ("decimal", "integer"):
        try:
            if attr_type == "integer":
                return int(float(value))
            return float(value)
        except (ValueError, TypeError):
            return str(value)

    # Default: string
    return str(value) if value is not None else None


def map_agent_response(
    agent_result: Any,
    input_record_id: str,
    attributes: dict[str, Any],
) -> dict[str, Any]:
    """Map agent response to CMW Platform attributes.

    Args:
        agent_result: The StructuredAgentResult from the agent
        input_record_id: The ID of the input record (for link field)
        attributes: Dictionary of attribute configs with from_agent mappings

    Returns:
        Dictionary of mapped values ready for CMW Platform
    """
    result = {}

    for attr_name, attr_config in attributes.items():
        from_agent = attr_config.get("from_agent")
        attr_type = attr_config.get("type", "string")

        # Handle special link field
        if from_agent == "_input_record_id":
            result[attr_name] = input_record_id
            continue

        if from_agent is None:
            continue

        # Extract value from agent result
        value = extract_value(agent_result, from_agent)

        # Serialize based on type
        serialized = serialize_value(value, attr_type)

        if serialized is not None:
            result[attr_name] = serialized

    return result
