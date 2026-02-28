"""Dynamic category enum loader from YAML config."""
from enum import Enum
from typing import Any

from rag_engine.cmw_platform import config as cmw_config


def _sanitize_enum_name(code: str) -> str:
    """Create valid enum member name from code.
    
    Args:
        code: The category code (e.g., 'documentation', 'migration_4.2_4.7._5.0.')
    
    Returns:
        Valid enum member name (uppercase, no special chars)
    """
    # Convert to uppercase and replace special chars
    name = code.upper()
    name = name.replace("-", "_")
    name = name.replace(".", "_")
    name = name.replace(" ", "_")
    name = name.replace(":", "_")
    name = name.replace("(", "_")
    name = name.replace(")", "_")
    
    # If starts with digit or underscore, prefix with underscore is handled by Enum
    # Just ensure it's valid
    return name


def load_category_enum() -> type[Enum]:
    """Load category enum from cmw_platform.yaml config.
    
    Returns:
        Enum class with codes as values and descriptions
    """
    yaml_config = cmw_config.load_cmw_config()
    category_enum_config = yaml_config.get("category_enum", {})
    
    if not category_enum_config:
        return _build_fallback_enum()
    
    # Build enum members from config - use member names and values
    # Format: {member_name: value, ...}
    enum_members = {}
    
    for code, _ in category_enum_config.items():
        member_name = _sanitize_enum_name(code)
        # Ensure unique member names
        if member_name in enum_members:
            # Add numeric suffix
            base_name = member_name
            counter = 1
            while member_name in enum_members:
                member_name = f"{base_name}_{counter}"
                counter += 1
        
        # Use value as-is (the code)
        enum_members[member_name] = code
    
    if not enum_members:
        return _build_fallback_enum()
    
    # Sort by member name for consistency
    sorted_members = {k: enum_members[k] for k in sorted(enum_members.keys())}
    
    return Enum("SGRCategory", sorted_members)


def _build_fallback_enum() -> type[Enum]:
    """Build fallback enum for when config is not available."""
    return Enum("SGRCategory", {"OTHER": "other"})


def get_category_description(code: str) -> str | None:
    """Get the description for a category code from config.
    
    Args:
        code: The category code
    
    Returns:
        Description string or None
    """
    yaml_config = cmw_config.load_cmw_config()
    category_enum_config = yaml_config.get("category_enum", {})
    return category_enum_config.get(code)


def get_category_choices_with_descriptions() -> str:
    """Get category choices formatted with descriptions for LLM.
    
    Returns:
        Formatted string with code: description for each category
    """
    yaml_config = cmw_config.load_cmw_config()
    category_enum_config = yaml_config.get("category_enum", {})
    
    if not category_enum_config:
        return "  - other: Other"
    
    lines = []
    # Sort by code for consistent output
    for code in sorted(category_enum_config.keys()):
        description = category_enum_config[code]
        lines.append(f"  - {code}: {description}")
    
    return "\n".join(lines)


def get_all_category_codes() -> list[str]:
    """Get all category codes as a list.
    
    Returns:
        List of category code strings
    """
    yaml_config = cmw_config.load_cmw_config()
    category_enum_config = yaml_config.get("category_enum", {})
    return list(category_enum_config.keys())
