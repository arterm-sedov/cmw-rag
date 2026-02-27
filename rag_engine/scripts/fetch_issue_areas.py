"""Utility to fetch RequestsIssueArea codes and generate YAML enum config."""
import os
import re
import yaml
from rag_engine.cmw_platform.api import _get_request

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "cmw_platform.yaml")

def fetch_issue_areas() -> list[dict]:
    """Fetch all issue areas from RequestsIssueArea template.
    
    Returns:
        List of {code, nameEn}
    """
    result = _get_request("/webapi/Records/Template@dima.RequestsIssueArea")
    if not result.get("success"):
        print(f"Error fetching: {result}")
        return []

    records = result.get("data", {}).get("response", [])
    areas = []
    for r in records:
        code = r.get("code")
        if code:
            areas.append({
                "code": code,
                "nameEn": r.get("nameEn", r.get("name", code)),
            })

    return sorted(areas, key=lambda x: x["code"])


def update_yaml_config(new_areas: list[dict]):
    """Update cmw_platform.yaml with new category_enum, merging with existing ones.
    
    Deleted categories are commented out to preserve their descriptions.
    New categories are added.
    Active categories are kept uncommented.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the category_enum block
    block_match = re.search(r"category_enum:.*?(?=\n\w+:|$)", content, re.DOTALL)
    if not block_match:
        print("Could not find category_enum block in YAML")
        return

    block_content = block_match.group(0)
    
    # Extract all keys and their descriptions, including commented ones
    # Pattern matches "key: description" or "# (DELETED) key: description"
    entries = {}
    entry_pattern = re.compile(r"^\s*(#\s*\(DELETED\)\s*)?([\wА-я]+):\s*\"(.*)\"", re.MULTILINE)
    for match in entry_pattern.finditer(block_content):
        is_commented = match.group(1) is not None
        key = match.group(2)
        desc = match.group(3)
        entries[key] = {"desc": desc, "existed": True}

    # Current codes from API
    new_codes = {a["code"] for a in new_areas}
    
    # Merge
    final_entries = entries.copy()
    for area in new_areas:
        code = area["code"]
        name_en = area["nameEn"]
        if code not in final_entries:
            final_entries[code] = {"desc": name_en, "existed": False}
            print(f"Added new category: {code}")

    # Generate new block
    lines = ["category_enum:"]
    # Sort keys for consistency
    for key in sorted(final_entries.keys()):
        desc = final_entries[key]["desc"].replace('"', '\\"')
        if key in new_codes:
            # Active
            lines.append(f'  {key}: "{desc}"')
        else:
            # Deleted - comment it out
            # Avoid double commenting if it was already commented
            clean_desc = desc.replace("(DELETED) ", "").strip()
            lines.append(f'  # (DELETED) {key}: "{clean_desc}"')
            if key in entries and not block_content.count(f"# (DELETED) {key}:"):
                 print(f"Category deleted from platform, commenting out: {key}")

    new_block_content = "\n".join(lines)
    updated_content = content.replace(block_content, new_block_content)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(updated_content)
    
    print(f"Updated {CONFIG_PATH} with {len(new_codes)} active categories.")


def main():
    """Fetch and update YAML enum config."""
    areas = fetch_issue_areas()
    
    if not areas:
        print("No areas fetched")
        return
    
    update_yaml_config(areas)


if __name__ == "__main__":
    main()
