# CMW Platform Enum Attribute Handling Analysis

**Date:** 2026-02-27  
**Status:** Open Issue - Requires Further Investigation

---

## Problem Statement

CMW Platform enum attributes require **enum value IDs** (e.g., `vv.13`, `vv.19`) to be set correctly. Passing system names (e.g., `block`, `info`) does NOT work reliably - the platform may display them incorrectly or not at all.

---

## Background

### Enum Value Structure

Each enum attribute has four identifiers:

| Identifier | Example | Source |
|------------|---------|--------|
| **ID** | `vv.13` | Internal platform ID - required for API |
| **System Name** | `block`, `how_to` | Defined in attribute metadata |
| **Display Name RU** | `Заблокировать`, `Инструкция` | Russian localization |
| **Display Name EN** | `Block`, `Instruction` | English localization |
| **Display Name DE** | `Sperren`, `Anleitung` | German localization |

### Current API Behavior

**Endpoint:** `GET /webapi/Attribute/List/Template@{app}.{template}`

Returns variants with:
```json
{
  "variants": [
    {
      "alias": {
        "type": "Variant",
        "owner": "action",
        "alias": "block"  // <-- System Name, NOT ID
      },
      "name": {
        "ru": "Заблокировать",
        "en": "Block",
        "de": "Sperren"
      },
      "color": "#b71c1c"
    }
  ]
}
```

**Issue:** API response does NOT include the `vv.xx` enum value IDs.

---

## Test Results

| Test | Input | Expected | Actual | Result |
|------|-------|----------|--------|--------|
| 1 | Set `action=block` | Display as BLOCK | Shows BLOCK | ✅ Works |
| 2 | Set `category=info` | Display as INFO | Shows INFO | ✅ Works |
| 3 | Set `category=general_question` | Invalid system name | Shows "general_question" | ❌ Wrong |

---

## Options for Resolution

### Option 1: Discover vv.xx IDs Programmatically (Recommended)

Create test records for each enum value system name, read back the resulting ID, and build a static mapping.

**Pros:** 
- Accurate mapping
- Works with current API

**Cons:**
- Requires one-time setup per template
- Must be refreshed if enum values change
- Adds startup overhead

**Implementation:**
```python
def _discover_enum_ids(app: str, template: str) -> dict:
    """Discover enum value IDs by creating test records."""
    # For each enum attribute:
    #   1. Create record with system name value
    #   2. Read back to get vv.xx ID
    #   3. Build mapping: system_name -> vv.xx ID
```

### Option 2: Use Alternative API Endpoint

Research if there's an API endpoint that returns enum value IDs.

**Status:** Unknown - requires research in CMW Platform documentation.

### Option 3: Hardcoded Static Mapping

Maintain a manual mapping in configuration.

**Pros:** Simple  
**Cons:** Must be manually maintained; fragile

---

## Recommended Approach

Use **Option 1** - programmatic discovery with caching:

1. On first use, fetch enum attribute metadata
2. For each enum variant with system name `X`:
   - Create test record with `{attr: X}`
   - Read back to get platform-assigned ID `vv.yy`
   - Store mapping: `X → vv.yy`
3. Cache the mapping for session lifetime
4. On subsequent requests: LLM output → system name lookup → ID lookup → pass vv.xx to API

---

## Implementation Notes

### Files to Modify

- `rag_engine/cmw_platform/api.py` - Add attribute metadata fetch
- `rag_engine/cmw_platform/config.py` - Add enum cache and lookup functions  
- `rag_engine/cmw_platform/attribute_types.py` - Update coerce_enum to use ID lookup
- `rag_engine/config/cmw_platform.yaml` - Mark enum attributes with `type: enum`

### Fallback Behavior

If enum value ID lookup fails:
- Return `None` (do NOT pass invalid string to platform)
- Log warning for debugging

### Alternative: Validate Only Mode

If vv.xx IDs cannot be discovered, implement validation-only mode:
- Validate LLM output against valid system names
- If invalid, fail explicitly
- If valid, pass system name through (let platform handle translation)

---

## Open Questions

1. Is there a CMW Platform API endpoint that directly returns enum value IDs?
2. Are vv.xx IDs stable across environments (dev/staging/prod)?
3. Should enum discovery run at startup or lazily on first use?

---

## References

- CMW Platform Agent: `D:\Repo\cmw-platform-agent\tools\templates_tools\tool_create_edit_record.py`
- CMW Platform Agent: `D:\Repo\cmw-platform-agent\tools\templates_tools\tool_list_attributes.py`
- CMW OpenAPI: `D:\Repo\cmw-platform-agent\cmw_open_api\web_api_v1.json`
