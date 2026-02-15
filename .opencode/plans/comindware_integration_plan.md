# Comindware Platform Integration Plan

## Goal

Bring Comindware Platform API integration to cmw-rag agent for:
- Reading records from the platform (by record ID)
- Creating new records in the platform (linked to original via foreign key)

## Source Code Reference

Copy from `C:\Repos\cmw-platform-agent\`:
- `tools/requests_.py` → adapted to this repo
- `tools/requests_models.py` → copy as-is
- `tools/templates_tools/tool_create_edit_record.py` → adapted to plain functions

## Architecture

### Session Strategy
- **Basic Authentication**: Each request uses `Authorization: Basic <base64(login:password)>` header
- No session pool, no login endpoint
- Each request is independent and authenticated via Basic Auth
- No need for session management - just add auth header to every request

### Directory Structure

```
rag_engine/integrations/
└── comindware/
    ├── __init__.py       # exports: create_record, read_record
    ├── models.py         # HTTPResponse, APIResponse
    ├── api.py            # _load_server_config, _basic_headers, _get_request, _post_request
    └── records.py        # create_record(), read_record()
```

## Files to Create

### 1. rag_engine/integrations/comindware/models.py
Copy from `C:\Repos\cmw-platform-agent\tools\requests_models.py`:
- `HTTPResponse`
- `APIResponse` 
- `RequestConfig`

### 2. rag_engine/integrations/comindware/api.py
Adapted from `C:\Repos\cmw-platform-agent\tools\requests_.py`:

| Function | Description |
|----------|-------------|
| `_load_server_config()` | Reads `.env` for CMW_BASE_URL, CMW_LOGIN, CMW_PASSWORD, CMW_TIMEOUT |
| `_basic_headers()` | Creates `Authorization: Basic <base64(login:password)>` header |
| `_get_request(endpoint)` | GET with Basic Auth header |
| `_post_request(body, endpoint)` | POST with Basic Auth header |

**Key changes from reference:**
- Remove session manager / session pool dependencies
- Remove `CMW_USE_DOTENV` logic (always use `.env`)
- Keep Basic Auth on every request (already how it works in reference)

### 3. rag_engine/integrations/comindware/records.py

| Function | Description |
|----------|-------------|
| `create_record(application_alias, template_alias, values)` | POST to `/webapi/Records/{templateGlobalAlias}` |
| `read_record(record_id, fields=None)` | POST to `api/public/system/TeamNetwork/ObjectService/GetPropertyValues` with server-side filtering |

**Why GetPropertyValues?**
- Accepts list of record IDs and field aliases
- Returns only requested fields (server-side filtering)
- Much more efficient than fetching full records via `/webapi/Record/{recordId}`

**Input format:**
```python
{
    "objects": ["record-id-1"],
    "propertiesByAlias": ["field_alias_1", "field_alias_2"]
}
```

**Return structure:**
```python
{
    "success": bool,
    "status_code": int,
    "record_id": str | None,  # for create
    "data": dict | list | None,  # for read
    "error": str | None
}
```

### 4. rag_engine/integrations/comindware/__init__.py
```python
from rag_engine.integrations.comindware.records import create_record, read_record

__all__ = ["create_record", "read_record"]
```

### 5. rag_engine/integrations/__init__.py
Empty or marker file to make it a package.

## Environment Variables

Add to `.env` and `.env-example`:

```bash
# Comindware Platform
CMW_BASE_URL=https://your-platform.comindware.com
CMW_LOGIN=your_username
CMW_PASSWORD=your_password
CMW_TIMEOUT=30
```

## Requirements Update

Add to `rag_engine/requirements.txt`:
```
requests>=2.28.0
```

## Usage Example

```python
from rag_engine.integrations.comindware import create_record, read_record

# Read record with specific fields (server-side filtering via GetPropertyValues)
result = read_record("record-uuid-123", fields=["user_question", "title"])
if result["success"]:
    data = result["data"]
    # Returns: {"record-uuid-123": {"user_question": "...", "title": "..."}}

# Create new record linked to original
result = create_record(
    application_alias="support_app",
    template_alias="resolution_output",
    values={
        "summary": "AI generated summary",
        "original_request": "record-uuid-123",  # foreign key
        "confidence": 0.95
    }
)
if result["success"]:
    new_record_id = result["record_id"]
```

## API Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `api/public/system/TeamNetwork/ObjectService/GetPropertyValues` | POST | Read specific fields from records (server-side filtering) |
| `/webapi/Records/{templateGlobalAlias}` | POST | Create new record(s) |

Note: All requests use Basic Authentication header. No login endpoint required.

## Implementation Order

1. Create `rag_engine/integrations/comindware/` directory
2. Copy `models.py` from reference
3. Create `api.py` (adapted from reference)
4. Create `records.py` with both functions
5. Create `__init__.py` with exports
6. Add `requests` to `requirements.txt`
7. Update `.env-example` with new variables
8. Create unit tests `rag_engine/tests/test_comindware_api.py`
9. (Optional) Create integration test script

## Verification

```bash
# Test import
python -c "from rag_engine.integrations.comindware import create_record, read_record; print('OK')"

# Run tests (if any created)
pytest rag_engine/tests/ -v
```

## Testing Strategy

Following AGENTS.md best practices: test behavior, not implementation.

### Unit Tests (`rag_engine/tests/test_comindware_api.py`)

**Test behavior, not implementation details:**

| Test | What it tests |
|------|---------------|
| `test_load_server_config_missing_env_vars` | Fails gracefully when CMW_BASE_URL/LOGIN/PASSWORD missing |
| `test_load_server_config_valid_env` | Returns valid RequestConfig with correct values |
| `test_basic_headers_creates_valid_auth` | Header contains valid base64 encoded credentials |
| `test_get_property_values_request_format` | Request body has correct structure (`objects`, `propertiesByAlias`) |
| `test_create_record_request_format` | Request body contains values with correct structure |
| `test_read_record_filters_fields` | Correctly filters response to requested fields |
| `test_api_response_success_structure` | Success response has expected keys |
| `test_api_response_error_structure` | Error response has expected keys |

**DO NOT test:**
- Specific ports or URLs (use patterns)
- Internal function calls
- Mock internal implementation details

### Integration Test Script (`rag_engine/scripts/test_comindware_integration.py`)

Optional script for manual testing against real API:

```python
if __name__ == "__main__":
    # Test read with specific fields
    result = read_record("test-record-id", fields=["user_question", "title"])
    print(result)
    
    # Test create
    result = create_record("app_alias", "template_alias", {"field": "value"})
    print(result)
```

### Mock Strategy

For unit tests, mock the `requests` library:
```python
from unittest.mock import patch, Mock

@patch('rag_engine.integrations.comindware.api.requests.post')
def test_create_record_success(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True, "data": "new-record-id"}
    mock_post.return_value = mock_response
    
    result = create_record("app", "template", {"field": "value"})
    assert result["success"] is True
```

## Dependent Features

- Platform webhook handler (future)
- Record processing pipeline (future)
