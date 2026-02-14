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
- **Per-request login**: Login → API call → close connection
- No session pool needed - Comindware platform handles concurrent API calls
- Each API request is independent

### Directory Structure

```
rag_engine/integrations/
└── comindware/
    ├── __init__.py       # exports: create_record, read_record
    ├── models.py         # HTTPResponse, APIResponse
    ├── api.py            # _login, _get_request, _post_request
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
| `_login()` | POST login, returns session cookie |
| `_get_request(endpoint)` | GET with session cookie, closes after |
| `_post_request(body, endpoint)` | POST with session cookie, closes after |

**Key changes from reference:**
- Remove session manager / session pool
- Remove `CMW_USE_DOTENV` logic (always use `.env`)
- Each request does fresh login

### 3. rag_engine/integrations/comindware/records.py

| Function | Description |
|----------|-------------|
| `create_record(application_alias, template_alias, values)` | POST to `/webapi/Records/{templateGlobalAlias}` |
| `read_record(record_id, fields=None)` | GET from `/webapi/Record/{recordId}`, optional field filter |

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

# Read record with specific fields
result = read_record("record-uuid-123", fields=["user_question", "title"])
if result["success"]:
    data = result["data"]

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
| `/api/auth/login` | POST | Authenticate, get session cookie |
| `/webapi/Record/{recordId}` | GET | Read single record |
| `/webapi/Records/{templateGlobalAlias}` | POST | Create new record(s) |

Note: API returns full record - filtering happens client-side in `read_record()`.

## Implementation Order

1. Create `rag_engine/integrations/comindware/` directory
2. Copy `models.py` from reference
3. Create `api.py` (adapted from reference)
4. Create `records.py` with both functions
5. Create `__init__.py` with exports
6. Add `requests` to `requirements.txt`
7. Update `.env-example` with new variables

## Verification

```bash
# Test import
python -c "from rag_engine.integrations.comindware import create_record, read_record; print('OK')"

# Run tests (if any created)
pytest rag_engine/tests/ -v
```

## Dependent Features

- Platform webhook handler (future)
- Record processing pipeline (future)
