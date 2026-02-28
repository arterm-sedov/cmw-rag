# CMW Platform API Endpoint Implementation Plan

**Date:** 2026-02-27  
**Status:** COMPLETED  
**Parent Branch:** `20260225_platform_integration`

---

## Goal

Create a production-ready API endpoint that the CMW Platform can call to process support requests. Keep the testbed script (`rag_engine/scripts/process_cmw_record.py`) as a reference implementation and comparison baseline.

---

## Requirements

### 1. API Contract

**Endpoint:** `POST /api/v1/cmw/process-support-request`

**Authorization:**
- Simple API key in header (`X-API-Key`) matching a config value
- Not OAuth-heavy — just robust enough to prevent random attacks
- Attack is harmless anyway: the response goes back to the platform

**Input (JSON Body):**
```json
{
  "request_id": "322919"
}
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "Request fetched, agent started",
  "error": null
}
```

**Error Response (4xx/5xx):**
```json
{
  "success": false,
  "message": null,
  "error": "Failed to fetch record: ..."
}
```

### 2. Pipeline Flow

1. **Receive** request ID from platform (POST body)
2. **Validate** API key
3. **Fetch** request record from TPAIModel (using config)
4. **Start** RAG agent asynchronously (don't wait)
4. **Return** success immediately after fetching the record — start agent in background

**Async Model:**
- The agent processes the full pipeline in the background
- Pipeline: Fetch → Build Request → Process → Map → Create Linked Response Record
- Platform doesn't wait for completion — it just needs confirmation that we started

### 3. Configuration Driven

All templates, fields, and mappings defined in `cmw_platform.yaml` — no hardcoding.

---

## Code Organization

### Final State

```
rag_engine/
├── cmw_platform/
│   ├── __init__.py         # Exports: Connector, ProcessResult, Records
│   ├── api.py              # _get_request, _post_request, _put_request
│   ├── attribute_types.py  # to_api_alias, coerce, metadata
│   ├── category_enum.py    # load_category_enum
│   ├── config.py           # load_cmw_config, get_*_config
│   ├── connector.py        # NEW: PlatformConnector orchestrator
│   ├── mapping.py          # map_agent_response
│   ├── models.py           # Pydantic models
│   ├── records.py          # create_record, read_record, update_record
│   └── request_builder.py  # build_request
├── api/
│   └── app.py              # MODIFIED: Added /api/v1/cmw/process-support-request endpoint
├── scripts/
│   └── process_cmw_record.py  # REFERENCE IMPLEMENTATION (KEPT)
```

---

## Implementation Details

### connector.py

- `ProcessResult` dataclass for return values
- `PlatformConnector` class with `start_request()` method
- Fire-and-forget: fetches record, starts agent in background thread
- Uses `threading.Thread` with daemon=True
- Full pipeline: fetch → build request → call agent → map → create response record

### API Endpoint

- FastAPI route via Gradio's underlying `demo.app`
- `POST /api/v1/cmw/process-support-request`
- Request model: `{"request_id": "string"}`
- Auth via `X-API-Key` header (optional if `CMW_API_KEY` is empty in .env)

### Settings

- `cmw_api_key: str` - mandatory in .env (empty = skip auth, present = require it)
- Added to settings.py with `validate_default=True`
- If not present in .env → fails on startup (ValidationError)

---

## Tests

All 23 tests pass:
- 14 existing tests
- 9 new tests for PlatformConnector and auth logic

---

## Validation

The endpoint:
- ✅ Accepts `{"request_id": "322919"}` via POST with optional `X-API-Key` header
- ✅ Returns `{"success": true, "message": "Request fetched, agent started", "error": null}`
- ✅ Starts agent in background after fetching the request record
- ✅ Creates linked record in `agent_responses` template asynchronously
- ✅ Logs all pipeline steps using existing logging engine
- ✅ Tests cover: valid request, missing request_id, invalid API key

---

## Notes

- **Session Isolation:** Gradio provides request isolation. Our agent is stateless between calls.
- **Multiple Responses:** We're okay with multiple responses per request — useful for A/B testing.
- **Async Model:** The platform sends a request ID and immediately detaches. We confirm we started.
- **Reference Kept:** The testbed script is a working, tested implementation we can compare against.
