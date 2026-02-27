# CMW Platform API Endpoint Implementation Plan

**Date:** 2026-02-27  
**Status:** PLANNED (Not Implemented)  
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

### Current State (Reference)

```
rag_engine/
├── cmw_platform/
│   ├── api.py              # _get_request, _post_request, _put_request
│   ├── attribute_types.py  # to_api_alias, coerce, metadata
│   ├── category_enum.py    # load_category_enum
│   ├── config.py           # load_cmw_config, get_*_config
│   ├── mapping.py          # map_agent_response
│   ├── models.py           # Pydantic models
│   ├── records.py          # create_record, read_record, update_record
│   └── request_builder.py # build_request
├── scripts/
│   └── process_cmw_record.py  # REFERENCE IMPLEMENTATION (KEPT)
```

### Target State (Production)

```
rag_engine/
├── cmw_platform/
│   ├── __init__.py         # Exports: Connector, Config, Records
│   ├── api.py              # Unchanged
│   ├── attribute_types.py  # Unchanged
│   ├── category_enum.py    # Unchanged
│   ├── config.py           # Unchanged
│   ├── connector.py        # NEW: PlatformConnector orchestrator
│   ├── mapping.py          # Unchanged
│   ├── models.py           # Unchanged
│   ├── records.py          # Unchanged
│   └── request_builder.py  # Unchanged
├── api/
│   └── app.py              # MODIFIED: Add /api/v1/cmw/process-support-request endpoint
```

### New File: `rag_engine/cmw_platform/connector.py`

```python
"""CMW Platform connector orchestrator.

Provides a single entry point for processing platform requests.
"""
import logging
from dataclasses import dataclass

from rag_engine.cmw_platform import config, records
from rag_engine.cmw_platform.mapping import map_agent_response
from rag_engine.cmw_platform.request_builder import build_request


logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    success: bool
    message: str | None = None
    error: str | None = None


class PlatformConnector:
    """Orchestrates the complete CMW Platform request → response pipeline."""

    def start_request(self, request_id: str) -> ProcessResult:
        """Start processing a TPAIModel record through the RAG pipeline.
        
        This method is ASYNC - it fetches the record, starts the agent, and returns.
        The agent runs in the background and creates the linked response record.
        
        Args:
            request_id: The TPAIModel request ID to process
            
        Returns:
            ProcessResult with success status (fetch succeeded, agent started)
        """
        try:
            # 1. Fetch input record
            input_config = config.get_input_config()
            fields = [f["name"] for f in input_config.get("fields", [])]
            record = records.read_record(request_id, fields=fields)
            
            if not record["success"]:
                logger.error(f"Failed to fetch record {request_id}: {record.get('error')}")
                return ProcessResult(
                    success=False,
                    error=f"Failed to fetch record: {record.get('error')}"
                )
            
            record_data = record["data"].get(request_id, {})
            
            # 2. Build markdown request
            md_request = build_request(record_data)
            
            # 3. START AGENT IN BACKGROUND (fire-and-forget)
            # Return success AFTER fetching record, not after agent completes
            import asyncio
            from rag_engine.api.app import ask_comindware_structured
            
            # Run in background thread so we don't block the API response
            def run_agent():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    agent_result = loop.run_until_complete(
                        ask_comindware_structured(md_request)
                    )
                    
                    # 4. Map to output template
                    output_config = config.get_output_config()
                    template_config = config.get_template_config(
                        output_config["application"],
                        output_config["template"]
                    )
                    mapped_values = map_agent_response(
                        agent_result=agent_result,
                        input_record_id=request_id,
                        attributes=template_config["attributes"],
                        md_request=md_request,
                    )
                    
                    # 5. Create response record
                    response = records.create_record(
                        application_alias=output_config["application"],
                        template_alias=output_config["template"],
                        values=mapped_values,
                    )
                    
                    if not response["success"]:
                        logger.error(f"Failed to create response: {response.get('error')}")
                    else:
                        logger.info(f"Created response record: {response.get('record_id')}")
                        
                finally:
                    loop.close()
            
            import threading
            thread = threading.Thread(target=run_agent, daemon=True)
            thread.start()
            
            logger.info(f"Started agent for request {request_id}")
            
            return ProcessResult(
                success=True,
                message="Request fetched, agent started",
            )
            
        except Exception as e:
            return ProcessResult(success=False, error=str(e))
```

### Integration: `rag_engine/api/app.py`

Add a new FastAPI endpoint (Gradio is based on FastAPI anyway):

```python
from rag_engine.cmw_platform.connector import PlatformConnector, ProcessResult

@app.post("/api/v1/cmw/process-support-request")
def process_support_request(request: dict) -> dict:
    """Process a CMW Platform support request.
    
    Expects: {"request_id": "123456"}
    Returns: {"success": true, "message": "Request fetched, agent started", "error": null}
    
    Note: This is a fire-and-forget endpoint. The agent runs asynchronously
    and creates the linked response record in the background.
    
    Authentication: CMW_API_KEY must be present in .env (empty = no auth, present = use it)
    """
    # 1. Get API key from settings (mandatory in .env, but can be empty to skip auth)
    # Use similar pattern as rag_agent: from rag_engine.config.settings import settings
    from rag_engine.config.settings import settings
    api_key = settings.cmw_api_key  # str | None, from settings
    
    if api_key:  # truthy = require auth
        # 2. Validate API key from header
        request_api_key = request.headers.get("X-API-Key")
        if request_api_key != api_key:
            logger.warning("Invalid API key attempt")
            return {
                "success": False,
                "message": None,
                "error": "Invalid API key"
            }
    
    # 3. Get request_id
    request_id = request.get("request_id")
    if not request_id:
        return {
            "success": False,
            "message": None,
            "error": "Missing request_id"
        }
    
    # 4. Start processing (async)
    connector = PlatformConnector()
    result = connector.start_request(request_id)
    
    return {
        "success": result.success,
        "message": result.message,
        "error": result.error,
    }

    
    # 2. Get request_id
    request_id = request.get("request_id")
    if not request_id:
        return {
            "success": False,
            "message": None,
            "error": "Missing request_id"
        }
    
    # 3. Start processing (async)
    connector = PlatformConnector()
    result = connector.start_request(request_id)
    
    return {
        "success": result.success,
        "message": result.message,
        "error": result.error,
    }
```

---

## Implementation Order

1. **Add tests first** (they define expected behavior)  
   - Test `PlatformConnector.start_request()` with mocked records  
   - Integration test with real platform connection (config in `.env`, templates in `cmw_platform.yaml`)  
   - Use `rag_engine/cmw_platform/records.py::list_records()` to get latest request from TPAIModel  
   - Or hardcode `322919` for quick validation  

2. **Create `rag_engine/cmw_platform/connector.py`**  
   - Wrap reference logic in `PlatformConnector` class  
   - Keep it minimal — delegate to existing modules  
   - Use threading for fire-and-forget async execution  

3. **Add API endpoint in `rag_engine/api/app.py`**  
   - Use FastAPI (built into Gradio)  
   - Add simple API key validation  
   - Return success immediately after starting agent  

4. **Keep reference implementation**  
   - `rag_engine/scripts/process_cmw_record.py` stays as comparison baseline  

---

## Cleanup Checklist

- [ ] KEEP `rag_engine/scripts/process_cmw_record.py` as reference
- [ ] Add `cmw_api_key: str | None = None` to `rag_engine/config/settings.py` (CMW Platform section, mandatory in .env but can be empty to skip auth)
- [ ] Add `CMW_API_KEY=` to `.env-example` (mandatory in .env, empty = skip auth, filled = use it)
- [ ] Generate key with `python -c "import secrets; print(secrets.token_hex(32))"` and add to actual `.env` (NEVER commit .env)
- [ ] Ensure all imports in `rag_engine/cmw_platform/__init__.py` are correct
- [ ] Verify no circular dependencies between `api.app` and `cmw_platform.connector`

---

## Validation

After implementation, the endpoint should:
- Accept `{"request_id": "322919"}` via POST with optional `X-API-Key` header (if configured)
- Return `{"success": true, "message": "Request fetched, agent started", "error": null}`
- Start agent in background after fetching the request record
- Create linked record in `agent_responses` template asynchronously
- Log all pipeline steps using existing logging engine
- Tests cover: valid request, missing request_id, invalid API key

---

## Notes

- **Session Isolation:** Gradio provides request isolation. Our agent is stateless between calls.
- **Multiple Responses:** We're okay with multiple responses per request — useful for A/B testing.
- **Async Model:** The platform sends a request ID and immediately detaches. We confirm we started.
- **Reference Kept:** The testbed script is a working, tested implementation we can compare against.
