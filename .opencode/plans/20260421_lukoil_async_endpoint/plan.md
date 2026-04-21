# Plan: Async Fire-and-Forget for Document Agent Endpoint

**Date:** 2026-04-21
**Status:** Draft
**Scope:** Update `/api/v1/cmw/summarize-document` to async pattern per `PlatformConnector`

## Problem

`DocumentSummaryConnector.process()` is synchronous — blocks HTTP response until LLM finishes (15–120s). The support agent endpoint (`PlatformConnector.start_request()`) already implements the correct async pattern: verify record → spawn background thread → return ACK immediately.

## Goal

Make the document agent endpoint async: verify record is readable, return ACK, process in background, write results back via API. No duplication — reuse existing patterns.

## Current State

### Support Agent (correct pattern — `connector.py`)

```
POST /api/v1/cmw/process-request
  → read_record()        # verify accessibility
  → if fail → return error
  → threading.Thread(target=_run_agent_background, daemon=True).start()
  → return ProcessResult(success=True, message="agent started")
```

### Document Agent (broken — `summary_connector.py`)

```
POST /api/v1/cmw/summarize-document
  → DocumentSummaryConnector.process()  # BLOCKS 15–120s
  → return {..., "summary": result.summary}  # summary in HTTP response
```

## Design

### Principle: Reuse, Don't Duplicate

`DocumentSummaryConnector` already does all the work in `process()`. The change is purely in **how it's called** — not what it does. Add a `start()` method that mirrors `PlatformConnector.start_request()`, then have the endpoint call `start()` instead of `process()`.

### Shared `ProcessResult`

Both connectors define their own `ProcessResult`. The one in `summary_connector.py` adds a `summary` field. For async ACK, `summary` is never returned in the HTTP response — it's written directly to the record. **Keep both dataclasses separate** — they serve different contracts. The endpoint ACK only needs `success`, `message`, `error`.

### Changes

#### 1. `summary_connector.py` — Add `start()` method

```python
def start(self, record_id: str) -> ProcessResult:
    """Verify record is readable, spawn background processing, return ACK.

    Mirrors PlatformConnector.start_request() pattern.
    Returns immediately; process() runs in background thread.
    """
    try:
        pipeline = config.load_pipeline_config(self.platform)
        attr_map = pipeline.get("input", {}).get("attributes", {})
        document_attr = attr_map.get("document_file", "")
        prompt_attr = attr_map.get("user_prompt", "")

        if not document_attr:
            return ProcessResult(success=False, error="No document attribute configured")

        record = records.read_record(
            record_id, fields=[document_attr, prompt_attr], platform=self.platform,
        )
        if not record.get("success"):
            return ProcessResult(success=False, error=f"Failed to read record: {record.get('error')}")

        thread = threading.Thread(
            target=self.process,
            args=(record_id,),
            daemon=True,
        )
        thread.start()

        logger.info("Started background document processing for %s", record_id)
        return ProcessResult(success=True, message="Начата обработка данных")
    except Exception as e:
        logger.exception("Failed to start document processing for %s", record_id)
        return ProcessResult(success=False, error=str(e))
```

Key decisions:
- `start()` verifies record readability (like `PlatformConnector.start_request()`)
- `start()` spawns `self.process()` in background thread — **reuses existing logic**, zero duplication
- `process()` stays unchanged — it already handles the full pipeline including write-back
- `ProcessResult` returned by `start()` has no `summary` — correct for ACK

#### 2. `app.py` — Endpoint calls `start()` instead of `process()`

```python
@fastapi_app.post("/api/v1/cmw/summarize-document")
async def summarize_document_endpoint(req: SummarizeDocumentRequest, http_req: Request) -> dict:
    """Summarize document attached to a CMW Platform record (Lukoil instance).

    Fire-and-forget: verifies record, returns ACK immediately,
    processes document in background thread, writes result to record via API.

    Args:
        request_id: Record ID in ArchitectureManagement.Zaprosinarazrabotky

    Returns:
        {"success": bool, "message": str, "error": str | None}
    """
    if settings.cmw2_api_key:
        provided_key = http_req.headers.get("X-API-Key")
        if provided_key != settings.cmw2_api_key:
            logger.warning("Invalid API key attempt to summarize-document endpoint")
            return {"success": False, "message": None, "error": "Invalid API key"}

    from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

    try:
        connector = DocumentSummaryConnector(platform="secondary")
        result = connector.start(req.request_id)

        return {
            "success": result.success,
            "message": result.message,
            "error": result.error,
        }
    except Exception:
        logger.exception("Document summarization failed to start")
        return {"success": False, "error": "Internal error"}
```

Key decisions:
- **No `summary` in response** — results are written directly to record attributes
- Response shape matches `PlatformConnector` endpoint response: `{success, message, error}`
- `process()` already calls `records.update_record()` at the end — no change needed

#### 3. Add `threading` import to `summary_connector.py`

Already has `logging` import. Add `threading` at top.

#### 4. `process()` stays unchanged

It already:
- Reads record → extracts document → summarizes → writes back via `records.update_record()`
- The only semantic change: `process()` now runs in a background thread
- Its `ProcessResult` (with `summary`) is consumed only by logging/error handling within the thread

### What Does NOT Change

- `process()` method — untouched, runs identically in background
- `ProcessResult` dataclass — both versions stay (different contracts)
- `_extract_text()`, `_to_html()`, `_summarize()` — untouched
- `PlatformConnector` — untouched
- Tests for `process()` — untouched (behavior unchanged)
- Integration tests — add new tests for `start()`

## Test Plan (TDD)

### New Tests — `test_cmw_platform_summary_connector.py`

```python
def test_document_summary_connector_start_exists():
    """start() method should exist on DocumentSummaryConnector."""
    from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector
    conn = DocumentSummaryConnector()
    assert hasattr(conn, "start")

def test_start_returns_process_result():
    """start() should return ProcessResult with no summary in ACK."""
    from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector, ProcessResult
    with patch("rag_engine.cmw_platform.summary_connector.records.read_record") as mock_read:
        mock_read.return_value = {"success": False, "error": "Not found", "data": {}}
        with patch("rag_engine.cmw_platform.summary_connector.config.load_pipeline_config") as mock_cfg:
            mock_cfg.return_value = {"input": {"attributes": {"document_file": "doc", "user_prompt": "p"}}, "output": {}}
            conn = DocumentSummaryConnector()
            result = conn.start("test-id")
            assert isinstance(result, ProcessResult)
            assert result.success is False

def test_start_success_returns_ack_without_summary():
    """start() should return ACK without summary when record is readable."""
    from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector, ProcessResult
    with patch("rag_engine.cmw_platform.summary_connector.records.read_record") as mock_read:
        mock_read.return_value = {"success": True, "data": {"test-id": {"doc": {"id": "file-1"}, "p": ""}}}
        with patch("rag_engine.cmw_platform.summary_connector.config.load_pipeline_config") as mock_cfg:
            mock_cfg.return_value = {"input": {"attributes": {"document_file": "doc", "user_prompt": "p"}}, "output": {}}
            with patch("rag_engine.cmw_platform.summary_connector.threading.Thread") as mock_thread:
                conn = DocumentSummaryConnector()
                result = conn.start("test-id")
                assert result.success is True
                assert result.summary is None
                mock_thread.assert_called_once()

def test_start_spawns_process_in_background_thread():
    """start() should spawn process() in a daemon thread."""
    from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector
    with patch("rag_engine.cmw_platform.summary_connector.records.read_record") as mock_read:
        mock_read.return_value = {"success": True, "data": {"test-id": {"doc": {"id": "file-1"}, "p": ""}}}
        with patch("rag_engine.cmw_platform.summary_connector.config.load_pipeline_config") as mock_cfg:
            mock_cfg.return_value = {"input": {"attributes": {"document_file": "doc", "user_prompt": "p"}}, "output": {}}
            with patch("rag_engine.cmw_platform.summary_connector.threading.Thread") as mock_thread:
                conn = DocumentSummaryConnector()
                conn.start("test-id")
                mock_thread.assert_called_once()
                args = mock_thread.call_args
                assert args[1]["target"] == conn.process
                assert args[1]["daemon"] is True
```

### Existing Tests — No Changes Expected

- `test_cmw_platform_summary_connector.py` — all `process()` tests unchanged
- `test_cmw_platform_summarize_integration.py` — all integration tests unchanged

## Verification

1. `ruff check rag_engine/cmw_platform/summary_connector.py`
2. `ruff check rag_engine/api/app.py`
3. `pytest rag_engine/tests/test_cmw_platform_summary_connector.py`
4. `pytest rag_engine/tests/test_cmw_platform_summarize_integration.py`
5. `pytest rag_engine/tests/test_cmw_platform.py` — verify support agent unchanged
6. `pytest` — full suite

## Summary of Changes

| File | Change |
|:--|:--|
| `rag_engine/cmw_platform/summary_connector.py` | Add `threading` import, add `start()` method |
| `rag_engine/api/app.py` | Endpoint: `connector.process()` → `connector.start()`, remove `summary` from response |
| `rag_engine/tests/test_cmw_platform_summary_connector.py` | Add 4 tests for `start()` |

**3 files, ~30 lines of new code, 0 lines of duplicated logic.**
