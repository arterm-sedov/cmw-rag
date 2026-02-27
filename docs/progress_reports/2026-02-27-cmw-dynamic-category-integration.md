# CMW Platform Dynamic Category Integration

**Date:** 2026-02-27  
**Status:** COMPLETED (Testbed Verified)  
**Branch:** `20260225_platform_integration`

---

## Goal

Integrate CMW Platform with the RAG agent pipeline to:
1. Read support request records from CMW Platform (TPAIModel template)
2. Process them through the agentic pipeline
3. Write response records back to CMW Platform (agent_responses template)
4. Use dynamic category enums from the platform's RequestsIssueArea reference data

---

## What Was Accomplished

### 1. Dynamic Category Enum Implementation ✅

**Problem:** The LLM needed to select from 69 valid category codes from the CMW Platform's `RequestsIssueArea` reference data.

**Solution:**
- Created `rag_engine/cmw_platform/category_enum.py` to dynamically load enum from YAML
- Updated `rag_engine/llm/schemas.py` to use the dynamic `SGRCategory` enum
- The schema now shows proper `enum` values to the LLM (not just a string with description)

**Files Modified:**
- `rag_engine/cmw_platform/category_enum.py` (NEW)
- `rag_engine/llm/schemas.py`

### 2. Curated Category Descriptions ✅

**Problem:** Category codes like `api`, `odata`, `api_gate_way` were ambiguous to the LLM.

**Solution:**
- Rewrote all 69 category descriptions in `cmw_platform.yaml` with concise, functional English descriptions
- Examples:
  - `api`: "Generic REST API usage and troubleshooting"
  - `odata`: "External data access via OData protocol"
  - `api_gate_way`: "API Gateway configuration and routing"

**Files Modified:**
- `rag_engine/config/cmw_platform.yaml`

### 3. Smart Sync Script ✅

**Problem:** Fetching categories from platform would overwrite curated descriptions.

**Solution:**
- Updated `rag_engine/scripts/fetch_issue_areas.py` with merging logic:
  - Preserves curated descriptions in YAML
  - Soft-deletes removed categories (comments out with `# (DELETED)` prefix)
  - Automatically reactivates categories if they reappear

**Files Modified:**
- `rag_engine/scripts/fetch_issue_areas.py`

### 4. JSON Serialization Fix ✅

**Problem:** `TypeError: Object of type SGRCategory is not JSON serializable` when the forced SGR tool call tried to serialize the plan to JSON.

**Solution:**
- Updated `rag_engine/tools/analyse_user_request.py` to convert Enum members to their `.value` strings before returning the JSON plan

**Files Modified:**
- `rag_engine/tools/analyse_user_request.py`

### 5. API Alias Normalization ✅

**Problem:** CMW Platform API expects `FirstCapital` aliases converted to `firstLowerCase` (e.g., `RequestsIssueArea` → `requestsIssueArea`).

**Solution:**
- Added `to_api_alias()` helper in `rag_engine/cmw_platform/attribute_types.py`
- Updated `rag_engine/cmw_platform/records.py` to convert keys automatically

**Files Modified:**
- `rag_engine/cmw_platform/attribute_types.py`
- `rag_engine/cmw_platform/records.py`

###  Request Support ✅

**Problem:** Needed6. PUT to update existing records (not just create new ones).

**Solution:**
- Added `_put_request()` to `rag_engine/cmw_platform/api.py`
- Added `update_record()` function to `rag_engine/cmw_platform/records.py`

**Files Modified:**
- `rag_engine/cmw_platform/api.py`

### 7. Full Pipeline Verification ✅

**Test Run:**
- Input: Record `322919` from TPAIModel ("Настройка интеграции с Outlook")
- SGR Planning: LLM selected category `how_to` from dynamic list
- RAG Retrieval: Retrieved 10 relevant articles
- Output: Created response record `322920` in agent_responses with linked category

---

## Files Changed

| File | Status | Description |
|------|--------|-------------|
| `rag_engine/cmw_platform/category_enum.py` | NEW | Dynamic enum loader from YAML |
| `rag_engine/cmw_platform/api.py` | MODIFIED | Added `_put_request()` |
| `rag_engine/cmw_platform/attribute_types.py` | MODIFIED | Added `to_api_alias()` |
| `rag_engine/cmw_platform/records.py` | MODIFIED | Added `update_record()`, key conversion |
| `rag_engine/cmw_platform/mapping.py` | MODIFIED | Cleanup, removed dead code |
| `rag_engine/config/cmw_platform.yaml` | MODIFIED | Added category_enum with curated descriptions |
| `rag_engine/llm/schemas.py` | MODIFIED | Uses dynamic SGRCategory enum |
| `rag_engine/scripts/fetch_issue_areas.py` | NEW | Smart sync with merge logic |
| `rag_engine/scripts/create_synthetic_request.py` | NEW | Test data generator |
| `rag_engine/tests/test_sgr_tool.py` | MODIFIED | Updated for dynamic enum |
| `rag_engine/tools/analyse_user_request.py` | MODIFIED | Fixed JSON serialization |

---

## Test Results

```
pytest rag_engine/tests/test_sgr_tool.py -v
============================= test session starts =============================
...
rag_engine/tests/test_sgr_tool.py::TestSGRPlanResultSchema::test_category_string_converted_to_enum PASSED
rag_engine/tests/test_sgr_tool.py::TestSGRPlanResultSchema::test_category_case_insensitive PASSED
rag_engine/tests/test_sgr_tool.py::TestSGRPlanResultSchema::test_category_invalid_returns_default PASSED
...
======================== 17 passed in 10.00s ===========================
```

---

## Integration Status

**Current State:** Testbed Verified  
**Pipeline Script:** `rag_engine/scripts/process_cmw_record.py`  
**Verified Record:** 322919 → 322920

The core agent "brain" (schemas, tools, planning) now understands CMW Platform metadata. The orchestration is still handled by the testbed script.

---

## Next Steps (NOT YET IMPLEMENTED)

See `.opencode/plans/20260227_cmw_agent_pipeline_plan.md` for the original plan.

See `.opencode/plans/20260227_cmw_api_endpoint_plan.md` (to be created) for the production integration plan.

---

## Commit

```
commit 946b834
feat: implement dynamic category enums with curated descriptions and merging logic

- Integrate dynamic SGRCategory enum from YAML into RAG pipeline schemas
- Add soft-delete and description-merging to fetch_issue_areas.py
- Fix JSON serialization error in forced SGR tool calls by using member values
- Enhance record operations with automatic PascalCase to camelCase alias conversion
- Verified full pipeline cycle from platform request to RAG response
```
