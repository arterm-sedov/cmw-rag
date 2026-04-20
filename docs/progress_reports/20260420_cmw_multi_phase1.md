# Phase 1 Complete: Config Refactoring

## Date: 2026-04-20

## Status: ✅ COMPLETE

## Checkpoint Status

- [x] Create test file `rag_engine/tests/test_cmw_platform_config_multi.py`
- [x] Write 12 tests for multi-platform config loading
- [x] Run tests — all 12 PASSED
- [x] Run lint — all checks passed
- [x] Verify existing CMW platform tests pass (28 passed)

---

## Tests Added/Updated

| File | Tests | Status |
|------|-------|--------|
| `rag_engine/tests/test_cmw_platform_config_multi.py` | 12 new | All PASSED |

### Test Results

```
test_load_cmw_config_default_returns_primary PASSED
test_load_cmw_config_with_explicit_primary PASSED
test_load_cmw_config_secondary_loads_different_file PASSED
test_get_input_config_accepts_platform_param PASSED
test_get_output_config_accepts_platform_param PASSED
test_get_input_attributes_accepts_platform_param PASSED
test_get_platform_attribute_accepts_platform_param PASSED
test_get_python_attribute_accepts_platform_param PASSED
test_get_request_template_accepts_platform_param PASSED
test_get_template_config_accepts_platform_param PASSED
test_get_attribute_metadata_accepts_platform_param PASSED
test_load_pipeline_config_accepts_platform_param PASSED
```

---

## Code Changes

| File | Lines | Description |
|------|-------|-------------|
| `rag_engine/cmw_platform/config.py` | +59/-0 | Added `platform` param to all config functions, platform-specific YAML loading, config caching |
| `rag_engine/config/cmw_platform_secondary.yaml` | +43 (new) | Secondary platform config for ArchitectureManagement |
| `rag_engine/tests/test_cmw_platform_config_multi.py` | +145 (new) | TDD tests for multi-platform config |
| `rag_engine/tests/test_cmw_platform.py` | 0 | All 28 existing tests still pass |

---

## Architecture Changes

### Before

```python
def load_cmw_config() -> dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config" / "cmw_platform.yaml"
    ...
```

### After

```python
DEFAULT_PLATFORM = os.getenv("CMW_PLATFORM_NAME", "primary")

def load_cmw_config(platform: str | None = None) -> dict[str, Any]:
    platform = platform or DEFAULT_PLATFORM
    # Loads cmw_platform.yaml or cmw_platform_{platform}.yaml
    ...
```

### Config Files

| Platform | Config File |
|----------|-------------|
| `primary` (default) | `cmw_platform.yaml` |
| `secondary` | `cmw_platform_secondary.yaml` |

---

## Files Modified

### `rag_engine/cmw_platform/config.py`

**New functions/parameters:**
- `DEFAULT_PLATFORM` — env var fallback, defaults to "primary"
- `_config_cache` — per-platform config caching
- `_get_config_path(platform)` — returns path to platform-specific YAML
- All existing functions now accept `platform: str | None = None` parameter
- Backward compatible: `load_cmw_config()` defaults to "primary"

**Functions updated:**
| Function | New Signature |
|----------|--------------|
| `load_cmw_config` | `(platform: str \| None = None)` |
| `load_pipeline_config` | `(platform: str \| None = None)` |
| `get_input_config` | `(platform: str \| None = None)` |
| `get_output_config` | `(platform: str \| None = None)` |
| `get_input_attributes` | `(platform: str \| None = None)` |
| `get_platform_attribute` | `(python_name: str, platform: str \| None = None)` |
| `get_python_attribute` | `(platform_name: str, platform: str \| None = None)` |
| `get_request_template` | `(platform: str \| None = None)` |
| `get_template_config` | `(app: str, template: str, platform: str \| None = None)` |
| `get_attribute_metadata` | `(app: str, template: str, platform: str \| None = None)` |
| `get_attribute_type` | `(app: str, template: str, attribute: str, platform: str \| None = None)` |
| `coerce_attribute_value` | `(app: str, template: str, attribute: str, value: Any, platform: str \| None = None)` |

---

## New File: `rag_engine/config/cmw_platform_secondary.yaml`

```yaml
pipeline:
  input:
    application: ArchitectureManagement
    template: Zaprosinarazrabotky
    attributes:
      document_file: Commerpredloshenie
      user_prompt: prompt
  output:
    summary_attribute: summary
```

---

## Issues/Notes

- **Backward compatibility preserved:** Existing code that calls `load_cmw_config()` without arguments gets "primary" platform
- **Config caching:** Added `_config_cache` dict to avoid re-loading YAML on every call
- **Lint passes:** `ruff check rag_engine/cmw_platform/config.py` — all clean

---

## Next Steps

**Phase 2:** API Refactoring — add `platform` parameter to HTTP client functions (`_load_server_config`, `_get_request`, `_post_request`, `_put_request`)

---

## Verification Commands

```bash
# Run config tests
.venv/bin/python -m pytest rag_engine/tests/test_cmw_platform_config_multi.py -v

# Run all CMW platform tests
.venv/bin/python -m pytest rag_engine/tests/test_cmw_platform.py -v

# Lint
.venv/bin/ruff check rag_engine/cmw_platform/config.py
```