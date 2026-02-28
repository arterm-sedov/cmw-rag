# CMW Platform Integration

**Date:** 2026-02-25

---

## Summary

Implemented Comindware Platform API integration for cmw-rag to enable reading and creating records. Follows battle-tested patterns from cmw-platform-agent with platform-agnostic architecture.

---

## What Was Done

### 1. Created Module Structure

```
rag_engine/cmw_platform/
├── __init__.py          # Exports: create_record, read_record
├── api.py               # HTTP client with Basic Auth
├── models.py            # Pydantic models (HTTPResponse├── config, RequestConfig)
.py            # Template/attribute config loader
└── attribute_types.py   # Immutable type coercion (string, boolean, record, etc.)
```

### 2. Platform Attribute Type System (Immutable, in Code)

`attribute_types.py` - Pydantic-based type coercion for all Comindware platform types:
- `string`, `text`, `document`, `image`, `drawing`
- `record` (references), `role`, `account`, `enum`
- `boolean`, `datetime`, `decimal`, `integer`

### 3. Template Configuration (Dynamic, in YAML)

`config/cmw_platform.yaml` - Defines templates and attribute aliases per app:

```yaml
templates:
  dima:
    TPAIModel:
      attributes:
        name: {type: string}
        title: {type: string}
        user_question: {type: string}
    response:
      attributes:
        request: {type: record, reference_to: TPAIModel}
        exampletext: {type: string}
```

### 4. API Integration

- **Read:** Uses `GetPropertyValues` endpoint for server-side field filtering
- **Create:** Uses `/webapi/Record/Template@{app}.{template}` endpoint
- **Auth:** Basic Auth header on every request
- **Config:** Loads from `.env` (CMW_BASE_URL, CMW_LOGIN, CMW_PASSWORD, CMW_TIMEOUT)

### 5. Testing

- **Unit tests:** 14 tests covering config, auth, API responses, CRUD operations
- **Integration test:** Successfully reads/creates records on actual platform
- **Linting:** Ruff clean

---

## Verified Functionality

| Test | Result |
|------|--------|
| Read record 322393 from TPAIModel | ✅ Returns title: "Сломались формы" |
| Create record in response template | ✅ Created with ID 322420 |
| Type coercion (record references) | ✅ Works |
| Unit tests | ✅ 14/14 passing |
| Ruff check | ✅ Clean |

---

## Key Design Decisions

1. **YAML for dynamic config only** - Template/app names and attribute aliases vary per deployment
2. **Pydantic for immutable types** - Platform attribute types are constant, defined in code
3. **No session management** - Basic Auth on every request (stateless)
4. **Lazy .env loading** - Avoids import-time errors in tests
5. **Follows cmw-platform-agent patterns** - Coercion, endpoint formats, error handling

---

## Files Changed/Created

| File | Status |
|------|--------|
| `rag_engine/cmw_platform/__init__.py` | Created |
| `rag_engine/cmw_platform/api.py` | Created |
| `rag_engine/cmw_platform/models.py` | Created |
| `rag_engine/cmw_platform/config.py` | Created |
| `rag_engine/cmw_platform/attribute_types.py` | Created |
| `rag_engine/config/cmw_platform.yaml` | Created |
| `rag_engine/tests/test_cmw_platform.py` | Created |
| `rag_engine/scripts/test_cmw_platform_integration.py` | Created |
| `rag_engine/config/settings.py` | Modified (added CMW settings) |
| `rag_engine/requirements.txt` | Modified (added requests) |
| `.env-example` | Modified (added CMW variables) |
