# Plan: Multi-Instance CMW Platform Integration

## Goal

Add a second Comindware Platform instance with independent credentials, schema (templates/attributes), and FastAPI endpoint.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           FastAPI Endpoints (app.py)                │
│  POST /api/v1/cmw/process-support-request          │
│  POST /api/v1/cmw2/process-support-request     │
└──────────────────┬────────────────┬──────────────┘
                   │               │
        ┌──────────▼──────┐  ┌───▼─────────────┐
        │   Platform 1    │  │   Platform 2    │
        │   (primary)     │  │   (secondary)   │
        │ .yaml + API    │  │ .yaml + API    │
        └────────┬───────┘  └───────┬─────────┘
                 │                 │
           ┌─────▼───────┐   ┌─────▼───────┐
           │ Connector(1) │   │ Connector(2) │
           │  (thread)  │   │  (thread)   │
           └───────────┘   └────────────┘
```

---

## Phase 1: Config Refactoring (TDD)

### Goal

Refactor `config.py` to support loading platform-specific YAML configs with backward compatibility.

### Tests First

#### Test 1.1: Config loads default platform

**File:** `rag_engine/tests/test_cmw_platform_config_multi.py` (new)

```python
def test_load_cmw_config_default_returns_primary():
    """Backward compat: default platform is 'primary'."""
    config = load_cmw_config()
    assert "pipeline" in config
    assert config["pipeline"]["input"]["application"] == "systemSolution"

def test_load_cmw_config_with_platform_name():
    """Can load config for named platform."""
    config = load_cmw_config("secondary")
    assert "pipeline" in config
```

#### Test 1.2: Platform config functions accept platform param

```python
def test_get_input_config_accepts_platform_param():
    """get_input_config returns platform-specific input config."""
    primary = get_input_config()
    secondary = get_input_config(platform="secondary")
    # Both should return dicts (actual values differ by platform)
    assert isinstance(primary, dict)
    assert isinstance(secondary, dict)

def test_get_input_attributes_accepts_platform_param():
    """get_input_attributes returns platform-specific mapping."""
    primary = get_input_attributes()
    secondary = get_input_attributes(platform="secondary")
    assert isinstance(primary, dict)
    assert isinstance(secondary, dict)
```

#### Test 1.3: Backward compatibility preserved

```python
def test_load_cmw_config_no_args_default():
    """Default call returns primary (backward compat)."""
    config = load_cmw_config()
    assert config["pipeline"]["input"]["application"] == "systemSolution"

def test_get_input_config_no_args_default():
    """Default call uses primary."""
    cfg = get_input_config()
    assert cfg.get("application") == "systemSolution"
```

### Checkpoint 1.1

- [ ] Create test file `rag_engine/tests/test_cmw_platform_config_multi.py`
- [ ] Write 5+ tests for multi-platform config loading
- [ ] Run tests — they should FAIL (no platform support yet)

### Implementation 1.1

**File:** `rag_engine/cmw_platform/config.py`

```python
import os
from pathlib import Path
from typing import Any

import yaml

# ── Platform Selection ─────────────────────────────────────────────────────
DEFAULT_PLATFORM = os.getenv("CMW_PLATFORM_NAME", "primary")

def _get_config_path(platform: str | None = None) -> Path:
    """Get path to platform config YAML."""
    platform = platform or DEFAULT_PLATFORM
    config_dir = Path(__file__).parent.parent / "config"

    if platform == "primary":
        return config_dir / "cmw_platform.yaml"

    return config_dir / f"cmw_platform_{platform}.yaml"


def load_cmw_config(platform: str | None = None) -> dict[str, Any]:
    """Load CMW Platform configuration from YAML.

    Args:
        platform: Platform name (e.g., "primary", "secondary").
                 Defaults to CMW_PLATFORM_NAME env var or "primary".

    Returns:
        Full config dict.
    """
    platform = platform or DEFAULT_PLATFORM
    config_path = _get_config_path(platform)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Cached Configs (per platform) ─────────────────────────────────────────
_config_cache: dict[str, dict[str, Any]] = {}


def _get_cached_config(platform: str) -> dict[str, Any]:
    """Get cached config or load fresh."""
    if platform not in _config_cache:
        _config_cache[platform] = load_cmw_config(platform)
    return _config_cache[platform]


# ── Config Accessors (updated to accept platform) ────────────────────
def load_pipeline_config(platform: str | None = None) -> dict[str, Any]:
    """Load pipeline configuration section from YAML."""
    cfg = _get_cached_config(platform or DEFAULT_PLATFORM)
    return cfg.get("pipeline", {})


def get_input_config(platform: str | None = None) -> dict[str, Any]:
    """Get input configuration (template to fetch from)."""
    pipeline = load_pipeline_config(platform)
    return pipeline.get("input", {})


def get_output_config(platform: str | None = None) -> dict[str, Any]:
    """Get output configuration (template to create in)."""
    pipeline = load_pipeline_config(platform)
    return pipeline.get("output", {})


def get_input_attributes(platform: str | None = None) -> dict[str, str]:
    """Get Python -> Platform attribute mapping."""
    return get_input_config(platform).get("attributes", {})


def get_platform_attribute(python_name: str, platform: str | None = None) -> str | None:
    """Map Python name to platform attribute name."""
    attrs = get_input_attributes(platform)
    return attrs.get(python_name)


def get_python_attribute(platform_name: str, platform: str | None = None) -> str | None:
    """Map platform attribute name to Python name."""
    attrs = get_input_attributes(platform)
    for python_name, platform_name_val in attrs.items():
        if platform_name_val == platform_name:
            return python_name
    return None


def get_request_template(platform: str | None = None) -> str:
    """Get the markdown request template."""
    pipeline = load_pipeline_config(platform)
    return pipeline.get("request_template", "")


# ── Template Config (existing - add platform param) ──────────────────
def get_template_config(app: str, template: str, platform: str | None = None) -> dict[str, Any] | None:
    """Get configuration for a specific template."""
    cfg = _get_cached_config(platform or DEFAULT_PLATFORM)
    return cfg.get("templates", {}).get(app, {}).get(template)


def get_attribute_metadata(
    app: str, template: str, platform: str | None = None
) -> dict[str, AttributeMetadata]:
    """Get full attribute metadata for a template."""
    from rag_engine.cmw_platform.attribute_types import AttributeMetadata

    template_config = get_template_config(app, template, platform)
    if not template_config:
        return {}

    attrs = template_config.get("attributes", {})
    result = {}
    for alias, cfg in attrs.items():
        if isinstance(cfg, str):
            attr_type = cfg
        else:
            attr_type = cfg.get("type", "string") if cfg else "string"

        result[alias] = AttributeMetadata(
            alias=alias,
            type=attr_type,
            is_system=False,
            is_multivalue=False,
        )
    return result
```

### Checkpoint 1.2

- [ ] Run tests — should PASS
- [ ] Run lint: `ruff check rag_engine/cmw_platform/config.py`

---

## Phase 2: API Refactoring (TDD)

### Goal

Refactor `api.py` (HTTP client) to support platform-specific credentials.

### Tests First

#### Test 2.1: API uses platform credentials

**File:** `rag_engine/tests/test_cmw_platform_api_multi.py` (new)

```python
def test_api_loads_primary_credentials():
    """Primary platform uses CMW_BASE_URL etc."""
    config = _load_server_config()
    assert config.base_url == "https://support.comindware.com/"

def test_api_loads_secondary_credentials():
    """Secondary platform uses CMW2_BASE_URL etc."""
    config = load_server_config(platform="secondary")
    assert config.base_url == os.getenv("CMW2_BASE_URL")

def test_api_basic_headers_platform_specific():
    """Headers differ per platform."""
    primary_headers = _basic_headers(platform="primary")
    secondary_headers = _basic_headers(platform="secondary")
    # Credentials differ, so headers differ
    assert primary_headers != secondary_headers
```

### Implementation 2.1

**File:** `rag_engine/cmw_platform/api.py`

```python
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from rag_engine.cmw_platform.models import HTTPResponse, RequestConfig


def _load_env_file(platform: str | None = None) -> None:
    """Load .env file, optionally platform-specific."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if platform and platform != "primary":
        platform_env = Path(__file__).parent.parent.parent / f".env.{platform}"
        if platform_env.exists():
            load_dotenv(platform_env)
            return
    load_dotenv(env_path)


def _load_server_config(platform: str | None = None) -> RequestConfig:
    """Load server configuration for platform from .env file.

    Args:
        platform: Platform name (e.g., "primary", "secondary").
                 Defaults to "primary".
    """
    _load_env_file(platform)
    platform = platform or "primary"

    # Build env var suffix: "" for primary, "2" for secondary (CMW_, CMW2_)
    suffix = "" if platform == "primary" else "2"
    base = f"CMW{suffix}_BASE_URL"
    login = f"CMW{suffix}_LOGIN"
    pw = f"CMW{suffix}_PASSWORD"
    timeout = f"CMW{suffix}_TIMEOUT"

    return RequestConfig(
        base_url=os.getenv(base, ""),
        login=os.getenv(login, ""),
        password=os.getenv(pw, ""),
        timeout=int(os.getenv(timeout, "30")),
    )


def _basic_headers(platform: str | None = None) -> dict[str, str]:
    """Create Basic Auth header for platform."""
    config = _load_server_config(platform)
    credentials = f"{config.login}:{config.password}"
    encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {encoded}"}


# ── HTTP Methods (updated to accept platform) ─────────────────────────────
def _get_request(endpoint: str, platform: str | None = None) -> dict[str, Any]:
    """Make GET request with platform-specific auth."""
    config = _load_server_config(platform)
    url = f"{config.base_url}{endpoint}"
    headers = _basic_headers(platform)

    try:
        response = requests.get(url, headers=headers, timeout=config.timeout)
        # ... (existing error handling)
    except requests.Timeout:
        return {"success": False, "status_code": 408, "error": "Request timeout", "data": None}
    # ...


    return {
        "success": http_response.success,
        "status_code": http_response.status_code,
        "data": http_response.raw_response,
        "error": http_response.error,
    }


def _post_request(body: dict[str, Any], endpoint: str, platform: str | None = None) -> dict[str, Any]:
    """Make POST request with platform-specific auth."""
    config = _load_server_config(platform)
    # ... (similar to existing, with platform param)
```

### Checkpoint 2.1

- [ ] Create test file `rag_engine/tests/test_cmw_platform_api_multi.py`
- [ ] Write tests for platform-specific API
- [ ] Run tests — should PASS
- [ ] Run lint: `ruff check rag_engine/cmw_platform/api.py`

---

## Phase 3: Connector Refactoring (TDD)

### Goal

Make `PlatformConnector` platform-aware.

### Tests First

#### Test 3.1: Connector uses platform config

**File:** `rag_engine/tests/test_cmw_platform_connector_multi.py` (new)

```python
def test_connector_accepts_platform_param():
    """Connector can be created with platform name."""
    conn = PlatformConnector(platform="secondary")
    assert conn.platform == "secondary"

def test_connector_default_is_primary():
    """Default platform is primary."""
    conn = PlatformConnector()
    assert conn.platform == "primary"

def test_connector_uses_correct_config(requests_mock):
    """Connector fetches using correct platform config."""
    conn = PlatformConnector(platform="secondary")
    # Verify it calls config with platform param
    with patch("rag_engine.cmw_platform.config.get_input_config") as mock_cfg:
        mock_cfg.return_value = {"application": "systemSolution", "template": "Requests"}
        conn.start_request("123")
        mock_cfg.assert_called_with(platform="secondary")
```

### Implementation 3.1

**File:** `rag_engine/cmw_platform/connector.py`

```python
from dataclasses import dataclass

from rag_engine.cmw_platform import config, records


@dataclass
class ProcessResult:
    """Result of a platform request processing operation."""
    success: bool
    message: str | None = None
    error: str | None = None


class PlatformConnector:
    """Orchestrates CMW Platform request → response pipeline.

    Args:
        platform: Platform name (e.g., "primary", "secondary").
                 Defaults to "primary".
    """

    def __init__(self, platform: str = "primary"):
        self.platform = platform = platform or "primary"

    def start_request(self, request_id: str) -> ProcessResult:
        """Start processing a request through the RAG pipeline."""
        input_config = config.get_input_config(platform=self.platform)
        # ...
```

### Checkpoint 3.1

- [ ] Create test file `rag_engine/tests/test_cmw_platform_connector_multi.py`
- [ ] Write tests for platform-aware connector
- [ ] Run tests
- [ ] Run lint

---

## Phase 4: Second Platform Document Processing (Core)

### New Platform Schema

Second platform instance with document summarization workflow:

| Field | Type | Description |
|-------|------|-------------|
| `Commerpredloshenie` | Document | Input: PDF/Word/Excel file attachment |
| `prompt` | Text | User prompt/instructions for summarization |
| `summary` | Text | Output: AI-generated summary |

#### Workflow

```
1. Read record → get document_id from "Commerpredloshenie" attribute
2. GET /webapi/Document/{documentId}/Content → base64 content
3. Decode base64 → detect file type → process
4. If PDF: PyMuPDF4LLM → markdown
5. LLM: summarize with {prompt}
6. Write summary → "summary" attribute
```

### Phase 4a: Document API Methods

**File:** `rag_engine/cmw_platform/document_api.py` (new)

#### Tests First

```python
def test_get_document_content_returns_base64():
    """Document API returns base64-encoded content."""
    result = get_document_content("doc-id-123", platform="secondary")
    assert result["success"] is True
    assert "content" in result  # base64 string
    assert result.get("mime_type") in ["application/pdf", ...]

def test_decode_base64_to_file():
    """Decode base64 to actual file bytes."""
    b64 = "JVBERi0xLjQK..."
    data = decode_base64_content(b64)
    assert data.startswith(b"%PDF")  # PDF magic bytes
```

#### Implementation

```python
def get_document_content(document_id: str, platform: str | None = None) -> dict:
    """Fetch document content from CMW Platform.

    Step 1: GET /webapi/Document/{documentId}/Content
    Step 2: Returns {"success": bool, "content": base64_string, "mime_type": str, "filename": str}

    Args:
        document_id: Document ID from document attribute value
        platform: Platform name (default: "secondary")

    Returns:
        Dict with base64 content and metadata
    """
    endpoint = f"/webapi/Document/{document_id}/Content"
    response = _get_request(endpoint, platform=platform)

    if not response["success"]:
        return {"success": False, "error": response.get("error")}

    raw = response.get("data", {})
    return {
        "success": True,
        "content": raw.get("content"),  # base64 encoded
        "mime_type": raw.get("mimeType") or raw.get("contentType"),
        "filename": raw.get("fileName"),
    }
```

### Phase 4b: Document Processor

**File:** `rag_engine/cmw_platform/document_processor.py` (new)

#### Tests First

```python
def test_process_pdf_document():
    """Process PDF document and extract text."""
    pdf_base64 = ...  # valid PDF in base64
    result = process_document(pdf_base64, mime_type="application/pdf")
    assert result["success"] is True
    assert "text" in result
    assert len(result["text"]) > 100

def test_process_word_document():
    """Process Word document."""
    docx_base64 = ...  # valid DOCX
    result = process_document(docx_base64, mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert result["success"] is True
    assert "text" in result
```

#### Implementation

```python
import base64
from typing import Optional

# Supported MIME types → extension mapping
MIME_TO_EXTENSION = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/zip": ".zip",
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


def decode_base64_content(base64_string: str) -> bytes:
    """Decode base64 string to bytes."""
    return base64.b64decode(base64_string)


def detect_mime_type(data: bytes) -> str:
    """Detect MIME type from file magic bytes."""
    magic_signatures = {
        b"%PDF": "application/pdf",
        b"PK\x03\x04": "application/vnd.openxmlformats-officedocument",  # DOCX/XLSX/ZIP
        b"\xd0\xcf\x11\xe0": "application/msword",  # DOC/XLS old format
    }
    for sig, mime in magic_signatures.items():
        if data.startswith(sig):
            return mime
    return "application/octet-stream"


def process_document(
    base64_content: str,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> dict:
    """Process document and extract text content.

    Supports: PDF, DOCX, XLSX, ZIP, images

    Args:
        base64_content: Base64-encoded document content
        mime_type: MIME type hint (detected from magic bytes if not provided)
        filename: Original filename for logging

    Returns:
        Dict: {"success": bool, "text": str, "page_count": int, "file_type": str, "error": str}
    """
    try:
        data = decode_base64_content(base64_content)
    except Exception as e:
        return {"success": False, "error": f"Failed to decode base64: {e}"}

    # Detect MIME type from content if not provided
    detected_mime = mime_type or detect_mime_type(data)

    # Route to appropriate processor
    if detected_mime == "application/pdf":
        return _process_pdf(data)
    elif "word" in detected_mime or "document" in detected_mime:
        return _process_docx(data)
    elif "sheet" in detected_mime or "excel" in detected_mime:
        return _process_xlsx(data)
    elif detected_mime == "application/zip":
        return _process_zip(data)
    elif detected_mime.startswith("image/"):
        return _process_image(data)

    return {"success": False, "error": f"Unsupported file type: {detected_mime}"}


def _process_pdf(data: bytes) -> dict:
    """Process PDF using PyMuPDF4LLM."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data)
        temp_path = f.name

    try:
        # PyMuPDF4LLM → markdown
        import pymupdf4llm

        md = pymupdf4llm.to_markdown(
            temp_path,
            ignore_images=True,
            ignore_graphics=True,
            page_chunks=False,
        )
        return {
            "success": True,
            "text": md,
            "page_count": 1,  # PyMuPDF processes all pages
            "file_type": "pdf",
        }
    except ImportError:
        return {"success": False, "error": "PyMuPDF4LLM not installed"}
    except Exception as e:
        return {"success": False, "error": f"PDF processing failed: {e}"}
    finally:
        os.unlink(temp_path)


def _process_docx(data: bytes) -> dict:
    """Process Word document."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(data)
        temp_path = f.name

    try:
        from docx import Document

        doc = Document(temp_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return {"success": True, "text": text, "page_count": 1, "file_type": "docx"}
    except ImportError:
        return {"success": False, "error": "python-docx not installed"}
    except Exception as e:
        return {"success": False, "error": f"DOCX processing failed: {e}"}
    finally:
        os.unlink(temp_path)


def _process_xlsx(data: bytes) -> dict:
    """Process Excel spreadsheet."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        f.write(data)
        temp_path = f.name

    try:
        import openpyxl

        wb = openpyxl.load_workbook(temp_path, data_only=True)
        sheets_text = []
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            rows = [str(cell.value or "") for row in ws.iter_rows() for cell in row]
            sheets_text.append(f"=== {sheet} ===\n" + "\n".join(rows))
        return {
            "success": True,
            "text": "\n\n".join(sheets_text),
            "page_count": len(wb.sheetnames),
            "file_type": "xlsx",
        }
    except ImportError:
        return {"success": False, "error": "openpyxl not installed"}
    except Exception as e:
        return {"success": False, "error": f"XLSX processing failed: {e}"}
    finally:
        os.unlink(temp_path)


def _process_zip(data: bytes) -> dict:
    """Process ZIP archive - extract file list only."""
    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
        return {"success": True, "text": "\n".join(names), "page_count": 0, "file_type": "zip"}
    except Exception as e:
        return {"success": False, "error": f"ZIP processing failed: {e}"}


def _process_image(data: bytes) -> dict:
    """Process image - extract basic info."""
    return {"success": True, "text": f"[Image: {len(data)} bytes]", "page_count": 1, "file_type": "image"}
```

### Phase 4c: Second Platform Config

**File:** `rag_engine/config/cmw_platform_secondary.yaml`

```yaml
# CMW Platform Secondary Instance - Document Summarization
# Application: ArchitectureManagement
# Template: Zaprosinarazrabotky (Запрос в разработку / Development Request)

pipeline:
  input:
    application: ArchitectureManagement
    template: Zaprosinarazrabotky
    attributes:
      # Input attributes
      document_file: Commerpredloshenie  # Document attribute - contains document_id
      user_prompt: prompt         # Text attribute - user instructions

  request_template: |
    # Document Summary Request

    Document file attached: {document_filename}
    User prompt: {user_prompt}

    ---

    Please summarize the document according to the user's request.

  output:
    application: ArchitectureManagement
    template: Zaprosinarazrabotky
    # Write summary back to same record
    summary_attribute: summary

# Template schema
templates:
  ArchitectureManagement:
    Zaprosinarazrabotky:
      attributes:
        document_file:
          type: document
          from_input: Commerpredloshenie
        user_prompt:
          type: text
          from_input: prompt
        summary:
          type: text
          from_agent: summary_result
          is_output: true
```

### Phase 4d: Document Summarization Connector

**File:** `rag_engine/cmw_platform/summary_connector.py` (new)

```python
class DocumentSummaryConnector:
    """Process document through LLM for summarization."""

    def __init__(self, platform: str = "secondary"):
        self.platform = platform

    def process(self, record_id: str) -> ProcessResult:
        """Process document: fetch → extract → summarize → write back."""
        # 1. Read record to get document_id and prompt
        record = read_record(record_id, fields=["Commerpredloshenie", "prompt"], platform=self.platform)
        document_id = record["data"].get(record_id, {}).get("Commerpredloshenie", {}).get("id")
        user_prompt = record["data"].get(record_id, {}).get("prompt", "")

        if not document_id:
            return ProcessResult(success=False, error="No document attached")

        # 2. Fetch document content
        doc_result = get_document_content(document_id, platform=self.platform)
        if not doc_result["success"]:
            return ProcessResult(success=False, error=doc_result.get("error"))

        # 3. Extract text from document
        text_result = process_document(
            doc_result["content"],
            mime_type=doc_result.get("mime_type"),
            filename=doc_result.get("filename"),
        )
        if not text_result["success"]:
            return ProcessResult(success=False, error=text_result.get("error"))

        # 4. Summarize with LLM
        summary = self._summarize(text_result["text"], user_prompt)

        # 5. Write summary back to record
        write_result = update_record(
            record_id=record_id,
            values={"summary": summary},
            platform=self.platform,
        )

        return ProcessResult(
            success=write_result["success"],
            message=f"Summary generated for {doc_result.get('filename')}",
        )

    def _summarize(self, text: str, user_prompt: str) -> str:
        """Call LLM to summarize text."""
        # Use existing LLM infrastructure
        from rag_engine.llm.llm_manager import LLMManager

        llm = LLMManager(provider="openrouter", model="qwen/qwen3.5-27b")
        prompt = f"Summarize the following document.\nUser request: {user_prompt}\n\nDocument:\n{text[:50000]}"
        result = llm.invoke(prompt)
        return result.content
```

---

**Phase 4 Summary:**

| Component | File | Tests |
|-----------|------|-------|
| Document API | `document_api.py` | 2+ tests |
| Document processor | `document_processor.py` | 4+ tests |
| Config YAML | `cmw_platform_secondary.yaml` | - |
| Summary connector | `summary_connector.py` | 2+ tests |
```

---

## Phase 5: Environment Variables (Both Platforms)

### File: `.env`

The current primary platform vars and new secondary platform vars:

```bash
# =============================================================================
# CMW PLATFORM PRIMARY INSTANCE (Default)
# =============================================================================
# Comindware Platform API credentials (required for platform integration)
CMW_BASE_URL=http://10.9.0.188:9999/   # or https://support.comindware.com/
CMW_LOGIN=monkey
CMW_PASSWORD=C0m1ndw4r3Pl@tf0rm
CMW_TIMEOUT=30
# API key for CMW Platform webhook endpoints (optional - empty = skip auth, present = require it)
CMW_API_KEY=336b119bf68dfe29898fc812bf52b5ffd1f8dd6963a897046a28aedcc3e82cd5

# =============================================================================
# CMW PLATFORM SECONDARY INSTANCE
# =============================================================================
# Comindware Platform API credentials (second instance, different schema)
CMW2_BASE_URL=https://second-instance.comindware.com/
CMW2_LOGIN=aiassintentcomindware
CMW2_PASSWORD=<password>
CMW2_TIMEOUT=30
# API key for CMW2 Platform webhook endpoints (optional)
CMW2_API_KEY=<key>
```

### Environment Variable Mapping

| Primary | Secondary | Description |
|---------|-----------|-------------|
| `CMW_BASE_URL` | `CMW2_BASE_URL` | Platform instance URL |
| `CMW_LOGIN` | `CMW2_LOGIN` | Username |
| `CMW_PASSWORD` | `CMW2_PASSWORD` | Password |
| `CMW_TIMEOUT` | `CMW2_TIMEOUT` | Request timeout in seconds |
| `CMW_API_KEY` | `CMW2_API_KEY` | Webhook API key (optional) |

**Note:** Primary platform vars use prefix `CMW_` (no number), secondary uses `CMW2_` — this matches the existing `.env` format.

### Tests First (Phase 2 - API)

**Already covered in Phase 2 tests:**

```python
def test_api_loads_primary_credentials():
    """Primary platform uses CMW_BASE_URL etc."""
    config = _load_server_config()
    assert config.base_url == "http://10.9.0.188:9999/"

def test_api_loads_secondary_credentials():
    """Secondary platform uses CMW2_BASE_URL etc."""
    config = _load_server_config(platform="secondary")
    assert "CMW2" in config.base_url or config.base_url ""
```

#### Test 5.1: Primary env vars load correctly

```python
def test_load_primary_env_vars():
    """Primary platform vars are loaded."""
    os.environ["CMW_BASE_URL"] = "http://test.local/"
    os.environ["CMW_LOGIN"] = "testuser"
    os.environ["CMW_PASSWORD"] = "testpass"

    config = _load_server_config()
    assert config.base_url == "http://test.local/"
    assert config.login == "testuser"
    assert config.password == "testpass"
```

#### Test 5.2: Secondary env vars load correctly

```python
def test_load_secondary_env_vars():
    """Secondary platform vars are loaded."""
    os.environ["CMW2_BASE_URL"] = "http://secondary.local/"
    os.environ["CMW2_LOGIN"] = "secondary_user"
    os.environ["CMW2_PASSWORD"] = "secondary_pass"

    config = _load_server_config(platform="secondary")
    assert config.base_url == "http://secondary.local/"
    assert config.login == "secondary_user"
    assert config.password == "secondary_pass"
```

#### Test 5.3: Env file loading per platform

```python
def test_env_file_loading_primary():
    """Primary uses default .env file."""
    _load_env_file("primary")
    assert os.getenv("CMW_BASE_URL") is not None

def test_env_file_loading_secondary():
    """Secondary can use .env.secondary if present."""
    # Create temp .env.secondary for test
    with patch("pathlib.Path.exists", return_value=True):
        with patch("dotenv.load_dotenv") as mock_load:
            _load_env_file("secondary")
            mock_load.assert_called()
```

---

## Phase 6b: Document Summarization Endpoint (TDD)

### New Endpoint for Document Summarization

Simple endpoint that receives request_id and processes the attached document:

```python
POST /api/v1/cmw/summarize-document
Body: {"request_id": "record-id-123"}
Response: {"success": bool, "summary": str, "error": str}
```

### Tests First

```python
def test_summarize_document_endpoint_exists():
    """POST /api/v1/cmw/summarize-document returns 200."""
    response = client.post(
        "/api/v1/cmw/summarize-document",
        json={"request_id": "123"},
    )
    assert response.status_code in (200, 400, 500)

def test_summarize_document_calls_connector():
    """Endpoint uses DocumentSummaryConnector."""
    with patch("rag_engine.api.app.DocumentSummaryConnector") as mock:
        mock.return_value.process.return_value = ProcessResult(
            success=True, message="Summary generated"
        )
        response = client.post("/api/v1/cmw/summarize-document", json={"request_id": "123"})
        mock.assert_called_once()
```

### Implementation

```python
class SummarizeDocumentRequest(BaseModel):
    request_id: str


@fastapi_app.post("/api/v1/cmw/summarize-document")
async def summarize_document_endpoint(
    req: SummarizeDocumentRequest,
    http_req: Request,
) -> dict:
    """Summarize document attached to a CMW Platform record.

    Gets document from Document attribute, extracts text,
    and generates summary using LLM.

    Args:
        request_id: Record ID in ArchitectureManagement.Zaprosinarazrabotky

    Returns:
        {"success": bool, "summary": str, "message": str, "error": str}
    """
    from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector

    try:
        connector = DocumentSummaryConnector(platform="secondary")
        result = connector.process(req.request_id)

        return {
            "success": result.success,
            "summary": result.summary if hasattr(result, "summary") else None,
            "message": result.message,
            "error": result.error,
        }
    except Exception as e:
        logger.exception("Document summarization failed")
        return {"success": False, "error": str(e)}
```

### Alternative: Query Parameter for Platform Selection

If you want to use query param instead of separate endpoint:

```python
# Single endpoint with platform selection
POST /api/v1/cmw/process?mode=summarize
# or
POST /api/v1/cmw/summarize?platform=secondary
```

---

## Phase 6: Second FastAPI Endpoint (TDD)

### Tests First

#### Test 6.1: Second endpoint exists

**File:** `rag_engine/tests/test_api_app.py` (extend)

```python
def test_cmw2_endpoint_exists(client):
    """POST /api/v1/cmw2/process-support-request returns 200."""
    response = client.post(
        "/api/v1/cmw2/process-support-request",
        json={"request_id": "123"},
    )
    assert response.status_code in (200, 404, 500)  # 404 if platform not configured

def test_cmw2_endpoint_passes_platform_to_connector():
    """Second endpoint uses secondary platform."""
    with patch("rag_engine.api.app.PlatformConnector") as mock_conn:
        mock_conn.return_value.start_request.return_value = ProcessResult(
            success=True, message="ok"
        )
        response = client.post(
            "/api/v1/cmw2/process-support-request",
            json={"request_id": "123"},
        )
        mock_conn.assert_called_with(platform="secondary")
```

### Implementation 6.1

**File:** `rag_engine/api/app.py`

```python
from rag_engine.cmw_platform import PlatformConnector

# ... (existing endpoint at ~line 4260)

class ProcessSupportRequest(BaseModel):
    request_id: str


@fastapi_app.post("/api/v1/cmw/process-support-request")
async def cmw_endpoint(req: ProcessSupportRequest, http_req: Request) -> dict:
    """Process CMW Platform (primary) support request."""
    return _process_cmw_request(req.request_id, platform="primary")


@fastapi_app.post("/api/v1/cmw2/process-support-request")
async def cmw2_endpoint(req: ProcessSupportRequest, http_req: Request) -> dict:
    """Process CMW Platform (secondary) support request."""
    return _process_cmw_request(req.request_id, platform="secondary")


def _process_cmw_request(request_id: str, platform: str) -> dict:
    """Process CMW request for specified platform."""
    connector = PlatformConnector(platform=platform)
    result = connector.start_request(request_id)
    return {
        "success": result.success,
        "message": result.message,
        "error": result.error,
    }
```

### Checkpoint 6.1

- [ ] Write/update tests for second endpoint
- [ ] Run tests
- [ ] Run lint

---

## Phase 7: Verification & Progress Tracking

### Verification Commands

```bash
# Run all platform tests
pytest rag_engine/tests/test_cmw_platform*.py -v

# Run API tests
pytest rag_engine/tests/test_api_app.py -k cmw -v

# Run lint
ruff check rag_engine/cmw_platform/
ruff check rag_engine/api/app.py
```

### Progress Files

Create progress report after each checkpoint:

| File | When |
|------|------|
| `docs/progress_reports/20260420_cmw_multi_phase1.md` | After Phase 1 |
| `docs/progress_reports/20260420_cmw_multi_phase2.md` | After Phase 2 |
| `docs/progress_reports/20260420_cmw_multi_phase3.md` | After Phase 3 |
| `docs/progress_reports/20260420_cmw_multi_phase4.md` | After Phase 4 |
| `docs/progress_reports/20260420_cmw_multi_phase5.md` | After Phase 5 |
| `docs/progress_reports/20260420_cmw_multi_phase6.md` | After Phase 6 |

### Progress Template

Each `.md` file:

```markdown
# Phase N: [Title]

## Date: YYYY-MM-DD HH:MM

## Checkpoint Status

- [x] Task description
- [ ] Pending task

## Tests Added/Updated

- [ ] `rag_engine/tests/test_cmw_platform_*.py`
- New tests: N
- Passing: M

## Code Changes

| File | Lines | Description |
|------|-------|-------------|
| `path/to/file` | +N/-M | Change |

## Issues/Notes

- Issue description and resolution

## Next Steps

- Next phase task list
```

---

## Summary

### Core Integration (Multi-Platform Support)

| Phase | Focus | Tests | Files Modified |
|-------|-------|-------|---------------|
| 1 | Config refactor (platform param) | 5+ | `config.py` |
| 2 | API refactor (platform credentials) | 4+ | `api.py` |
| 3 | Connector refactor (platform param) | 3+ | `connector.py` |
| 5 | Environment vars | 3+ | `.env` (add CMW2_* vars) |
| 7 | Verification | - | All + progress files |

### Document Processing (New Endpoint)

| Phase | Focus | Tests | Files Modified |
|-------|-------|-------|---------------|
| 4a | Document API (get content) | 2+ | `document_api.py` (new) |
| 4b | Document processor (PDF/DOCX/XLSX) | 4+ | `document_processor.py` (new) |
| 4c | Secondary platform config | 0 | `cmw_platform_secondary.yaml` (new) |
| 4d | Summary connector | 2+ | `summary_connector.py` (new) |
| 6b | Summarization endpoint | 2+ | `app.py` |

## Timeline

- Phase 1 (Config): ~30 min
- Phase 2 (API): ~20 min
- Phase 3 (Connector): ~20 min
- Phase 4a-d (Document): ~60 min
- Phase 5 (Env vars): ~10 min
- Phase 6b (Endpoint): ~15 min
- Phase 7 (Verification): ~15 min

**Total: ~170 min**

---

## Document Processing Workflow Summary

```
Record ID → Read Record → Get document_id from "Commerpredloshenie"
                         ↓
        GET /webapi/Document/{documentId}/Content
                         ↓
        Base64 Content → Decode → Detect MIME type
                         ↓
        Route to processor:
          - PDF → PyMuPDF4LLM → Markdown
          - DOCX → python-docx → Text
          - XLSX → openpyxl → Text
          - ZIP → List files
                         ↓
        LLM (Qwen3.5) → Summarize with prompt
                         ↓
        Write "summary" → Update record
```

## Environment Variables

```bash
# PRIMARY (current support assistant)
CMW_BASE_URL=http://10.9.0.188:9999/
CMW_LOGIN=monkey
CMW_PASSWORD=C0m1ndw4r3Pl@tf0rm

# SECONDARY (document summarization)
CMW2_BASE_URL=https://...
CMW2_LOGIN=...
CMW2_PASSWORD=...
```