# Progress Report: Unified `read_file` Tool Implementation

**Date:** 2026-04-20
**Task:** Bring Text/PDF reading and parsing utilities from cmw-platform-agent to cmw-rag

---

## Summary

Successfully implemented a unified `read_file` LangChain tool that auto-detects file type and handles both text files and PDFs. The tool is now available in cmw-rag without being bound to any agent yet.

---

## Background

The cmw-platform-agent had PDF and text file reading utilities that were tightly coupled with CMW platform operations. The goal was to extract these utilities and bring them to cmw-rag in a modular way.

---

## Implementation

### Files Created

| File | Lines | Description |
|------|-------|-------------|
| `rag_engine/tools/pdf_utils.py` | 104 | PDF extraction using PyMuPDF4LLM |
| `rag_engine/tools/file_utils.py` | 253 | Text file reading with encoding fallback, URL download |
| `rag_engine/tools/read_file.py` | 121 | Unified LangChain `@tool` - auto-detects text vs PDF |
| `rag_engine/tests/test_tools_read_file.py` | 144 | 11 comprehensive tests |
| `rag_engine/tests/fixtures/` | - | Test fixtures (test.txt, test.md, test.json, sample.pdf) |

### Files Modified

| File | Change |
|------|--------|
| `rag_engine/requirements.txt` | Added `pymupdf4llm>=0.0.1` |
| `rag_engine/tools/__init__.py` | Added `read_file` export |

---

## Features

- **Auto-detection:** Automatically detects file type by extension
- **Text files:** `.txt`, `.md`, `.json`, `.py`, `.html`, `.yaml`, `.yml`, etc.
- **PDF files:** Extracts text as Markdown using PyMuPDF4LLM
- **URL support:** Downloads files from URLs automatically
- **Encoding fallback:** Tries UTF-8 → Latin-1 → CP1252 for text files
- **JSON output:** Returns standardized JSON with content + metadata

---

## Test Results

```
11 passed in 11.51s
```

### Test Coverage

- ✅ Read .txt file
- ✅ Read .md file
- ✅ Read .json file
- ✅ Read .pdf file (extracts Markdown)
- ✅ Missing file error handling
- ✅ Unsupported file type error handling
- ✅ Text file detection
- ✅ Encoding fallback
- ✅ PDF utils availability check
- ✅ PDF file detection
- ✅ PDF text extraction

---

## Usage

```python
from rag_engine.tools import read_file

# Read local file
result = read_file.invoke({"file_reference": "document.pdf"})

# Read from URL
result = read_file.invoke({"file_reference": "https://example.com/file.txt"})
```

### Output Format

```json
{
  "type": "tool_response",
  "tool_name": "read_file",
  "result": "File: document.pdf (100 KB)\n\nContent:\n# Markdown extracted...",
  "file_info": {
    "exists": true,
    "name": "document.pdf",
    "size": 102400,
    "extension": ".pdf"
  }
}
```

---

## Verification

- [x] Lint passes: `ruff check rag_engine/tools/`
- [x] All tests pass
- [x] PDF extraction works with real PDF files
- [x] Text file reading with encoding fallback works
- [x] URL download works

---

## Next Steps (Not Bound Yet)

The tool is ready but not yet bound to any agent. To bind it:

1. Import `read_file` in your agent configuration
2. Add to agent's tool list
3. Agent can then read PDF/text files upon user request

---

## Dependencies

- `pymupdf4llm>=0.0.1` - For PDF text extraction to Markdown

---

## References

- Source: `D:/Repo/cmw-platform-agent/tools/pdf_utils.py`
- Source: `D:/Repo/cmw-platform-agent/tools/file_utils.py`
- Source: `D:/Repo/cmw-platform-agent/tools/tools.py` (lines 797-869, `read_text_based_file`)
