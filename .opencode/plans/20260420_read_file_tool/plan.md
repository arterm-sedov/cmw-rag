# Plan: Unified `read_file` Tool for cmw-rag

## Goal

Create a single LangChain tool that automatically detects file type and handles both text files and PDFs (with URL support).

---

## Tasks

### Phase 1: Dependencies

- [ ] **1.1** Add `pymupdf4llm>=0.0.1` to `rag_engine/requirements.txt`
- [ ] **1.2** Run `pip install -r rag_engine/requirements.txt` to install new dependency

---

### Phase 2: PDF Utilities (pdf_utils.py)

- [ ] **2.1** Create `rag_engine/tools/pdf_utils.py` (~80 lines)
- [ ] **2.2** Implement `PDFTextResult` Pydantic model:
  - `success: bool`
  - `text_content: str`
  - `page_count: int`
  - `error_message: Optional[str]`
- [ ] **2.3** Implement `PDFUtils` class:
  - `is_pdf_file(file_path: str) -> bool` - validate via PDF header `%PDF`
  - `extract_text_from_pdf(file_path: str) -> PDFTextResult` - extract as Markdown using PyMuPDF4LLM
  - `is_available() -> bool` - check if PyMuPDF4LLM is installed
- [ ] **2.4** Add convenience functions: `extract_pdf_text()`, `is_pdf_file()`

---

### Phase 3: File Utilities (file_utils.py)

- [ ] **3.1** Create `rag_engine/tools/file_utils.py` (~150 lines)
- [ ] **3.2** Implement Pydantic models:
  - `FileInfo`: exists, path, name, size, extension, error
  - `TextFileResult`: success, content, encoding, file_info, error
- [ ] **3.3** Implement `FileUtils` class:
  - `file_exists(file_path: str) -> bool`
  - `get_file_size(file_path: str) -> int`
  - `get_file_info(file_path: str) -> FileInfo`
  - `read_text_file(file_path: str, encodings: list) -> TextFileResult` - with encoding fallback (utf-8 → latin-1 → cp1252 → iso-8859-1)
  - `is_text_file(file_path: str) -> bool` - extension detection
  - `format_file_size(size_bytes: int) -> str` - human-readable
  - `download_file_to_path(url: str, target_path: str) -> str` - URL download with smart extension detection

---

### Phase 4: Unified Tool (read_file.py)

- [ ] **4.1** Create `rag_engine/tools/read_file.py` (~80 lines)
- [ ] **4.2** Import dependencies:
  - `from langchain.tools import tool`
  - `from pydantic import BaseModel, Field`
- [ ] **4.3** Implement `ReadFileSchema` Pydantic model (input schema):
  - `file_reference: str` - filename or URL
- [ ] **4.4** Implement `@tool read_file` function:
  - **Step 1:** Resolve file reference via `FileUtils.resolve_file_reference()` (handles local + URL)
  - **Step 2:** Get file info via `FileUtils.get_file_info()`
  - **Step 3:** Auto-detect type via extension:
    - If `.pdf` → use `PDFUtils.extract_text_from_pdf()`
    - If text file → use `FileUtils.read_text_file()`
  - **Step 4:** Return JSON with tool response
- [ ] **4.5** Return format (JSON):
  ```json
  {
    "type": "tool_response",
    "tool_name": "read_file",
    "result": "File: {name} ({size})\n\nContent:\n{content}",
    "file_info": {
      "exists": true,
      "name": "document.pdf",
      "size": 102400,
      "extension": ".pdf"
    }
  }
  ```

---

### Phase 5: Integration

- [ ] **5.1** Update `rag_engine/tools/__init__.py`:
  - Add `from rag_engine.tools.read_file import read_file`
  - Add `read_file` to `__all__`

---

### Phase 6: Testing

- [ ] **6.1** Download test PDF:
  ```bash
  curl -L -o rag_engine/tests/fixtures/sample.pdf "https://file-examples.com/storage/fef2d12481c6a002ce00cf92/2018/05/file-example_100kb.pdf"
  ```
- [ ] **6.2** Create test files:
  - `rag_engine/tests/fixtures/test.txt` - plain text
  - `rag_engine/tests/fixtures/test.md` - markdown
  - `rag_engine/tests/fixtures/test.json` - JSON
- [ ] **6.3** Create `rag_engine/tests/test_tools_read_file.py`:
  - [ ] **Test 1:** `test_read_text_file_txt` - read .txt file
  - [ ] **Test 2:** `test_read_text_file_md` - read .md file
  - [ ] **Test 3:** `test_read_text_file_json` - read .json file
  - [ ] **Test 4:** `test_read_pdf_file` - read .pdf file (real PDF)
  - [ ] **Test 5:** `test_read_missing_file` - error handling for missing file
  - [ ] **Test 6:** `test_read_unsupported_file` - error handling for unsupported type (e.g., .exe)
  - [ ] **Test 7:** `test_read_file_from_url` - download and read from URL

---

### Phase 7: Verification

- [ ] **7.1** Run lint: `ruff check rag_engine/tools/`
- [ ] **7.2** Run tests: `pytest rag_engine/tests/test_tools_read_file.py -v`
- [ ] **7.3** Verify all endpoints still work (non-breaking check)

---

## Order of Execution (Sequential Within Phase)

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7
  ↓
[1.1, 1.2]
         ↓
[2.1→2.4] → [3.1→3.3] → [4.1→4.5] → [5.1] → [6.1→6.3] → [7.1→7.3]
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|---------|-----------|
| PDF library fails | Low | High | Graceful fallback to error message |
| URL download fails | Medium | Medium | Test with real URL, handle exceptions |
| Encoding issues | Medium | Low | Multiple encoding fallback |

---

## References

- Source (cmw-platform-agent):
  - `D:/Repo/cmw-platform-agent/tools/pdf_utils.py`
  - `D:/Repo/cmw-platform-agent/tools/file_utils.py`
  - `D:/Repo/cmw-platform-agent/tools/tools.py` (lines 797-869, `read_text_based_file`)

---

## Success Criteria

1. Tool reads text files (.txt, .md, .json, .py, .html, .yaml, etc.)
2. Tool reads PDF files and extracts as Markdown
3. Tool downloads files from URLs
4. Returns JSON with content + metadata
5. Handles errors gracefully
6. All tests pass
7. Lint passes