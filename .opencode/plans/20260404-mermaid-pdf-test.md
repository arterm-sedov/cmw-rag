# План: Mermaid Rendering Test for PDF

**Дата:** 2026-04-04
**Цель:** Определить оптимальный подход рендеринга Mermaid-диаграмм в PDF через `with-pdf` (WeasyPrint).

---

## Final Verdict

**Use `mkdocs-mermaid-to-svg` + `mmdc`.**

- ✅ **Works** — tested, produces crisp SVG diagrams in PDF
- 📦 **Dependencies:** `pip install mkdocs-mermaid-to-svg` + `npm install -g @mermaid-js/mermaid-cli`
- 🚫 **`render_js: true` does NOT work** — bug in `mkdocs-with-pdf` v0.9.3 (incompatible with modern BeautifulSoup)
- 📝 **Documented in:** `cbap-mkdocs-ru/readme.md` and `install/requirements.txt`

---

## Context

- `cbap-mkdocs-ru` использует `with-pdf` (orzih) — движок WeasyPrint, **без JavaScript**
- Mermaid-диаграммы рендерятся через JS в браузере, поэтому WeasyPrint их **не видит**
- Нужно решение, которое конвертирует Mermaid в статическое изображение до PDF-генерации

---

## Test Results

### Environment

- **Python:** 3.13.2 (cbap-mkdocs-ru venv)
- **MkDocs:** 1.6.1
- **WeasyPrint:** 68.1
- **BeautifulSoup4:** 4.14.3
- **Node.js:** v18.20.7
- **mmdc:** 11.12.0
- **Chrome:** installed at `C:\Program Files\Google\Chrome\Application\chrome.exe`

### Approach A: `mkdocs-mermaid-to-image` plugin

**Result: ❌ FAILED** — Package not available on PyPI for Python 3.13.
```
ERROR: No matching distribution found for mkdocs-mermaid-to-image
```

### Approach B: `mkdocs-mermaid-to-svg` plugin + `mmdc`

**Result: ✅ SUCCESS** — Works perfectly.

**Setup required:**
```bash
pip install mkdocs-mermaid-to-svg
npm install -g @mermaid-js/mermaid-cli
```

**Config:**
```yaml
plugins:
  mermaid-to-svg:
    output_dir: _mermaid_assets
```

**Test build result:**
- PDF generated successfully: `test-mermaid-render.pdf` (2.7 MB)
- Mermaid diagram converted to SVG (14.9 KB)
- SVG embedded in HTML: `<img alt="Mermaid Diagram" src="../_mermaid_assets/...svg" />`
- WeasyPrint included SVG in final PDF
- Build time: 36 seconds (including Mermaid conversion)

**Pros:**
- SVG output — infinite scaling, crisp at any resolution
- Automatic conversion during MkDocs build
- Clean integration with existing pipeline

**Cons:**
- Requires Node.js + `mmdc` (global npm install)
- Slightly slower build due to external process

### Approach C: `with-pdf` `render_js: true` flag

**Result: ❌ FAILED** — Bug in `mkdocs-with-pdf` v0.9.3.

**Config tested:**
```yaml
plugins:
  with-pdf:
    render_js: true
    headless_chrome_path: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
```

**Error:**
```
AttributeError: property 'text' of 'Tag' object has no setter
```

**Root cause:** The `_render_js` method in `mkdocs_with_pdf/generator.py` (line 381) tries to set the `.text` property of a BeautifulSoup `Tag` object, which is read-only in BeautifulSoup 4.14.3. The plugin was last updated in 2021 and is incompatible with modern BeautifulSoup versions.

**Conclusion:** This flag does not work and cannot be used without patching the plugin source code.

---

## Comparison

| Criteria | `mkdocs-mermaid-to-svg` + `mmdc` | `render_js: true` |
|----------|----------------------------------|-------------------|
| Works? | ✅ Yes | ❌ No (bug in plugin) |
| Setup complexity | `pip install` + `npm install -g` | Single flag (but broken) |
| Quality | SVG (infinite scaling) | N/A |
| Build speed | ~36s for full doc | N/A |
| Maintenance | Zero — runs automatically | N/A |

---

## Decision

**Use `mkdocs-mermaid-to-svg` + `mmdc`.**

**Rationale:**
1. SVG output is superior to PNG for PDF — infinite scaling, no pixelation
2. Plugin integrates cleanly with MkDocs build pipeline
3. Only external dependency is `mmdc` (Node.js), which is already available
4. No custom script maintenance required
5. The alternative `render_js: true` flag is broken due to a bug in the unmaintained `with-pdf` plugin

**Prerequisites for PDF build:**
1. `pip install mkdocs-mermaid-to-svg` (added to `install/requirements.txt`)
2. `npm install -g @mermaid-js/mermaid-cli` (one-time setup)
3. Node.js must be in PATH

---

## Definition of Done

- [x] Approach A (`mkdocs-mermaid-to-image`) tested — not available
- [x] Approach B (`mkdocs-mermaid-to-svg`) tested — works perfectly
- [x] Approach C (`render_js: true`) tested — broken due to BeautifulSoup incompatibility
- [x] PDF output verified — SVG renders correctly
- [x] Decision documented
- [x] Dependencies added to `install/requirements.txt`
- [x] Documentation updated in `readme.md`
