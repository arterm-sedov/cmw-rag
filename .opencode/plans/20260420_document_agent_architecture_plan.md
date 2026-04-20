# Plan: Document Agent Architecture Report

## Goal

Create a 5-page executive report explaining the document processing agent architecture for business stakeholders (Igor & Samarsky).

## Target Format

Follow `20260409-research-quick-start-rag-agent-ru.md` patterns:
- YAML front matter with title, date, status, tags, hide
- Markdown headings with anchors (`{: #anchor}`)
- MkDocs admonitions (`!!! note`, `!!! tip`, `!!! warning`)
- Tables in Markdown format
- No personal names (client-neutral)

## File Location

`docs/research/executive-research-technology-transfer/report-pack/20260420-document-agent-architecture-ru.md`

## Structure (5 pages)

### Page 1: Резюме для руководства (Executive Summary)

**SCQA:**
- **Ситуация:** Client sends commercial offer (PDF/DOCX) → needs summary in 30 sec
- **Вызов:** Manual processing takes 15–20 min, errors in data extraction
- **Решение:** Autonomous agent → reads document → generates structured summary → writes back to platform
- **Результат:** 99% automation, 15 sec processing time

### Page 2: Архитектура системы (System Architecture)

**Components table:**

| Компонент | Роль | notes |
| :--- | :--- | :--- |
| CMW Platform (secondary) | Document source and result receiver | Records with attachments |
| REST API endpoint | `/api/v1/cmw/summarize-document` | Accepts request_id |
| Document Connector | Extracts document from CMW | Base64 → text |
| Document Processor | Converts PDF/DOCX/XLSX → Markdown | PyMuPDF4LLM, python-docx |
| LLM Agent (glm-5) | Generates summary | Z-AI GLM via OpenRouter |
| Web Search (Tavily) | Competitive intelligence (prices, weather) | Automatic tool call |
| Output | Summary → attribute `summary` in CMW | Markdown or HTML |

**Diagram (Mermaid):**
```
[CMW Secondary]
       ↓ (API call with request_id)
[REST Endpoint /summarize-document]
       ↓
[Document Connector → Document Processor]
       ↓ (text extraction)
[LLM Agent + Web Search Tool]
       ↓ (summary + optional web search)
[CMW Secondary → attribute summary]
```

### Page 3: Инфраструктура (Infrastructure)

**Requirements table:**

| Параметр | Значение |
| :--- | :--- |
| Where it runs | Single server with RAG engine (`localhost:7860`) |
| Model | Z-AI GLM-5 (via OpenRouter) |
| Search | Tavily API (limits apply) |
| Documents | PDF, DOCX, XLSX, ZIP |
| Processing time | 15–60 sec/doc |
| API Key protection | X-API-Key header |

### Page 4: Интеграция с CMW Platform (CMW Integration)

**Input:**
- Record in template `ArchitectureManagement.Zaprosinarazrabotky`
- Attribute `Commerpredloshenie` — document reference
- Attribute `promt` — user instructions

**Output:**
- Attribute `summary` (Markdown or HTML)

**Key point:** Agent writes result back to CMW automatically — no user action after triggering

### Page 5: Демонстрация и следующие шаги (Demo & Next Steps)

**How to demonstrate:**
1. curl to endpoint → pass request_id
2. Agent reads document from CMW
3. Generates summary with tables and analysis
4. Writes to CMW → check in interface

**Next steps:**
- Scale to other document types
- Different template support
- Other CMW instances

## Content Rules

- **NO personal names** — "Lukoil" → "Отраслевой клиент" / "Заказчик"
- **NO Igor/Samarsky** — not mentioned
- **Client-neutral** — reusable for any client
- **PDF-ready** — use same scaffolding as research reports
- **Executive level** — no technical jargon, focus on business value

## Implementation

1. Write Markdown file following template
2. Add to git
3. PDF export via MkDocs

## Checkpoints

- [ ] Front matter correct (title, date, status, tags)
- [ ] All 5 sections present
- [ ] Tables in Markdown format
- [ ] No personal names
- [ ] No technical details about support agent or frontend
- [ ] Mermaid diagram renders
- [ ] Passes lint