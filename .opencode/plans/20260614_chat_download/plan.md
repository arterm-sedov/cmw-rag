# Plan: Chat Download Button

## Goal

Add a download button next to the send/submit button inside the textbox area of the KB assistant (`kb_assist_demo`). The button exports the chat as a Markdown file. It appears only when the chat has content and the assistant is idle (not streaming).

## Design

### Layout

Replace `submit_btn=True` on the Textbox with a manual `gr.Row`:

```
[ textarea.......................... ] [→ submit] [↓ download]
```

The download button is small, icon-only, and visually part of the input box.

### Visibility

- Hidden by default (`visible=False`)
- Shown after the first user message is sent
- Hidden during streaming
- Re-shown when streaming completes
- Hidden when chat is cleared

### Export Format

Markdown with:
- Title and timestamp frontmatter
- Messages formatted as `## User` / `## Assistant` headings
- Content extracted from Gradio 6 message dicts (`role` + `content`)
- Saved to temp file, served via `gr.DownloadButton`

## Files to Modify

| File | Change |
|------|--------|
| `rag_engine/api/app.py` | Add `DownloadButton`, export function, visibility wiring, replace `submit_btn=True` with Row |
| `rag_engine/resources/css/cmw_copilot_theme.css` | Style download button to match send button |

## Step-by-Step Tasks

### Step 1: Add export function

```python
import tempfile, datetime

def _kb_export_chat(history: list[dict]) -> str | None:
    if not history:
        return None
    lines = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        heading = "User" if role == "user" else "Assistant"
        lines.append(f"## {heading}\n\n{content}\n")
    md = f"# Диалог с ИИ-ассистентом\n\n*{datetime.datetime.now():%Y-%m-%d %H:%M}*\n\n" + "\n---\n\n".join(lines)
    path = os.path.join(tempfile.mkdtemp(), "chat_export.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path
```

### Step 2: Replace Textbox with Row

```python
with gr.Row(elem_id="input-row", elem_classes=["input-row"]):
    msg = gr.Textbox(
        label="Сообщение",
        placeholder="Введите ваш вопрос...",
        lines=1,
        max_lines=4,
        show_label=False,
        elem_id="message-input",
        elem_classes=["message-card"],
        scale=4,
    )
    download_btn = gr.DownloadButton(
        "↓",
        elem_id="chat-download-btn",
        elem_classes=["chat-download-btn"],
        visible=False,
        scale=0,
        min_width=40,
    )
```

Note: `submit_btn=True` is removed — the Textbox default submit behavior (Enter key) still works without `submit_btn`.

### Step 3: Wire visibility

- After bot response completes → `download_btn.visible = True` if history non-empty
- After stop → same
- On chatbot.clear → `download_btn.visible = False`
- During streaming → `download_btn.visible = False`

### Step 4: Wire download

- `download_btn.click` → `_kb_export_chat(chatbot)` → returns file path → triggers download

### Step 5: CSS styling

```css
#chat-download-btn {
    min-width: 40px !important;
    padding: 0 8px !important;
    /* match submit button style */
}
```

## Verification

1. Start local Gradio: `python rag_engine/api/app.py`
2. Open `http://localhost:7862/kb_assist`
3. Send a message → download button appears after response
4. Click download → Markdown file downloads with full conversation
5. During streaming → download button hidden
6. Clear chat → download button hidden
7. Run `ruff check rag_engine/api/app.py`
