# Plan: KB Assist — Second Gradio GUI + Native Widget Injection

## Problem

The existing AI assistant widget at `/platform/v5.0/chat/` works but has two flaws:

1. **Iframe approach** — a standalone page that frames the KB in the background. Users leave the real KB, no deep-linking, no URL sync, no navigation context.
2. **Full Gradio UI** — metadata panels (Analysis, Guardian, SGR, SRP, Articles) take ~40% of a 450×650px widget, all empty by default.

## Goal

- Stripped-down Gradio app at `/kb_assist` — same RAG agent, no metadata panels, no version selector, no SRP
- Floating chat widget injected natively into KB pages
- Same toggle/resize/Gradio-embed behavior as the existing iframe widget
- KB context aware: knows version, article, category from the page
- "Explain selected text" floating button
- "Explain this article" button in widget

## Architecture

```
KB Page
  ├── PHPKB renders normally (header, sidebar, article, footer)
  ├── ai-assistant.php
  │     ├── PHP: determine version + article ID from category hierarchy
  │     ├── <script>window.KB_CONTEXT = {version, article_id, article_title}</script>
  │     ├── CSS: widget toggle + panel (injected, position: fixed)
  │     ├── HTML: toggle button + panel + <gradio-app>
  │     └── JS: toggle, resize, localStorage, Gradio init,
  │              text selection listener ("Explain" button),
  │              "Explain this article" button in widget
  └── </body>

RAG Backend (same port)
  ├── /           → main app (full metadata, SRP, MCP)
  └── /kb_assist  → lite app (chatbot only, no header controls)
```

---

## Phase 1: Backend — `/kb_assist` Gradio App

### File: `rag_engine/api/app.py`

### Step 1.1: Add `kb_assist_handler` wrapper

Thin async generator wrapping `chat_with_metadata`, stripping the 5 metadata outputs. No agent logic duplication — SRP already gated by `SRP_ENABLED=false` (default).

```python
async def kb_assist_handler(message, history, cancel_state=None, request=None):
    async for result in chat_with_metadata(message, history, cancel_state, request):
        yield result[0]  # chatbot only
```

### Step 1.2: Build second `gr.Blocks`

| Property | `/` | `/kb_assist` |
|---|---|---|
| Title | "Ассистент инженера поддержки" | "Ассистент базы знаний" |
| Header | Hero title + version selector | **None** (no header at all) |
| Chatbot | `elem_id="chatbot-main"`, `elem_classes=["chatbot-card"]` | Same |
| Metadata panels | 5 panels | **None** |
| Handler | `chat_with_metadata` | `kb_assist_handler` |
| Output count | 6 (chatbot + 5 panels) | **1** (chatbot only) |
| MCP server | yes | no |

The `/kb_assist` UI is minimal: chatbot + textbox + stop button. No version selector — version is derived from the KB page context (Phase 2).

### Step 1.3: Mount at `/kb_assist`

In `__main__`, after the first `mount_gradio_app` call:

```python
kb_assist_demo = ...  # second gr.Blocks instance

app = mount_gradio_app(
    fastapi_app, kb_assist_demo, path="/kb_assist",
    mcp_server=False, footer_links=[],
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
)
```

### Step 1.4 (optional): Theme CSS overrides

Minimal CSS file for compact widget-friendly spacing.

---

## Phase 2: Frontend — Widget Injection into KB Pages

### File: `ai-assistant.php` (new, KB repo root)

Four parts: PHP context detection, CSS, HTML, JS.

#### Part A: PHP — context detection

The KB uses a **category hierarchy** for versioning. Categories have `parent_id` relations forming a tree. Each article belongs to a leaf category. We walk up to find the version:

```
Product (298) → Version (896=v6, 798=v5) → Section → Article
```

A small PHP lookup in `ai-assistant.php`:

1. Get current category ID from `$catid` (article page) or `$row['categoryid']`
2. Walk up `$parent_id` chain using PHPKB functions until we match a known version category
3. Output `window.KB_CONTEXT` with version string and article ID

```php
// Known version root categories
$VERSION_ROOTS = [896 => 'v6', 798 => 'v5', 378 => 'v4.7', 377 => 'v3.5'];
// Walk parent chain, find match
// Emit JS:
<script>window.KB_CONTEXT = {version:"v6", article_id:"5162", article_title:"..."};</script>
```

For standalone pages (index, product-*) that aren't article-specific, just detect version from the category context.

#### Part B: CSS — widget styling

Extracted from `platform/v5.0/chat/index.php`:
- Toggle button (60px circle, bottom-right, blue, shadow)
- Widget panel (position: fixed, bottom: 90px, right: 20px, width: 450px, height: 650px)
- Panel header (blue bar, title, close button)
- Gradio container sizing, chatbot scroll area
- Resize handle (top-left corner)
- Loading indicator
- Responsive breakpoints (tablet, mobile)
- Dark theme support
- `#cmw-explain-btn` — floating "Explain" button near text selection (hidden by default)
- `.cmw-widget-explain-article` — "Explain article" button inside widget panel

#### Part C: HTML — widget structure

```
<div id="cmw-widget" class="cmw-widget">
    <button id="cmw-widget-toggle">        ← floating circle
    <div id="cmw-widget-container">         ← resizable panel
        <div id="cmw-widget-resize-handle"> ← drag corner
        <div class="cmw-widget-header">
            <h3>Ассистент</h3>
            <button class="cmw-widget-explain-article">Объяснить эту статью</button>
            <button class="cmw-widget-close">×</button>
        </div>
        <div class="cmw-widget-gradio-container">
            <div id="cmw-widget-loading">Загрузка...</div>
            <gradio-app id="cmw-widget-gradio-app" src="" style="display:none;">
        </div>
    </div>
</div>
<div id="cmw-explain-btn" class="cmw-explain-btn" style="display:none;">
    Объяснить с помощью ИИ
</div>
```

#### Part D: JS — widget logic + context features

**Reused from `chat/index.php`:**
- `toggleWidget(forceState)` — open/close with animation
- `initializeGradioApp(url)` — set `<gradio-app>` src, loading state
- `updateChatbotHeight()` — calculate `--chatbot-height` from available space (66%/76% formula)
- Resize handlers — pointer-event drag from top-left
- `localStorage` state — persist open/closed, size, position
- GRADIO_URL loading — `fetch('gradio-config.json')` → fallback `window.GRADIO_URL` → append `/kb_assist`

**New: "Explain selected text" feature**
```javascript
document.addEventListener('mouseup', (e) => {
    const selection = window.getSelection();
    const text = selection.toString().trim();
    if (text.length < 5) { explainBtn.style.display = 'none'; return; }
    // Position button near selection end
    const rect = selection.getRangeAt(0).getBoundingClientRect();
    explainBtn.style.top = rect.bottom + 5 + 'px';
    explainBtn.style.left = rect.right - 100 + 'px';
    explainBtn.style.display = 'block';
    explainBtn.textContent = 'Объяснить с помощью ИИ';
});
explainBtn.addEventListener('click', () => {
    const text = window.getSelection().toString().trim();
    toggleWidget(true);  // open widget
    sendToGradio(`Объясни: "${text}"`);  // send message
    explainBtn.style.display = 'none';
    window.getSelection().removeAllRanges();
});
```

CSS for the explain button: `position: fixed`, `z-index: 2000` (above everything), small pill shape with blue background, appears near selection, hides on click-away.

**New: "Explain this article" button**
```javascript
document.querySelector('.cmw-widget-explain-article').addEventListener('click', () => {
    const ctx = window.KB_CONTEXT;
    if (!ctx.article_id) return;
    const msg = `Объясни эту статью. ID статьи: ${ctx.article_id}, версия ПО: ${ctx.version}`;
    sendToGradio(msg);
});
```

**Messaging Gradio**: Since `<gradio-app>` is a web component, we need to send messages programmatically. The Gradio JS client exposes an API:

```javascript
// Option A: Use Gradio's JS client to call the chat handler
import { Client } from '.../gradio.js';
const client = await Client.connect(gradioUrl);
client.predict(message);

// Option B: Dispatch a custom event the Gradio app listens for
// Option C: Use the Gradio app's internal textbox + submit
//    → querySelector the textbox inside shadow DOM, set value, click submit
```

The simplest approach: **Option C** — reach into the Gradio shadow DOM, set the textbox value, and dispatch an Enter key or click the submit button. The existing `updateChatbotHeight()` already traverses shadow DOM, so we have the pattern.

#### `gradio-config.json` (gitignored, per-deployment)

```json
{"GRADIO_URL": "<deployment-url>"}
```

JS appends `/kb_assist` automatically. Example resolved URL: `<deployment-url>/kb_assist`.

### Injection points

Add one line per page before `</body>`:

| Page | After | Why |
|---|---|---|
| `article.php` | `live-search.php` | Most visited — no chat widget currently |
| `category.php` | `live-search.php` | Most visited — no chat widget currently |
| `index.php` | `live-search.php` | Home page |
| `product-platform.php` | `jivo.php` | Product landing |
| `product-project.php` | `jivo.php` | Product landing |
| `product-tracker.php` | `jivo.php` | Product landing |

```php
<?php include('ai-assistant.php'); ?>
```

---

## Phase 3: Version Handling Strategy

The KB version selector (below search bar) navigates between category pages — it's a page changer, not a state toggle. We need the AI widget to know which version the user is viewing **without** duplicating a version selector in the widget.

### How it works

1. **PHP side** (`ai-assistant.php`): On article/category pages, walk the category parent chain to find the version root (896=v6, 798=v5, etc.). Output `window.KB_CONTEXT.version`.
2. **Widget JS**: When "Explain this article" is clicked, construct the message including the version: `"Объясни эту статью. ID: 5162, версия: v6"`
3. **Agent backend**: The agent's `retrieve_context` tool already filters by `product_version`. The SGR planner (`analyse_user_request`) parses the user message and sets the version. No backend changes needed — the agent already handles `product_version` as a tool parameter.
4. **Free-form chat**: If the user types their own question (not using the "Explain" buttons), the agent defaults to v6 (current version) — same as the main app's default.

This means `/kb_assist` has **no version dropdown** in its Gradio UI — it's purely a chatbot.

---

## Phase 4: Verification

| Check | How |
|---|---|
| Backend lint | `ruff check rag_engine/api/app.py` |
| Backend tests | `pytest -m "not slow"` — confirm no regressions |
| `/` unchanged | Main app loads, all panels still work |
| `/kb_assist` serves | Returns HTML, chatbot renders |
| Widget loads on article | Toggle visible, panel opens, Gradio loads, chat works |
| Text selection | Select text → "Explain" button appears → click → widget opens with selected text |
| "Explain article" | Click button in widget → agent receives article ID + version → responds with article content |

---

## Non-breaking Guarantees

- `/` unchanged — no code removed from main demo or handler
- Same `LLMManager`, `ChromaStore`, `RAGRetriever` — shared backend
- Same agent (`create_rag_agent`) — same tools, prompts, retrieval
- SRP off by default — no-op in both GUIs
- Session isolation via `gr.Request.session_hash` — separate Gradio instances, separate memory
- KB pages: one include line per template, no existing code modified

---

## Files Changed

| Repo | File | Action | Lines |
|---|---|---|---|
| rag | `rag_engine/api/app.py` | Add `kb_assist_handler` + 2nd `gr.Blocks` + 2nd mount | ~100 |
| rag | `rag_engine/resources/css/kb_assist_theme.css` | New (optional overrides) | 0+ |
| kb | `ai-assistant.php` | New (PHP context + CSS + HTML + JS) | ~200 |
| kb | `article.php` | Add include | 1 |
| kb | `category.php` | Add include | 1 |
| kb | `index.php` | Add include | 1 |
| kb | `product-platform.php` | Add include | 1 |
| kb | `product-project.php` | Add include | 1 |
| kb | `product-tracker.php` | Add include | 1 |
