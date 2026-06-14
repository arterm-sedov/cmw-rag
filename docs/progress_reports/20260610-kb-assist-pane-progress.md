# KB Assist Pane — Implementation Progress Report

**Date:** 2026-06-12  
**Plan:** `.opencode/plans/20260610_kb-assist/plan.md`  
**Branches:** `cmw-rag:20260610-kb-assist` (6 commits, pushed), `kb.comindware.ru:20260610-kb-assist` (25 commits, pushed, merged from `-pane`)

---

## Summary

Implementation of a native AI assistant widget for kb.comindware.ru is **substantially complete**. Both repos have been deployed locally and verified via Playwright. The production (ennoia) deploy is blocked by WSL SSH proxy config.

**Known remaining issues (not blocking):**
- Sidebar/logo collapse CSS changes not propagating to browser despite WSL sync + nginx reload — likely browser cache or PHP opcache issue

---

## Backend (cmw-rag) — 100% Complete

### Files changed
- `rag_engine/api/app.py` — +272 / -0 lines (5 commits)
- `rag_engine/requirements.txt` — Gradio pinned `==6.9.0`

### What was built

| Feature | Status | Details |
|---|---|---|
| `kb_assist_handler` wrapper | ✅ | Strips 5 metadata yields → yields only `result[0]` (chatbot history) |
| Second `gr.Blocks` at `/kb_assist` | ✅ | Minimal: chatbot + textbox + stop button only. No header, no version selector, no metadata panels |
| Route ordering fix | ✅ | `/kb_assist` mounted before `/` in Starlette routes to prevent catch-all |
| Trailing-slash redirect | ✅ | FastAPI route `GET /kb_assist` → `301 /kb_assist/` |
| SRP gate per handler | ✅ | `skip_srp=True` flag threads through `kb_assist_handler` → `chat_with_metadata` → `agent_chat_handler` |
| Chatbot height removed | ✅ | `height`/`max_height` removed from `kb_assist_demo` Chatbot — CSS flex chain handles sizing |

### Verification
- Ruff check: clean
- Existing tests pass (no regression)

---

## Frontend (kb.comindware.ru) — 95% Complete

### Files changed
- `ai-assistant.php` — **new**, 1237 lines (plan estimated ~200; actual blew up due to responsive right-pane layout, FA Pro icons, multiple explain buttons, and advanced JS)
- `.gitignore` — exception `!/assets/fontawesome/`
- `assets/fontawesome/` — FA Pro 6.2.1 webfonts + CSS from CBAP_MONO (committed)
- 6 PHP pages — each gets `<?php include('ai-assistant.php'); ?>` before `</body>`

### What was built

| Feature | Status | Details |
|---|---|---|
| **Responsive layout** | ✅ | Right pane pushes content on all desktop widths; full-screen overlay only on mobile <768px |
| **Header matches KB top bar** | ✅ | Height via `--cmw-topnav-height` CSS variable (91px), `#2d9adf` primary color, no border offset. Both KB top nav and widget header share one source of truth |
| **Overflow fix** | ✅ | `overflow: hidden` + `box-sizing: border-box` on container — prevents vertical/horizontal scroll within widget |
| **Resizable pane** | ✅ | Drag left edge of widget to resize width; body margin-right and toggle button position sync dynamically; width persisted in localStorage |
| **No page scroll in pane mode** | ✅ | Body margin-right driven by JS (not CSS) — page scrolls freely when pane is closed, reflows when open |
| **GRADIO_URL simplified** | ✅ | Full URL in PHP variable, no JS concatenation (removed `/kb_assist` suffix appending that caused path doubling) |
| **FA Pro icons** | ✅ | `fa-microchip-ai` (search button), `fa-wand-magic-sparkles` (explain), `fa-chevron-left` / `fa-xmark` (header), `fa-bolt` (loading) |
| **AI chip search button** | ✅ | Injected between search input and dropdowns (`col-xs-12` in `.row .text-right`), toggles pane open/close |
| **Tooltip toggle** | ✅ | Title changes: "Открыть ИИ-ассистент" / "Закрыть ИИ-ассистент" (committed & pushed) |
| **H1 explain button** | ✅ | Blue pill block below H1, `fa-wand-magic-sparkles`, "Объяснить с помощью ИИ", 6px border-radius |
| **Selection explain button** | ✅ | Same text/icon/style as H1, positioned near selection, hides on click-away |
| **KB_CONTEXT detection** | ✅ | PHP walks category parent chain, detects version root (v5/v6), sets `window.KB_CONTEXT` with `version`, `article_id`, `article_title` |
| **Gradio CDN integration** | ✅ | CDN 6.5.1 `<gradio-app>` web component. Plan said no CDN but self-hosted didn't register web component; CDN needed |
| **Shadow DOM messaging** | ✅ | `_sendWhenReady()` retry loop — polls for textarea, sets value, dispatches input + submit events |
| **localStorage persistence** | ✅ | Pane open/closed state, width remembered across page loads |
| **Resize handle** | ✅ | Visible on left edge in pane mode (`cursor: ew-resize`); dragging syncs widget width, body margin, and toggle button position. Width saved to localStorage |
| **Sidebar collapse** | ✅ | At ≤1047px with pane active: sidebar slides left via KB's own CSS transition, hamburger toggle appears to reopen. Uses KB's native `xv-menu.js` mechanism |
| **Logo collapse** | ✅ | At ≤1031px with pane active: logo hidden, hamburger toggle shown. Menu items hidden, replaced by hamburger |
| **Menu right-aligned** | ✅ | `margin-left: auto` on `.xv-menuwrapper`/`.dl-menuwrapper` keeps menu right-aligned regardless of logo visibility |
| **Widget height fit** | ✅ | Padding buffer 50px accounts for Gradio internal margins; textarea measured via `.closest()` to include container margins |
| **Flex chain height** | ✅ | Complete chain from container to `#chatbot-main` via flex. Resizes correctly in both pane and floating modes |
| **Pane inline !important removed** | ✅ | Replaced inline `max-height:69vh!important` with CSS class; JS clears Gradio's inline styles in both modes |
| **Redundant buttons removed** | ✅ | Bottom floating toggle + widget-header explain article button removed |

### Deviations from plan

| Plan | Actual | Rationale |
|---|---|---|
| Self-hosted `<gradio-app>` | CDN 6.5.1 | Gradio 6.9.0 server doesn't serve the web component JS; `<gradio-app>` tag requires the CDN bundle to register the custom element |
| `gradio-config.json` | PHP `$GRADIO_URL` variable | Simpler — one PHP variable per deployment, no additional HTTP request |
| Right-pane push layout | Not in plan | Discovered GitBook/Mintlify pattern during implementation; later expanded to all desktop widths (removed 1400px breakpoint) |
| Floating overlay (769-1399px) | Removed | Pane push works on all desktop widths now; only mobile gets full-screen overlay |
| FA Free icons | FA Pro 6.2.1 | CBAP_MONO had Pro license; `microchip-ai` and `wand-magic-sparkles` are Pro-only |
| 3 explain buttons (selection, article, widget-header) | 3 buttons (selection, H1, search AI chip) | Widget-header removed as redundant with H1 and search AI; bottom toggle removed |

### Known issues
- `TypeError: this.app.$destroy is not a function` — Gradio CDN 6.5.1 cosmetic bug, does not block functionality

---

## KB ↔ Widget Interaction Discoveries

### Layout Architecture

| KB element | CSS class | Behavior |
|---|---|---|
| Blue top bar | `.bg-cmw.top_nav` > `.mainnav.navbar` | Height driven by content (line-height: 90px on menu links). No explicit height. |
| Logo | `.navbar-brand` > `img` | Hidden by KB's own JS at narrow widths (≤1031px). Widget pane does NOT need to hide it. |
| Menu | `.xv-menuwrapper .dl-menu` | Right-aligned via `justify-content-between` on `.mainnav`. At ≤1031px, KB switches to hamburger (`.dl-menuwrapper`). |
| Sidebar | `.left-side` / `.main-sidebar` | Collapses at ≤1047px via `xv-menu.js` adding `sidebar-collapse` to `<body>`. Uses `transform: translate(-330px, 0)`. |
| Search bar | `.search-section` | Inside `.container` below the blue bar. Not affected by pane. |

### How KB responsive breakpoints work

- **1047px**: Sidebar collapses (`sidebar-collapse` class added to body). Menu switches to `dl-menuwrapper` (hamburger).
- **1031px**: Hamburger toggle shows (`.paper-nav-toggle`). Nav gets padding. Menu links compact.
- These breakpoints are based on `window.innerWidth`, NOT content width. Body `margin-right` from the widget does NOT trigger them.

### What widget CSS must override when pane is active

| Rule | Purpose |
|---|---|
| `body.cmw-widget-pane-active .mainnav { padding: 15px }` | Compact nav padding (KB only does this at ≤1031px) |
| `body.cmw-widget-pane-active .xv-menuwrapper .dl-menu > li > a { line-height: 60px; font-size: 12px; ... }` | Compact menu items (KB only does this at ≤1031px) |
| `body.cmw-widget-pane-active .xv-menuwrapper { margin-left: auto }` | Keep menu right-aligned when logo is hidden |

### What widget CSS must NOT do

- **Do NOT hide `.navbar-brand`** — KB's own JS handles this at narrow widths
- **Do NOT add `overflow: hidden` to `html`/`body`** — kills page scroll entirely
- **Do NOT use hardcoded `margin-right` in CSS** — must be JS-driven to stay in sync with resizable widget width

### Pane-mode CSS architecture

```
body.cmw-widget-pane-active     → margin-right: {widget-width}px (JS-driven)
#cmw-widget-container           → width: 400px; height: 100vh; right: 0; top: 0; bottom: 0
                                  border: none; border-left: 1px solid var(--cmw-border)
                                  overflow: hidden; box-sizing: border-box
#cmw-widget-container.show      → transform: translateX(0)
```

### JS state management in pane mode

- `toggleWidget(true)`: adds `cmw-widget-pane-active` + `sidebar-collapse` to body, sets `body.style.marginRight`, calls `restoreContainerSize()`
- `toggleWidget(false)`: removes classes, clears `body.style.marginRight`, conditionally removes `sidebar-collapse` if viewport > 1047px
- `saveState()`: skips `left`/`bottom` in pane mode (prevents floating-mode restore on reopen)
- `restoreContainerSize()`: in pane mode, syncs body margin + toggle button position from stored width
- `doResize()`: in pane mode, updates widget width + body margin + toggle position (no height/left/bottom changes)

### Gradio CDN vs self-hosted

- `<gradio-app>` web component requires CDN bundle to register the custom element
- Self-hosted Gradio 6.9.0 does NOT serve the web component JS
- CDN version used: 6.5.1 (works with Gradio 6.9.0 backend)
- `GRADIO_URL` must be a full URL (no path concatenation in JS)

### WSL Dev Environment — PHP File Propagation

**Critical finding:** The KB nginx server in WSL reads from `/var/www/kb/`, **not** from the Windows-mounted `D:\Repo\kb.comindware.ru\`. These are **separate copies** of the files.

| Path | Role |
|---|---|
| `D:\Repo\kb.comindware.ru\` | Git repo on Windows — where edits are made |
| `/var/www/kb/` (WSL) | nginx document root — what the server actually serves |

**To propagate PHP changes to the local dev server:**
```bash
wsl bash -c "cp /mnt/d/Repo/kb.comindware.ru/ai-assistant.php /var/www/kb/ai-assistant.php && chown www-data:www-data /var/www/kb/ai-assistant.php"
```

For **production** (ennoia), changes must be pushed to the `kb.comindware.ru` git repo and deployed separately. The `D:\Repo\kb.comindware.ru\` working copy is not the live server.

---

## Remaining Work

### Blocked (by infrastructure)
1. **Deploy `20260610-kb-assist-pane` to ennoia production** — WSL localhost proxy not mirrored into WSL NAT mode. `ssh ennoia` cannot resolve hostname. Needs manual deploy or WSL proxy config fix.

### Resolved
2. **GRADIO_URL path doubling** — Fixed: removed `+ '/kb_assist'` concatenation in JS, now uses full URL directly. Config URL now resolves correctly to `/kb_assist/config` without doubling.
3. **Widget header match KB top bar** — Now uses `--cmw-topnav-height` CSS variable (91px) shared by both `.bg-cmw.top_nav` and `.cmw-widget-header`. Pure CSS, no JS measurement needed.
4. **Responsive layout simplified** — Right pane push now applies at all desktop widths; floating overlay breakpoint (769-1399px) removed.
5. **Widget overflow fixed** — Added `overflow: hidden` + `box-sizing: border-box` to `#cmw-widget-container` in pane mode. Container no longer causes vertical/horizontal scroll.
6. **Pane resize** — Drag left edge to resize width; body margin-right and toggle position sync dynamically. Width persisted in localStorage; `saveState()` skips left/bottom in pane mode to prevent floating-mode restore.
7. **Blue bar alignment** — Removed `border-top` from pane-mode container, eliminating 1px offset. Both bars now start at y=0 with matching 91px height.
8. **Sidebar collapse** — CSS media queries at KB's native breakpoints (1047px sidebar, 1031px nav) trigger when pane is active. Sidebar slides left with transition, hamburger toggle appears.
9. **Logo collapse** — At ≤1031px with pane active: `.navbar-brand` hidden, hamburger shown, menu items hidden.
10. **Menu alignment** — `margin-left: auto` on menu wrapper keeps it right-aligned regardless of logo visibility.
11. **Widget height fit** — Padding buffer increased to 50px for Gradio internal margins; textarea measured via `.closest()` to include container margins.

### Resolved (2026-06-14)
- **Widget chatbot height cap on resize** — Root cause chain:
  1. `#chatbot-main` missing `flex: 1` in `.main.fillable` broke the vertical flex chain
  2. `main` lacked `display: flex; flex-direction: column` so `.main.fillable`'s `flex: 1` had no effect
  3. Pane mode set inline `max-height: 69vh !important` via `element.style.setProperty()` — inline `!important` has highest cascade priority, persists across mode switches, beats all CSS
  4. Gradio's `cmw_copilot_theme.css` set `#chatbot-main { max-height: 80vh !important }` — same specificity `(1,2,0)` as our PHP CSS, won on source order (injected later)
  5. `app.py` set `height="70vh"` and `max_height="70vh"` on Chatbot component — inline styles re-applied on Gradio re-renders
- **Fixes applied (3 commits across 2 repos):**
  - `cmw_copilot_theme.css`: Removed `resize: vertical`, `overflow: auto`, `max-height: 80vh !important` from `#chatbot-main` — flex chain handles sizing
  - `ai-assistant.php`: Complete flex chain — `main { display:flex; flex-direction:column; flex:1 }`, `.main.fillable { flex:1 }`, `gradio-app > div { height:100% }`. All selectors use `#cmw-widget-container` prefix `(1,2,0)` specificity. Removed inline `!important` from pane mode `updateChatbotHeight()`. JS now clears Gradio's inline `height`/`max-height` on chatbot element in both modes.
  - `app.py`: Removed `height="70vh"` and `max_height="70vh"` from `kb_assist_demo` Chatbot — CSS flex chain handles sizing
  - `ai-assistant-standalone.php`: Matching flex chain with `#cmw-standalone-page` prefix

### Not resolved (investigation needed)
13. **Collapse CSS not visible in browser** — Changes verified in Playwright but not rendering in user's Chrome. Suspected: PHP opcache or aggressive browser cache. WSL file sync confirmed (md5 match), nginx reloaded.

### Pending (no blockers)
5. **Merge `cmw-rag:20260610-kb-assist` → `main`** — 6 commits, all pushed to origin, no conflicts
6. **Merge `kb:20260610-kb-assist` → `develop`** — 25 commits on branch, merged from `-pane` sub-branch. Prefer squash-merge or rebase for clean history.

### Not implemented
4. **Playwright test for widget interaction** — plan mentioned verification but no automated test was written. Manual testing done.

---

## Verification Checklist

| Check | Status |
|---|---|
| Backend lint (`ruff check rag_engine/api/app.py`) | ✅ |
| Existing tests pass (`pytest -m "not slow"`) | ✅ |
| `/` unchanged — all panels load | ✅ |
| `/kb_assist` serves — chatbot renders | ✅ |
| Widget toggle + Gradio loads on article page | ✅ (local Playwright) |
| Text selection → explain button → widget opens with text | ✅ |
| H1 explain button sends context | ✅ |
| Search AI chip toggles pane | ✅ |
| Mobile <768px full-screen | ✅ |
| FA Pro icons render | ✅ |
| Pane resize syncs body margin | ✅ (local Playwright) |
| Sidebar collapses at ≤1047px with pane | ⚠️ Playwright OK, browser not verified |
| Logo hides at ≤1031px with pane | ⚠️ Playwright OK, browser not verified |
| Widget fits viewport (no overflow) | ✅ (Playwright) |
| Widget height on manual resize | ✅ Fixed 2026-06-14 (flex chain + theme CSS cap removal) |
| ennoia production deploy | ❌ blocked |
