# KB Assist Pane — Implementation Progress Report

**Date:** 2026-06-10  
**Plan:** `.opencode/plans/20260610_kb-assist/plan.md`  
**Branches:** `cmw-rag:20260610-kb-assist` (5 commits, pushed), `kb.comindware.ru:20260610-kb-assist-pane` (19 commits, pushed)

---

## Summary

Implementation of a native AI assistant widget for kb.comindware.ru is **substantially complete**. Both repos have been deployed locally and verified via Playwright. The production (ennoia) deploy is blocked by WSL SSH proxy config.

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
| **Responsive layout** | ✅ | Right pane >1400px (pushes content), floating overlay 769-1399px, full-screen <768px |
| **FA Pro icons** | ✅ | `fa-microchip-ai` (search button), `fa-wand-magic-sparkles` (explain), `fa-chevron-left` / `fa-xmark` (header), `fa-bolt` (loading) |
| **AI chip search button** | ✅ | Injected between search input and dropdowns (`col-xs-12` in `.row .text-right`), toggles pane open/close |
| **Tooltip toggle** | ✅ | Title changes: "Открыть ИИ-ассистент" / "Закрыть ИИ-ассистент" (committed & pushed) |
| **H1 explain button** | ✅ | Blue pill block below H1, `fa-wand-magic-sparkles`, "Объяснить с помощью ИИ", 6px border-radius |
| **Selection explain button** | ✅ | Same text/icon/style as H1, positioned near selection, hides on click-away |
| **KB_CONTEXT detection** | ✅ | PHP walks category parent chain, detects version root (v5/v6), sets `window.KB_CONTEXT` with `version`, `article_id`, `article_title` |
| **Gradio CDN integration** | ✅ | CDN 6.5.1 `<gradio-app>` web component. Plan said no CDN but self-hosted didn't register web component; CDN needed |
| **Shadow DOM messaging** | ✅ | `_sendWhenReady()` retry loop — polls for textarea, sets value, dispatches input + submit events |
| **localStorage persistence** | ✅ | Pane open/closed state, width remembered across page loads |
| **Resize handle** | ✅ | Top-left corner drag for pane width (hidden on mobile) |
| **Redundant buttons removed** | ✅ | Bottom floating toggle + widget-header explain article button removed |

### Deviations from plan

| Plan | Actual | Rationale |
|---|---|---|
| Self-hosted `<gradio-app>` | CDN 6.5.1 | Gradio 6.9.0 server doesn't serve the web component JS; `<gradio-app>` tag requires the CDN bundle to register the custom element |
| `gradio-config.json` | PHP `$GRADIO_URL` variable | Simpler — one PHP variable per deployment, no additional HTTP request |
| Right-pane push layout | Not in plan | Discovered GitBook/Mintlify pattern during implementation; much better UX than floating popup |
| FA Free icons | FA Pro 6.2.1 | CBAP_MONO had Pro license; `microchip-ai` and `wand-magic-sparkles` are Pro-only |
| 3 explain buttons (selection, article, widget-header) | 3 buttons (selection, H1, search AI chip) | Widget-header removed as redundant with H1 and search AI; bottom toggle removed |

### Known issues
- `TypeError: this.app.$destroy is not a function` — Gradio CDN 6.5.1 cosmetic bug, does not block functionality

---

## Remaining Work

### Blocked (by infrastructure)
1. **Deploy `20260610-kb-assist-pane` to ennoia production** — WSL localhost proxy not mirrored into WSL NAT mode. `ssh ennoia` cannot resolve hostname. Needs manual deploy or WSL proxy config fix.

### Pending (no blockers)
2. **Merge `cmw-rag:20260610-kb-assist` → `main`** — 5 commits, all pushed to origin, no conflicts
3. **Merge `kb:20260610-kb-assist-pane` → `develop`** — merge-base is `440f6e1` (shared with develop at `5254370` and master at `95b7805`). Branch has 302 commits ahead of master (includes full develop history) but only 19 unique to our feature. Prefer squash-merge or rebase for clean history.

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
| ennoia production deploy | ❌ blocked |
