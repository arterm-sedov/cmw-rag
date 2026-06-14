# AI Assistant Widget UI Patterns — Live Examples

**Date:** 2026-06-10
**Method:** Live browser inspection via Playwright + web scraping

---

## 1. GitHub Copilot Docs — Inline Search Overlay

**URL:** `https://docs.github.com/en/copilot`

### Pattern: Search-first overlay dialog with AI Q&A

| Aspect | Detail |
|--------|--------|
| **Widget position** | Center-screen dialog overlay, triggered from header. No persistent floating button. |
| **Panel size & behavior** | Full-screen modal overlay (dialog "Search overlay"). Appears as a centered panel with large input area. |
| **Toggle mechanism** | Button "Search or ask Copilot" in the header nav bar. URL reflects state: `?search-overlay-open=true`. |
| **Mobile behavior** | Likely same full-screen dialog pattern (already optimized for it). |
| **Cover vs resize** | **Covers content** — it's a full-screen overlay. Main page is dimmed behind it. |

### Panel contents
- Combobox input "Search or ask Copilot" (auto-focused on open)
- "Ask Copilot" section header with icon
- List of 4 suggested AI questions (e.g., "How do I connect to GitHub with SSH?")
- Privacy disclaimer at bottom
- "Clear" button next to input

### Key design decisions
- **AI is not a separate widget** — it's integrated into the search experience. One input for both "Search" and "Ask Copilot".
- The overlay is also reachable via `Ctrl+K` (standard GitHub keyboard shortcut).
- No permanent floating button or sidebar.

---

## 2. Intercom Messenger — Floating Bubble + Slide-up Overlay

**URL:** `https://www.intercom.com/` (blocked from automated access; pattern documented from public knowledge)

### Pattern: Classic floating chat bubble → slide-up panel

| Aspect | Detail |
|--------|--------|
| **Widget position** | Fixed floating button (circular "chat bubble" or branded launcher) at **bottom-right** of viewport. |
| **Panel size & behavior** | Overlay panel slides up from the bottom-right corner. ~376px wide, ~80% viewport height. Rounded top corners. |
| **Toggle mechanism** | Click the floating launcher button. Also triggerable programmatically via Intercom JS API. |
| **Mobile behavior** | Full-screen overlay on mobile devices. |
| **Cover vs resize** | **Covers content** — overlay does not push or resize the page. Background scroll may be locked when open. |

### Panel contents
- Branded header bar with product/team name and avatar
- Conversation thread area (scrollable)
- Composer with text input, emoji picker, and attachment button
- Optional: bot greeting, quick-reply buttons, help article suggestions
- Top-right close/minimize button

### Key design decisions
- **Always visible launcher** — persistent floating presence on every page.
- The launcher is often styled as a branded icon or chat bubble with unread badge.
- Can be configured as "Help" button, "Chat with us", or custom launcher text.
- Standard z-index: typically `2147483000` (maximum to sit above everything).
- This is the **canonical** floating chat widget pattern, widely copied by Crisp, Drift, Zendesk, etc.

---

## 3. GitBook Assistant — Right Sidebar Panel (Push Layout)

**URL:** `https://docs.gitbook.com/`

### Pattern: Right sidebar panel that pushes/squeezes main content

| Aspect | Detail |
|--------|--------|
| **Widget position** | Right sidebar panel (ARIA `complementary` landmark). No floating button on published docs. |
| **Panel size & behavior** | Fixed-width right panel. Opens by **pushing/squeezing** the main content left. The panel slides in from the right, narrowing the content area. |
| **Toggle mechanism** | ① Header button "Ask GitBook Assistant" (in the top nav bar). ② Inline "Ask" button in the main content area. URL reflects state: `?ask=`. Close via ✕ button in panel header. |
| **Mobile behavior** | (Not directly observed — likely switches to full-screen overlay on small viewports.) |
| **Cover vs resize** | **Resizes page** — the panel pushes existing content to the left. Content remains visible and accessible alongside the panel. |

### Panel contents
- Header: "GitBook Assistant" with AI icon and close ✕ button
- Greeting: "Good afternoon" + "I'm here to help you with the docs."
- Suggested questions as clickable pill buttons (3 examples shown)
- Chat input: textbox "Ask, search, or explain..." with:
  - "AI" badge
  - "Based on your context" indicator with info icon
  - Send button (disabled when empty)
- Scrollable conversation history area

### Key design decisions
- **Push layout, not overlay** — the panel is a first-class layout element that resizes the page. Users can reference docs while using the assistant.
- Uses `complementary` ARIA landmark for accessibility.
- "Based on your context" indicator tells users the AI is aware of the current page.
- The inline "Ask" button in the main content is a secondary trigger point.
- The AI assistant is tightly integrated with the docs content — it knows which page you're on.

---

## 4. Mintlify Assistant — Right Sidebar Panel (Push Layout)

**URL:** `https://docs.perplexity.ai/` (Mintlify-powered docs)

### Pattern: Right sidebar panel similar to GitBook, with header toggle

| Aspect | Detail |
|--------|--------|
| **Widget position** | Right sidebar panel. Toggle button in the top header bar. |
| **Panel size & behavior** | Right panel that pushes/compresses the main content area. Viewport adjusts — the main content and navigation both shift left. |
| **Toggle mechanism** | Header button "Toggle assistant panel" with icon. Close button "Close assistant panel" in panel header. No URL state change observed. The toggle button gets `[active]` state when panel is open. |
| **Mobile behavior** | (Not directly observed — likely overlay on mobile.) |
| **Cover vs resize** | **Resizes page** — pushes main content left. Content remains visible alongside panel. |

### Panel contents
- Header: "Assistant" label with icon + close ✕ button
- Disclaimer: "Responses are generated using AI and may contain mistakes."
- "Suggestions" section with 3 clickable suggested questions
- Input area: textbox "Ask a question..." with:
  - "Add attachment" button (paperclip icon)
  - "Send message" button (disabled when empty)

### Key design decisions
- Same push-layout paradigm as GitBook — panel is not a floating overlay.
- Clean, minimal panel design with no role-playing greeting text.
- Suggestions are presented as plain text buttons (not pills/chips).
- Attachment support (unique among the four examples).
- Toggle button has a visual `[active]` state.
- Source verified by the presence of Mintlify-specific navigation patterns (`Navigation Getting Started Overview` breadcrumb in header, `docs.perplexity.ai` domain).

---

## Comparative Summary

| Feature | GitHub Copilot | Intercom | GitBook | Mintlify |
|---------|:---:|:---:|:---:|:---:|
| **Position** | Center overlay | Bottom-right floating | Right sidebar | Right sidebar |
| **Layout impact** | Overlay covers page | Overlay covers content | Pushes content left | Pushes content left |
| **Always visible?** | No (hidden until triggered) | Yes (floating launcher) | Yes (header button) | Yes (header button) |
| **Toggle** | Header button + `?search-...=true` | Click bubble | Header button + inline button + `?ask=` | Header button |
| **AI + Search united?** | Yes — one combined input | No — chat only | Yes — "Ask, search, or explain..." | No — "Ask a question..." only |
| **Context-aware** | Implicit (current docs page) | Implicit (page URL) | Explicit "Based on your context" | Not indicated |
| **Suggested questions** | Yes (4) | Varies (configurable) | Yes (3 pills) | Yes (3 buttons) |
| **Privacy/Disclaimer** | Yes (footer note) | Configurable | No (not observed) | Yes ("may contain mistakes") |

### Key Patterns Observed

1. **Floating Chat Bubble (Intercom):** The oldest and most widely copied pattern. A persistent, always-visible launcher in the bottom-right corner. Best for: **customer support / sales** where proactive engagement matters. Drawback: takes up screen real estate and can feel intrusive on documentation pages.

2. **Full-Screen Search Overlay (GitHub Copilot):** Treats AI as an extension of search — one input for both. The UI is essentially a search dialog augmented with Q&A. Best for: **large knowledge bases** where finding content and asking questions are intertwined. Clean — no persistent widget.

3. **Right Sidebar with Push Layout (GitBook & Mintlify):** The AI assistant lives as a persistent sidebar panel that **resizes the page** rather than covering it. Users can read docs and chat simultaneously. Best for: **technical documentation** where the assistant should augment rather than replace the reading experience.

4. **None of these** use a "cover without resizing" overlay pattern outside of mobile contexts. The push-layout approach (GitBook/Mintlify) is emerging as the preferred pattern for documentation AI assistants, as it preserves access to the source content while enabling back-and-forth questioning.

### Design Implications for Documentation AI Assistants

- **Push layout > Overlay** for doc contexts: users need to reference the docs while questioning.
- **Context awareness matters**: explicitly telling users the assistant knows which page they're on builds trust (GitBook's "Based on your context").
- **Combined search + ask is powerful**: GitHub's single-input pattern reduces cognitive load — users don't have to choose between "searching" and "asking".
- **Persistent launcher vs. header button**: the floating bubble feels necessary for support tools but header buttons are sufficient and cleaner for docs.
- **Suggested questions are universal**: all four patterns include them — they seed the interaction and demonstrate what the assistant can do.
