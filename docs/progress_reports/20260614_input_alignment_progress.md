# Message Input Alignment — Progress Notes

**Date:** 2026-06-14  
**CSS:** `rag_engine/resources/css/cmw_copilot_theme.css`  
**Surfaces:** embedded KB widget (`#cmw-widget-container`), standalone `/kb_assist`, main `/` demo  
**Scratch artifacts (local only):** `.scratch/20260614_input_alignment/` — Playwright measurement scripts and before/after PNGs; gitignored.

---

## Problem

The message input row (`#input-row`) did not line up with the chatbot card (`#chatbot-main`) in the embedded widget pane: horizontal edge mismatch, uneven card chrome (shadow/padding), and extra bottom spacing. kb_assist and standalone had similar Gradio DOM layering issues.

---

## Root causes

1. **Gradio `.row` gutter** — Bootstrap-style `margin-left/right: -15px` on `#input-row` pulled the input wider/narrower than the chatbot card above it.
2. **Wrong shadow target** — Card styles (`.message-card`, padding, shadow) applied to `#message-input` outer shell, but Gradio renders visible chrome on inner `label.container` → shadow looked missing or clipped.
3. **Double padding shell** — `#message-input.message-card.padded` inherited card padding while inner layers also had min-height/padding → textarea sat shorter than the visible card; shell-minus-textarea gap ~16px.
4. **Overflow clipping** — Gradio sets `overflow: hidden` on `.block`; box-shadow on inner container was clipped unless outer shell had `overflow: visible`.
5. **Flex chain regression** — Later chatbot height tweaks broke `#assistant-column` flex; input row lost bottom alignment until flex chain restored.

---

## What worked (commits on `main`)

| Commit | Fix |
|--------|-----|
| `496cc15` | `#input-row` `margin-left/right: 0` — cancel Gradio row gutter |
| `08f89de` | Card shadow on visible field (first attempt on outer shell) |
| `0cba575` | Move card chrome to `#message-input label.container`; zero outer shell padding/border/shadow |
| `919b2bd` | Shadow/height parity: `min-height: 48px` on label + input-container; `overflow: visible` on message-input stack |
| `a37af0b` | `#input-row` `margin-bottom: 0` — remove extra bottom gap |
| `6d082b4` | Restore `#assistant-column` / `#chatbot-main` flex chain after height calc changes |
| `7b11806` | Download button 30×30 circle beside submit — row `align-items: center` |

**Stable pattern:** treat `#message-input` as a transparent flex shell; put border, background, shadow, and `min-height` on `label.container`; keep `#message-input.message-card` padding at zero.

---

## Dead ends (do not retry)

- **Patching only `#message-input.message-card` padding/shadow** without retargeting `label.container` — shadow still invisible or misaligned; outer box and inner visible field are different elements.
- **Relying on `.message-card` group selector alone** — `:not(#message-input)` excludes input from shared card rules; input needs its own block (see lines ~459–515 in theme CSS).
- **Measuring only `#input-row` vs `#chatbot-main` rects** — delta can look fine while inner `label.container` shadow still clips; measure shell-minus-textarea and computed `box-shadow` on `label.container`.
- **Adding padding to outer `#message-input` to “match” chatbot** — doubles with Gradio inner structure; zero outer padding + chrome on `label.container` is correct.
- **Svelte hash selectors** — brittle across Gradio builds; use stable `elem_id`s (`#message-input`, `#input-row`, `#chatbot-main`, `#assistant-column`).
- **Committing Playwright scratch under `docs/progress_reports/`** — one-off scripts; keep in `.scratch/` (see AGENTS.md).

---

## Verification (manual)

1. Embedded widget on KB article page: open pane, compare left/right edges of chatbot card and input card.
2. `http://127.0.0.1:7862/kb_assist/` — same alignment with download + submit buttons.
3. Resize textarea vertically — card grows, shadow not clipped.
4. Dark mode — border/shadow use `--cmw-border` / `--cmw-shadow-soft`.

Optional local remeasure: `.scratch/20260614_input_alignment/verify_input_fix.js` (requires local app + Playwright; not CI).

---

## Related

- Broader KB widget context: `docs/progress_reports/20260610-kb-assist-pane-progress.md`
- Plan (chat download row layout): `.opencode/plans/20260614_chat_download/plan.md`
