# Chat Persistence Across Article Navigations — Feasibility Research

**Date:** 2026-06-13

**Scope:** Can the AI assistant chatbot retain conversation history when the user navigates between KB articles (full page reloads)? Without modifying Gradio source code.

## Architecture Constraints

1. **Full page reloads:** PHPKB is traditional server-rendered. Every article click destroys the `<gradio-app>` custom element and rebuilds it from scratch.

2. **Cross-origin Gradio:** Gradio runs at `https://ennoia.slickjump.org/kb_assist`, KB at `kb.comindware.ru`. No shared `localStorage`, cookies, or DOM access between them.

3. **Gradio session hash is random:** `Math.random().toString(36).substring(2)` generated in `Client` constructor (`client.ts:53`) on every page load. The `<gradio-app>` SPA never passes a persisted `session_hash`.

4. **Backend sessions are in-memory** and TTL-bound. Even with a stable hash, server restart or TTL expiry loses state.

5. **Chatbot `value` is internal Svelte 5 reactive state** — not exposed as DOM property, custom element attribute, or public API on `<gradio-app>`. Cannot be read directly from outside.

## Can We Read Chat History Without Gradio Source Changes?

**Direct access to the `value` prop: No.** The `get_data` callback (`$state.snapshot(this.props)`) is Svelte-internal. There is no `ga.chatHistory` or similar public property on the custom element.

**BUT we can parse the rendered DOM** because chatbot messages render as HTML elements inside `<gradio-app>`. The `sendMessageToGradio` function already proves we can read and write to the DOM from the KB page (light DOM — no shadow root).

## Approach: Save → POST → Restore via Message Replay

### Step 1: On `beforeunload`, serialize chatbot DOM

```
beforeunload listener:
  1. Read all message bubbles from rendered chatbot DOM
  2. Extract user/assistant text pairs
  3. POST to a PHP endpoint on kb.comindware.ru (same origin — no CORS)
  4. PHP saves to MySQL or PHP session
```

**Risks:**
- DOM parsing is fragile (code blocks, file attachments, markdown formatting)
- Large histories may cause `beforeunload` to delay navigation

### Step 2: On next page load, PHP injects saved history

```
article.php (PHP):
  1. Read saved history from MySQL/session
  2. Inject as JS array in KB_CONTEXT or a <script> tag

ai-assistant.php (JS):
  1. After Gradio loads, iterate saved message pairs
  2. Replay each via sendMessageToGradio()
```

**Caveats:**
- Replaying via textbox re-processes messages on the Gradio backend (each message triggers a backend call)
- User sees messages appear one by one (not instant restore)
- Timing: need to wait for Gradio to be fully ready before replaying

## Alternative: localStorage-only (no backend)

| Step | What |
|------|------|
| 1 | On `beforeunload`, serialize chatbot DOM → JSON → `localStorage.setItem('cmw_chat_history', ...)` |
| 2 | On page load, read `localStorage` → inject as `window.KB_CONTEXT.saved_history` |
| 3 | After Gradio init, replay pairs via `sendMessageToGradio` |

Avoids any backend work. History stays in the browser. But still has the same DOM-parsing and replay timing challenges.

## Limitation: Replay Triggers Backend Processing

Every `sendMessageToGradio` call dispatches a Gradio submit event, sending the full conversation history + one new message to the backend. The backend re-processes and returns the updated history. This means:

- N previous messages = N backend round trips
- Each call waits for the previous response before proceeding (or they'd race)
- Network latency × N can be slow for long histories

## Could the History Be Injected as Initial Value?

If we could set the chatbot's initial `value` before Gradio renders, we'd skip replay entirely. But this requires either:

1. Modifying the Gradio SPA to accept `initial_value` via URL param → **excluded**
2. Modifying the Gradio Python backend to restore from a database → **backend work, not source modification**
3. Finding a window between `gradio-app` mount and first render where we can `postMessage` an initial state → **unreliable, undocumented**

None of these are viable without touching Gradio source or backend.

## Recommendation

The viable path for **engineering effort vs. reliability** is:

1. **POST-based save (PHP endpoint, MySQL):** Stable, survives browser clear, works across devices for logged-in users.
2. **Replay via `sendMessageToGradio`:** Already proven to work (the chat-clearing bug was fixed). The known limitation is speed (N round trips) but functionally correct.
3. **Accept the trade-off:** History between articles is restored with a brief replay animation. For any serious use case, keeping the Gradio session alive (one-page KB app) would be the correct architectural fix.

## Source References

- Gradio session hash generation: `client/js/src/client.ts:53`
- `Client.connect()` accepts `session_hash` option: `client/js/src/client.ts:263-275`
- `<gradio-app>` mount sequence: `js/spa/src/main.ts`
- `gather_state` reads `value` from component props: `js/app/src/dependency.ts:749-761`
- Chatbot `value` is Svelte 5 `$state`: `js/chatbot/Index.svelte`
- PHP session starts in `kb.comindware.ru/include/db-conn.php:3`
- Widget localStorage key: `cmw_chat_widget_state` (ai-assistant.php line 756)
