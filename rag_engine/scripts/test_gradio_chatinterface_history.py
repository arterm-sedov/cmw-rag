"""Test script to test regular gr.Chatbot approach (like reference agent).

This script uses regular gr.Chatbot with manual button clicks instead of ChatInterface.
This matches the reference agent pattern which doesn't duplicate history.

Based on reference agent: .reference-repos/.cmw-platform-agent/agent_ng/tabs/chat_tab.py

Usage:
    python -m rag_engine.scripts.test_gradio_chatinterface_history

    Or activate venv first:
    .venv\\Scripts\\Activate.ps1  # Windows PowerShell
    .venv-wsl/bin/activate         # WSL/Linux
    python -m rag_engine.scripts.test_gradio_chatinterface_history

The app will launch on http://127.0.0.1:7861

Key Pattern (from reference agent):
- Use regular gr.Chatbot (NOT ChatInterface)
- Build working_history = history + [new messages]
- Always yield full working_history list
- Update messages in-place by index
- Don't check for duplicates - trust history parameter
"""

import time
from collections.abc import Generator
from uuid import uuid4

import gradio as gr


def simulate_search_started(query: str) -> dict:
    """Simulate search started thinking block."""
    return {
        "role": "assistant",
        "content": f"Ищу: {query}",
        "metadata": {"title": "🧠 Поиск информации в базе знаний"},
    }


def simulate_search_completed(count: int) -> dict:
    """Simulate search completed thinking block."""
    return {
        "role": "assistant",
        "content": f"Найдено статей: {count}.",
        "metadata": {"title": "✅ Поиск завершен"},
    }


def clear_history():
    """Clear chatbot history and reset state."""
    print("History cleared!")
    # Return empty list for chatbot, empty MultimodalTextbox value, and a new UUID for state
    return [], {"text": "", "files": []}, uuid4()


def chat_handler(multimodal_value, history: list[dict], session_uuid) -> Generator[tuple[list[dict], dict], None, None]:
    """Chat handler using reference agent pattern.

    Reference agent pattern:
    1. Extract text from MultimodalTextbox format (dict with 'text' and 'files')
    2. Build working_history = history + [new messages]
    3. Always yield full working_history list
    4. Update messages in-place by index
    5. Don't check for duplicates - trust history parameter
    """
    # Extract text from MultimodalTextbox format (like reference agent)
    if isinstance(multimodal_value, dict):
        message = multimodal_value.get("text", "")
        # files = multimodal_value.get("files", [])  # Not used in test, but available for file handling
    else:
        # Fallback for non-dict values
        message = str(multimodal_value) if multimodal_value else ""

    print(f"[Chat] Message: {message}, History length: {len(history)}, Session: {session_uuid}")

    if not message or not message.strip():
        yield history, {"text": "", "files": []}
        return

    # Reference agent pattern: working_history = history + [new messages]
    # Start with provided history and build incrementally
    working_history = list(history) if history else []

    # Add user message (reference agent: always add, don't check)
    user_msg = {"role": "user", "content": message}
    working_history.append(user_msg)

    # Add search started
    search_started = simulate_search_started(message)
    working_history.append(search_started)
    # Yield full history (reference agent pattern: always yield full working_history)
    # Return empty MultimodalTextbox value to clear input
    yield working_history, {"text": "", "files": []}

    # Simulate tool execution
    time.sleep(0.5)

    # Update search started with LLM query (reference agent: update in-place by index)
    search_started["content"] = f"Ищу: {message.upper()}"
    # Yield full history with updated search
    yield working_history, {"text": "", "files": []}

    # Add search completed
    search_completed = simulate_search_completed(5)
    working_history.append(search_completed)
    # Yield full history with search completed
    yield working_history, {"text": "", "files": []}

    # Stream answer - update last message in-place (reference agent pattern)
    answer = ""
    assistant_message_index = len(working_history) - 1  # Track assistant message index
    for chunk in ["Это ", "ответ ", "на ", "вопрос: ", message]:
        answer += chunk
        # Update last message in-place (reference agent pattern)
        if assistant_message_index >= 0 and assistant_message_index < len(working_history):
            working_history[assistant_message_index] = {"role": "assistant", "content": answer}
        else:
            working_history.append({"role": "assistant", "content": answer})
            assistant_message_index = len(working_history) - 1
        # Yield full history - always yield full working_history
        # Return empty MultimodalTextbox value to clear input
        yield working_history, {"text": "", "files": []}
        time.sleep(0.2)


with gr.Blocks(title="Gradio Chatbot Test (Reference Agent Pattern)") as demo:
    gr.Markdown("# Gradio Chatbot Test - Reference Agent Pattern")
    gr.Markdown(
        """
        **Testing with Gradio 6.1.0** - Using regular gr.Chatbot (NOT ChatInterface)

        This matches the reference agent pattern:
        - Regular gr.Chatbot component (not ChatInterface)
        - Manual button clicks (send button, clear button)
        - Build working_history = history + [new messages]
        - Always yield full working_history list
        - Update messages in-place by index

        **Instructions:**
        1. Send first message: "Привет"
        2. Wait for response to complete
        3. Send second message: "Как дела?"
        4. Check if first conversation is duplicated (should NOT duplicate)
        5. Check if thinking blocks persist during streaming (should persist)
        6. Click "Clear" to reset history and state
        """
    )

    # Create chatbot component (like reference agent)
    # In Gradio 6, Chatbot uses messages format by default (no type parameter)
    # show_copy_button removed - use buttons parameter instead
    chatbot = gr.Chatbot(
        label="Test Chat",
        height=400,
        show_label=True,
        container=True,
        buttons=["copy"],
    )

    # Create message input (like reference agent)
    msg = gr.MultimodalTextbox(
        label="Message",
        placeholder="Type your message...",
        lines=2,
        max_lines=4,
    )

    # Create buttons (like reference agent)
    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

    # Create state for session tracking
    session_state = gr.State(value=uuid4())

    # Connect events (like reference agent)
    # Send button click
    send_btn.click(
        fn=chat_handler,
        inputs=[msg, chatbot, session_state],
        outputs=[chatbot, msg],
    )

    # Message submit (Enter key)
    msg.submit(
        fn=chat_handler,
        inputs=[msg, chatbot, session_state],
        outputs=[chatbot, msg],
    )

    # Clear button
    clear_btn.click(
        fn=clear_history,
        outputs=[chatbot, msg, session_state]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
