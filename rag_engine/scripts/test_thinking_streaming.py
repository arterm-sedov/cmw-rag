"""Test script for Gradio thinking blocks and streaming persistence.

This script tests how thinking blocks (metadata messages) persist when streaming
answer text in Gradio Chatbot. It simulates the agent pipeline to identify
issues with block persistence.

Usage:
    python -m rag_engine.scripts.test_thinking_streaming

    Or activate venv first:
    .venv\\Scripts\\Activate.ps1  # Windows PowerShell
    .venv-wsl/bin/activate         # WSL/Linux
    python -m rag_engine.scripts.test_thinking_streaming

The app will launch on http://127.0.0.1:7861

Test Cases:
    - V1: Original approach (yields dicts then strings) - may have issues
    - V2: Full history approach (always yields complete message list) - potential fix
    - V3: ChatMessage approach (uses ChatMessage dataclass) - potential fix

Expected Behavior:
    - AI disclaimer should remain visible throughout
    - Thinking blocks (search started/completed) should remain visible
    - Answer should stream below thinking blocks without replacing them
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import time
from collections.abc import Generator

import gradio as gr

from rag_engine.utils.message_utils import get_message_content


# Mock AI disclaimer
AI_DISCLAIMER = "⚠️ This is an AI-generated response. Please verify important information."


def mock_ai_pipeline(message: str, history: list[dict]) -> Generator[list[dict], None, None]:
    """Mock AI pipeline that simulates thinking blocks and streaming.

    This function mimics the behavior of agent_chat_handler:
    1. Yields AI disclaimer
    2. Yields thinking block (search started)
    3. Simulates tool execution delay
    4. Yields thinking block (search completed)
    5. Streams answer text chunks

    Args:
        message: User message (may be empty if extracted from history)
        history: Chat history (already contains user message)

    Yields:
        Full message history list (required for Chatbot component streaming)
    """
    # Build message history from provided history
    messages = list(history) if history else []
    
    # Extract user message text properly (handles structured content format)
    if message and message.strip():
        user_msg_text = message
    elif messages:
        # Extract from last user message in history
        last_msg = messages[-1]
        user_msg_text = get_message_content(last_msg) or ""
    else:
        user_msg_text = ""
    
    # Ensure user message is in history (should already be there from submit handler)
    # But if not, add it
    if not messages or messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": user_msg_text})

    # Step 1: Add AI disclaimer
    messages.append({
        "role": "assistant",
        "content": AI_DISCLAIMER,
    })
    yield messages.copy()

    # Step 2: Add thinking block for search started
    messages.append({
        "role": "assistant",
        "content": f"Searching for: {user_msg_text}",
        "metadata": {
            "title": "🧠 Searching the knowledge base",
            "status": "pending",
        },
    })
    yield messages.copy()

    # Step 3: Simulate tool execution delay
    time.sleep(1.5)

    # Step 4: Add thinking block completed
    messages.append({
        "role": "assistant",
        "content": "Found articles: 3.",
        "metadata": {
            "title": "✅ Search completed",
            "status": "done",
        },
    })
    yield messages.copy()

    # Step 5: Stream answer text chunks
    # Create a new message for the answer and update it
    answer_chunks = [
        "Based on the articles I found, ",
        "here is the answer to your question: ",
        f"{user_msg_text} can be configured by following these steps:\n\n",
        "1. First, you need to access the settings.\n",
        "2. Then, configure the relevant parameters.\n",
        "3. Finally, save your changes.\n\n",
        "This should help you get started!",
    ]

    accumulated_answer = ""
    answer_message_added = False
    
    for chunk in answer_chunks:
        accumulated_answer += chunk
        # Create new message for answer (separate from thinking blocks)
        if not answer_message_added:
            messages.append({"role": "assistant", "content": accumulated_answer})
            answer_message_added = True
        else:
            # Update the last message (which is the answer message)
            messages[-1] = {"role": "assistant", "content": accumulated_answer}
        yield messages.copy()

        # Small delay to simulate streaming
        time.sleep(0.3)


def mock_ai_pipeline_v2(message: str, history: list[dict]) -> Generator[list[dict], None, None]:
    """Alternative implementation: Always yield full message history.

    This version maintains the full message history and appends to it,
    which should preserve all thinking blocks.

    Args:
        message: User message (may be empty if extracted from history)
        history: Chat history (already contains user message)

    Yields:
        Full message history list with all messages preserved
    """
    # Build message history from provided history
    messages = list(history) if history else []
    
    # Extract user message text properly (handles structured content format)
    if message and message.strip():
        user_msg_text = message
    elif messages:
        # Extract from last user message in history
        last_msg = messages[-1]
        user_msg_text = get_message_content(last_msg) or ""
    else:
        user_msg_text = ""
    
    # Ensure user message is in history (should already be there from submit handler)
    if not messages or messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": user_msg_text})

    # Step 1: Add AI disclaimer
    messages.append({
        "role": "assistant",
        "content": AI_DISCLAIMER,
    })
    yield messages.copy()

    # Step 2: Add thinking block for search started
    messages.append({
        "role": "assistant",
        "content": f"Searching for: {user_msg_text}",
        "metadata": {
            "title": "🧠 Searching the knowledge base",
            "status": "pending",
        },
    })
    yield messages.copy()

    # Step 3: Simulate tool execution delay
    time.sleep(1.5)

    # Step 4: Update thinking block to completed
    messages.append({
        "role": "assistant",
        "content": "Found articles: 3.",
        "metadata": {
            "title": "✅ Search completed",
            "status": "done",
        },
    })
    yield messages.copy()

    # Step 5: Stream answer by creating a new message and updating it
    answer_chunks = [
        "Based on the articles I found, ",
        "here is the answer to your question: ",
        f"{user_msg_text} can be configured by following these steps:\n\n",
        "1. First, you need to access the settings.\n",
        "2. Then, configure the relevant parameters.\n",
        "3. Finally, save your changes.\n\n",
        "This should help you get started!",
    ]

    accumulated_answer = ""
    answer_message_added = False
    
    for chunk in answer_chunks:
        accumulated_answer += chunk
        # Create new message for answer (separate from thinking blocks)
        if not answer_message_added:
            messages.append({"role": "assistant", "content": accumulated_answer})
            answer_message_added = True
        else:
            # Update the last message (which is the answer message)
            messages[-1] = {"role": "assistant", "content": accumulated_answer}
        yield messages.copy()

        time.sleep(0.3)


def mock_ai_pipeline_v3(message: str, history: list[dict]) -> Generator[list[dict], None, None]:
    """Version 3: Use ChatMessage dataclass but yield full history.

    This version uses Gradio's ChatMessage pattern but yields full history
    to ensure compatibility with Chatbot component streaming.

    Args:
        message: User message (may be empty if extracted from history)
        history: Chat history (already contains user message)

    Yields:
        Full message history list with ChatMessage-style dicts
    """
    # Build message history from provided history
    messages = list(history) if history else []
    
    # Extract user message text properly (handles structured content format)
    if message and message.strip():
        user_msg_text = message
    elif messages:
        # Extract from last user message in history
        last_msg = messages[-1]
        user_msg_text = get_message_content(last_msg) or ""
    else:
        user_msg_text = ""
    
    # Ensure user message is in history (should already be there from submit handler)
    if not messages or messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": user_msg_text})

    # Step 1: Add AI disclaimer using ChatMessage-style dict
    messages.append({
        "role": "assistant",
        "content": AI_DISCLAIMER,
    })
    yield messages.copy()

    # Step 2: Add thinking block for search started
    messages.append({
        "role": "assistant",
        "content": f"Searching for: {user_msg_text}",
        "metadata": {
            "title": "🧠 Searching the knowledge base",
            "status": "pending",
        },
    })
    yield messages.copy()

    # Step 3: Simulate tool execution delay
    time.sleep(1.5)

    # Step 4: Add thinking block completed
    messages.append({
        "role": "assistant",
        "content": "Found articles: 3.",
        "metadata": {
            "title": "✅ Search completed",
            "status": "done",
        },
    })
    yield messages.copy()

    # Step 5: Stream answer - create new message and update it
    answer_chunks = [
        "Based on the articles I found, ",
        "here is the answer to your question: ",
        f"{user_msg_text} can be configured by following these steps:\n\n",
        "1. First, you need to access the settings.\n",
        "2. Then, configure the relevant parameters.\n",
        "3. Finally, save your changes.\n\n",
        "This should help you get started!",
    ]

    accumulated_answer = ""
    answer_message_added = False
    
    for chunk in answer_chunks:
        accumulated_answer += chunk
        if not answer_message_added:
            # Create new message for answer
            messages.append({"role": "assistant", "content": accumulated_answer})
            answer_message_added = True
        else:
            # Update the last message (which is the answer message)
            messages[-1] = {"role": "assistant", "content": accumulated_answer}
        yield messages.copy()

        time.sleep(0.3)


def chat_handler_v1(message: str, history: list[dict]) -> Generator[list[dict], None, None]:
    """Handler using full history approach (V1 variant).
    
    This version yields complete message history to ensure all messages
    (disclaimer, thinking blocks, answer) are preserved during streaming.
    """
    # Message extraction is handled inside mock_ai_pipeline
    # Pass empty string if message was cleared (it's already in history)
    user_msg = message if message and message.strip() else ""
    yield from mock_ai_pipeline(user_msg, history)


def chat_handler_v2(message: str, history: list[dict]) -> Generator[list[dict], None, None]:
    """Handler using full history approach (yields complete message lists).
    
    This approach always yields the complete message history, ensuring
    all previous messages (disclaimer, thinking blocks) are preserved.
    
    Potential solution: Always yield full history instead of individual messages/strings.
    """
    # Message extraction is handled inside mock_ai_pipeline_v2
    # Pass empty string if message was cleared (it's already in history)
    user_msg = message if message and message.strip() else ""
    yield from mock_ai_pipeline_v2(user_msg, history)


def chat_handler_v3(message: str, history: list[dict]) -> Generator[list[dict], None, None]:
    """Handler using full history approach (V3 variant).
    
    Uses full history approach with ChatMessage-style message structure.
    Functionally similar to V2 but demonstrates the pattern with explicit structure.
    """
    # Message extraction is handled inside mock_ai_pipeline_v3
    # Pass empty string if message was cleared (it's already in history)
    user_msg = message if message and message.strip() else ""
    yield from mock_ai_pipeline_v3(user_msg, history)


# Create test interface
with gr.Blocks(title="Thinking Blocks Streaming Test") as demo:
    gr.Markdown("""
    # Testing Thinking Blocks Persistence During Streaming

    This app tests different approaches to keep thinking blocks visible when streaming answer text.

    **Test Cases:**
    All versions now use the full history approach (yielding complete message lists)
    to ensure compatibility with Chatbot component streaming:
    1. **V1**: Full history approach (tests if this fixes the original issue)
    2. **V2**: Full history approach (baseline implementation)
    3. **V3**: Full history approach with ChatMessage-style structure

    **Expected Behavior:**
    - AI disclaimer should remain visible
    - Thinking blocks (search started/completed) should remain visible
    - Answer should stream below thinking blocks
    """)

    with gr.Tabs():
        with gr.Tab("V1: Dicts then Strings"):
            chatbot_v1 = gr.Chatbot(
                label="V1: Original Approach",
                render_markdown=True,
            )
            msg_v1 = gr.Textbox(
                label="Message",
                placeholder="Type a question...",
            )
            msg_v1.submit(
                lambda msg, hist: ("", hist + [{"role": "user", "content": msg}]),
                inputs=[msg_v1, chatbot_v1],
                outputs=[msg_v1, chatbot_v1],
            ).then(
                chat_handler_v1,
                inputs=[msg_v1, chatbot_v1],
                outputs=chatbot_v1,
            )

        with gr.Tab("V2: Full History"):
            chatbot_v2 = gr.Chatbot(
                label="V2: Full History Approach",
                render_markdown=True,
            )
            msg_v2 = gr.Textbox(
                label="Message",
                placeholder="Type a question...",
            )
            msg_v2.submit(
                lambda msg, hist: ("", hist + [{"role": "user", "content": msg}]),
                inputs=[msg_v2, chatbot_v2],
                outputs=[msg_v2, chatbot_v2],
            ).then(
                chat_handler_v2,
                inputs=[msg_v2, chatbot_v2],
                outputs=chatbot_v2,
            )

        with gr.Tab("V3: ChatMessage"):
            chatbot_v3 = gr.Chatbot(
                label="V3: ChatMessage Approach",
                render_markdown=True,
            )
            msg_v3 = gr.Textbox(
                label="Message",
                placeholder="Type a question...",
            )
            msg_v3.submit(
                lambda msg, hist: ("", hist + [{"role": "user", "content": msg}]),
                inputs=[msg_v3, chatbot_v3],
                outputs=[msg_v3, chatbot_v3],
            ).then(
                chat_handler_v3,
                inputs=[msg_v3, chatbot_v3],
                outputs=chatbot_v3,
            )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7861)

