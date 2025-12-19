"""Demo script to visualize Gradio spinner implementation.

This script creates a simple Gradio interface that demonstrates how spinners
appear and disappear during different operations, without requiring the full
RAG system to be running.

Run with:
    python rag_engine/scripts/demo_spinners.py
"""

from __future__ import annotations

import asyncio
import logging

import gradio as gr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_handler(
    message: str,
    history: list[dict],
) -> list[dict]:
    """Demo handler that simulates agent operations with spinners.

    Shows the visual flow of spinners appearing and disappearing during
    different operations.

    Args:
        message: User message
        history: Chat history

    Yields:
        Updated history with spinner demonstrations
    """
    # Build working history
    working_history = list(history) if history else []

    # 1. Show thinking block with spinner (pending)
    thinking_msg = {
        "role": "assistant",
        "content": "Analyzing your question...",
        "metadata": {
            "title": "🧠 Thinking...",
            "ui_type": "thinking",
            "status": "pending",  # ← Shows spinner
        },
    }
    working_history.append(thinking_msg)
    yield working_history
    await asyncio.sleep(2)  # Simulate thinking time

    # Update thinking to done before moving to search
    thinking_msg["metadata"]["status"] = "done"  # ← Stop spinner
    yield working_history

    # 2. Show search started with spinner (pending)
    search_msg = {
        "role": "assistant",
        "content": f'Searching for: "{message}"',
        "metadata": {
            "title": "🔍 Searching...",
            "ui_type": "search_started",
            "status": "pending",  # ← Shows spinner
        },
    }
    working_history.append(search_msg)
    yield working_history
    await asyncio.sleep(2)  # Simulate search time

    # Update search to done before showing completed message
    search_msg["metadata"]["status"] = "done"  # ← Stop spinner
    yield working_history

    # 3. Show search completed (no spinner, stays open to show sources)
    completed_msg = {
        "role": "assistant",
        "content": "Found 5 relevant articles\n\nSources:\n1. [Article A](https://example.com/a)\n2. [Article B](https://example.com/b)\n3. [Article C](https://example.com/c)",
        "metadata": {
            "title": "✅ Search completed",
            "ui_type": "search_completed",
            # NO status - accordion stays open to show clickable article links
        },
    }
    working_history.append(completed_msg)
    yield working_history
    await asyncio.sleep(0.5)

    # 4. Show "Generating answer" with spinner (simulates LLM processing time)
    generating_msg = {
        "role": "assistant",
        "content": "Composing response based on retrieved information...",
        "metadata": {
            "title": "✍️ Generating answer",
            "ui_type": "generating_answer",
            "status": "pending",  # ← Shows spinner during LLM processing
        },
    }
    working_history.append(generating_msg)
    yield working_history
    await asyncio.sleep(2)  # Simulate slow LLM response time

    # Stop generating spinner before streaming answer
    generating_msg["metadata"]["status"] = "done"  # ← Stop spinner
    yield working_history

    # 6. Stream the final answer
    answer = f"Here's what I found about '{message}':\n\n"
    answer += "This is a demonstration of the Gradio spinner feature. "
    answer += "Notice how the spinners appeared next to 'Thinking', 'Searching', "
    answer += "and 'Generating answer' messages, then disappeared when operations completed.\n\n"
    answer += "**Key Points:**\n"
    answer += "- Spinners appear automatically when `status='pending'`\n"
    answer += "- Spinners disappear when status is updated to `status='done'`\n"
    answer += "- Messages must be updated in place (not just add new messages)\n"
    answer += "- The 'Generating answer' spinner is especially useful for slow LLM responses\n"
    answer += "- This is a native Gradio feature (no custom code needed)\n"

    # Stream the answer character by character for effect
    assistant_msg = {"role": "assistant", "content": ""}
    working_history.append(assistant_msg)

    for char in answer:
        assistant_msg["content"] += char
        working_history[-1] = assistant_msg
        yield working_history
        await asyncio.sleep(0.01)  # Simulate streaming

    # Final yield
    yield working_history


# Create demo interface
with gr.Blocks(title="Gradio Spinners Demo") as demo:
    gr.Markdown("# 🎯 Gradio Native Spinners Demo")
    gr.Markdown(
        """
        This demo shows how spinners appear during agent operations.
        
        **Try it:**
        1. Type any question (e.g., "authentication", "workflows", "API")
        2. Watch for spinners appearing next to metadata messages
        3. See how spinners disappear when operations complete
        
        **What to observe:**
        - 🧠 Thinking... [spinner] → appears first
        - 🔍 Searching... [spinner] → appears during search
        - ✅ Search completed → spinner disappears
        - ✍️ Generating answer... [spinner] → appears while LLM processes
        - Answer streams after all operations complete
        """
    )

    chatbot = gr.Chatbot(
        label="Demo Chat",
        height="500px",
        show_label=True,
        container=True,
        buttons=["copy"],
    )

    msg = gr.Textbox(
        label="Message",
        placeholder="Type any question to see spinners in action...",
        lines=1,
        max_lines=1,
        show_label=False,
        submit_btn=True,
        stop_btn=True,
    )

    # State for saved input
    saved_input = gr.State()

    def clear_and_save(message: str) -> tuple[gr.Textbox, str]:
        """Clear textbox and save message."""
        return gr.Textbox(value="", interactive=False), message

    # Submit event chain
    submit_event = (
        msg.submit(
            fn=clear_and_save,
            inputs=[msg],
            outputs=[msg, saved_input],
            queue=False,
        )
        .then(
            lambda message, history: history + [{"role": "user", "content": message}],
            inputs=[saved_input, chatbot],
            outputs=[chatbot],
            queue=False,
        )
        .then(
            fn=demo_handler,
            inputs=[saved_input, chatbot],
            outputs=[chatbot],
        )
        .then(
            lambda: gr.Textbox(value="", interactive=True),
            outputs=[msg],
        )
    )

    # Stop button
    msg.stop(fn=None, cancels=[submit_event])

if __name__ == "__main__":
    logger.info("Starting Gradio Spinners Demo at http://localhost:7860")
    logger.info("Watch for spinners appearing and disappearing during operations!")

    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )

