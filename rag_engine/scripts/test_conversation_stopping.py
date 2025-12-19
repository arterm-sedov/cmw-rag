"""Test script to rehearse conversation stopping with mocked agent features."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncGenerator, Generator
from queue import Empty, Queue

import gradio as gr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockConversationStore:
    """Mock conversation store for testing memory saving."""

    def __init__(self):
        self._sessions: dict[str, list[tuple[str, str]]] = {}

    def get(self, session_id: str) -> list[tuple[str, str]]:
        """Get conversation history for a session."""
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, role: str, content: str) -> None:
        """Append a turn to conversation history."""
        history = self._sessions.setdefault(session_id, [])
        history.append((role, content))
        logger.info(
            f"MockConversationStore: Saved {role} turn to session {session_id[:8]}... ({len(content)} chars)"
        )

    def get_formatted_history(self, session_id: str) -> str:
        """Get formatted conversation history for display."""
        history = self.get(session_id)
        if not history:
            return "No conversation memory saved yet."
        lines = []
        for role, content in history:
            role_emoji = "👤" if role == "user" else "🤖"
            # Truncate long content for display
            display_content = content[:200] + "..." if len(content) > 200 else content
            lines.append(f"{role_emoji} **{role.upper()}**: {display_content}")
        return "\n\n".join(lines)

    def clear(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"MockConversationStore: Cleared session {session_id[:8]}...")


class MockLLMManager:
    """Mock LLMManager for testing memory saving functionality."""

    def __init__(self):
        self._conversations = MockConversationStore()

    def save_assistant_turn(self, session_id: str | None, content: str) -> None:
        """Save assistant turn to conversation memory."""
        if session_id and content:
            self._conversations.append(session_id, "assistant", content)
        else:
            logger.warning(
                f"MockLLMManager: Cannot save - session_id={session_id}, content_empty={not content}"
            )

    def save_user_turn(self, session_id: str | None, content: str) -> None:
        """Save user turn to conversation memory."""
        if session_id and content:
            self._conversations.append(session_id, "user", content)

    def get_memory_display(self, session_id: str | None) -> str:
        """Get formatted memory for display."""
        if not session_id:
            return "No session ID available."
        return self._conversations.get_formatted_history(session_id)

    def clear_memory(self, session_id: str | None) -> None:
        """Clear conversation memory for a session."""
        if session_id:
            self._conversations.clear(session_id)


# Global mock LLMManager instance
_mock_llm_manager = MockLLMManager()
# Use a fixed test session ID for simplicity
_TEST_SESSION_ID = "test_session_001"


class MockAgentStreamer:
    """Mock agent streamer that simulates agent behavior with stopping support."""

    def __init__(self):
        self._cancelled = False
        self._cancellation_lock = threading.Lock()
        self._cancellation_event = threading.Event()

    def cancel(self):
        with self._cancellation_lock:
            self._cancelled = True
        self._cancellation_event.set()
        logger.info("Mock streamer: Cancellation requested")

    def reset(self):
        with self._cancellation_lock:
            self._cancelled = False
        self._cancellation_event.clear()

    def is_cancelled(self) -> bool:
        if self._cancellation_event.is_set():
            return True
        with self._cancellation_lock:
            return self._cancelled

    async def mock_stream_chat(
        self, message: str, history: list[dict]
    ) -> AsyncGenerator[dict, None]:
        self.reset()
        # Reference agent pattern: working_history = history + [new messages]
        # Preserve ALL messages from history including meta blocks (thinking, searching, etc.)
        # Don't normalize - just make a copy to preserve everything
        working_history = list(history) if history else []

        # Don't add user message here - it's already in history from ChatInterface pattern
        # The submit_event chain adds the user message to chatbot before calling mock_chat_handler

        if self.is_cancelled():
            return

        # Reference agent pattern: append meta blocks, don't replace them
        # Each meta block persists as a separate message in history
        thinking_msg = {
            "role": "assistant",
            "content": "Processing your request...",
            "metadata": {"title": "🧠 Thinking", "ui_type": "thinking"},
        }
        working_history.append(thinking_msg)  # Append, don't replace
        yield {"history": working_history, "status": "thinking"}

        for _ in range(6):
            if self.is_cancelled():
                logger.info("Mock streamer: Cancelled during thinking (outer loop)")
                return
            for _ in range(10):
                if self.is_cancelled() or self._cancellation_event.is_set():
                    logger.info("Mock streamer: Cancelled during thinking (inner loop)")
                    return
                await asyncio.sleep(0.05)
                if self.is_cancelled() or self._cancellation_event.is_set():
                    logger.info("Mock streamer: Cancelled during thinking (after sleep)")
                    return

        if self.is_cancelled():
            return

        # Append search block (don't replace thinking)
        search_msg = {
            "role": "assistant",
            "content": f"Searching for: {message}",
            "metadata": {"title": "🔍 Searching", "ui_type": "search_started"},
        }
        working_history.append(search_msg)  # Append, don't replace
        yield {"history": working_history, "status": "searching"}

        for _ in range(10):
            if self.is_cancelled() or self._cancellation_event.is_set():
                logger.info("Mock streamer: Cancelled during tool execution (outer loop)")
                return
            for _ in range(10):
                if self.is_cancelled() or self._cancellation_event.is_set():
                    logger.info("Mock streamer: Cancelled during tool execution (inner loop)")
                    return
                await asyncio.sleep(0.05)
                if self.is_cancelled() or self._cancellation_event.is_set():
                    logger.info("Mock streamer: Cancelled during tool execution (after sleep)")
                    return

        if self.is_cancelled():
            return

        # Append completion block (don't replace search)
        completion_msg = {
            "role": "assistant",
            "content": "Found 3 relevant articles.",
            "metadata": {"title": "✅ Search completed", "ui_type": "search_completed"},
        }
        working_history.append(completion_msg)  # Append, don't replace
        yield {"history": working_history, "status": "tool_completed"}

        if self.is_cancelled():
            return

        # Append empty assistant message for streaming response (don't remove completion)
        working_history.append({"role": "assistant", "content": ""})
        yield {"history": working_history, "status": "streaming"}

        response_parts = [
            "Based on the search results, ",
            "I can provide you with the following information: ",
            "The topic you asked about is important because ",
            "it relates to several key concepts. ",
            "First, ",
            "there are multiple approaches to consider. ",
            "Second, ",
            "the implementation details matter significantly. ",
            "Finally, ",
            "best practices suggest following these guidelines. ",
            "In summary, ",
            "the answer to your question involves understanding ",
            "the relationship between different components ",
            "and how they interact together. ",
            "This concludes the response.",
        ]

        accumulated_response = ""
        for part in response_parts:
            if self.is_cancelled() or self._cancellation_event.is_set():
                logger.info("Mock streamer: Cancelled during response streaming")
                return

            accumulated_response += part
            working_history[-1] = {"role": "assistant", "content": accumulated_response}
            yield {"history": working_history, "status": "streaming"}
            for _ in range(6):
                if self.is_cancelled() or self._cancellation_event.is_set():
                    logger.info("Mock streamer: Cancelled during response streaming (during sleep)")
                    return
                await asyncio.sleep(0.05)

        logger.info("Mock streamer: Streaming completed successfully")


class AsyncStreamAdapter:
    """Adapter to run async streaming in background thread and yield to sync generator."""

    def __init__(self, streamer: MockAgentStreamer):
        self.streamer = streamer
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._ensure_loop()

    def _ensure_loop(self):
        if self._loop is not None and self._loop.is_running():
            return
        self._loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

    def stream_chat_sync(
        self, message: str, history: list[dict]
    ) -> Generator[dict, None, None]:
        queue: Queue[dict | StopIteration] = Queue()

        async def producer():
            try:
                async for result in self.streamer.mock_stream_chat(message, history):
                    if self.streamer.is_cancelled():
                        logger.info("Producer: Cancellation detected during iteration, stopping")
                        break
                    try:
                        queue.put_nowait(result)
                    except Exception:
                        if self.streamer.is_cancelled():
                            logger.info("Producer: Cancellation detected while queue full, stopping")
                            break
                        queue.put(result)
            except asyncio.CancelledError:
                logger.info("Producer: Coroutine cancelled by asyncio")
                self.streamer.cancel()
            except Exception as e:
                logger.error(f"Producer error: {e}")
                if not self.streamer.is_cancelled():
                    try:
                        queue.put_nowait({"history": history, "status": "error", "error": str(e)})
                    except Exception:
                        pass
            finally:
                try:
                    queue.put_nowait(StopIteration)
                except Exception:
                    pass

        future = asyncio.run_coroutine_threadsafe(producer(), self._loop)

        try:
            while True:
                if self.streamer.is_cancelled():
                    logger.info("Consumer: Cancellation detected at start of loop, stopping")
                    break
                try:
                    item = queue.get(timeout=0.05)
                    if item is StopIteration:
                        break
                    if self.streamer.is_cancelled():
                        logger.info("Consumer: Cancellation detected before yield, stopping immediately")
                        break
                    yield item
                except Empty:
                    if self.streamer.is_cancelled():
                        logger.info("Consumer: Cancellation detected in empty check, stopping")
                        break
                    if future.done():
                        try:
                            future.result()
                        except asyncio.CancelledError:
                            logger.info("Consumer: Future was cancelled")
                        except Exception:
                            pass
                        break
                    continue
        except GeneratorExit:
            logger.info("Consumer: GeneratorExit - Gradio cancelled the event")
            self.streamer.cancel()
        finally:
            if not future.done():
                try:
                    future.cancel()
                    self.streamer.cancel()
                    logger.info("Consumer: Cancelled producer coroutine and set cancellation flag")
                    try:
                        future.result(timeout=0.1)
                    except (asyncio.CancelledError, TimeoutError):
                        pass
                except Exception as e:
                    logger.debug(f"Error cancelling producer: {e}")


_mock_streamer = MockAgentStreamer()
_adapter = AsyncStreamAdapter(_mock_streamer)


def _validate_and_normalize_history(history: list[dict]) -> list[dict]:
    """Validate and normalize history to ensure all messages have correct format."""
    if not history:
        return []
    normalized = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        if "role" not in msg or "content" not in msg:
            continue
        # Extract content as plain string (handles both string and structured formats)
        content_raw = msg.get("content", "")
        content = _extract_content_string(content_raw)
        normalized_msg = {"role": msg["role"], "content": content}
        # Preserve metadata if it exists
        if "metadata" in msg:
            normalized_msg["metadata"] = msg["metadata"]
        normalized.append(normalized_msg)
    return normalized


def _extract_content_string(content: str | list | None) -> str:
    """Extract plain text string from Gradio message content.

    Handles both string format and structured format (list of content blocks).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        # If it's already a string, return as-is
        return content
    if isinstance(content, list):
        # Handle structured content format (Gradio 6): [{"type": "text", "text": "..."}]
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        # Recursively handle nested structures (double-encoded JSON)
                        # Handle case where text field contains a JSON string: "[{'text': '...', 'type': 'text'}]"
                        if isinstance(text, str) and text.strip().startswith("[") and text.strip().endswith("]"):
                            try:
                                parsed = json.loads(text)
                                if isinstance(parsed, list):
                                    # Recursively extract from nested structure
                                    extracted = _extract_content_string(parsed)
                                    if extracted:
                                        text_parts.append(extracted)
                                        continue
                            except (json.JSONDecodeError, ValueError):
                                # Not valid JSON, treat as plain text
                                pass
                        text_parts.append(str(text))
        return " ".join(text_parts) if text_parts else ""
    # Fallback: convert to string
    return str(content)


def handle_stop_click(history: list[dict]) -> list[dict]:
    """Handle built-in stop button click - cancel stream and update history."""
    logger.info("Stop button clicked - cancelling stream")
    _mock_streamer.cancel()
    if history:
        last_msg = history[-1]
        if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
            content_raw = last_msg.get("content", "")
            content_str = _extract_content_string(content_raw)
            if not content_str or not content_str.strip().endswith("."):
                cancellation_msg = {
                    "role": "assistant",
                    "content": content_str + "\n\n⚠️ Response cancelled by user.",
                    "metadata": {"title": "⏹️ Cancelled", "ui_type": "cancelled"},
                }
                history[-1] = cancellation_msg
    return history


def _validate_and_normalize_history(history: list[dict]) -> list[dict]:
    """Validate and normalize history to ensure all messages have correct format."""
    if not history:
        return []
    normalized = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        if "role" not in msg or "content" not in msg:
            continue
        # Extract content as plain string (handles both string and structured formats)
        content_raw = msg.get("content", "")
        content = _extract_content_string(content_raw)
        normalized_msg = {"role": msg["role"], "content": content}
        # Preserve metadata if it exists
        if "metadata" in msg:
            normalized_msg["metadata"] = msg["metadata"]
        normalized.append(normalized_msg)
    return normalized


def mock_chat_handler(message: str, history: list[dict]) -> Generator[list[dict], None, None]:
    logger.info(f"Mock handler: Processing message: {message}")
    # Reset cancellation state at start of new message
    _mock_streamer.reset()
    # Reference agent pattern: preserve ALL messages including meta blocks
    # Don't normalize - just use history as-is to preserve meta blocks
    # Only normalize when we need to extract content for memory saving
    gradio_history = list(history) if history else []

    # Save user message to memory (extract from normalized version for memory)
    normalized_for_memory = _validate_and_normalize_history([{"role": "user", "content": message}])
    if normalized_for_memory:
        user_content = normalized_for_memory[0].get("content", "")
        if user_content:
            _mock_llm_manager.save_user_turn(_TEST_SESSION_ID, user_content)
            logger.info("Mock handler: Saved user message to memory")

    incomplete_response = None
    final_response = None

    try:
        # Pass gradio_history (with all meta blocks preserved) to streamer
        for result in _adapter.stream_chat_sync(message, gradio_history):
            if "history" in result:
                # Use history directly from streamer (preserves all meta blocks)
                # Only normalize for memory extraction, not for display
                working_history = result["history"]
                # Track incomplete response for memory saving if cancelled
                if working_history:
                    last_msg = working_history[-1]
                    if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
                        content = last_msg.get("content", "")
                        # Only track non-meta messages for memory
                        if content and not last_msg.get("metadata"):
                            incomplete_response = content
                            final_response = content  # Update final response as we stream
                yield working_history
            else:
                yield gradio_history

        # Streaming completed successfully - save complete response to memory
        if final_response:
            _mock_llm_manager.save_assistant_turn(_TEST_SESSION_ID, final_response)
            logger.info(f"Mock handler: Saved complete response to memory ({len(final_response)} chars)")
        logger.info("Mock handler: Streaming completed")
    except GeneratorExit:
        # Stream was cancelled - save incomplete response to memory if available
        logger.info("Mock handler: Stream cancelled (GeneratorExit)")
        if incomplete_response:
            _mock_llm_manager.save_assistant_turn(_TEST_SESSION_ID, incomplete_response)
            logger.info(
                f"Mock handler: Saved INCOMPLETE response to memory: {incomplete_response[:50]}... ({len(incomplete_response)} chars)"
            )
        raise


def create_test_interface() -> gr.Blocks:
    """Create Gradio test interface with stop button.

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="Conversation Stopping Test") as demo:
        gr.Markdown("# 🧪 Conversation Stopping Test")

        # State to store saved message (like ChatInterface)
        saved_input = gr.State()

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Test Chat",
                    height=500,
                    show_label=True,
                    container=True,
                    elem_id="test-chatbot",
                    buttons=["copy","copy_all"],  # Hide share button, only show copy_all
                )

                msg = gr.Textbox(
                    label="Message",
                    placeholder="Enter a message to test streaming and stopping...",
                    lines=1,
                    max_lines=1,
                    show_label=False,
                    submit_btn=True,
                    stop_btn=False,  # Start hidden, will be shown when streaming starts
                )

            with gr.Column(scale=1):
                gr.Markdown("### 💾 Conversation Memory")
                memory_display = gr.Markdown(
                    value=_mock_llm_manager.get_memory_display(_TEST_SESSION_ID),
                    label="Saved Memory",
                    container=True,
                )

                with gr.Row():
                    refresh_memory_btn = gr.Button("🔄 Refresh Memory", size="sm")
                    clear_memory_btn = gr.Button("🗑️ Clear Memory", variant="secondary", size="sm")

        def refresh_memory():
            """Refresh memory display."""
            return _mock_llm_manager.get_memory_display(_TEST_SESSION_ID)

        def clear_memory():
            """Clear conversation memory."""
            _mock_llm_manager.clear_memory(_TEST_SESSION_ID)
            logger.info("Memory cleared by user")
            return _mock_llm_manager.get_memory_display(_TEST_SESSION_ID)

        def handle_chatbot_clear():
            """Handle chatbot clear event - also clear memory when chat is cleared."""
            _mock_llm_manager.clear_memory(_TEST_SESSION_ID)
            logger.info("Memory cleared via chatbot clear button")
            return _mock_llm_manager.get_memory_display(_TEST_SESSION_ID)

        # Store original stop_btn value (True in this case, but we start with False)
        original_stop_btn = True

        def clear_and_save_textbox(message: str) -> tuple[gr.Textbox, str]:
            """Clear textbox and save message to state (pattern from ChatInterface)."""
            return (
                gr.Textbox(value="", interactive=False, placeholder=""),
                message,
            )

        # Submit event - main handler
        # Pattern from ChatInterface: show stop button after submit succeeds, hide when streaming completes
        user_submit = msg.submit(
            fn=clear_and_save_textbox,
            inputs=[msg],
            outputs=[msg, saved_input],  # Clear textbox and save message to state
            queue=False,
        )

        # Show stop button when submit succeeds (before streaming starts)
        # Pattern from ChatInterface: after_success.success() shows stop button
        user_submit.success(
            lambda: gr.Textbox(submit_btn=False, stop_btn=original_stop_btn),
            outputs=[msg],
            queue=False,
        )

        # Main streaming handler - chained from user_submit
        # Pattern from ChatInterface: append user message to chatbot first, then call handler
        # The handler (mock_chat_handler) receives history with user message already added
        # But mock_stream_chat also adds it, so we need to NOT add it in mock_stream_chat
        submit_event = user_submit.then(
            lambda message, history: history + [{"role": "user", "content": message}],
            inputs=[saved_input, chatbot],
            outputs=[chatbot],
            queue=False,
        ).then(
            fn=mock_chat_handler,
            inputs=[saved_input, chatbot],  # Use saved message from state, chatbot now has user message
            outputs=[chatbot],
            concurrency_limit=1,
        ).then(
            lambda: gr.Textbox(value="", interactive=True),  # Clear and re-enable input after completion
            outputs=[msg],
        ).then(
            fn=refresh_memory,
            outputs=[memory_display],
        )

        # Hide stop button when streaming completes
        # Pattern from ChatInterface: events_to_cancel.then() hides stop button
        submit_event.then(
            lambda: gr.Textbox(submit_btn=True, stop_btn=False),
            outputs=[msg],
            queue=False,
        )

        # Built-in stop button automatically cancels submit_event when stop_btn=True
        # Wire up the stop event to handle cancellation and update history/memory
        # Also hide stop button when stop is clicked
        msg.stop(
            fn=handle_stop_click,
            inputs=[chatbot],
            outputs=[chatbot],
            cancels=[submit_event],  # Explicitly cancel submit event (though it's automatic)
        ).then(
            lambda: gr.Textbox(submit_btn=True, stop_btn=False),  # Hide stop button after cancellation
            outputs=[msg],
            queue=False,
        ).then(
            fn=refresh_memory,
            outputs=[memory_display],
        )

        refresh_memory_btn.click(fn=refresh_memory, outputs=[memory_display])
        clear_memory_btn.click(fn=clear_memory, outputs=[memory_display])

        # Bind to the built-in clear button's clear event
        # The function takes no inputs - it's just triggered when clear is clicked
        chatbot.clear(
            fn=handle_chatbot_clear,
            inputs=[],  # Explicitly no inputs
            outputs=[memory_display],
        )

    return demo


if __name__ == "__main__":
    logger.info("Starting conversation stopping test interface...")
    demo = create_test_interface()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        theme=gr.themes.Soft(),
    )
