"""Test spinner lifecycle: spinners start and stop correctly.

This test verifies that spinners transition from pending to done at the right times.
"""

from rag_engine.api.stream_helpers import (
    update_message_status_in_history,
    yield_search_completed,
    yield_search_started,
    yield_thinking_block,
)


def test_spinner_lifecycle_simple():
    """Test spinner lifecycle: pending -> done transition."""
    history = []
    
    # 1. Add thinking message with spinner
    thinking = yield_thinking_block("test_tool")
    history.append(thinking)
    assert history[-1]["metadata"]["status"] == "pending", "Thinking should start with pending"
    
    # 2. Update thinking to done
    result = update_message_status_in_history(history, "thinking", "done")
    assert result is True, "Should successfully update thinking status"
    assert history[0]["metadata"]["status"] == "done", "Thinking should be updated to done"
    

def test_spinner_lifecycle_search_flow():
    """Test complete search flow: thinking -> search -> completed."""
    history = []
    
    # 1. Thinking with spinner
    thinking = yield_thinking_block("retrieve_context")
    history.append(thinking)
    assert thinking["metadata"]["status"] == "pending"
    
    # 2. Search started with spinner
    search = yield_search_started("test query")
    history.append(search)
    assert search["metadata"]["status"] == "pending"
    
    # At this point, both thinking and search should have pending spinners
    assert history[0]["metadata"]["status"] == "pending"
    assert history[1]["metadata"]["status"] == "pending"
    
    # 3. When tool completes, update both to done
    update_message_status_in_history(history, "thinking", "done")
    update_message_status_in_history(history, "search_started", "done")
    
    assert history[0]["metadata"]["status"] == "done", "Thinking spinner should stop"
    assert history[1]["metadata"]["status"] == "done", "Search spinner should stop"
    
    # 4. Add search completed (no status - stays open for article links)
    completed = yield_search_completed(count=5)
    history.append(completed)
    assert "status" not in completed["metadata"], "Search completed should stay open"
    
    # Final check: thinking and search_started should have status="done"
    assert history[0]["metadata"]["status"] == "done", "Thinking spinner should be stopped"
    assert history[1]["metadata"]["status"] == "done", "Search spinner should be stopped"
    assert "status" not in history[2]["metadata"], "Search completed should stay open"


def test_spinner_multiple_thinking_blocks():
    """Test that update affects only the most recent thinking block."""
    history = []
    
    # Add multiple thinking blocks (different tools)
    thinking1 = yield_thinking_block("tool_a")
    history.append(thinking1)
    
    thinking2 = yield_thinking_block("tool_b")
    history.append(thinking2)
    
    # Both should start as pending
    assert history[0]["metadata"]["status"] == "pending"
    assert history[1]["metadata"]["status"] == "pending"
    
    # Update thinking status (should update most recent only)
    result = update_message_status_in_history(history, "thinking", "done")
    
    assert result is True, "Should find and update a message"
    # Only the most recent should be updated
    assert history[0]["metadata"]["status"] == "pending", "First thinking should remain pending"
    assert history[1]["metadata"]["status"] == "done", "Most recent thinking should be done"
    
    # Note: Calling update again would update the MOST RECENT message again (history[1]),
    # not the first one. This is the correct behavior - we always update the most recent message.
    # In practice, each tool execution creates one thinking block and updates it when done.
    
    # To update the first one, we'd need to update it directly:
    history[0]["metadata"]["status"] = "done"
    assert history[0]["metadata"]["status"] == "done"


def test_spinner_lifecycle_with_answer():
    """Test complete flow including final answer (no spinner for answer)."""
    history = []
    
    # 1. Thinking
    thinking = yield_thinking_block("retrieve_context")
    history.append(thinking)
    
    # 2. Search
    search = yield_search_started("query")
    history.append(search)
    
    # 3. Complete search and stop spinners
    update_message_status_in_history(history, "thinking", "done")
    update_message_status_in_history(history, "search_started", "done")
    
    completed = yield_search_completed(count=3)
    history.append(completed)
    
    # 4. Add final answer (regular message, no metadata)
    answer = {"role": "assistant", "content": "Here's the answer..."}
    history.append(answer)
    
    # Verify spinner messages have stopped, others stay open
    for msg in history:
        if "metadata" in msg:
            ui_type = msg["metadata"].get("ui_type")
            if ui_type in ["thinking", "search_started"]:
                assert msg["metadata"]["status"] == "done", \
                    f"Spinner messages should have status=done: {ui_type}"
            elif ui_type == "search_completed":
                assert "status" not in msg["metadata"], \
                    f"Search completed should stay open (no status): {ui_type}"


def test_no_spinner_for_regular_messages():
    """Test that regular messages (without metadata) don't interfere with spinners."""
    history = []
    
    # Add thinking with spinner
    thinking = yield_thinking_block("test")
    history.append(thinking)
    
    # Add regular message (no metadata)
    regular = {"role": "assistant", "content": "Regular message"}
    history.append(regular)
    
    # Update thinking status
    result = update_message_status_in_history(history, "thinking", "done")
    
    assert result is True
    assert history[0]["metadata"]["status"] == "done"
    assert "metadata" not in history[1], "Regular message should not have metadata"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

