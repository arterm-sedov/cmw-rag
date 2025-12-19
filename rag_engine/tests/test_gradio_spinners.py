"""Test Gradio native spinner implementation in metadata messages.

This test verifies that metadata messages include the correct status field
for displaying native Gradio spinners during agent operations.
"""

from rag_engine.api.stream_helpers import (
    update_message_status_in_history,
    yield_cancelled,
    yield_generating_answer,
    yield_model_switch_notice,
    yield_search_completed,
    yield_search_started,
    yield_thinking_block,
)


def test_thinking_block_has_pending_status():
    """Verify thinking block has status='pending' for spinner."""
    msg = yield_thinking_block("retrieve_context")
    
    assert msg["role"] == "assistant"
    assert "metadata" in msg
    assert msg["metadata"]["ui_type"] == "thinking"
    assert msg["metadata"]["status"] == "pending", "Thinking block should show spinner"


def test_search_started_has_pending_status():
    """Verify search started message has status='pending' for spinner."""
    msg = yield_search_started("test query")
    
    assert msg["role"] == "assistant"
    assert "metadata" in msg
    assert msg["metadata"]["ui_type"] == "search_started"
    assert msg["metadata"]["status"] == "pending", "Search started should show spinner"


def test_search_completed_no_status():
    """Verify search completed message has no status (stays open for article links)."""
    msg = yield_search_completed(count=5, articles=None)
    
    assert msg["role"] == "assistant"
    assert "metadata" in msg
    assert msg["metadata"]["ui_type"] == "search_completed"
    assert "status" not in msg["metadata"], "Search completed should stay open (no status field)"


def test_model_switch_no_status():
    """Verify model switch notice has no status (stays open for visibility)."""
    msg = yield_model_switch_notice("gemini-2.0-flash-exp")
    
    assert msg["role"] == "assistant"
    assert "metadata" in msg
    assert msg["metadata"]["ui_type"] == "model_switch"
    assert "status" not in msg["metadata"], "Model switch should stay open (important info)"


def test_cancelled_no_status():
    """Verify cancellation message has no status (stays open for visibility)."""
    msg = yield_cancelled()
    
    assert msg["role"] == "assistant"
    assert "metadata" in msg
    assert msg["metadata"]["ui_type"] == "cancelled"
    assert "status" not in msg["metadata"], "Cancelled message should stay open (important notice)"


def test_update_message_status_in_history():
    """Verify helper function can update status in existing messages."""
    # Create history with a pending search message
    history = [
        {
            "role": "user",
            "content": "test question"
        },
        {
            "role": "assistant",
            "content": "Searching...",
            "metadata": {
                "ui_type": "search_started",
                "status": "pending"
            }
        }
    ]
    
    # Update status to done
    result = update_message_status_in_history(history, "search_started", "done")
    
    assert result is True, "Should find and update the message"
    assert history[1]["metadata"]["status"] == "done", "Status should be updated to done"


def test_update_message_status_not_found():
    """Verify helper function returns False when message type not found."""
    history = [
        {
            "role": "assistant",
            "content": "Some message",
            "metadata": {
                "ui_type": "thinking",
                "status": "pending"
            }
        }
    ]
    
    # Try to update non-existent message type
    result = update_message_status_in_history(history, "search_started", "done")
    
    assert result is False, "Should return False when message type not found"
    assert history[0]["metadata"]["status"] == "pending", "Original message should be unchanged"


def test_update_message_status_updates_most_recent():
    """Verify helper function updates the most recent message of given type."""
    history = [
        {
            "role": "assistant",
            "content": "First search",
            "metadata": {
                "ui_type": "search_started",
                "status": "pending"
            }
        },
        {
            "role": "assistant",
            "content": "Second search",
            "metadata": {
                "ui_type": "search_started",
                "status": "pending"
            }
        }
    ]
    
    # Update status - should update most recent (second) message
    result = update_message_status_in_history(history, "search_started", "done")
    
    assert result is True
    assert history[0]["metadata"]["status"] == "pending", "First message should remain pending"
    assert history[1]["metadata"]["status"] == "done", "Most recent message should be updated"


def test_search_completed_with_articles():
    """Verify search completed includes articles in content and stays open."""
    articles = [
        {"title": "Article 1", "url": "https://example.com/1"},
        {"title": "Article 2", "url": "https://example.com/2"},
    ]
    
    msg = yield_search_completed(count=2, articles=articles)
    
    assert msg["role"] == "assistant"
    assert "metadata" in msg
    assert "status" not in msg["metadata"], "Search completed should stay open for clickable links"
    # Check that articles are included in content
    assert "Article 1" in msg["content"]
    assert "Article 2" in msg["content"]
    assert "https://example.com/1" in msg["content"]
    assert "https://example.com/2" in msg["content"]


def test_spinner_vs_non_spinner_messages():
    """Verify which messages have spinners (status field) vs stay open (no status)."""
    # Messages WITH spinners (status="pending")
    spinner_messages = [
        yield_thinking_block("test_tool"),
        yield_search_started("test query"),
        yield_generating_answer(),
    ]
    
    for msg in spinner_messages:
        assert "metadata" in msg, f"Message should have metadata: {msg}"
        assert "status" in msg["metadata"], f"Spinner message should have status: {msg}"
        assert msg["metadata"]["status"] == "pending", \
            f"Spinner message should have status='pending': {msg['metadata'].get('ui_type')}"
    
    # Messages WITHOUT spinners (stay open for visibility)
    no_spinner_messages = [
        yield_search_completed(count=5),
        yield_model_switch_notice("test-model"),
        yield_cancelled(),
    ]
    
    for msg in no_spinner_messages:
        assert "metadata" in msg, f"Message should have metadata: {msg}"
        assert "status" not in msg["metadata"], \
            f"Non-spinner message should NOT have status (stays open): {msg['metadata'].get('ui_type')}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

