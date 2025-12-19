"""Test generating answer spinner functionality."""

from rag_engine.api.stream_helpers import (
    update_message_status_in_history,
    yield_generating_answer,
    yield_search_completed,
)


def test_generating_answer_has_pending_status():
    """Verify generating answer message has status='pending' for spinner."""
    msg = yield_generating_answer()
    
    assert msg["role"] == "assistant"
    assert "metadata" in msg
    assert msg["metadata"]["ui_type"] == "generating_answer"
    assert msg["metadata"]["status"] == "pending", "Generating answer should show spinner"


def test_complete_flow_with_generating_spinner():
    """Test complete flow: search → completed → generating → answer."""
    history = []
    
    # 1. Search completed (no status - stays open for article links)
    completed = yield_search_completed(count=5)
    history.append(completed)
    assert "status" not in completed["metadata"], "Search completed should stay open"
    
    # 2. Add generating answer with spinner
    generating = yield_generating_answer()
    history.append(generating)
    assert generating["metadata"]["status"] == "pending", "Should show spinner while LLM processes"
    
    # 3. First text chunk arrives - stop spinner
    update_message_status_in_history(history, "generating_answer", "done")
    assert history[1]["metadata"]["status"] == "done", "Spinner should stop when text arrives"
    
    # 4. Add final answer (regular message, no metadata)
    answer = {"role": "assistant", "content": "Here's the answer..."}
    history.append(answer)
    
    # Verify flow
    assert len(history) == 3
    assert history[0]["metadata"]["ui_type"] == "search_completed"
    assert history[1]["metadata"]["ui_type"] == "generating_answer"
    assert "metadata" not in history[2], "Final answer should be regular message"


def test_generating_answer_i18n():
    """Test that generating answer message has i18n translations."""
    msg = yield_generating_answer()
    
    # Check that title and content are strings (not i18n objects)
    assert isinstance(msg["metadata"]["title"], str)
    assert isinstance(msg["content"], str)
    
    # Check that content mentions composing/generating
    content_lower = msg["content"].lower()
    assert any(word in content_lower for word in ["composing", "generating", "формирую", "генерация"]), \
        "Content should mention answer generation"


def test_ui_only_message_filtering():
    """Test that generating_answer is filtered from agent context."""
    from rag_engine.api.app import _is_ui_only_message
    
    # Generating answer message should be UI-only
    generating_msg = yield_generating_answer()
    assert _is_ui_only_message(generating_msg) is True, \
        "Generating answer message should be filtered from agent context"
    
    # Regular answer should NOT be UI-only
    regular_answer = {"role": "assistant", "content": "Here's the answer"}
    assert _is_ui_only_message(regular_answer) is False, \
        "Regular answer should be sent to agent"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

