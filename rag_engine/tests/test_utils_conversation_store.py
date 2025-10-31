"""Tests for conversation store."""
from __future__ import annotations

import pytest

from rag_engine.utils.conversation_store import ConversationStore


class TestConversationStore:
    """Tests for ConversationStore."""

    def test_initialization(self):
        """Test store initializes empty."""
        store = ConversationStore()
        assert store.get("any_id") == []

    def test_append_and_get(self):
        """Test appending and retrieving conversation turns."""
        store = ConversationStore()
        session_id = "session-1"

        store.append(session_id, "user", "Hello")
        store.append(session_id, "assistant", "Hi there")

        history = store.get(session_id)
        assert len(history) == 2
        assert history[0] == ("user", "Hello")
        assert history[1] == ("assistant", "Hi there")

    def test_set_overwrites_history(self):
        """Test set() overwrites existing history."""
        store = ConversationStore()
        session_id = "session-1"

        store.append(session_id, "user", "First")
        store.append(session_id, "assistant", "Response")

        new_history = [("user", "New"), ("assistant", "Reply")]
        store.set(session_id, new_history)

        assert store.get(session_id) == new_history
        assert len(store.get(session_id)) == 2

    def test_session_isolation(self):
        """Test different sessions have isolated memory."""
        store = ConversationStore()

        store.append("session-1", "user", "Hello 1")
        store.append("session-1", "assistant", "Hi 1")

        store.append("session-2", "user", "Hello 2")
        store.append("session-2", "assistant", "Hi 2")

        h1 = store.get("session-1")
        h2 = store.get("session-2")

        assert h1 != h2
        assert h1[0] == ("user", "Hello 1")
        assert h2[0] == ("user", "Hello 2")

    def test_empty_session_returns_empty_list(self):
        """Test getting non-existent session returns empty list."""
        store = ConversationStore()
        assert store.get("non-existent") == []

    def test_get_returns_copy(self):
        """Test get() returns a copy, not a reference."""
        store = ConversationStore()
        session_id = "session-1"

        store.append(session_id, "user", "Test")
        history1 = store.get(session_id)
        history2 = store.get(session_id)

        # Modifying one shouldn't affect the other
        history1.append(("assistant", "Modified"))
        assert len(store.get(session_id)) == 1  # Original unchanged

    def test_set_with_empty_list(self):
        """Test set() with empty list clears history."""
        store = ConversationStore()
        session_id = "session-1"

        store.append(session_id, "user", "Hello")
        store.set(session_id, [])

        assert store.get(session_id) == []

