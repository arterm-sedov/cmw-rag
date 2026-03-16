"""Tests for usage_accounting module (accumulate_conversation_usage, turn timing)."""

from __future__ import annotations

from rag_engine.llm.usage_accounting import accumulate_conversation_usage


def test_accumulate_conversation_usage_sums_turn_time_ms_per_session():
    """accumulate_conversation_usage sums turn_time_ms into total_conversation_time_ms per session."""
    session_id = "test-session-turn-time-accumulation"
    # First turn: 100 ms
    out1 = accumulate_conversation_usage(
        session_id=session_id,
        turn_summary={"turn_time_ms": 100},
    )
    assert out1.get("total_conversation_time_ms") == 100.0

    # Second turn: 50 ms -> total 150 ms
    out2 = accumulate_conversation_usage(
        session_id=session_id,
        turn_summary={"turn_time_ms": 50},
    )
    assert out2.get("total_conversation_time_ms") == 150.0

    # Third turn: no turn_time_ms in summary -> total unchanged
    out3 = accumulate_conversation_usage(
        session_id=session_id,
        turn_summary={"prompt_tokens": 10},
    )
    assert out3.get("total_conversation_time_ms") == 150.0


def test_accumulate_conversation_usage_none_session_id_returns_turn_time_only():
    """With session_id None, total_conversation_time_ms is turn's turn_time_ms only (no accumulation)."""
    out = accumulate_conversation_usage(
        session_id=None,
        turn_summary={"turn_time_ms": 200},
    )
    assert out.get("total_conversation_time_ms") == 200.0

    # Second call with None session_id does not accumulate
    out2 = accumulate_conversation_usage(
        session_id=None,
        turn_summary={"turn_time_ms": 50},
    )
    assert out2.get("total_conversation_time_ms") == 50.0


def test_accumulate_conversation_usage_turn_time_ms_int_or_float():
    """turn_time_ms can be int or float."""
    session_id = "test-session-turn-time-types"
    out_int = accumulate_conversation_usage(
        session_id=session_id,
        turn_summary={"turn_time_ms": 100},
    )
    assert out_int.get("total_conversation_time_ms") == 100.0

    out_float = accumulate_conversation_usage(
        session_id=session_id,
        turn_summary={"turn_time_ms": 25.5},
    )
    assert out_float.get("total_conversation_time_ms") == 125.5
