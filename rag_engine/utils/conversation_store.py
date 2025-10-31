from __future__ import annotations

from typing import Dict, List, Tuple


class ConversationStore:
    """Minimal in-memory conversation store keyed by session_id.

    Stores turns as a list of (role, content) tuples. This is per-process and
    resets on restart. Thread-safe access is not implemented; rely on Gradio's
    single-threaded request handling or add locking if needed.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, List[Tuple[str, str]]] = {}

    def get(self, session_id: str) -> List[Tuple[str, str]]:
        return list(self._sessions.get(session_id, []))

    def set(self, session_id: str, turns: List[Tuple[str, str]]) -> None:
        self._sessions[session_id] = list(turns)

    def append(self, session_id: str, role: str, content: str) -> None:
        history = self._sessions.setdefault(session_id, [])
        history.append((role, content))


