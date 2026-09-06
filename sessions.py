"""sessions/store — in-memory session state for follow-ups.

Maya ships a `session_id` + `ui_state` snapshot every turn. This store keeps
the server-authoritative conversation per session so anaphora ("increase it a
little", "only the liability changes") resolves against everything that has
happened, not just the client's last few turns. No Redis — a TTL + size-capped
in-memory map is the honest minimum for a modular monolith.
"""

import threading
import time

TTL_SECONDS = 30 * 60
MAX_SESSIONS = 1000
MAX_CONVERSATION = 40


class _Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}

    def get(self, session_id: str) -> dict | None:
        now = time.time()
        with self._lock:
            rec = self._data.get(session_id)
            if rec is None:
                return None
            if now - rec["updated"] > TTL_SECONDS:
                del self._data[session_id]
                return None
            return {
                "conversation": list(rec["conversation"]),
                "ui_state": rec["ui_state"],
            }

    def update(
        self,
        session_id: str,
        conversation: list[dict],
        ui_state: dict | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            if session_id not in self._data and len(self._data) >= MAX_SESSIONS:
                oldest = min(self._data, key=lambda k: self._data[k]["updated"])
                del self._data[oldest]
            self._data[session_id] = {
                "conversation": conversation[-MAX_CONVERSATION:],
                "ui_state": ui_state,
                "updated": now,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


Sessions = _Store()