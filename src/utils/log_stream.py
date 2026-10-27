"""
Real-time log streaming — thread-safe, in-memory ring buffer.

Pipeline nodes write log events here; the API reads them via SSE/WebSocket.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass
class LogEvent:
    ts: str
    level: str        # info | warn | error | step | progress
    source: str       # node name or system component
    message: str
    data: dict | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["data"] is None:
            del d["data"]
        return d


class LogStream:
    """
    Global, thread-safe log ring buffer.

    Nodes call  LogStream.emit(...)  to add events.
    The dashboard SSE endpoint calls  LogStream.tail(...)  to read them.
    """

    _buffer: deque[LogEvent] = deque(maxlen=2000)
    _lock = threading.Lock()
    _seq = 0                    # monotonic counter for cursor-based polling
    _seq_map: dict[int, int] = {}  # seq -> index mapping
    _subscribers: list[Callable[[dict], None]] = []

    @classmethod
    def subscribe(cls, callback: Callable[[dict], None]) -> None:
        """
        Register a callback invoked synchronously with each event's dict
        right after it's buffered. Used by the WebSocket connection manager
        to push live events without this module knowing about WebSockets.
        Callbacks must not raise or block — emit() is called from pipeline
        worker threads.
        """
        cls._subscribers.append(callback)

    @classmethod
    def emit(
        cls,
        message: str,
        level: str = "info",
        source: str = "system",
        data: dict | None = None,
    ) -> None:
        with cls._lock:
            cls._seq += 1
            evt = LogEvent(
                ts=datetime.now(timezone.utc).isoformat(),
                level=level,
                source=source,
                message=message,
                data=data,
            )
            cls._buffer.append(evt)
            event_dict = evt.to_dict()

        for callback in cls._subscribers:
            try:
                callback(event_dict)
            except Exception:
                pass  # a dead/misbehaving subscriber must never break pipeline logging

    @classmethod
    def tail(cls, last_n: int = 100) -> list[dict]:
        """Return the most recent `last_n` log events."""
        with cls._lock:
            items = list(cls._buffer)[-last_n:]
            return [e.to_dict() for e in items]

    @classmethod
    def since(cls, after_ts: str | None = None, limit: int = 200) -> list[dict]:
        """Return events newer than `after_ts`."""
        with cls._lock:
            items = list(cls._buffer)
        if after_ts:
            items = [e for e in items if e.ts > after_ts]
        return [e.to_dict() for e in items[-limit:]]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._buffer.clear()
            cls._seq = 0
