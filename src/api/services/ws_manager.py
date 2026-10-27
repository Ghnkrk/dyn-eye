"""
WebSocket connection manager — topic-based broadcast.

Replaces SSE log polling + frontend status polling with a single push
channel. Clients connect to /ws/stream and send {"subscribe": [...]} to
pick topics ("logs", "pipeline-status", "clusters"); the server pushes
{"topic": ..., "event": ..., "data": ...} frames.

emit() on LogStream (and pipeline-status updates) happen from background
worker threads, not the asyncio event loop thread, so broadcast has to be
scheduled onto the loop with run_coroutine_threadsafe — see
broadcast_threadsafe().
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastapi import WebSocket

from src.utils import get_logger

log = get_logger("ws_manager")


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[WebSocket, set[str]] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once at FastAPI startup so background threads can schedule onto it."""
        self._loop = loop

    async def connect(self, websocket: WebSocket, topics: set[str]) -> None:
        await websocket.accept()
        with self._lock:
            self._connections[websocket] = topics

    def disconnect(self, websocket: WebSocket) -> None:
        with self._lock:
            self._connections.pop(websocket, None)

    def set_topics(self, websocket: WebSocket, topics: set[str]) -> None:
        with self._lock:
            if websocket in self._connections:
                self._connections[websocket] = topics

    async def broadcast(self, topic: str, event: str, data: Any) -> None:
        with self._lock:
            targets = [ws for ws, topics in self._connections.items() if topic in topics]

        payload = {"topic": topic, "event": event, "data": data}
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)

        if dead:
            with self._lock:
                for ws in dead:
                    self._connections.pop(ws, None)

    def broadcast_threadsafe(self, topic: str, event: str, data: Any) -> None:
        """Call from any thread (pipeline workers, orchestrator loop, etc.)."""
        if self._loop is None:
            return  # app not started yet / no event loop bound
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(topic, event, data), self._loop)
        except RuntimeError:
            pass  # loop closed (shutdown race) — nothing to deliver to


# Module-level singleton, mirroring the orchestrator pattern in
# src/pipeline/orchestrator.py.
ws_manager = ConnectionManager()
