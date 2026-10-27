"""Log access: recent-logs snapshot and the live WebSocket channel."""
from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.services.ws_manager import ws_manager
from src.utils import LogStream

router = APIRouter(tags=["logs"])

_VALID_TOPICS = {"logs", "pipeline-status", "clusters", "models"}


@router.get("/api/v2/logs/recent")
async def recent_logs(n: int = 100):
    return LogStream.tail(n)


@router.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """
    Topic-based push channel. On connect, subscribes to all topics by
    default; send {"subscribe": ["logs", "pipeline-status", ...]} to narrow.
    """
    await ws_manager.connect(websocket, set(_VALID_TOPICS))
    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            topics = msg.get("subscribe")
            if isinstance(topics, list):
                ws_manager.set_topics(websocket, set(topics) & _VALID_TOPICS)
    finally:
        ws_manager.disconnect(websocket)
