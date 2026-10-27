"""
Log access: recent-logs snapshot, legacy SSE stream, and the new
WebSocket channel.

The SSE endpoint is kept temporarily for the old dashboard/static UI
during the migration to Next.js (see README migration plan, step 2) and
will be removed once the new frontend is the only consumer.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from src.api.services.ws_manager import ws_manager
from src.utils import LogStream

router = APIRouter(tags=["logs"])

_VALID_TOPICS = {"logs", "pipeline-status", "clusters", "models"}


@router.get("/api/v2/logs/recent")
async def recent_logs(n: int = 100):
    return LogStream.tail(n)


@router.get("/api/logs/stream")
async def log_stream_sse(after_ts: str | None = None):
    """Legacy SSE endpoint — superseded by /ws/stream."""
    async def event_generator():
        last_ts = after_ts
        while True:
            events = LogStream.since(after_ts=last_ts, limit=50)
            for evt in events:
                last_ts = evt["ts"]
                yield f"data: {json.dumps(evt)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


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
