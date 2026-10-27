"""
DYN-EYE API — FastAPI backend (v2)

Restructured from the old dashboard/app.py monolith into routers +
services (src/api/routers, src/api/services), backed by the JSON->DB
sync layer in src/db/.

Temporarily still serves the legacy dashboard/static UI and the legacy
SSE log endpoint for continuity while the Next.js frontend is built out
(see README migration plan) — both are removed once that cutover happens.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import config as cfg
from src.api.routers import clusters, faiss, images, inference, logs, models, pipeline, system, vlm
from src.api.services.ws_manager import ws_manager
from src.db.session import get_session, init_db
from src.db import sync as db_sync
from src.utils import LogStream, get_logger

log = get_logger("api")

app = FastAPI(
    title="DYN-EYE — Unknown Defect Discovery",
    description="Autonomous industrial defect discovery pipeline",
    version="2.0.0",
)

# Dashboard and API currently run on different origins until the Next.js
# frontend proxies /api/* through its own origin (see migration plan step 4).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router)
app.include_router(clusters.router)
app.include_router(models.router)
app.include_router(faiss.router)
app.include_router(system.router)
app.include_router(vlm.router)
app.include_router(images.router)
app.include_router(inference.router)
app.include_router(logs.router)

# ── Legacy static dashboard (kept until the Next.js cutover) ───
_legacy_static_dir = Path(__file__).resolve().parent.parent.parent / "dashboard" / "static"
if _legacy_static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_legacy_static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = _legacy_static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>DYN-EYE API</h1><p>No static frontend mounted.</p>")


@app.on_event("startup")
async def on_startup() -> None:
    init_db()
    with get_session() as session:
        db_sync.sync_all(session)

    loop = asyncio.get_running_loop()
    ws_manager.bind_loop(loop)
    LogStream.subscribe(lambda event: ws_manager.broadcast_threadsafe("logs", "log", event))

    log.info("DYN-EYE API started: DB synced, WebSocket manager bound to event loop")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=cfg.DASHBOARD_HOST, port=cfg.DASHBOARD_PORT, reload=True)
