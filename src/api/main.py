"""
DYN-EYE API — FastAPI backend (v2)

Restructured from the old dashboard/app.py monolith into routers +
services (src/api/routers, src/api/services), backed by the JSON->DB
sync layer in src/db/. API-only: the UI is the Next.js app in web/,
which talks to this service over /api/v2/* (same-origin, proxied by
Next's rewrites) and /ws/stream (direct WebSocket connection).
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import clusters, faiss, images, inference, logs, models, pipeline, system, vlm
from src.api.services.ws_manager import ws_manager
from src.db.session import get_session, init_db
from src.db import sync as db_sync
from src.utils import LogStream, get_logger

log = get_logger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    with get_session() as session:
        db_sync.sync_all(session)

    loop = asyncio.get_running_loop()
    ws_manager.bind_loop(loop)
    LogStream.subscribe(lambda event: ws_manager.broadcast_threadsafe("logs", "log", event))

    log.info("DYN-EYE API started: DB synced, WebSocket manager bound to event loop")
    yield


app = FastAPI(
    title="DYN-EYE — Unknown Defect Discovery",
    description="Autonomous industrial defect discovery pipeline (API only — UI is the Next.js app in web/)",
    version="2.0.0",
    lifespan=lifespan,
)

# The Next.js app proxies /api/v2/* through its own origin in production, so
# this stays permissive for local development against any frontend port.
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


@app.get("/")
async def root():
    return {"service": "dyn-eye-api", "version": app.version, "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    import config as cfg

    uvicorn.run("src.api.main:app", host=cfg.DASHBOARD_HOST, port=cfg.DASHBOARD_PORT, reload=True)
