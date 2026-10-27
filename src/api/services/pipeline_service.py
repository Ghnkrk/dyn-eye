"""
Pipeline trigger/status service.

Carries over the background-thread runners and shared status dict from
the old dashboard/app.py, adding:
  - WebSocket broadcast of status transitions (topic "pipeline-status")
  - a DB sync of the run's metrics once it completes (so /api/v2/pipeline/runs
    can be served from SQLite instead of re-parsing logs/pipeline_runs/*.json)
"""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import config as cfg
from src.api.services.ws_manager import ws_manager
from src.db.session import get_session
from src.db import sync as db_sync
from src.utils import get_logger, LogStream

log = get_logger("pipeline_service")

_pipeline_status: dict[str, Any] = {
    "discovery": {"status": "idle", "run_id": None, "result": None},
    "retraining": {"status": "idle", "run_id": None, "result": None},
    "orchestrator": {"status": "idle"},
}
_clusters_ready = True  # False while pipeline is running (prevents stale cluster display)
_lock = threading.Lock()


def _sanitize(obj: Any) -> Any:
    """Make a result dict JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items() if not isinstance(k, str) or not k.startswith("_")}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def get_status() -> dict:
    with _lock:
        status = dict(_pipeline_status)
        status["clusters_ready"] = _clusters_ready
        return status


def is_running(kind: str) -> bool:
    with _lock:
        return _pipeline_status[kind]["status"] == "running"


def _broadcast_status() -> None:
    ws_manager.broadcast_threadsafe("pipeline-status", "status", get_status())


def _resync_after_run() -> None:
    """Mirror whatever JSON this run touched into the DB."""
    try:
        with get_session() as session:
            db_sync.sync_clusters(session)
            db_sync.sync_model_registry(session)
            db_sync.sync_known_defects(session)
            db_sync.sync_cluster_fingerprints(session)
    except Exception as e:
        log.warning(f"Post-run DB resync failed (non-fatal): {e}")


# ── Discovery ─────────────────────────────────────────────────

def trigger_discovery(
    use_sample_run: bool,
    input_images_dir: str | None,
    yolo_model_path: str | None,
    use_cache: bool,
) -> None:
    """Fire-and-forget: runs _run_discovery in a background thread."""
    thread = threading.Thread(
        target=_run_discovery,
        args=(use_sample_run, input_images_dir, yolo_model_path, use_cache),
        daemon=True,
    )
    thread.start()


def _run_discovery(
    use_sample_run: bool,
    input_images_dir: str | None,
    yolo_model_path: str | None,
    use_cache: bool,
) -> None:
    global _clusters_ready

    with _lock:
        _pipeline_status["discovery"]["status"] = "running"
        _pipeline_status["discovery"]["started_at"] = datetime.now().isoformat()
        _pipeline_status["discovery"]["error"] = None
        _clusters_ready = False  # Hide stale clusters while pipeline runs
    _broadcast_status()

    LogStream.emit(
        f"Discovery pipeline triggered (Sample Run: {use_sample_run}, Cache: {use_cache})",
        level="step", source="dashboard",
    )

    try:
        from src.pipeline.graph import run_discovery_pipeline
        input_dir = str(cfg.SAMPLE_RUN_DIR) if use_sample_run else input_images_dir
        result = run_discovery_pipeline(
            input_images_dir=input_dir,
            yolo_model_path=yolo_model_path,
            use_cache=use_cache,
        )
        with _lock:
            _pipeline_status["discovery"]["status"] = "complete"
            _pipeline_status["discovery"]["result"] = _sanitize(result)
            _pipeline_status["discovery"]["completed_at"] = datetime.now().isoformat()
            _clusters_ready = True  # Clusters are now fresh
        LogStream.emit("Discovery pipeline finished successfully", level="info", source="dashboard")
    except Exception as e:
        with _lock:
            _pipeline_status["discovery"]["status"] = "failed"
            _pipeline_status["discovery"]["error"] = str(e)
            _pipeline_status["discovery"]["completed_at"] = datetime.now().isoformat()
            _clusters_ready = True  # Re-enable cluster display even on failure
        LogStream.emit(f"Discovery pipeline failed: {e}", level="error", source="dashboard")
        log.error(f"Discovery pipeline failed: {e}")

    _broadcast_status()
    _resync_after_run()
    ws_manager.broadcast_threadsafe("clusters", "updated", {})


# ── Retraining ────────────────────────────────────────────────

def trigger_retraining(
    project_id: int,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch_size: int | None = None,
    freeze: int | None = None,
) -> None:
    thread = threading.Thread(
        target=_run_retraining,
        args=(project_id, epochs, imgsz, batch_size, freeze),
        daemon=True,
    )
    thread.start()


def _run_retraining(
    project_id: int,
    epochs: int | None,
    imgsz: int | None,
    batch_size: int | None,
    freeze: int | None,
) -> None:
    with _lock:
        _pipeline_status["retraining"]["status"] = "running"
        _pipeline_status["retraining"]["started_at"] = datetime.now().isoformat()
        _pipeline_status["retraining"]["error"] = None
    _broadcast_status()

    LogStream.emit("Retraining pipeline triggered", level="step", source="dashboard")

    try:
        from src.retraining.agent import run_retraining_pipeline
        result = run_retraining_pipeline(
            project_id=project_id,
            epochs=epochs,
            imgsz=imgsz,
            batch_size=batch_size,
            freeze=freeze,
        )
        with _lock:
            _pipeline_status["retraining"]["status"] = "complete"
            _pipeline_status["retraining"]["result"] = _sanitize(result)
            _pipeline_status["retraining"]["completed_at"] = datetime.now().isoformat()
        LogStream.emit("Retraining pipeline finished", level="info", source="dashboard")
    except Exception as e:
        with _lock:
            _pipeline_status["retraining"]["status"] = "failed"
            _pipeline_status["retraining"]["error"] = str(e)
            _pipeline_status["retraining"]["completed_at"] = datetime.now().isoformat()
        LogStream.emit(f"Retraining pipeline failed: {e}", level="error", source="dashboard")
        log.error(f"Retraining pipeline failed: {e}")

    _broadcast_status()
    _resync_after_run()
    ws_manager.broadcast_threadsafe("models", "updated", {})


def reset_status() -> None:
    """Used by /api/v2/system/reset."""
    global _pipeline_status, _clusters_ready
    with _lock:
        _pipeline_status["discovery"] = {"status": "idle", "run_id": None, "result": None}
        _pipeline_status["retraining"] = {"status": "idle", "run_id": None, "result": None}
        _pipeline_status["orchestrator"] = {"status": "idle"}
        _clusters_ready = True
    _broadcast_status()


def set_clusters_ready(value: bool) -> None:
    global _clusters_ready
    with _lock:
        _clusters_ready = value


def clusters_ready() -> bool:
    with _lock:
        return _clusters_ready
