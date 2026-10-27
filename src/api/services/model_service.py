"""
Model registry service — thin wrapper around src/retraining/model_registry.py
that adds a DB resync + WebSocket broadcast after every mutation.
"""
from __future__ import annotations

from src.api.services.ws_manager import ws_manager
from src.db.session import get_session
from src.db import sync as db_sync
from src.retraining.model_registry import ModelRegistry
from src.utils import get_logger, LogStream

log = get_logger("model_service")


def _resync_and_broadcast() -> None:
    with get_session() as session:
        db_sync.sync_model_registry(session)
        db_sync.sync_known_defects(session)
        db_sync.sync_cluster_fingerprints(session)
    ws_manager.broadcast_threadsafe("models", "updated", {})


def rollback(version_id: str, confirmed_by: str = "dashboard") -> dict:
    LogStream.emit(f"Rolling back to model version: {version_id}", level="step", source="model_registry")
    registry = ModelRegistry()
    result = registry.rollback(version_id, confirmed_by=confirmed_by)
    if result["success"]:
        LogStream.emit(
            f"Rollback complete: {version_id} is now active (FAISS: {result.get('faiss_vectors', 0)} vectors)",
            level="info", source="model_registry",
        )
    else:
        LogStream.emit(f"Rollback failed: {result.get('error', 'Unknown')}", level="error", source="model_registry")
    _resync_and_broadcast()
    return result


def deploy_confirm(version_id: str) -> dict:
    LogStream.emit(f"Deploying model version: {version_id} (user confirmed)", level="step", source="model_registry")
    registry = ModelRegistry()
    result = registry.deploy_version(version_id, confirmed_by="dashboard-user")
    if result["success"]:
        LogStream.emit(f"Model {version_id} deployed successfully", level="info", source="model_registry")
    _resync_and_broadcast()
    return result
