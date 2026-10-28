"""Model version registry, deployment, and rollback."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import config as cfg
from src.api.schemas.models import DeployConfirmRequest, RollbackRequest
from src.api.services import model_service
from src.retraining.model_registry import ModelRegistry

router = APIRouter(prefix="/api/v2/models", tags=["models"])


def _save_upload_to_temp(file: UploadFile) -> Path:
    if not file.filename.lower().endswith(".pt"):
        raise HTTPException(400, "Model file must be a .pt checkpoint")
    tmp_dir = Path(tempfile.mkdtemp(prefix="model_upload_"))
    tmp_path = tmp_dir / Path(file.filename).name
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return tmp_path


@router.get("")
async def get_model_versions():
    registry = ModelRegistry()
    return {"versions": registry.list_versions(), "current": registry.get_current()}


@router.get("/active")
async def get_active_model():
    registry = ModelRegistry()
    entry = registry.get_current_entry()
    return {
        "active_version": registry.get_current(),
        "entry": entry,
        "model_exists": cfg.YOLO_MODEL_PATH.exists(),
        "model_path": str(cfg.YOLO_MODEL_PATH),
    }


@router.post("/rollback")
async def rollback_model(req: RollbackRequest):
    return model_service.rollback(req.version_id, confirmed_by=req.confirmed_by)


@router.post("/deploy-confirm")
async def deploy_confirm(req: DeployConfirmRequest):
    return model_service.deploy_confirm(req.version_id)


@router.get("/history")
async def get_deployment_history():
    registry = ModelRegistry()
    return {"history": registry.get_deployment_history()}


@router.post("/upload")
async def upload_model(file: UploadFile = File(...), notes: str = Form("")):
    """Create: register a new model version from an uploaded .pt file."""
    tmp_path = _save_upload_to_temp(file)
    try:
        entry = model_service.create_version(str(tmp_path), notes=notes)
        if not entry:
            raise HTTPException(500, "Failed to register uploaded model")
        return entry
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


@router.post("/{version_id}/replace")
async def replace_model(version_id: str, file: UploadFile = File(...)):
    """Update: replace the weights file backing an existing version."""
    tmp_path = _save_upload_to_temp(file)
    try:
        result = model_service.replace_version(version_id, str(tmp_path))
        if not result["success"]:
            raise HTTPException(404, result.get("error", "Replace failed"))
        return result
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


@router.delete("/{version_id}")
async def delete_model(version_id: str):
    result = model_service.delete_version(version_id)
    if not result["success"]:
        raise HTTPException(400, result.get("error", "Delete failed"))
    return result
