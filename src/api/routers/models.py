"""Model version registry, deployment, and rollback."""
from __future__ import annotations

from fastapi import APIRouter

import config as cfg
from src.api.schemas.models import DeployConfirmRequest, RollbackRequest
from src.api.services import model_service
from src.retraining.model_registry import ModelRegistry

router = APIRouter(prefix="/api/v2/models", tags=["models"])


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
