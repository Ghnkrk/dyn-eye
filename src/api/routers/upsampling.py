"""Upsampling page: settings, dataset balance, preview/generate, and the image viewer."""
from __future__ import annotations

import mimetypes

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.api.services import upsampling_service as svc
from src.retraining.tools import upsample as up

router = APIRouter(prefix="/api/v2/upsampling", tags=["upsampling"])


class Transforms(BaseModel):
    hflip: bool = True
    vflip: bool = True
    rot180: bool = True
    rot90: bool = False
    contrast: bool = True
    gamma: bool = True
    blur: bool = True
    noise: bool = True


class UpsamplingSettings(BaseModel):
    enabled: bool | None = None
    target_ratio: float = Field(0.5, gt=0, le=1)
    max_factor: float = Field(3.0, ge=1, le=10)
    imbalance_threshold: float = Field(2.0, ge=1, le=20)
    seed: int = 42
    transforms: Transforms = Transforms()


def _call(fn, *args):
    try:
        return fn(*args)
    except svc.Busy as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/status")
def get_status():
    return svc.status()


@router.get("/settings")
def get_settings():
    return up.load_settings()


@router.put("/settings")
def put_settings(req: UpsamplingSettings):
    body = req.model_dump()
    if body["enabled"] is None:
        body["enabled"] = up.load_settings()["enabled"]
    return _call(up.save_settings, body)


@router.post("/prepare")
def prepare():
    return _call(svc.prepare_dataset)


@router.post("/preview")
def preview(req: UpsamplingSettings):
    return _call(svc.preview, req.model_dump())


@router.post("/generate")
def generate(req: UpsamplingSettings):
    return _call(svc.generate, req.model_dump())


@router.delete("")
def clear():
    return _call(svc.clear)


@router.get("/groups")
def get_groups():
    return svc.groups()


@router.get("/image/{name}")
def get_image(name: str):
    path = svc.image_path(name)
    if path is None:
        raise HTTPException(404, "Image not found")
    media = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    # Augmented names are reused across runs, so never cache them for long
    return FileResponse(str(path), media_type=media, headers={"Cache-Control": "no-cache"})
