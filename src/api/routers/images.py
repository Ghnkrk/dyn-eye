"""Image upload and serving: crops, cluster crops, and input images."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

import config as cfg
from src.utils import LogStream

router = APIRouter(prefix="/api/v2", tags=["images"])


@router.post("/images/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload images to run the pipeline on, clearing existing input images."""
    try:
        input_dir = cfg.INPUT_IMAGES_DIR

        if input_dir.exists():
            for item in input_dir.iterdir():
                if item.is_file() and item.name != ".gitkeep":
                    try:
                        item.unlink()
                    except Exception:
                        pass
        else:
            input_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for file in files:
            if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
                continue

            base_name = Path(file.filename).name  # prevent subdirectory write issues
            dest_path = input_dir / base_name
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_count += 1

        LogStream.emit(f"Uploaded {saved_count} new images to input directory", level="info", source="system")
        return {"success": True, "count": saved_count}
    except Exception as e:
        LogStream.emit(f"Image upload failed: {e}", level="error", source="system")
        raise HTTPException(500, f"Failed to upload images: {e}")


@router.get("/crops/{filename}")
async def serve_crop_image(filename: str):
    image_path = cfg.CROPS_DIR / filename
    if not image_path.exists():
        raise HTTPException(404, f"Crop image not found: {filename}")
    return FileResponse(str(image_path), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@router.get("/input-images")
async def list_input_images():
    if not cfg.INPUT_IMAGES_DIR.exists():
        return []
    return [f.name for f in sorted(cfg.INPUT_IMAGES_DIR.glob("*")) if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]


@router.get("/input-images/{filename}")
async def serve_input_image(filename: str):
    image_path = cfg.INPUT_IMAGES_DIR / filename
    if not image_path.exists():
        raise HTTPException(404, f"Image not found: {filename}")
    return FileResponse(str(image_path), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


# Two path segments after /clusters/, so this never collides with the
# single-segment routes in clusters.py (/clusters/manifest, /clusters/name, ...).
@router.get("/clusters/{cluster_name}/{filename}")
async def serve_cluster_image(cluster_name: str, filename: str):
    image_path = cfg.CLUSTERS_DIR / cluster_name / filename
    if not image_path.exists():
        raise HTTPException(404, f"Image not found: {cluster_name}/{filename}")
    return FileResponse(str(image_path), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
