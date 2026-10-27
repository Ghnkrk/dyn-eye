"""FAISS novelty index admin: setup/reset/rebuild."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas.faiss import FAISSSetupRequest
from src.utils import LogStream

router = APIRouter(prefix="/api/v2/faiss", tags=["faiss"])


@router.post("/setup")
async def faiss_setup(req: FAISSSetupRequest):
    LogStream.emit("Setting up FAISS index...", level="step", source="faiss")
    try:
        from src.features.faiss_index import FAISSIndexManager
        manager = FAISSIndexManager()
        count = manager.setup(known_crops_dir=req.known_crops_dir, class_subdirs=req.class_subdirs)
        LogStream.emit(f"FAISS index built with {count} vectors", level="info", source="faiss")
        return {"message": f"FAISS index built with {count} vectors", "count": count}
    except Exception as e:
        LogStream.emit(f"FAISS setup failed: {e}", level="error", source="faiss")
        raise HTTPException(500, f"FAISS setup failed: {e}")


@router.post("/reset")
async def faiss_reset():
    LogStream.emit("Resetting FAISS index to empty state...", level="step", source="faiss")
    try:
        from src.features.faiss_index import FAISSIndexManager
        manager = FAISSIndexManager()
        manager.reset()
        LogStream.emit(
            "FAISS index successfully reset/deleted. All defects will be treated as novel.",
            level="info", source="faiss",
        )
        return {"message": "FAISS index cleared successfully", "success": True}
    except Exception as e:
        LogStream.emit(f"FAISS reset failed: {e}", level="error", source="faiss")
        raise HTTPException(500, f"FAISS reset failed: {e}")


@router.post("/rebuild")
async def faiss_rebuild():
    LogStream.emit("Manually triggering FAISS index rebuild...", level="step", source="faiss")
    try:
        from src.features.faiss_index import FAISSIndexManager
        manager = FAISSIndexManager()
        count = manager.setup()
        LogStream.emit(f"FAISS index manually rebuilt successfully with {count} vectors.", level="info", source="faiss")
        return {"message": f"FAISS index rebuilt successfully with {count} vectors", "success": True, "count": count}
    except Exception as e:
        LogStream.emit(f"FAISS rebuild failed: {e}", level="error", source="faiss")
        raise HTTPException(500, f"FAISS rebuild failed: {e}")
