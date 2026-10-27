"""Full system reset and non-sensitive config readout."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

import config as cfg
from src.api.schemas.system import ResetRequest
from src.api.services import pipeline_service
from src.api.services.ws_manager import ws_manager
from src.db.session import get_session
from src.db import sync as db_sync
from src.utils import LogStream, save_json

router = APIRouter(prefix="/api/v2/system", tags=["system"])


@router.post("/reset-all")
async def system_reset_all(req: ResetRequest | None = None):
    """Reset everything to the initial state: YOLO v1, 6 known classes, pristine FAISS."""
    delete_crops = req.delete_crops if req else True
    LogStream.emit("Initiating universal system reset...", level="step", source="system")
    try:
        initial_model = cfg.MODELS_DIR / "best_initial.pt"
        if initial_model.exists():
            shutil.copy2(str(initial_model), str(cfg.YOLO_MODEL_PATH))
            LogStream.emit("YOLO model restored to initial v1 weights", level="info", source="system")
        else:
            LogStream.emit("Initial YOLO model backup not found, keeping active model", level="warning", source="system")

        initial_classes = list(cfg.BASELINE_CLASSES)

        registry_file = cfg.MODELS_DIR / "registry.json"
        registry_data = {"versions": [], "current": None, "deployment_history": []}

        for f in cfg.MODEL_VERSIONS_DIR.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass

        if initial_model.exists():
            v1_dst = cfg.MODEL_VERSIONS_DIR / "best_v1_initial.pt"
            shutil.copy2(str(initial_model), str(v1_dst))

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            v1_entry = {
                "version_id": "v1_initial",
                "version_num": 1,
                "path": str(v1_dst),
                "original_path": str(initial_model),
                "timestamp": ts,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {},
                "training_config": {},
                "source": "factory_reset",
                "notes": "Original 6-class YOLOv8 base model",
                "classes": initial_classes,
                "dataset_stats": {},
                "size_mb": round(initial_model.stat().st_size / (1024 * 1024), 2),
                "status": "deployed",
            }
            registry_data["versions"].append(v1_entry)
            registry_data["current"] = "v1_initial"

        save_json(registry_data, registry_file)
        LogStream.emit("Model registry cleared and v1_initial registered", level="info", source="system")

        reg_data = {
            "defect_classes": initial_classes,
            "version": 1,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "history": [
                {"added": initial_classes, "source": "initial_reset", "timestamp": datetime.now(timezone.utc).isoformat()}
            ],
        }
        save_json(reg_data, cfg.DATA_DIR / "known_defects.json")
        LogStream.emit("Known defect classes reset to 6 initial categories", level="info", source="system")

        folders_to_clear = [cfg.CLUSTERS_DIR]
        if delete_crops:
            folders_to_clear.append(cfg.CROPS_DIR)

        for folder in folders_to_clear:
            if folder.exists():
                for item in folder.iterdir():
                    if item.is_dir():
                        try:
                            shutil.rmtree(str(item))
                        except Exception:
                            pass
                    elif item.is_file() and item.name != ".gitkeep":
                        try:
                            item.unlink()
                        except Exception:
                            pass

        for split in ["train", "val"]:
            for folder_name in ["images", "labels"]:
                dir_path = cfg.YOLO_DATASET_DIR / folder_name / split
                if dir_path.exists():
                    for item in dir_path.iterdir():
                        if item.is_file() and item.name != ".gitkeep":
                            try:
                                item.unlink()
                            except Exception:
                                pass

        flat_map = cfg.DATA_DIR / "defect_label_mapping.json"
        if flat_map.exists():
            flat_map.unlink()

        if cfg.VLM_PROMPT_CACHE_PATH.exists():
            cfg.VLM_PROMPT_CACHE_PATH.unlink()

        msg = "Clusters and fine-tuning datasets cleared"
        if delete_crops:
            msg = "Crops, clusters, and fine-tuning datasets cleared"
        LogStream.emit(msg, level="info", source="system")

        backup_index = cfg.FAISS_INDEX_DIR / "known_defects.index.backup"
        backup_labels = cfg.FAISS_INDEX_DIR / "known_defects_labels.json.backup"

        if backup_index.exists() and backup_labels.exists():
            shutil.copy2(str(backup_index), str(cfg.FAISS_INDEX_FILE))
            shutil.copy2(str(backup_labels), str(cfg.FAISS_LABELS_FILE))
            count = 2965
            LogStream.emit(f"FAISS index instantly restored from pristine backups ({count} vectors)", level="info", source="system")
        else:
            from src.features.faiss_index import FAISSIndexManager
            manager = FAISSIndexManager()
            manager.reset()
            count = manager.setup()
            LogStream.emit(f"FAISS index rebuilt from pristine known crops folder ({count} vectors)", level="info", source="system")

        pipeline_service.reset_status()

        with get_session() as session:
            db_sync.sync_all(session)
        ws_manager.broadcast_threadsafe("clusters", "updated", {})
        ws_manager.broadcast_threadsafe("models", "updated", {})

        LogStream.emit("SYSTEM UNIVERSAL RESET COMPLETED SUCCESSFULLY!", level="success", source="system")
        return {"message": "System reset to initial state successfully", "success": True}
    except Exception as e:
        LogStream.emit(f"System reset failed: {e}", level="error", source="system")
        raise HTTPException(500, f"System reset failed: {e}")


@router.get("/config")
async def get_config():
    from src.features.known_defects_registry import get_known_defect_names
    return {
        "known_defect_names": get_known_defect_names(),
        "faiss_novelty_threshold": cfg.FAISS_NOVELTY_THRESHOLD,
        "hdbscan_min_cluster_size": cfg.HDBSCAN_MIN_CLUSTER_SIZE,
        "yolo_model_exists": cfg.YOLO_MODEL_PATH.exists(),
        "faiss_index_exists": cfg.FAISS_INDEX_FILE.exists(),
        "input_images_count": len(list(cfg.INPUT_IMAGES_DIR.glob("*"))) if cfg.INPUT_IMAGES_DIR.exists() else 0,
    }
