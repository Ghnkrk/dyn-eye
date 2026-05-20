"""
Tool: YOLO Fine-Tuning

Runs YOLO fine-tuning using Ultralytics API.
Takes the current best.pt model and trains on the new dataset
with specified hyperparameters.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("train_yolo")


def train_yolo(
    data_yaml: str | None = None,
    model_path: str | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    project_name: str | None = None,
) -> dict:
    """
    Fine-tune a YOLO model on the given dataset.

    Args:
        data_yaml: Path to data.yaml (defaults to config)
        model_path: Path to base YOLO model (defaults to config)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        project_name: Name for training run output

    Returns:
        {
            "success": bool,
            "model_path": str,  # Path to best trained model
            "metrics": {...},
            "error": str | None,
        }
    """
    from ultralytics import YOLO

    data = data_yaml or str(cfg.YOLO_DATASET_DIR / "data.yaml")
    base_model = model_path or str(cfg.YOLO_MODEL_PATH)
    ep = epochs or cfg.YOLO_TRAIN_EPOCHS
    img = imgsz or cfg.YOLO_TRAIN_IMGSZ
    bs = batch or cfg.YOLO_TRAIN_BATCH
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    proj = project_name or f"finetune_{ts}"

    log.info(f"Starting YOLO fine-tuning:")
    log.info(f"  Base model: {base_model}")
    log.info(f"  Data: {data}")
    log.info(f"  Epochs: {ep}, ImgSz: {img}, Batch: {bs}")

    if not Path(base_model).exists():
        return {
            "success": False,
            "model_path": "",
            "metrics": {},
            "error": f"Base model not found: {base_model}",
        }

    if not Path(data).exists():
        return {
            "success": False,
            "model_path": "",
            "metrics": {},
            "error": f"data.yaml not found: {data}",
        }

    try:
        model = YOLO(base_model)
        results = model.train(
            data=data,
            epochs=ep,
            imgsz=img,
            batch=bs,
            project=str(cfg.PROJECT_ROOT / "runs" / "train"),
            name=proj,
            exist_ok=True,
            verbose=True,
        )

        # Find the best model
        train_dir = cfg.PROJECT_ROOT / "runs" / "train" / proj
        best_model = train_dir / "weights" / "best.pt"

        if not best_model.exists():
            return {
                "success": False,
                "model_path": "",
                "metrics": {},
                "error": "Training completed but best.pt not found",
            }

        # Version the model
        version_name = f"best_v{ts}.pt"
        version_path = cfg.MODEL_VERSIONS_DIR / version_name
        shutil.copy2(str(best_model), str(version_path))

        # Replace current best.pt
        shutil.copy2(str(best_model), str(cfg.YOLO_MODEL_PATH))

        # Extract metrics
        metrics = {}
        try:
            metrics = {
                "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            }
        except Exception:
            pass

        log.info(f"Training complete. Best model: {version_path}")
        log.info(f"Metrics: {metrics}")

        return {
            "success": True,
            "model_path": str(version_path),
            "replaced_model": str(cfg.YOLO_MODEL_PATH),
            "metrics": metrics,
            "error": None,
        }

    except Exception as e:
        log.error(f"Training failed: {e}")
        return {
            "success": False,
            "model_path": "",
            "metrics": {},
            "error": str(e),
        }
