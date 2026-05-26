"""
Tool: YOLO Fine-Tuning (Full Parameters)

Runs YOLO fine-tuning using Ultralytics API.
Takes the current best.pt model and trains on the new dataset
with all configurable hyperparameters — designed to accept
LLM-recommended or user-specified training configs.
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
    # ── Learning rate & optimizer ──
    lr0: float | None = None,
    lrf: float | None = None,
    momentum: float | None = None,
    weight_decay: float | None = None,
    warmup_epochs: float | None = None,
    patience: int | None = None,
    optimizer: str | None = None,
    cos_lr: bool | None = None,
    # ── Backbone freeze ──
    freeze: int | None = None,
    # ── Augmentation ──
    augment: bool | None = None,
    mosaic: float | None = None,
    mixup: float | None = None,
    degrees: float | None = None,
    translate: float | None = None,
    scale: float | None = None,
    flipud: float | None = None,
    fliplr: float | None = None,
    hsv_h: float | None = None,
    hsv_s: float | None = None,
    hsv_v: float | None = None,
) -> dict:
    """
    Fine-tune a YOLO model on the given dataset with full hyperparameter control.

    All parameters default to config values if not provided.
    The LLM advisor or dashboard can override any of them.

    Returns:
        {
            "success": bool,
            "model_path": str,
            "metrics": {...},
            "training_config": {...},
            "error": str | None,
        }
    """
    from ultralytics import YOLO
    from src.utils import LogStream

    data = data_yaml or str(cfg.YOLO_DATASET_DIR / "data.yaml")
    base_model = model_path or str(cfg.YOLO_MODEL_PATH)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    proj = project_name or f"finetune_{ts}"

    # Resolve all hyperparameters with config defaults
    train_cfg = {
        "epochs": epochs or cfg.YOLO_TRAIN_EPOCHS,
        "imgsz": imgsz or cfg.YOLO_TRAIN_IMGSZ,
        "batch": batch or cfg.YOLO_TRAIN_BATCH,
        "lr0": lr0 if lr0 is not None else cfg.YOLO_TRAIN_LR0,
        "lrf": lrf if lrf is not None else cfg.YOLO_TRAIN_LRF,
        "momentum": momentum if momentum is not None else cfg.YOLO_TRAIN_MOMENTUM,
        "weight_decay": weight_decay if weight_decay is not None else cfg.YOLO_TRAIN_WEIGHT_DECAY,
        "warmup_epochs": warmup_epochs if warmup_epochs is not None else cfg.YOLO_TRAIN_WARMUP_EPOCHS,
        "patience": patience if patience is not None else cfg.YOLO_TRAIN_PATIENCE,
        "optimizer": optimizer or cfg.YOLO_TRAIN_OPTIMIZER,
        "cos_lr": cos_lr if cos_lr is not None else cfg.YOLO_TRAIN_COS_LR,
    }

    # Optional: freeze backbone layers
    freeze_val = freeze if freeze is not None else cfg.YOLO_TRAIN_FREEZE
    if freeze_val is not None:
        train_cfg["freeze"] = freeze_val

    # Augmentation parameters (only set if explicitly provided)
    aug_params = {}
    if augment is not None:
        aug_params["augment"] = augment
    if mosaic is not None:
        aug_params["mosaic"] = mosaic
    if mixup is not None:
        aug_params["mixup"] = mixup
    if degrees is not None:
        aug_params["degrees"] = degrees
    if translate is not None:
        aug_params["translate"] = translate
    if scale is not None:
        aug_params["scale"] = scale
    if flipud is not None:
        aug_params["flipud"] = flipud
    if fliplr is not None:
        aug_params["fliplr"] = fliplr
    if hsv_h is not None:
        aug_params["hsv_h"] = hsv_h
    if hsv_s is not None:
        aug_params["hsv_s"] = hsv_s
    if hsv_v is not None:
        aug_params["hsv_v"] = hsv_v

    log.info(f"Starting YOLO fine-tuning:")
    log.info(f"  Base model: {base_model}")
    log.info(f"  Data: {data}")
    log.info(f"  Core config: {train_cfg}")
    if aug_params:
        log.info(f"  Augmentation: {aug_params}")

    LogStream.emit(
        f"YOLO training config: epochs={train_cfg['epochs']}, batch={train_cfg['batch']}, "
        f"lr0={train_cfg['lr0']}, optimizer={train_cfg['optimizer']}",
        level="info", source="yolo_train"
    )

    if not Path(base_model).exists():
        return {
            "success": False,
            "model_path": "",
            "metrics": {},
            "training_config": train_cfg,
            "error": f"Base model not found: {base_model}",
        }

    if not Path(data).exists():
        return {
            "success": False,
            "model_path": "",
            "metrics": {},
            "training_config": train_cfg,
            "error": f"data.yaml not found: {data}",
        }

    try:
        model = YOLO(base_model)

        # Register callbacks to stream training progress to the dashboard
        def on_train_start(trainer):
            LogStream.emit("YOLO training loop started", level="step", source="yolo_train")

        def on_train_epoch_start(trainer):
            LogStream.emit(
                f"Epoch {trainer.epoch + 1}/{trainer.epochs} starting...",
                level="info", source="yolo_train"
            )

        def on_fit_epoch_end(trainer):
            loss = getattr(trainer, "loss", 0.0)
            if hasattr(loss, "item"):
                loss = loss.item()
            loss_items = getattr(trainer, "loss_items", None)
            loss_str = f"Loss: {loss:.4f}"
            if loss_items is not None:
                try:
                    box_l = float(loss_items[0])
                    cls_l = float(loss_items[1])
                    loss_str = f"Loss: {loss:.4f} (Box: {box_l:.3f}, Cls: {cls_l:.3f})"
                except Exception:
                    pass
            LogStream.emit(
                f"Epoch {trainer.epoch + 1}/{trainer.epochs} completed. {loss_str}",
                level="progress",
                source="yolo_train"
            )

        def on_train_end(trainer):
            LogStream.emit(
                "YOLO training loop finished. Finalizing weights...",
                level="step", source="yolo_train"
            )

        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.add_callback("on_train_end", on_train_end)

        # Merge all training arguments
        train_args = {
            "data": data,
            "project": str(cfg.PROJECT_ROOT / "runs" / "train"),
            "name": proj,
            "exist_ok": True,
            "verbose": True,
            **train_cfg,
            **aug_params,
        }

        results = model.train(**train_args)

        # Find the best model
        train_dir = cfg.PROJECT_ROOT / "runs" / "train" / proj
        best_model = train_dir / "weights" / "best.pt"

        if not best_model.exists():
            return {
                "success": False,
                "model_path": "",
                "metrics": {},
                "training_config": train_cfg,
                "error": "Training completed but best.pt not found",
            }

        # Version the model
        version_name = f"best_v{ts}.pt"
        version_path = cfg.MODEL_VERSIONS_DIR / version_name
        version_path.parent.mkdir(parents=True, exist_ok=True)
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
        LogStream.emit(
            f"Training complete! mAP50={metrics.get('map50', 0):.3f}, "
            f"Precision={metrics.get('precision', 0):.3f}, Recall={metrics.get('recall', 0):.3f}",
            level="info", source="yolo_train"
        )

        return {
            "success": True,
            "model_path": str(version_path),
            "replaced_model": str(cfg.YOLO_MODEL_PATH),
            "metrics": metrics,
            "training_config": {**train_cfg, **aug_params},
            "error": None,
        }

    except Exception as e:
        log.error(f"Training failed: {e}")
        LogStream.emit(f"YOLO training failed: {e}", level="error", source="yolo_train")
        return {
            "success": False,
            "model_path": "",
            "metrics": {},
            "training_config": train_cfg,
            "error": str(e),
        }
