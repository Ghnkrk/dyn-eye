"""
Node 1 — YOLO Batch Inference

Runs YOLO (best.pt) over all images in the input directory.
Separates images into known vs unknown defects based on
confidence threshold and the known defect class-name list.
Saves unknown filenames to a JSON file for downstream use.
"""
from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger, save_json, list_images
from src.features.known_defects_registry import get_known_defect_names

log = get_logger("yolo_inference")


def yolo_inference_node(state: dict) -> dict:
    """
    LangGraph node: YOLO batch inference.

    Reads:
        state["input_images_dir"]
        state["use_cache"]

    Writes:
        state["all_image_paths"]
        state["known_image_paths"]
        state["unknown_image_paths"]
        state["unknown_defects_json"]
        state["yolo_raw_results"]
        state["known_defect_names"]  (populated from registry)
    """
    # Always hot-read from the registry — picks up newly deployed classes
    known_names = get_known_defect_names()

    # ── Cache mode: skip YOLO entirely ───────────────────────
    if state.get("use_cache"):
        log.info("Cache mode — skipping YOLO inference (reusing existing crops).")
        # Reconstruct unknown_image_paths from cached VLM annotations if available
        vlm_annotations = state.get("vlm_annotations", [])
        unknown_paths = list({
            ann["image_path"] for ann in vlm_annotations
            if "image_path" in ann
        })
        return {
            "all_image_paths": unknown_paths,
            "known_image_paths": [],
            "unknown_image_paths": unknown_paths,
            "unknown_defects_json": str(cfg.UNKNOWN_DEFECTS_JSON),
            "yolo_raw_results": [],
            "known_defect_names": known_names,
            "_cached": True,
        }

    # ── Normal mode: full YOLO inference ─────────────────────
    images_dir = Path(state.get("input_images_dir", str(cfg.INPUT_IMAGES_DIR)))
    conf_thresh = cfg.YOLO_CONFIDENCE_THRESHOLD
    known_names_lower = {n.lower() for n in known_names}

    model_path = state.get("yolo_model_path", str(cfg.YOLO_MODEL_PATH))

    log.info(f"Loading YOLO model from {model_path}")
    if not Path(model_path).exists():
        msg = f"YOLO model not found at {model_path}. Place best.pt in models/ directory."
        log.error(msg)
        return {
            "errors": state.get("errors", []) + [msg],
            "all_image_paths": [],
            "known_image_paths": [],
            "unknown_image_paths": [],
            "unknown_defects_json": "",
            "yolo_raw_results": [],
        }

    model = YOLO(str(model_path))
    image_paths = list_images(images_dir)
    log.info(f"Found {len(image_paths)} images in {images_dir}")

    all_paths: list[str] = []
    known_paths: list[str] = []
    unknown_paths: list[str] = []
    raw_results: list[dict] = []

    # Batch inference
    batch_size = 16
    for batch_start in range(0, len(image_paths), batch_size):
        batch = image_paths[batch_start : batch_start + batch_size]
        batch_str = [str(p) for p in batch]

        results = model.predict(source=batch_str, conf=conf_thresh, verbose=False)

        for img_path, result in zip(batch, results):
            img_str = str(img_path)
            all_paths.append(img_str)

            detections = []
            is_known = False

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()

                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox_xyxy": [round(c, 2) for c in xyxy],
                    })

                    if cls_name.lower() in known_names_lower and conf >= conf_thresh:
                        is_known = True

            if is_known:
                known_paths.append(img_str)
            else:
                unknown_paths.append(img_str)

            raw_results.append({
                "image_path": img_str,
                "detections": detections,
                "classified_as": "known" if is_known else "unknown",
            })

    # Save unknown defect filenames
    unknown_json_path = str(cfg.UNKNOWN_DEFECTS_JSON)
    unknown_data = {
        "count": len(unknown_paths),
        "image_paths": unknown_paths,
        "image_names": [Path(p).name for p in unknown_paths],
    }
    save_json(unknown_data, unknown_json_path)

    log.info(
        f"YOLO inference complete: {len(known_paths)} known, "
        f"{len(unknown_paths)} unknown out of {len(all_paths)} total"
    )

    return {
        "all_image_paths": all_paths,
        "known_image_paths": known_paths,
        "unknown_image_paths": unknown_paths,
        "unknown_defects_json": unknown_json_path,
        "yolo_raw_results": raw_results,
        "known_defect_names": known_names,
    }
