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


def _build_known_crop_index(known_names_lower: set[str]) -> list[tuple[str, str]]:
    """Scan the known-defect crop registry ONCE → [(class_name, crop_file_stem), ...]."""
    index: list[tuple[str, str]] = []
    known_dir = Path(cfg.KNOWN_DEFECTS_DIR)
    if not known_dir.exists():
        return index
    for sub in known_dir.iterdir():
        if sub.is_dir() and sub.name.lower() in known_names_lower:
            for crop_file in sub.iterdir():
                if crop_file.is_file():
                    index.append((sub.name, crop_file.stem))
    return index


def _match_known_crop(img_stem: str, index: list[tuple[str, str]]) -> str | None:
    """
    Return the class whose crop was cut from this image, else None.
    Matches 'img1' / 'img1_crop_0' but NOT 'img10' (next char must be a separator).
    """
    for class_name, crop_stem in index:
        if crop_stem.startswith(img_stem) and (
            len(crop_stem) == len(img_stem) or not crop_stem[len(img_stem)].isalnum()
        ):
            return class_name
    return None


def _save_known_replay(raw_results: list[dict], known_names_lower: set[str]) -> None:
    """
    Persist confident detections on KNOWN images (YOLO-normalised boxes) so the
    retraining export can replay them as pseudo-labels for the old classes.
    """
    replay = []
    for r in raw_results:
        if r["classified_as"] != "known":
            continue
        dets = [
            {"class_name": d["class_name"], "bbox_xywhn": d["bbox_xywhn"]}
            for d in r["detections"]
            if d.get("class_id", -1) >= 0
            and "bbox_xywhn" in d
            and d["confidence"] >= cfg.REPLAY_MIN_CONF
            and d["class_name"].lower() in known_names_lower
        ]
        if dets:
            replay.append({"image_path": r["image_path"], "detections": dets})
    try:
        save_json(replay, str(cfg.KNOWN_REPLAY_JSON))
        log.info(f"Saved replay pseudo-labels for {len(replay)} known images")
    except Exception as e:
        log.warning(f"Could not save replay file: {e}")


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

    known_crop_index = _build_known_crop_index(known_names_lower)

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
                    xywhn = box.xywhn[0].tolist()

                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox_xyxy": [round(c, 2) for c in xyxy],
                        "bbox_xywhn": [round(c, 6) for c in xywhn],
                    })

                    if cls_name.lower() in known_names_lower and conf >= conf_thresh:
                        is_known = True

            # Fallback Stage 2: Check if this image has a crop in the known_defect_crops registry
            # (Bootstraps filtering for known defects before the model is fully fine-tuned)
            if not is_known:
                matched_class = _match_known_crop(img_path.stem, known_crop_index)
                if matched_class:
                    is_known = True
                    log.info(f"Fallback matched known crop for {img_path.name} in registry class '{matched_class}'")
                    detections.append({
                        "class_id": -99,
                        "class_name": matched_class,
                        "confidence": 1.0,
                        "bbox_xyxy": [0.0, 0.0, 0.0, 0.0],
                        "note": "Matched via known defect crops registry"
                    })

            if is_known:
                known_paths.append(img_str)
            else:
                unknown_paths.append(img_str)

            raw_results.append({
                "image_path": img_str,
                "detections": detections,
                "classified_as": "known" if is_known else "unknown",
            })

    _save_known_replay(raw_results, known_names_lower)

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
