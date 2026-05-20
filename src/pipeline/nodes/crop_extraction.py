"""
Node 3 — Crop Extraction

Uses VLM bounding-box annotations to crop defect regions from
original images and saves them to data/crops/.
"""
from __future__ import annotations

import cv2
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("crop_extraction")


def crop_extraction_node(state: dict) -> dict:
    """
    LangGraph node: crop extraction from VLM bounding boxes.

    Reads:
        state["vlm_annotations"]

    Writes:
        state["crop_paths"]
        state["crop_metadata"]
    """
    annotations = state.get("vlm_annotations", [])
    crops_dir = cfg.CROPS_DIR
    crops_dir.mkdir(parents=True, exist_ok=True)

    crop_paths: list[str] = []
    crop_metadata: list[dict] = []
    crop_idx = 0

    for ann in annotations:
        img_path = ann.get("image_path", "")
        findings = ann.get("findings", [])

        if not findings or not Path(img_path).exists():
            continue

        img = cv2.imread(img_path)
        if img is None:
            log.warning(f"Could not read image: {img_path}")
            continue

        h, w = img.shape[:2]
        img_stem = Path(img_path).stem

        for f_idx, finding in enumerate(findings):
            box = finding.get("box_2d", [])
            if len(box) != 4:
                continue

            # box_2d is [ymin, xmin, ymax, xmax] in 0-1000 scale
            y1 = int(box[0] * h / 1000)
            x1 = int(box[1] * w / 1000)
            y2 = int(box[2] * h / 1000)
            x2 = int(box[3] * w / 1000)

            # Clamp to image boundaries
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = min(h, y2), min(w, x2)

            # Geometric area guard (skip tiny/pixelated overzoomed crops < 30x30 pixels)
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < 30 or crop_h < 30 or (crop_w * crop_h) < 900:
                log.info(f"Skipping tiny/overzoomed crop from {img_stem} (dimensions: {crop_w}x{crop_h} below 30x30 threshold)")
                continue

            # Overzoom guard (covers >85% of original image dimensions - typical VLM failure mode)
            if crop_w > 0.85 * w and crop_h > 0.85 * h:
                log.info(f"Skipping overzoomed crop from {img_stem} (covers >85% of original image dimensions: {crop_w}x{crop_h})")
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Blur guard (Laplacian variance < 25.0 - filters out out-of-focus or uninformative regions)
            try:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                if blur_score < 25.0:
                    log.info(f"Skipping blurry crop from {img_stem} (Laplacian variance: {blur_score:.2f} < 25.0)")
                    continue
            except Exception as e:
                log.warning(f"Error checking blur on crop: {e}")

            crop_name = f"{img_stem}_crop_{crop_idx:04d}.jpg"
            crop_path = str(crops_dir / crop_name)
            cv2.imwrite(crop_path, crop)

            crop_paths.append(crop_path)
            crop_metadata.append({
                "crop_path": crop_path,
                "source_image": img_path,
                "source_image_name": Path(img_path).name,
                "finding_index": f_idx,
                "box_2d_raw": box,
                "box_2d_pixels": [y1, x1, y2, x2],
                "physical_traits": finding.get("physical_traits", ""),
                "crop_width": x2 - x1,
                "crop_height": y2 - y1,
            })
            crop_idx += 1

    log.info(f"Extracted {len(crop_paths)} crops from {len(annotations)} images")

    return {
        "crop_paths": crop_paths,
        "crop_metadata": crop_metadata,
    }
