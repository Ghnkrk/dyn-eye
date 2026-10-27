"""Side-by-side baseline vs finetuned YOLO inference comparison."""
from __future__ import annotations

import base64

from fastapi import APIRouter, File, HTTPException, UploadFile

import config as cfg

router = APIRouter(prefix="/api/v2/inference", tags=["inference"])

_PALETTE = [
    (16, 185, 129),   # emerald
    (59, 130, 246),   # blue
    (245, 158, 11),   # amber
    (239, 68, 68),    # red
    (168, 85, 247),   # purple
    (236, 72, 153),   # pink
    (20, 184, 166),   # teal
    (251, 191, 36),   # yellow
]


def _run_model(model_path, img, conf: float):
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.predict(img, conf=conf, verbose=False)
    result = results[0]

    annotated = img.copy()
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf_val = float(box.conf[0])
        label = result.names.get(cls_id, f"class_{cls_id}")
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        color = _PALETTE[cls_id % len(_PALETTE)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf_val:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        detections.append({"label": label, "confidence": round(conf_val, 4), "box": [x1, y1, x2, y2]})

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    class_names = list(result.names.values()) if result.names else []
    return b64, detections, class_names


@router.post("/compare")
async def inference_compare(file: UploadFile = File(...), conf: float = 0.10):
    """Run both the baseline and finetuned YOLO models on one image, side by side."""
    import cv2
    import numpy as np

    baseline_path = cfg.MODELS_DIR / "best_initial.pt"
    finetuned_path = cfg.YOLO_MODEL_PATH  # models/best.pt

    if not baseline_path.exists():
        raise HTTPException(404, "Baseline model (best_initial.pt) not found")
    if not finetuned_path.exists():
        raise HTTPException(404, "Finetuned model (best.pt) not found")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_original is None:
        raise HTTPException(400, "Invalid image file")

    baseline_b64, baseline_dets, baseline_classes = _run_model(baseline_path, img_original, conf)
    finetuned_b64, finetuned_dets, finetuned_classes = _run_model(finetuned_path, img_original, conf)

    return {
        "baseline": {
            "image_b64": baseline_b64,
            "detections": baseline_dets,
            "num_detections": len(baseline_dets),
            "classes": baseline_classes,
            "model": "Baseline (v1 — 6 classes)",
        },
        "finetuned": {
            "image_b64": finetuned_b64,
            "detections": finetuned_dets,
            "num_detections": len(finetuned_dets),
            "classes": finetuned_classes,
            "model": "Finetuned (latest — expanded classes)",
        },
    }
