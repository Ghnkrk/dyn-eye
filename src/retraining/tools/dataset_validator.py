"""
Tool: YOLO Dataset Format Validator

Validates that a YOLO-format dataset directory has:
  - Correct folder structure (images/train, images/val, labels/train, labels/val)
  - Valid data.yaml with required fields
  - Matching image-label pairs
  - Valid label file format (class_id cx cy w h, all normalized)
"""
from __future__ import annotations

import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("dataset_validator")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def validate_yolo_dataset(
    dataset_dir: str | None = None,
) -> dict:
    """
    Validate a YOLO-format dataset.

    Returns:
        {
            "valid": bool,
            "errors": [...],
            "warnings": [...],
            "stats": {
                "train_images": int,
                "val_images": int,
                "train_labels": int,
                "val_labels": int,
                "classes": [...],
                "orphan_images": [...],
                "orphan_labels": [...],
            }
        }
    """
    ds = Path(dataset_dir or cfg.YOLO_DATASET_DIR)
    errors: list[str] = []
    warnings: list[str] = []
    stats = {}

    # 1. Check directory structure
    required_dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val",
    ]
    for d in required_dirs:
        if not (ds / d).is_dir():
            errors.append(f"Missing required directory: {d}")

    # 2. Check data.yaml
    yaml_path = ds / "data.yaml"
    if not yaml_path.exists():
        errors.append("Missing data.yaml")
        class_names = []
        num_classes = 0
    else:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data_yaml = yaml.safe_load(f)

            required_keys = ["nc", "names", "train", "val"]
            for key in required_keys:
                if key not in data_yaml:
                    errors.append(f"data.yaml missing required key: '{key}'")

            class_names = data_yaml.get("names", [])
            num_classes = data_yaml.get("nc", 0)

            if len(class_names) != num_classes:
                errors.append(
                    f"data.yaml: nc={num_classes} but names has {len(class_names)} entries"
                )
        except Exception as e:
            errors.append(f"Failed to parse data.yaml: {e}")
            class_names = []
            num_classes = 0

    stats["classes"] = class_names

    # 3. Check image-label pairs
    for split in ["train", "val"]:
        img_dir = ds / "images" / split
        lbl_dir = ds / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        images = {f.stem: f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS}
        labels = {f.stem: f for f in lbl_dir.iterdir() if f.suffix == ".txt"}

        stats[f"{split}_images"] = len(images)
        stats[f"{split}_labels"] = len(labels)

        # Orphan images (no matching label)
        orphan_imgs = set(images.keys()) - set(labels.keys())
        if orphan_imgs:
            warnings.append(
                f"{split}: {len(orphan_imgs)} images without labels: {list(orphan_imgs)[:5]}..."
            )

        # Orphan labels (no matching image)
        orphan_lbls = set(labels.keys()) - set(images.keys())
        if orphan_lbls:
            warnings.append(
                f"{split}: {len(orphan_lbls)} labels without images: {list(orphan_lbls)[:5]}..."
            )

        # 4. Validate label file contents
        for stem, label_path in labels.items():
            try:
                lines = label_path.read_text(encoding="utf-8").strip().split("\n")
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(
                            f"{split}/{label_path.name}:{line_num} — "
                            f"expected 5 values, got {len(parts)}"
                        )
                        continue

                    try:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                    except ValueError:
                        errors.append(
                            f"{split}/{label_path.name}:{line_num} — "
                            f"invalid numeric values: {line}"
                        )
                        continue

                    if cls_id < 0 or cls_id >= max(num_classes, 1):
                        errors.append(
                            f"{split}/{label_path.name}:{line_num} — "
                            f"class_id {cls_id} out of range [0, {num_classes})"
                        )

                    for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                        if val < 0 or val > 1:
                            errors.append(
                                f"{split}/{label_path.name}:{line_num} — "
                                f"{name}={val:.4f} out of [0, 1] range"
                            )
            except Exception as e:
                errors.append(f"Failed to read {split}/{label_path.name}: {e}")

    stats["orphan_images"] = []  # Populated above in warnings
    stats["orphan_labels"] = []

    is_valid = len(errors) == 0

    result = {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }

    if is_valid:
        log.info(f"Dataset validation PASSED: {stats}")
    else:
        log.error(f"Dataset validation FAILED with {len(errors)} errors")
        for err in errors[:10]:
            log.error(f"  ✗ {err}")

    return result
