"""
Export Label Studio Annotations to YOLO Format

After cluster naming in Label Studio, export the annotations
and images to YOLO-compatible directory structure.

Output structure:
    yolo_dataset/
        data.yaml
        images/
            train/
            val/
        labels/
            train/
            val/
"""
from __future__ import annotations

import json
import random
import shutil
import requests
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.utils import get_logger, save_json
from src.features.known_defects_registry import register_defects

log = get_logger("ls_export")


def _get_headers() -> dict:
    return {
        "Authorization": f"Token {cfg.LABEL_STUDIO_API_KEY}",
    }


def fetch_annotations(project_id: int) -> list[dict]:
    """Fetch all completed annotations from a Label Studio project."""
    try:
        resp = requests.get(
            f"{cfg.LABEL_STUDIO_URL}/api/projects/{project_id}/export?exportType=JSON",
            headers=_get_headers(),
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error(f"Failed to fetch annotations: {e}")
        return []


def export_local_to_yolo(
    output_dir: str | Path | None = None,
    val_split: float = 0.2,
) -> dict:
    """
    Directly export local dashboard cluster assignments to YOLO format.
    Does not depend on Label Studio!
    """
    from PIL import Image

    output = Path(output_dir or cfg.YOLO_DATASET_DIR)

    # Create directory structure
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    if not manifest_path.exists():
        log.error("No cluster manifest found for local export")
        return {"total": 0, "train": 0, "val": 0}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        log.error(f"Failed to load manifest: {e}")
        return {"total": 0, "train": 0, "val": 0}

    # Gather clusters that have a name
    clusters = manifest.get("clusters", {})
    class_names = []
    class_map = {}

    # Map from source_image_path to list of dicts: {class_id, bbox: [ymin, xmin, ymax, xmax]}
    image_annotations = {}

    for cname, entry in clusters.items():
        defect_name = entry.get("defect_name")
        if not defect_name:
            continue

        if defect_name not in class_map:
            class_map[defect_name] = len(class_names)
            class_names.append(defect_name)

        class_id = class_map[defect_name]

        for crop in entry.get("crops", []):
            src_image = crop.get("source_image")
            if not src_image or not Path(src_image).exists():
                continue

            box = crop.get("box_2d_pixels")
            if not box or len(box) != 4:
                continue

            image_annotations.setdefault(src_image, []).append({
                "class_id": class_id,
                "box": box
            })

    if not image_annotations:
        log.warning("No named defect clusters or crops found in manifest")
        return {"total": 0, "train": 0, "val": 0}

    # Split images into train/val
    src_images = list(image_annotations.keys())
    random.shuffle(src_images)
    split_idx = int(len(src_images) * (1 - val_split))
    train_images = src_images[:split_idx]
    val_images = src_images[split_idx:]

    processed = {"train": 0, "val": 0}

    for split, images in [("train", train_images), ("val", val_images)]:
        for src_path_str in images:
            src_path = Path(src_path_str)
            image_name = src_path.name

            # Copy source image
            dst_img = output / "images" / split / image_name
            shutil.copy2(src_path, dst_img)

            # Read image dimensions
            try:
                with Image.open(src_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                log.error(f"Failed to read dimensions of {src_path}: {e}")
                continue

            label_lines = []
            for ann in image_annotations[src_path_str]:
                class_id = ann["class_id"]
                ymin, xmin, ymax, xmax = ann["box"]

                w = xmax - xmin
                h = ymax - ymin

                cx = (xmin + w / 2) / img_width
                cy = (ymin + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height

                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))

                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}")

            if label_lines:
                label_name = src_path.stem + ".txt"
                label_path = output / "labels" / split / label_name
                label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                processed[split] += 1

    # Write data.yaml
    data_yaml = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = output / "data.yaml"
    import yaml
    yaml_path.write_text(
        yaml.dump(data_yaml, default_flow_style=False),
        encoding="utf-8",
    )

    save_json(data_yaml, output / "data.json")
    save_json({"class_map": class_map, "class_names": class_names}, output / "class_map.json")

    log.info(
        f"Local export complete: {processed['train']} train, "
        f"{processed['val']} val, {len(class_names)} classes"
    )

    if class_names:
        added = register_defects(class_names, source="local_dashboard_export")
        if added:
            log.info(f"Registered {len(added)} new defect classes: {added}")

    return {
        "total": processed["train"] + processed["val"],
        "train": processed["train"],
        "val": processed["val"],
        "classes": class_names,
        "output_dir": str(output),
        "data_yaml": str(yaml_path),
    }


def export_to_yolo(
    project_id: int,
    output_dir: str | Path | None = None,
    val_split: float = 0.2,
) -> dict:
    """
    Export Label Studio annotations to YOLO format.
    Falls back to local dashboard manifest if project_id is -1 or fetching annotations fails.

    Args:
        project_id: Label Studio project ID
        output_dir: Output directory (defaults to config)
        val_split: Fraction of data for validation (0-1)

    Returns:
        Summary dict with counts and paths.
    """
    if project_id == -1:
        log.info("Label Studio project_id is -1. Using local dashboard cluster manifest directly.")
        return export_local_to_yolo(output_dir, val_split)

    output = Path(output_dir or cfg.YOLO_DATASET_DIR)

    # Fetch annotations
    tasks = fetch_annotations(project_id)
    if not tasks:
        log.warning("No annotations found in Label Studio. Falling back to local dashboard cluster manifest.")
        return export_local_to_yolo(output_dir, val_split)

    # Build class map from annotations
    class_names: list[str] = []
    class_map: dict[str, int] = {}

    for task in tasks:
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                labels = result.get("value", {}).get("rectanglelabels", [])
                for label in labels:
                    if label not in class_map:
                        class_map[label] = len(class_names)
                        class_names.append(label)

    log.info(f"Found {len(class_names)} classes: {class_names}")

    # Shuffle and split
    random.shuffle(tasks)
    split_idx = int(len(tasks) * (1 - val_split))
    train_tasks = tasks[:split_idx]
    val_tasks = tasks[split_idx:]

    processed = {"train": 0, "val": 0}

    for split, split_tasks in [("train", train_tasks), ("val", val_tasks)]:
        for task in split_tasks:
            image_url = task.get("data", {}).get("image", "")
            image_name = Path(image_url).name if image_url else None
            if not image_name:
                continue

            # Resolve source image
            source_path = _resolve_source_image(image_url, task)
            if not source_path:
                continue

            # Copy image
            dst_img = output / "images" / split / image_name
            shutil.copy2(source_path, str(dst_img))

            # Generate YOLO label file
            label_lines = []
            img_width = None
            img_height = None

            for ann in task.get("annotations", []):
                for result in ann.get("result", []):
                    if result.get("type") != "rectanglelabels":
                        continue

                    value = result.get("value", {})
                    img_width = result.get("original_width", 100)
                    img_height = result.get("original_height", 100)

                    labels = value.get("rectanglelabels", [])
                    if not labels:
                        continue

                    class_id = class_map.get(labels[0], 0)

                    # Convert percentage to YOLO format (normalized center + w/h)
                    x_pct = value.get("x", 0)
                    y_pct = value.get("y", 0)
                    w_pct = value.get("width", 0)
                    h_pct = value.get("height", 0)

                    cx = (x_pct + w_pct / 2) / 100.0
                    cy = (y_pct + h_pct / 2) / 100.0
                    w = w_pct / 100.0
                    h = h_pct / 100.0

                    # Clamp to [0, 1]
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))

                    label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if label_lines:
                label_name = Path(image_name).stem + ".txt"
                label_path = output / "labels" / split / label_name
                label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                processed[split] += 1

    # Write data.yaml
    data_yaml = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = output / "data.yaml"
    import yaml
    yaml_path.write_text(
        yaml.dump(data_yaml, default_flow_style=False),
        encoding="utf-8",
    )

    # Also save as JSON for convenience
    save_json(data_yaml, output / "data.json")
    save_json({"class_map": class_map, "class_names": class_names},
              output / "class_map.json")

    log.info(
        f"YOLO export complete: {processed['train']} train, "
        f"{processed['val']} val, {len(class_names)} classes"
    )

    # Auto-register exported class names into the known-defects registry
    if class_names:
        added = register_defects(class_names, source="label_studio_export")
        if added:
            log.info(f"Registered {len(added)} new defect classes: {added}")

    return {
        "total": processed["train"] + processed["val"],
        "train": processed["train"],
        "val": processed["val"],
        "classes": class_names,
        "output_dir": str(output),
        "data_yaml": str(yaml_path),
    }


def _resolve_source_image(url: str, task: dict) -> str | None:
    """Resolve an image URL/path to a local file."""
    # Try local-files pattern
    if "/data/local-files/" in url:
        relative = url.split("?d=")[-1] if "?d=" in url else ""
        candidate = cfg.DATA_DIR / relative
        if candidate.exists():
            return str(candidate)

    # Try source_file from task metadata
    source = task.get("data", {}).get("source_file", "")
    if source:
        for search_dir in [cfg.INPUT_IMAGES_DIR, cfg.CROPS_DIR, cfg.CLUSTERS_DIR]:
            candidate = search_dir / source
            if candidate.exists():
                return str(candidate)
            # Search subdirectories
            for match in search_dir.rglob(source):
                return str(match)

    # Try filename from URL
    fname = Path(url).name
    for search_dir in [cfg.INPUT_IMAGES_DIR, cfg.CROPS_DIR, cfg.CLUSTERS_DIR]:
        candidate = search_dir / fname
        if candidate.exists():
            return str(candidate)
        for match in search_dir.rglob(fname):
            return str(match)

    return None
