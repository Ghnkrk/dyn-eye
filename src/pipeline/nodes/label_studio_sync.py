"""
Node 7 — Manifest Save

Saves a cluster manifest JSON that maps:
  cluster_name → [{crop_file, source_image, bbox_pixels, bbox_raw, traits}]

This is the master mapping used by the dashboard for cluster editing
and later for propagating human-assigned cluster names back to
full-frame source images with bounding boxes for YOLO retraining.

(Label Studio integration has been removed — all cluster editing
is handled directly in the DYN-EYE dashboard.)
"""
from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger, save_json
from src.utils.io_helpers import list_images

log = get_logger("manifest_save")


def _save_cluster_manifest(
    run_id: str,
    cluster_folders: dict[int, str],
    crop_metadata: list[dict],
) -> Path:
    """
    Save a manifest JSON that maps:
      cluster_name → [{crop_file, source_image, bbox_pixels, bbox_raw, traits}]

    This is the master mapping used later to propagate human-assigned
    cluster names back to full-frame source images with bounding boxes.
    """
    meta_lookup: dict[str, dict] = {}
    for meta in crop_metadata:
        crop_name = Path(meta.get("crop_path", "")).name
        meta_lookup[crop_name] = meta

    manifest = {
        "run_id": run_id,
        "clusters": {},
    }

    for cluster_id, folder_path in cluster_folders.items():
        folder = Path(folder_path)
        cluster_name = folder.name

        if not folder.exists():
            continue

        images = list_images(folder)

        cluster_entry = {
            "cluster_id": int(cluster_id),
            "crop_count": len(images),
            "defect_name": None,  # Will be filled after human naming in dashboard
            "crops": [],
        }

        for img_path in images:
            meta = meta_lookup.get(img_path.name, {})
            cluster_entry["crops"].append({
                "crop_file": img_path.name,
                "crop_path": str(img_path),
                "source_image": meta.get("source_image", ""),
                "source_image_name": meta.get("source_image_name", ""),
                "box_2d_pixels": meta.get("box_2d_pixels", []),
                "box_2d_raw": meta.get("box_2d_raw", []),
                "physical_traits": meta.get("physical_traits", ""),
                "crop_width": meta.get("crop_width", 0),
                "crop_height": meta.get("crop_height", 0),
            })

        manifest["clusters"][cluster_name] = cluster_entry

    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    save_json(manifest, manifest_path)
    log.info(f"Cluster manifest saved to {manifest_path}")
    return manifest_path


def label_studio_sync_node(state: dict) -> dict:
    """
    LangGraph node: save cluster manifest.

    Reads:
        state["cluster_folders"]
        state["crop_metadata"]
        state["run_id"]

    Writes:
        state["label_studio_project_id"]  (always None — LS removed)
        state["label_studio_task_ids"]     (always [] — LS removed)
    """
    cluster_folders = state.get("cluster_folders", {})
    crop_metadata = state.get("crop_metadata", [])
    run_id = state.get("run_id", "unknown_run")

    if not cluster_folders:
        log.warning("No clusters to save manifest for")
        return {
            "label_studio_project_id": None,
            "label_studio_task_ids": [],
        }

    _save_cluster_manifest(run_id, cluster_folders, crop_metadata)

    # Save crop-to-source mapping for retraining annotation back-mapper
    crop_mapping = {}
    for meta in crop_metadata:
        crop_path_obj = Path(meta.get("crop_path", ""))
        crop_name = crop_path_obj.stem
        
        box = meta.get("box_2d_raw", [])
        if len(box) == 4:
            ymin, xmin, ymax, xmax = box[0] / 1000, box[1] / 1000, box[2] / 1000, box[3] / 1000
            w = xmax - xmin
            h = ymax - ymin
            x_center = xmin + w / 2
            y_center = ymin + h / 2
            bbox_normalized = [x_center, y_center, w, h]
        else:
            bbox_normalized = [0.5, 0.5, 1.0, 1.0]
            
        crop_mapping[crop_name] = {
            "source_image": meta.get("source_image", ""),
            "bbox_normalized": bbox_normalized,
        }
    
    save_json(crop_mapping, cfg.DATA_DIR / "crop_to_source.json")
    log.info(f"Crop-to-source mapping saved to {cfg.DATA_DIR / 'crop_to_source.json'}")

    # Save sync-pending info for reference
    sync_info = {
        "run_id": run_id,
        "cluster_folders": {str(k): v for k, v in cluster_folders.items()},
        "instruction": "Use the DYN-EYE dashboard to review and name clusters.",
    }
    save_json(sync_info, cfg.DATA_DIR / "manifest_save_pending.json")

    log.info(f"Manifest saved for {len(cluster_folders)} clusters. Ready for review in dashboard.")

    return {
        "label_studio_project_id": None,
        "label_studio_task_ids": [],
    }
