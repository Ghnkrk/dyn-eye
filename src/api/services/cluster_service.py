"""
Cluster manifest service — curation logic migrated from dashboard/app.py.

The manifest JSON (data/clusters/cluster_manifest.json) stays the source
of truth; every mutating function here ends by calling db_sync.sync_clusters
and broadcasting a "clusters updated" WebSocket event.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import HTTPException

import config as cfg
from src.api.services.ws_manager import ws_manager
from src.db.session import get_session
from src.db import sync as db_sync
from src.utils import get_logger, load_json, save_json
from src.utils.io_helpers import list_images

log = get_logger("cluster_service")


def _resync_and_broadcast() -> None:
    with get_session() as session:
        db_sync.sync_clusters(session)
    ws_manager.broadcast_threadsafe("clusters", "updated", {})


def load_and_sync_manifest() -> dict:
    """
    Load the cluster manifest and dynamically synchronize it with the
    directories physically present on disk. Automatically backfills
    missing folders like 'unassigned' or 'noise' to prevent UI edit
    failures.
    """
    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "No cluster manifest found. Run the discovery pipeline first.")

    try:
        manifest = load_json(manifest_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to load cluster manifest: {e}")

    modified = False
    if "vlm_system_prompt" not in manifest:
        from src.pipeline.nodes.vlm_annotation import SYSTEM_PROMPT as STATIC_PROMPT
        manifest["vlm_system_prompt"] = STATIC_PROMPT
        modified = True

    clusters_dir = cfg.CLUSTERS_DIR
    if clusters_dir.exists():
        crop_to_source = {}
        cts_path = cfg.DATA_DIR / "crop_to_source.json"
        if cts_path.exists():
            try:
                crop_to_source = load_json(cts_path)
            except Exception:
                pass

        for folder in clusters_dir.iterdir():
            if folder.is_dir() and folder.name not in ("__pycache__",):
                folder_name = folder.name
                if folder_name not in manifest.setdefault("clusters", {}):
                    images = list_images(folder)

                    if folder_name == "unassigned":
                        cluster_id = -2
                    elif folder_name == "noise":
                        cluster_id = -1
                    else:
                        try:
                            cluster_id = int(folder_name.replace("cluster_", ""))
                        except ValueError:
                            cluster_id = -99

                    cluster_entry = {
                        "cluster_id": cluster_id,
                        "crop_count": len(images),
                        "defect_name": None,
                        "crops": [],
                    }

                    for img_path in images:
                        crop_name = img_path.stem
                        cts = crop_to_source.get(crop_name, {})
                        source_img = cts.get("source_image", "")

                        box_2d_raw = []
                        if "bbox_normalized" in cts:
                            cx, cy, w, h = cts["bbox_normalized"]
                            ymin, xmin = cy - h / 2, cx - w / 2
                            ymax, xmax = cy + h / 2, cx + w / 2
                            box_2d_raw = [int(ymin * 1000), int(xmin * 1000), int(ymax * 1000), int(xmax * 1000)]

                        cluster_entry["crops"].append({
                            "crop_file": img_path.name,
                            "crop_path": str(img_path),
                            "source_image": source_img,
                            "source_image_name": Path(source_img).name if source_img else "",
                            "box_2d_pixels": [],
                            "box_2d_raw": box_2d_raw,
                            "physical_traits": "",
                            "crop_width": 0,
                            "crop_height": 0,
                        })

                    manifest["clusters"][folder_name] = cluster_entry
                    modified = True
                    log.info(f"Backfilled missing cluster '{folder_name}' into manifest")

        if modified:
            try:
                save_json(manifest, manifest_path)
            except Exception as e:
                log.warning(f"Failed to save auto-synced manifest: {e}")

    return manifest


def build_label_mapping(manifest: dict) -> list[dict]:
    """Build flat label mapping from manifest for retraining."""
    label_mapping = []
    for cluster_name, entry in manifest.get("clusters", {}).items():
        defect_name = entry.get("defect_name")
        if defect_name:
            for crop in entry.get("crops", []):
                label_mapping.append({
                    "defect_name": defect_name,
                    "crop_file": crop["crop_file"],
                    "source_image": crop.get("source_image", ""),
                    "source_image_name": crop.get("source_image_name", ""),
                    "box_2d_pixels": crop.get("box_2d_pixels", []),
                    "box_2d_raw": crop.get("box_2d_raw", []),
                })
    return label_mapping


def _save_mapping(manifest: dict) -> list[dict]:
    label_mapping = build_label_mapping(manifest)
    mapping_path = cfg.DATA_DIR / "defect_label_mapping.json"
    save_json({"run_id": manifest.get("run_id"), "labels": label_mapping}, mapping_path)
    return label_mapping


def name_clusters(names: dict[str, str]) -> dict:
    """Assign human-readable defect names to clusters."""
    manifest = load_and_sync_manifest()

    clusters_to_remove = []
    for cname, entry in list(manifest.get("clusters", {}).items()):
        folder = cfg.CLUSTERS_DIR / cname
        imgs = list_images(folder) if folder.exists() else []
        if len(entry.get("crops", [])) == 0 or len(imgs) == 0:
            clusters_to_remove.append(cname)

    for cname in clusters_to_remove:
        manifest.get("clusters", {}).pop(cname, None)
        folder = cfg.CLUSTERS_DIR / cname
        if folder.exists():
            try:
                shutil.rmtree(str(folder))
            except Exception:
                pass
        log.info(f"Erased empty cluster: {cname}")

    updated = []
    for cluster_name, defect_name in names.items():
        if cluster_name in manifest.get("clusters", {}):
            manifest["clusters"][cluster_name]["defect_name"] = defect_name
            from src.features import cluster_registry as creg
            creg.set_label(manifest["clusters"][cluster_name].get("fingerprint_id"), defect_name)
            updated.append(cluster_name)
            log.info(f"Named cluster '{cluster_name}' -> '{defect_name}'")

    save_json(manifest, cfg.CLUSTERS_DIR / "cluster_manifest.json")
    label_mapping = _save_mapping(manifest)
    _resync_and_broadcast()

    return {
        "message": f"Named {len(updated)} clusters",
        "updated_clusters": updated,
        "total_labeled_crops": len(label_mapping),
    }


def merge_clusters(source_cluster: str, target_cluster: str, merged_label: str | None) -> dict:
    manifest = load_and_sync_manifest()
    clusters = manifest.get("clusters", {})

    src, dst = source_cluster, target_cluster
    if src not in clusters:
        raise HTTPException(404, f"Source cluster '{src}' not found in manifest")
    if dst not in clusters:
        raise HTTPException(404, f"Target cluster '{dst}' not found in manifest")
    if src == dst:
        raise HTTPException(400, "Source and target clusters must be different")

    src_entry = clusters[src]
    dst_entry = clusters[dst]

    src_crops = src_entry.get("crops", [])
    dst_crops = dst_entry.setdefault("crops", [])

    for crop in src_crops:
        cfile = crop["crop_file"]
        src_path = cfg.CLUSTERS_DIR / src / cfile
        dst_path = cfg.CLUSTERS_DIR / dst / cfile

        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))

        if "crop_path" in crop:
            crop["crop_path"] = str(dst_path)

        dst_crops.append(crop)

    dst_entry["crop_count"] = len(dst_crops)

    if merged_label:
        dst_entry["defect_name"] = merged_label
        from src.features import cluster_registry as creg
        creg.set_label(dst_entry.get("fingerprint_id"), merged_label)

    clusters.pop(src, None)
    src_folder = cfg.CLUSTERS_DIR / src
    if src_folder.exists():
        try:
            shutil.rmtree(str(src_folder))
        except Exception as e:
            log.warning(f"Could not delete source cluster folder '{src}': {e}")

    log.info(f"Merged cluster '{src}' ({len(src_crops)} crops) into '{dst}' (now {dst_entry['crop_count']} crops)")

    save_json(manifest, cfg.CLUSTERS_DIR / "cluster_manifest.json")
    label_mapping = _save_mapping(manifest)
    _resync_and_broadcast()

    return {
        "message": f"Merged '{src}' into '{dst}'",
        "target_cluster": dst,
        "merged_crop_count": dst_entry["crop_count"],
        "total_labeled_crops": len(label_mapping),
    }


def batch_edit_crops(
    crop_files: list[str],
    source_cluster: str,
    target_cluster: str | None,
    action: str,
) -> dict:
    manifest = load_and_sync_manifest()

    if source_cluster not in manifest.get("clusters", {}):
        raise HTTPException(404, f"Source cluster '{source_cluster}' not found in manifest")

    src_entry = manifest["clusters"][source_cluster]
    crops = src_entry.get("crops", [])

    target_crops_map = {c["crop_file"]: c for c in crops if c["crop_file"] in crop_files}
    remaining_crops = [c for c in crops if c["crop_file"] not in target_crops_map]

    if not target_crops_map:
        raise HTTPException(404, "No matching crops found in source cluster")

    src_entry["crops"] = remaining_crops
    src_entry["crop_count"] = len(remaining_crops)

    if action == "move":
        if not target_cluster or target_cluster not in manifest.get("clusters", {}):
            raise HTTPException(404, f"Target cluster '{target_cluster}' not found in manifest")

        dst_entry = manifest["clusters"][target_cluster]
        for cfile, target_crop in target_crops_map.items():
            dst_entry.setdefault("crops", []).append(target_crop)

            src_path = cfg.CLUSTERS_DIR / source_cluster / cfile
            dst_path = cfg.CLUSTERS_DIR / target_cluster / cfile
            if src_path.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(dst_path))
                if "crop_path" in target_crop:
                    target_crop["crop_path"] = str(dst_path)

        dst_entry["crop_count"] = len(dst_entry["crops"])
        log.info(f"Batch moved {len(target_crops_map)} crops from '{source_cluster}' to '{target_cluster}'")

    elif action == "drop":
        for cfile in target_crops_map.keys():
            src_path = cfg.CLUSTERS_DIR / source_cluster / cfile
            if src_path.exists():
                src_path.unlink()
        log.info(f"Batch dropped {len(target_crops_map)} crops from '{source_cluster}'")
    else:
        raise HTTPException(400, f"Unknown action '{action}' (expected 'move' or 'drop')")

    save_json(manifest, cfg.CLUSTERS_DIR / "cluster_manifest.json")
    label_mapping = _save_mapping(manifest)
    _resync_and_broadcast()

    return {
        "message": f"Batch action '{action}' completed on {len(target_crops_map)} crops",
        "total_labeled_crops": len(label_mapping),
    }


def edit_crop(
    crop_file: str,
    source_cluster: str,
    target_cluster: str | None,
    action: str,
) -> dict:
    manifest = load_and_sync_manifest()

    if source_cluster not in manifest.get("clusters", {}):
        raise HTTPException(404, f"Source cluster '{source_cluster}' not found in manifest")

    src_entry = manifest["clusters"][source_cluster]
    crops = src_entry.get("crops", [])

    target_crop = None
    remaining_crops = []
    for c in crops:
        if c["crop_file"] == crop_file:
            target_crop = c
        else:
            remaining_crops.append(c)

    if not target_crop:
        raise HTTPException(404, f"Crop '{crop_file}' not found in cluster '{source_cluster}'")

    src_entry["crops"] = remaining_crops
    src_entry["crop_count"] = len(remaining_crops)

    if action == "move":
        if not target_cluster or target_cluster not in manifest.get("clusters", {}):
            raise HTTPException(404, f"Target cluster '{target_cluster}' not found in manifest")

        dst_entry = manifest["clusters"][target_cluster]
        dst_entry.setdefault("crops", []).append(target_crop)
        dst_entry["crop_count"] = len(dst_entry["crops"])

        src_path = cfg.CLUSTERS_DIR / source_cluster / crop_file
        dst_path = cfg.CLUSTERS_DIR / target_cluster / crop_file
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            if "crop_path" in target_crop:
                target_crop["crop_path"] = str(dst_path)

        log.info(f"Moved crop '{crop_file}' from '{source_cluster}' to '{target_cluster}'")

    elif action == "drop":
        src_path = cfg.CLUSTERS_DIR / source_cluster / crop_file
        if src_path.exists():
            src_path.unlink()
        log.info(f"Dropped crop '{crop_file}' from '{source_cluster}'")
    else:
        raise HTTPException(400, f"Unknown action '{action}' (expected 'move' or 'drop')")

    save_json(manifest, cfg.CLUSTERS_DIR / "cluster_manifest.json")
    label_mapping = _save_mapping(manifest)
    _resync_and_broadcast()

    return {
        "message": f"Crop action '{action}' completed successfully",
        "total_labeled_crops": len(label_mapping),
    }
