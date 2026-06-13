"""
Node 7 — Manifest Save + ICC Metrics

Saves the cluster manifest JSON used by the DYN-EYE dashboard for
cluster review, naming, and YOLO retraining.

Also runs ICC (Intra-Cluster Consistency) multi-sample annotation
per cluster and logs VLM performance metrics.
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
    vlm_system_prompt: str | None = None,
) -> Path:
    """
    Save a manifest JSON:
      cluster_name → [{crop_file, source_image, bbox_pixels, bbox_raw, traits}]
    """
    meta_lookup: dict[str, dict] = {
        Path(m.get("crop_path", "")).name: m for m in crop_metadata
    }

    from src.pipeline.nodes.vlm_annotation import SYSTEM_PROMPT as STATIC_PROMPT
    manifest: dict = {
        "run_id": run_id,
        "vlm_system_prompt": vlm_system_prompt or STATIC_PROMPT,
        "clusters": {}
    }

    # Copy to avoid mutating original dictionary in state
    folders_to_process = dict(cluster_folders)

    # Explicitly check for and include unassigned folder so crops inside can be edited/moved in dashboard
    unassigned_path = cfg.CLUSTERS_DIR / "unassigned"
    if unassigned_path.exists():
        folders_to_process[-2] = str(unassigned_path)

    # Explicitly check for and include noise folder if not already present
    noise_path = cfg.CLUSTERS_DIR / "noise"
    if noise_path.exists() and -1 not in folders_to_process:
        folders_to_process[-1] = str(noise_path)

    for cluster_id, folder_path in folders_to_process.items():
        folder = Path(folder_path)
        if not folder.exists():
            continue

        images = list_images(folder)
        cluster_entry = {
            "cluster_id": int(cluster_id),
            "crop_count": len(images),
            "defect_name": None,
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

        manifest["clusters"][folder.name] = cluster_entry

    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    save_json(manifest, manifest_path)
    log.info(f"Cluster manifest saved to {manifest_path}")
    return manifest_path


def manifest_save_node(state: dict) -> dict:
    """
    LangGraph node: save cluster manifest, run ICC annotation, log VLM metrics.

    Reads:
        state["cluster_folders"]
        state["crop_metadata"]
        state["run_id"]
        state["vlm_system_prompt"]   (optional — dynamic prompt from dataset_context)
        state["dbcv_score"]
        state["registry_hits"]
        state["registry_total"]

    Writes:
        state["vlm_cluster_results"]
    """
    cluster_folders = state.get("cluster_folders", {})
    crop_metadata   = state.get("crop_metadata", [])
    run_id          = state.get("run_id", "unknown_run")
    vlm_system_prompt = state.get("vlm_system_prompt")

    if not cluster_folders:
        log.warning("No clusters to save manifest for")
        return {"vlm_cluster_results": []}

    # ── 1. Save cluster manifest ─────────────────────────────
    _save_cluster_manifest(run_id, cluster_folders, crop_metadata, vlm_system_prompt)

    # ── 2. Save crop-to-source mapping ───────────────────────
    crop_mapping: dict = {}
    for meta in crop_metadata:
        crop_name = Path(meta.get("crop_path", "")).stem
        box = meta.get("box_2d_raw", [])
        if len(box) == 4:
            ymin, xmin, ymax, xmax = (v / 1000 for v in box)
            w, h = xmax - xmin, ymax - ymin
            bbox_normalized = [xmin + w / 2, ymin + h / 2, w, h]
        else:
            bbox_normalized = [0.5, 0.5, 1.0, 1.0]
        crop_mapping[crop_name] = {
            "source_image": meta.get("source_image", ""),
            "bbox_normalized": bbox_normalized,
        }

    save_json(crop_mapping, cfg.DATA_DIR / "crop_to_source.json")
    log.info(f"Crop-to-source mapping saved to {cfg.DATA_DIR / 'crop_to_source.json'}")
    log.info(f"Manifest saved for {len(cluster_folders)} clusters. Ready for review in dashboard.")

    # ── 3. ICC annotation per cluster ────────────────────────
    cluster_results: list[dict] = []
    vlm_system_prompt = state.get("vlm_system_prompt")
    use_cache = state.get("use_cache", False)
    cache_path = cfg.DATA_DIR / "vlm_icc_cache.json"

    if use_cache and cache_path.exists():
        try:
            from src.utils import load_json
            cluster_results = load_json(cache_path)
            msg_cache = f"Cache mode — loaded {len(cluster_results)} ICC results from cache"
            log.info(msg_cache)
            from src.utils import LogStream
            LogStream.emit(msg_cache, level="info", source="manifest_save")
        except Exception as e:
            log.warning(f"Failed to load ICC cache: {e}. Falling back to live API.")
            cluster_results = []

    if not cluster_results:
        try:
            import config as _cfg
            from src.pipeline.nodes.vlm_annotation import (
                _annotate_cluster_for_icc,
                SYSTEM_PROMPT as _STATIC_PROMPT,
            )
            from google import genai
            from google.genai import types
            from pydantic import BaseModel

            class _Report(BaseModel):
                anomalies_found: bool
                findings: list

            client  = genai.Client()
            gen_cfg = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_Report,
                temperature=_cfg.VLM_TEMPERATURE,
            )
            active_prompt = vlm_system_prompt or _STATIC_PROMPT

            msg_start = f"Starting ICC verification on {len(cluster_folders)} clusters ({_cfg.VLM_ICC_SAMPLES} samples/cluster)"
            log.info(msg_start)
            from src.utils import LogStream
            LogStream.emit(msg_start, level="info", source="manifest_save")

            for cid_key, folder in cluster_folders.items():
                cid = int(cid_key) if isinstance(cid_key, str) else cid_key
                if cid == -1:
                    continue
                try:
                    result = _annotate_cluster_for_icc(
                        folder, client, gen_cfg, active_prompt,
                        n_samples=_cfg.VLM_ICC_SAMPLES,
                    )
                    cluster_results.append(result)
                    log.info(
                        f"  Cluster {cid}: label='{result['label']}', "
                        f"ICC={result['icc']:.2f}, confidence={result['confidence']:.2f}"
                    )
                except Exception as e:
                    log.warning(f"ICC scoring failed for cluster {cid}: {e}")
                    cluster_results.append({
                        "label": "error", "confidence": 0.0,
                        "icc": 1.0, "n_samples": 0, "labels_seen": [],
                    })

            if cluster_results:
                avg_icc = sum(r.get("icc", 1.0) for r in cluster_results) / len(cluster_results)
                msg_end = f"ICC verification complete. Average consistency: {avg_icc:.2f}"
                log.info(msg_end)
                LogStream.emit(msg_end, level="info", source="manifest_save")
                
                # Save to cache file
                try:
                    save_json(cluster_results, cache_path)
                    log.info(f"Saved {len(cluster_results)} ICC results to cache at {cache_path}")
                except Exception as e:
                    log.warning(f"Failed to save ICC cache: {e}")

        except Exception as e:
            log.warning(f"ICC annotation phase failed (non-fatal): {e}")

    # ── 4. Log VLM metrics ───────────────────────────────────
    if cluster_results:
        try:
            from src.utils.vlm_metrics import log_run_metrics
            log_run_metrics(
                cluster_results=cluster_results,
                dbcv_score=state.get("dbcv_score", -1.0),
                registry_hits=state.get("registry_hits", 0),
                registry_total=state.get("registry_total", 0),
                run_id=run_id,
            )
        except Exception as e:
            log.warning(f"VLM metrics logging failed (non-fatal): {e}")

    return {"vlm_cluster_results": cluster_results}
