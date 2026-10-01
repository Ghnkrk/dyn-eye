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
    cohesions: dict[str, float] = None,
    global_icc: float = 0.0,
    global_silhouette: float = 0.0,
    vars_result: dict | None = None,
    cluster_fingerprints: dict[int, str] | None = None,
    inherited_labels: dict[int, str] | None = None,
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
        "global_icc": round(global_icc, 4),
        "global_silhouette": round(global_silhouette, 4),
        "vars": vars_result or {},
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
        cohesion_val = cohesions.get(folder.name) if cohesions else None
        cluster_entry = {
            "cluster_id": int(cluster_id),
            "crop_count": len(images),
            # Pre-labelled when this group matches a fingerprint a human already named
            "defect_name": (inherited_labels or {}).get(int(cluster_id)),
            "fingerprint_id": (cluster_fingerprints or {}).get(int(cluster_id)),
            "cohesion": cohesion_val,
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
    # NOTE: names are NOT copied over from the previous manifest by folder name:
    # cluster_000 is re-numbered every run and may be a different defect now.
    # Stable identity comes from the fingerprint registry (see inherited_labels).
    save_json(manifest, manifest_path)
    log.info(f"Cluster manifest saved to {manifest_path}")
    return manifest_path


def _normalize_box(box: list) -> list[float] | None:
    """
    VLM box [ymin, xmin, ymax, xmax] (0-1000) → YOLO [cx, cy, w, h] (0-1).
    Clamps to the image and returns None for degenerate boxes.
    """
    if len(box) != 4:
        return None
    ymin, xmin, ymax, xmax = (min(max(float(v), 0.0), 1000.0) / 1000.0 for v in box)
    w, h = xmax - xmin, ymax - ymin
    if w <= 0.0 or h <= 0.0:
        return None
    return [xmin + w / 2, ymin + h / 2, w, h]


def manifest_save_node(state: dict) -> dict:
    """
    LangGraph node: save cluster manifest, run ICC annotation, log VLM/run metrics.
    """
    cluster_folders   = state.get("cluster_folders", {})
    crop_metadata     = state.get("crop_metadata", [])
    run_id            = state.get("run_id", "unknown_run")
    vlm_system_prompt = state.get("vlm_system_prompt")
    vlm_annotations   = state.get("vlm_annotations", [])

    if not cluster_folders:
        log.warning("No clusters to save manifest for")
        return {"vlm_cluster_results": []}

    # ── 1. Calculate cohesion & global metrics from embeddings ──
    from src.utils import LogStream
    import numpy as np
    from sklearn.metrics import silhouette_score

    cluster_results: list[dict] = []
    cohesions: dict[str, float] = {}
    global_icc = 0.0
    global_silhouette = 0.0

    try:
        feature_vectors = state.get("feature_vectors")   # (N, 384)
        cluster_labels  = state.get("cluster_labels")    # (N,) int, -1 = noise
        novel_indices   = state.get("novel_indices", [])

        if feature_vectors is not None and cluster_labels is not None:
            feat = np.asarray(feature_vectors)
            labels = np.asarray(cluster_labels)

            if novel_indices:
                feat = feat[novel_indices]

            # L2-normalise so cosine similarity = dot product
            norms = np.linalg.norm(feat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            feat_norm = feat / norms

            valid_ids = sorted(set(int(l) for l in labels) - {-1})

            if valid_ids:
                msg_start = f"Computing statistical ICC for {len(valid_ids)} clusters using embedding cohesion"
                log.info(msg_start)
                LogStream.emit(msg_start, level="info", source="manifest_save")

                # Per-cluster cohesion score
                group_vecs: list[np.ndarray] = []
                for cid in valid_ids:
                    mask = labels == cid
                    g = feat_norm[mask]
                    group_vecs.append(g)

                    if len(g) == 0:
                        cohesion = 0.0
                    elif len(g) == 1:
                        cohesion = 1.0
                    else:
                        centroid = g.mean(axis=0)
                        centroid /= max(np.linalg.norm(centroid), 1e-9)
                        cohesion = float(np.clip(g @ centroid, 0, 1).mean())

                    cohesions[f"cluster_{cid:03d}"] = round(cohesion, 4)
                    cluster_results.append({
                        "cluster_id": cid,
                        "label": f"cluster_{cid:03d}",
                        "icc": round(cohesion, 4),
                        "confidence": round(cohesion, 4),
                        "n_samples": int(mask.sum()),
                        "labels_seen": [],
                    })
                    log.info(f"  Cluster {cid}: cohesion={cohesion:.4f}  (n={mask.sum()})")

                # Global ANOVA ICC
                all_vecs = feat_norm
                grand_mean = all_vecs.mean(axis=0)
                K = len(valid_ids)
                N = len(all_vecs)

                ss_between = sum(
                    len(g) * float(np.sum((g.mean(axis=0) - grand_mean) ** 2))
                    for g in group_vecs
                )
                ss_within = sum(
                    float(np.sum((g - g.mean(axis=0)) ** 2))
                    for g in group_vecs
                )
                df_b = K - 1
                df_w = N - K

                if df_b > 0 and df_w > 0 and ss_within > 0:
                    ms_b = ss_between / df_b
                    ms_w = ss_within / df_w
                    k0 = (N - sum(len(g) ** 2 for g in group_vecs) / N) / df_b
                    denom = ms_b + max(k0 - 1, 0) * ms_w
                    global_icc = float(np.clip((ms_b - ms_w) / denom if denom > 0 else 0.0, 0.0, 1.0))
                else:
                    global_icc = 1.0 if K == 1 else 0.0

                # Global Silhouette score
                mask_non_noise = labels != -1
                if mask_non_noise.sum() >= 4 and len(set(labels[mask_non_noise])) >= 2:
                    global_silhouette = float(silhouette_score(feat_norm[mask_non_noise], labels[mask_non_noise], metric="cosine"))
                else:
                    global_silhouette = 0.0

                avg_cohesion = sum(r["icc"] for r in cluster_results) / len(cluster_results)
                msg_end = (
                    f"ICC complete — global ICC={global_icc:.4f}, global Silhouette={global_silhouette:.4f}, "
                    f"mean per-cluster cohesion={avg_cohesion:.4f}"
                )
                log.info(msg_end)
                LogStream.emit(msg_end, level="info", source="manifest_save")

                # Attach global score to state so vlm_metrics can log it
                state["global_icc"] = global_icc
                state["global_silhouette"] = global_silhouette

    except Exception as e:
        log.warning(f"Statistical ICC failed (non-fatal): {e}")

    # ── 2. Compute VARS (VLM Annotation Reliability Score) ───
    vars_result: dict = {}
    try:
        from src.utils.vlm_score import compute_vars
        feature_vectors = state.get("feature_vectors")
        novel_indices   = state.get("novel_indices", [])

        # Use only the novel-subset feature vectors (same slice used for clustering)
        fv_for_vars = None
        if feature_vectors is not None:
            import numpy as np
            fv_arr = np.asarray(feature_vectors)
            if novel_indices:
                fv_for_vars = fv_arr[novel_indices]
            else:
                fv_for_vars = fv_arr

        vars_result = compute_vars(
            crop_metadata=crop_metadata,
            vlm_annotations=vlm_annotations,
            crop_feature_vectors=fv_for_vars,
        )

        vars_msg = (
            f"[VARS] VLM Reliability: {vars_result['vars_pct']}% — "
            f"CDS={vars_result['cds']:.3f}  BQS={vars_result['bqs']:.3f}  DRS={vars_result['drs']:.3f} — "
            f"{vars_result['interpretation']}"
        )
        log.info(vars_msg)
        LogStream.emit(vars_msg, level="info", source="manifest_save")
        state["vars_result"] = vars_result

    except Exception as e:
        log.warning(f"VARS computation failed (non-fatal): {e}")

    # ── 3. Save cluster manifest with metrics ────────────────
    _save_cluster_manifest(
        run_id=run_id,
        cluster_folders=cluster_folders,
        crop_metadata=crop_metadata,
        vlm_system_prompt=vlm_system_prompt,
        cohesions=cohesions,
        global_icc=global_icc,
        global_silhouette=global_silhouette,
        vars_result=vars_result,
        cluster_fingerprints=state.get("cluster_fingerprints"),
        inherited_labels=state.get("cluster_inherited_labels"),
    )

    # ── 4. Save crop-to-source mapping ───────────────────────
    crop_mapping: dict = {}
    skipped_boxes = 0
    for meta in crop_metadata:
        crop_name = Path(meta.get("crop_path", "")).stem
        bbox_normalized = _normalize_box(meta.get("box_2d_raw", []))
        if bbox_normalized is None:
            skipped_boxes += 1   # no valid box → cannot become a YOLO label
            continue
        crop_mapping[crop_name] = {
            "source_image": meta.get("source_image", ""),
            "bbox_normalized": bbox_normalized,
        }
    if skipped_boxes:
        log.warning(f"{skipped_boxes} crops had no valid bounding box and were left out of the label mapping")

    save_json(crop_mapping, cfg.DATA_DIR / "crop_to_source.json")
    log.info(f"Crop-to-source mapping saved to {cfg.DATA_DIR / 'crop_to_source.json'}")
    log.info(f"Manifest saved for {len(cluster_folders)} clusters. Ready for review in dashboard.")

    # ── 5. Log VLM/run metrics ───────────────────────────────
    if cluster_results:
        try:
            from src.utils.vlm_metrics import log_run_metrics
            log_run_metrics(
                cluster_results=cluster_results,
                dbcv_score=state.get("dbcv_score", -1.0),
                registry_hits=state.get("registry_hits", 0),
                registry_total=state.get("registry_total", 0),
                run_id=run_id,
                global_icc=state.get("global_icc"),
                vars_result=state.get("vars_result"),
            )
        except Exception as e:
            log.warning(f"VLM metrics logging failed (non-fatal): {e}")

    return {"vlm_cluster_results": cluster_results}
