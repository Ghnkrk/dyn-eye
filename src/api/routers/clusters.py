"""Cluster listing, manifest, naming, merging, and crop editing."""
from __future__ import annotations

from fastapi import APIRouter

import config as cfg
from src.api.schemas.clusters import (
    BatchEditCropsRequest,
    ClusterNamingRequest,
    EditCropRequest,
    MergeClustersRequest,
)
from src.api.services import cluster_service, pipeline_service
from src.utils.io_helpers import list_images

router = APIRouter(prefix="/api/v2/clusters", tags=["clusters"])


@router.get("")
async def get_clusters():
    """
    Current cluster information. Returns empty while pipeline is running
    to prevent stale data display.
    """
    status = pipeline_service.get_status()
    if not pipeline_service.clusters_ready() or status["discovery"]["status"] != "complete":
        return {"clusters": [], "pipeline_running": status["discovery"]["status"] == "running"}

    try:
        manifest = cluster_service.load_and_sync_manifest()
    except Exception:
        manifest = {}

    clusters_dir = cfg.CLUSTERS_DIR
    if not clusters_dir.exists():
        return {"clusters": [], "global_icc": 0.0, "global_silhouette": 0.0, "mean_cohesion": 0.0}

    clusters = []
    manifest_clusters = manifest.get("clusters", {})
    cohesions_list = []

    for d in sorted(clusters_dir.iterdir()):
        if d.is_dir() and d.name not in ("__pycache__",):
            images = list_images(d)
            m_entry = manifest_clusters.get(d.name, {})
            defect_name = m_entry.get("defect_name")
            cohesion_val = m_entry.get("cohesion")

            if cohesion_val is not None:
                cohesions_list.append(cohesion_val)

            clusters.append({
                "name": d.name,
                "image_count": len(images),
                "images": [img.name for img in images],
                "defect_name": defect_name,
                "cohesion": cohesion_val,
            })

    mean_cohesion = sum(cohesions_list) / len(cohesions_list) if cohesions_list else 0.0

    return {
        "clusters": clusters,
        "global_icc": manifest.get("global_icc", 0.0),
        "global_silhouette": manifest.get("global_silhouette", 0.0),
        "mean_cohesion": round(mean_cohesion, 4),
    }


@router.get("/manifest")
async def get_cluster_manifest():
    return cluster_service.load_and_sync_manifest()


@router.post("/name")
async def name_clusters(req: ClusterNamingRequest):
    return cluster_service.name_clusters(req.names)


@router.post("/merge")
async def merge_clusters(req: MergeClustersRequest):
    return cluster_service.merge_clusters(req.source_cluster, req.target_cluster, req.merged_label)


@router.post("/batch-edit-crops")
async def batch_edit_crops(req: BatchEditCropsRequest):
    return cluster_service.batch_edit_crops(
        req.crop_files, req.source_cluster, req.target_cluster, req.action
    )


@router.post("/edit-crop")
async def edit_crop(req: EditCropRequest):
    return cluster_service.edit_crop(
        req.crop_file, req.source_cluster, req.target_cluster, req.action
    )
