"""VLM Annotation Reliability Score (VARS) readout."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

import config as cfg
from src.utils import load_json

router = APIRouter(prefix="/api/v2/vlm", tags=["vlm"])


@router.get("/score")
async def get_vlm_score():
    """
    Latest VARS from the cluster manifest: vars_score, CDS, BQS, DRS.
    """
    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    if not manifest_path.exists():
        return {"available": False, "reason": "No manifest found — run discovery pipeline first."}
    try:
        manifest = load_json(manifest_path)
        vars_data = manifest.get("vars", {})
        if not vars_data:
            return {"available": False, "reason": "VARS not yet computed in this run."}
        return {"available": True, **vars_data}
    except Exception as e:
        raise HTTPException(500, f"Failed to read VLM score: {e}")
