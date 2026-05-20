"""
DYN-EYE Dashboard — FastAPI Backend (v2)

Fully autonomous pipeline dashboard:
  - Real-time log streaming via SSE
  - One-click pipeline trigger (then hands-off)
  - Autonomous orchestrator background daemon
  - Cluster monitoring and Label Studio integration
  - FAISS setup endpoint
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from src.utils import get_logger, save_json, load_json, LogStream
from src.utils.metrics import MetricsTracker

log = get_logger("dashboard")

app = FastAPI(
    title="DYN-EYE — Unknown Defect Discovery",
    description="Autonomous industrial defect discovery pipeline",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Global State ─────────────────────────────────────────────
_pipeline_status: dict[str, Any] = {
    "discovery": {"status": "idle", "run_id": None, "result": None},
    "retraining": {"status": "idle", "run_id": None, "result": None},
    "orchestrator": {"status": "idle"},
}
_lock = threading.Lock()


# ── Request Models ───────────────────────────────────────────

class DiscoveryRequest(BaseModel):
    use_sample_run: bool = False
    input_images_dir: str | None = None
    confidence_threshold: float | None = None
    yolo_model_path: str | None = None


class RetrainingRequest(BaseModel):
    project_id: int
    epochs: int | None = None
    imgsz: int | None = None
    batch_size: int | None = None


class FAISSSetupRequest(BaseModel):
    known_crops_dir: str | None = None
    class_subdirs: bool = True


class OrchestratorRequest(BaseModel):
    project_id: int | None = None


# ── Background Pipeline Runners ─────────────────────────────

def _run_discovery_bg(req: DiscoveryRequest):
    """Run discovery pipeline in background thread."""
    with _lock:
        _pipeline_status["discovery"]["status"] = "running"
        _pipeline_status["discovery"]["started_at"] = datetime.now().isoformat()
        _pipeline_status["discovery"]["error"] = None

    LogStream.emit(f"Discovery pipeline triggered (Sample Run: {req.use_sample_run})", level="step", source="dashboard")

    try:
        from src.pipeline.graph import run_discovery_pipeline
        input_dir = str(cfg.SAMPLE_RUN_DIR) if req.use_sample_run else req.input_images_dir
        result = run_discovery_pipeline(
            input_images_dir=input_dir,
            confidence_threshold=req.confidence_threshold,
            yolo_model_path=req.yolo_model_path,
        )
        with _lock:
            _pipeline_status["discovery"]["status"] = "complete"
            _pipeline_status["discovery"]["result"] = _sanitize(result)
            _pipeline_status["discovery"]["completed_at"] = datetime.now().isoformat()
        LogStream.emit("Discovery pipeline finished successfully", level="info", source="dashboard")
    except Exception as e:
        with _lock:
            _pipeline_status["discovery"]["status"] = "failed"
            _pipeline_status["discovery"]["error"] = str(e)
            _pipeline_status["discovery"]["completed_at"] = datetime.now().isoformat()
        LogStream.emit(f"Discovery pipeline failed: {e}", level="error", source="dashboard")
        log.error(f"Discovery pipeline failed: {e}")


def _run_retraining_bg(req: RetrainingRequest):
    """Run retraining pipeline in background thread."""
    with _lock:
        _pipeline_status["retraining"]["status"] = "running"
        _pipeline_status["retraining"]["started_at"] = datetime.now().isoformat()
        _pipeline_status["retraining"]["error"] = None

    LogStream.emit("Retraining pipeline triggered", level="step", source="dashboard")

    try:
        from src.retraining.agent import run_retraining_pipeline
        result = run_retraining_pipeline(
            project_id=req.project_id,
            epochs=req.epochs,
            imgsz=req.imgsz,
            batch_size=req.batch_size,
        )
        with _lock:
            _pipeline_status["retraining"]["status"] = "complete"
            _pipeline_status["retraining"]["result"] = _sanitize(result)
            _pipeline_status["retraining"]["completed_at"] = datetime.now().isoformat()
        LogStream.emit("Retraining pipeline finished", level="info", source="dashboard")
    except Exception as e:
        with _lock:
            _pipeline_status["retraining"]["status"] = "failed"
            _pipeline_status["retraining"]["error"] = str(e)
            _pipeline_status["retraining"]["completed_at"] = datetime.now().isoformat()
        LogStream.emit(f"Retraining pipeline failed: {e}", level="error", source="dashboard")
        log.error(f"Retraining pipeline failed: {e}")


def _sanitize(obj: Any) -> Any:
    """Make a result dict JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items() if not isinstance(k, str) or not k.startswith("_")}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj


# ── API Routes ───────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard frontend."""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>DYN-EYE Dashboard</h1><p>Static files not found.</p>")


# ── SSE Log Stream ───────────────────────────────────────────

@app.get("/api/logs/stream")
async def log_stream():
    """Server-Sent Events endpoint for real-time log streaming."""
    async def event_generator():
        last_ts = None
        while True:
            events = LogStream.since(after_ts=last_ts, limit=50)
            for evt in events:
                last_ts = evt["ts"]
                yield f"data: {json.dumps(evt)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/logs/recent")
async def recent_logs(n: int = 100):
    """Get the most recent N log events."""
    return JSONResponse(LogStream.tail(n))


# ── Pipeline Triggers ────────────────────────────────────────

@app.post("/api/discovery/trigger")
async def trigger_discovery(req: DiscoveryRequest, background_tasks: BackgroundTasks):
    """Trigger the discovery pipeline."""
    with _lock:
        if _pipeline_status["discovery"]["status"] == "running":
            raise HTTPException(400, "Discovery pipeline is already running")

    background_tasks.add_task(_run_discovery_bg, req)
    return {"message": "Discovery pipeline triggered", "status": "starting"}


@app.post("/api/retraining/trigger")
async def trigger_retraining(req: RetrainingRequest, background_tasks: BackgroundTasks):
    """Trigger the retraining pipeline."""
    with _lock:
        if _pipeline_status["retraining"]["status"] == "running":
            raise HTTPException(400, "Retraining pipeline is already running")

    background_tasks.add_task(_run_retraining_bg, req)
    return {"message": "Retraining pipeline triggered", "status": "starting"}


@app.post("/api/faiss/setup")
async def faiss_setup(req: FAISSSetupRequest):
    """Setup FAISS index from known defect crops."""
    LogStream.emit("Setting up FAISS index...", level="step", source="faiss")
    try:
        from src.features.faiss_index import FAISSIndexManager
        manager = FAISSIndexManager()
        count = manager.setup(
            known_crops_dir=req.known_crops_dir,
            class_subdirs=req.class_subdirs,
        )
        LogStream.emit(f"FAISS index built with {count} vectors", level="info", source="faiss")
        return {"message": f"FAISS index built with {count} vectors", "count": count}
    except Exception as e:
        LogStream.emit(f"FAISS setup failed: {e}", level="error", source="faiss")
        raise HTTPException(500, f"FAISS setup failed: {e}")


# ── Orchestrator Control ─────────────────────────────────────

@app.post("/api/orchestrator/start")
async def start_orchestrator(req: OrchestratorRequest):
    """Start the autonomous orchestrator daemon."""
    from src.pipeline.orchestrator import orchestrator
    if orchestrator.is_running:
        return {"message": "Orchestrator is already running"}
    orchestrator.start(project_id=req.project_id)
    with _lock:
        _pipeline_status["orchestrator"]["status"] = "running"
    return {"message": "Orchestrator started"}


@app.post("/api/orchestrator/stop")
async def stop_orchestrator():
    """Stop the autonomous orchestrator daemon."""
    from src.pipeline.orchestrator import orchestrator
    orchestrator.stop()
    with _lock:
        _pipeline_status["orchestrator"]["status"] = "idle"
    return {"message": "Orchestrator stopped"}


# ── Status & Metrics ─────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Get current pipeline status."""
    with _lock:
        return JSONResponse(_pipeline_status)


@app.get("/api/metrics/{run_id}")
async def get_metrics(run_id: str):
    """Get metrics for a specific pipeline run."""
    try:
        data = MetricsTracker.load_run(run_id)
        return JSONResponse(data)
    except FileNotFoundError:
        raise HTTPException(404, f"Run {run_id} not found")


@app.get("/api/runs")
async def list_runs():
    """List all pipeline run IDs."""
    runs = MetricsTracker.list_runs()
    results = []
    for run_id in runs[:50]:
        try:
            data = MetricsTracker.load_run(run_id)
            results.append({
                "run_id": run_id,
                "steps": list(data.get("steps", {}).keys()),
                "summary": {
                    k: v.get("status")
                    for k, v in data.get("steps", {}).items()
                },
            })
        except Exception:
            results.append({"run_id": run_id, "steps": [], "summary": {}})
    return JSONResponse(results)


@app.get("/api/config")
async def get_config():
    """Get current configuration (non-sensitive)."""
    from src.features.known_defects_registry import get_known_defect_names
    return {
        "label_studio_url": cfg.LABEL_STUDIO_URL,
        "mlflow_tracking_uri": cfg.MLFLOW_TRACKING_URI,
        "yolo_confidence_threshold": cfg.YOLO_CONFIDENCE_THRESHOLD,
        "known_defect_names": get_known_defect_names(),
        "faiss_novelty_threshold": cfg.FAISS_NOVELTY_THRESHOLD,
        "hdbscan_min_cluster_size": cfg.HDBSCAN_MIN_CLUSTER_SIZE,
        "yolo_model_exists": cfg.YOLO_MODEL_PATH.exists(),
        "faiss_index_exists": cfg.FAISS_INDEX_FILE.exists(),
        "input_images_count": len(list(cfg.INPUT_IMAGES_DIR.glob("*")))
            if cfg.INPUT_IMAGES_DIR.exists() else 0,
    }


@app.get("/api/clusters")
async def get_clusters():
    """Get current cluster information."""
    clusters_dir = cfg.CLUSTERS_DIR
    if not clusters_dir.exists():
        return {"clusters": []}

    clusters = []
    for d in sorted(clusters_dir.iterdir()):
        if d.is_dir():
            from src.utils.io_helpers import list_images
            images = list_images(d)
            clusters.append({
                "name": d.name,
                "image_count": len(images),
                "images": [img.name for img in images],
            })
    return {"clusters": clusters}


@app.get("/api/model-versions")
async def get_model_versions():
    """List all model versions."""
    versions_dir = cfg.MODEL_VERSIONS_DIR
    if not versions_dir.exists():
        return {"versions": []}

    versions = []
    for f in sorted(versions_dir.iterdir(), reverse=True):
        if f.suffix == ".pt":
            versions.append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
            })
    return {"versions": versions}


# ── Image Serving (for Label Studio) ─────────────────────────

@app.get("/api/crops/{filename}")
async def serve_crop_image(filename: str):
    """Serve a crop image from the crops directory."""
    image_path = cfg.CROPS_DIR / filename
    if not image_path.exists():
        raise HTTPException(404, f"Crop image not found: {filename}")
    return FileResponse(
        str(image_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ── Cluster Manifest & Naming ────────────────────────────────
# NOTE: These specific routes MUST be defined BEFORE the
# parameterized /api/clusters/{cluster_name}/{filename} route

@app.get("/api/clusters/manifest")
async def get_cluster_manifest():
    """Get the cluster manifest with full traceability info."""
    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "No cluster manifest found. Run the discovery pipeline first.")
    return JSONResponse(load_json(manifest_path))


class ClusterNamingRequest(BaseModel):
    """Request to assign defect names to clusters."""
    names: dict[str, str]  # cluster_name → defect_name


@app.post("/api/clusters/name")
async def name_clusters(req: ClusterNamingRequest):
    """
    Assign human-readable defect names to clusters.
    Updates the cluster manifest and returns the mapping
    from crop files to source images with bbox coordinates.
    """
    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "No cluster manifest found")

    manifest = load_json(manifest_path)

    updated = []
    for cluster_name, defect_name in req.names.items():
        if cluster_name in manifest.get("clusters", {}):
            manifest["clusters"][cluster_name]["defect_name"] = defect_name
            updated.append(cluster_name)
            log.info(f"Named cluster '{cluster_name}' → '{defect_name}'")

    save_json(manifest, manifest_path)

    # Build a flat mapping for downstream use
    label_mapping = []
    for cluster_name, entry in manifest.get("clusters", {}).items():
        defect_name = entry.get("defect_name")
        if defect_name:
            for crop in entry.get("crops", []):
                label_mapping.append({
                    "defect_name": defect_name,
                    "crop_file": crop["crop_file"],
                    "source_image": crop["source_image"],
                    "source_image_name": crop["source_image_name"],
                    "box_2d_pixels": crop["box_2d_pixels"],
                    "box_2d_raw": crop["box_2d_raw"],
                })

    # Save the flat mapping for retraining
    mapping_path = cfg.DATA_DIR / "defect_label_mapping.json"
    save_json({"run_id": manifest.get("run_id"), "labels": label_mapping}, mapping_path)
    log.info(f"Saved defect label mapping ({len(label_mapping)} entries) to {mapping_path}")

    return {
        "message": f"Named {len(updated)} clusters",
        "updated_clusters": updated,
        "total_labeled_crops": len(label_mapping),
        "mapping_path": str(mapping_path),
    }


class BatchEditCropsRequest(BaseModel):
    """Request to perform a batch action on multiple crops in a cluster."""
    crop_files: list[str]
    source_cluster: str
    target_cluster: str | None = None
    action: str  # "move" or "drop"


@app.post("/api/clusters/batch-edit-crops")
async def batch_edit_crops(req: BatchEditCropsRequest):
    """
    Perform a batch operation (move or drop) on multiple crops from a cluster.
    """
    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "No cluster manifest found")

    manifest = load_json(manifest_path)

    src_cluster = req.source_cluster
    if src_cluster not in manifest.get("clusters", {}):
        raise HTTPException(404, f"Source cluster '{src_cluster}' not found in manifest")

    src_entry = manifest["clusters"][src_cluster]
    crops = src_entry.get("crops", [])

    # Filter target crops
    target_crops_map = {c["crop_file"]: c for c in crops if c["crop_file"] in req.crop_files}
    remaining_crops = [c for c in crops if c["crop_file"] not in target_crops_map]

    if not target_crops_map:
        raise HTTPException(404, "No matching crops found in source cluster")

    # Update source cluster
    src_entry["crops"] = remaining_crops
    src_entry["crop_count"] = len(remaining_crops)

    # Perform action
    if req.action == "move":
        dst_cluster = req.target_cluster
        if not dst_cluster or dst_cluster not in manifest.get("clusters", {}):
            raise HTTPException(404, f"Target cluster '{dst_cluster}' not found in manifest")

        dst_entry = manifest["clusters"][dst_cluster]
        for cfile, target_crop in target_crops_map.items():
            dst_entry.setdefault("crops", []).append(target_crop)

            # Physically move the crop file in the filesystem
            src_path = cfg.CLUSTERS_DIR / src_cluster / cfile
            dst_path = cfg.CLUSTERS_DIR / dst_cluster / cfile
            if src_path.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.move(str(src_path), str(dst_path))
                if "crop_path" in target_crop:
                    target_crop["crop_path"] = str(dst_path)

        dst_entry["crop_count"] = len(dst_entry["crops"])
        log.info(f"Batch moved {len(target_crops_map)} crops from '{src_cluster}' to '{dst_cluster}'")

    elif req.action == "drop":
        # Physically delete target crop files
        for cfile in target_crops_map.keys():
            src_path = cfg.CLUSTERS_DIR / src_cluster / cfile
            if src_path.exists():
                src_path.unlink()
        log.info(f"Batch dropped {len(target_crops_map)} crops from '{src_cluster}'")

    save_json(manifest, manifest_path)

    # Build flat mapping for downstream use
    label_mapping = []
    for cluster_name, entry in manifest.get("clusters", {}).items():
        defect_name = entry.get("defect_name")
        if defect_name:
            for crop in entry.get("crops", []):
                label_mapping.append({
                    "defect_name": defect_name,
                    "crop_file": crop["crop_file"],
                    "source_image": crop["source_image"],
                    "source_image_name": crop["source_image_name"],
                    "box_2d_pixels": crop["box_2d_pixels"],
                    "box_2d_raw": crop["box_2d_raw"],
                })

    mapping_path = cfg.DATA_DIR / "defect_label_mapping.json"
    save_json({"run_id": manifest.get("run_id"), "labels": label_mapping}, mapping_path)

    return {
        "message": f"Batch action '{req.action}' completed on {len(target_crops_map)} crops",
        "total_labeled_crops": len(label_mapping),
        "mapping_path": str(mapping_path)
    }


class EditCropRequest(BaseModel):
    """Request to reassign or drop a crop image in a cluster."""
    crop_file: str
    source_cluster: str
    target_cluster: str | None = None
    action: str  # "move" or "drop"


@app.post("/api/clusters/edit-crop")
async def edit_crop(req: EditCropRequest):
    """
    Move a crop to another cluster or drop/delete it from the cluster.
    Updates the manifest and flat defect label mapping.
    """
    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "No cluster manifest found")

    manifest = load_json(manifest_path)
    
    # 1. Find the crop in source cluster
    src_cluster = req.source_cluster
    if src_cluster not in manifest.get("clusters", {}):
        raise HTTPException(404, f"Source cluster '{src_cluster}' not found in manifest")

    src_entry = manifest["clusters"][src_cluster]
    crops = src_entry.get("crops", [])
    
    target_crop = None
    remaining_crops = []
    for c in crops:
        if c["crop_file"] == req.crop_file:
            target_crop = c
        else:
            remaining_crops.append(c)

    if not target_crop:
        raise HTTPException(404, f"Crop '{req.crop_file}' not found in cluster '{src_cluster}'")

    # Update source cluster
    src_entry["crops"] = remaining_crops
    src_entry["crop_count"] = len(remaining_crops)

    # 2. Perform action
    if req.action == "move":
        dst_cluster = req.target_cluster
        if not dst_cluster or dst_cluster not in manifest.get("clusters", {}):
            raise HTTPException(404, f"Target cluster '{dst_cluster}' not found in manifest")
        
        dst_entry = manifest["clusters"][dst_cluster]
        dst_entry.setdefault("crops", []).append(target_crop)
        dst_entry["crop_count"] = len(dst_entry["crops"])
        
        # Physically move the crop file in the filesystem so everything is synced
        src_path = cfg.CLUSTERS_DIR / src_cluster / req.crop_file
        dst_path = cfg.CLUSTERS_DIR / dst_cluster / req.crop_file
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.move(str(src_path), str(dst_path))
            # update local paths inside crop entry if they are stored
            if "crop_path" in target_crop:
                target_crop["crop_path"] = str(dst_path)
            
        log.info(f"Moved crop '{req.crop_file}' from '{src_cluster}' to '{dst_cluster}'")

    elif req.action == "drop":
        # Physically delete the crop file from clusters directory
        src_path = cfg.CLUSTERS_DIR / src_cluster / req.crop_file
        if src_path.exists():
            src_path.unlink()
        log.info(f"Dropped crop '{req.crop_file}' from '{src_cluster}'")

    save_json(manifest, manifest_path)

    # 3. Build a flat mapping for downstream use
    label_mapping = []
    for cluster_name, entry in manifest.get("clusters", {}).items():
        defect_name = entry.get("defect_name")
        if defect_name:
            for crop in entry.get("crops", []):
                label_mapping.append({
                    "defect_name": defect_name,
                    "crop_file": crop["crop_file"],
                    "source_image": crop["source_image"],
                    "source_image_name": crop["source_image_name"],
                    "box_2d_pixels": crop["box_2d_pixels"],
                    "box_2d_raw": crop["box_2d_raw"],
                })

    # Save the flat mapping for retraining
    mapping_path = cfg.DATA_DIR / "defect_label_mapping.json"
    save_json({"run_id": manifest.get("run_id"), "labels": label_mapping}, mapping_path)

    return {
        "message": f"Crop action '{req.action}' completed successfully",
        "total_labeled_crops": len(label_mapping),
        "mapping_path": str(mapping_path)
    }


@app.post("/api/clusters/pull-from-ls")
async def pull_from_label_studio():
    """
    Pull annotations from Label Studio and update the cluster manifest.

    Reads all completed annotations from the latest LS project and:
    1. Applies defect name assignments
    2. Applies cluster reassignments
    3. Removes crops flagged as 'drop'
    4. Regenerates defect_label_mapping.json for retraining
    """
    import requests as req

    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "No cluster manifest found. Run the pipeline first.")

    manifest = load_json(manifest_path)
    project_id = manifest.get("project_id")
    if not project_id:
        raise HTTPException(400, "No Label Studio project ID in manifest")

    api_key = cfg.LABEL_STUDIO_API_KEY
    if not api_key:
        raise HTTPException(400, "LABEL_STUDIO_API_KEY not configured")

    headers = {"Authorization": f"Token {api_key}"}

    # Fetch all tasks with annotations
    try:
        resp = req.get(
            f"{cfg.LABEL_STUDIO_URL}/api/projects/{project_id}/tasks",
            headers=headers,
            params={"page_size": 500},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        tasks = data if isinstance(data, list) else data.get("tasks", [])
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch tasks from Label Studio: {e}")

    # Process annotations
    reassignments = 0
    drops = 0
    names_found = {}

    for task in tasks:
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        # Use the latest annotation
        anno = annotations[-1]
        results = anno.get("result", [])
        task_data = task.get("data", {})
        crop_file = task_data.get("source_file", "")
        original_cluster = task_data.get("cluster_name", "")

        defect_name = None
        new_cluster = None
        quality = "keep"

        for r in results:
            r_type = r.get("type")
            r_from = r.get("from_name")
            value = r.get("value", {})

            if r_from == "defect_name" and r_type == "textarea":
                texts = value.get("text", [])
                if texts:
                    defect_name = texts[0].strip()

            elif r_from == "cluster_assignment" and r_type == "choices":
                choices = value.get("choices", [])
                if choices and choices[0] != original_cluster:
                    new_cluster = choices[0]

            elif r_from == "quality" and r_type == "choices":
                choices = value.get("choices", [])
                if choices:
                    quality = choices[0]

        # Apply defect name to the original cluster
        if defect_name and original_cluster:
            names_found[original_cluster] = defect_name

        # Handle drop
        if quality == "drop" and original_cluster in manifest.get("clusters", {}):
            cluster_entry = manifest["clusters"][original_cluster]
            cluster_entry["crops"] = [
                c for c in cluster_entry.get("crops", [])
                if c["crop_file"] != crop_file
            ]
            cluster_entry["crop_count"] = len(cluster_entry["crops"])
            drops += 1

        # Handle reassignment
        if new_cluster and quality != "drop":
            src = manifest["clusters"].get(original_cluster, {})
            dst = manifest["clusters"].get(new_cluster)

            if src and dst:
                crop_entry = None
                new_crops = []
                for c in src.get("crops", []):
                    if c["crop_file"] == crop_file:
                        crop_entry = c
                    else:
                        new_crops.append(c)

                if crop_entry:
                    src["crops"] = new_crops
                    src["crop_count"] = len(new_crops)
                    dst["crops"].append(crop_entry)
                    dst["crop_count"] = len(dst["crops"])
                    reassignments += 1

    # Apply collected names
    for cluster_name, defect_name in names_found.items():
        if cluster_name in manifest.get("clusters", {}):
            manifest["clusters"][cluster_name]["defect_name"] = defect_name
            log.info(f"LS → Named cluster '{cluster_name}' as '{defect_name}'")

    # Save updated manifest
    save_json(manifest, manifest_path)

    # Regenerate the flat label mapping
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

    mapping_path = cfg.DATA_DIR / "defect_label_mapping.json"
    save_json({"run_id": manifest.get("run_id"), "labels": label_mapping}, mapping_path)

    log.info(
        f"Pulled from LS: {len(names_found)} names, "
        f"{reassignments} reassignments, {drops} drops, "
        f"{len(label_mapping)} labeled crops"
    )

    return {
        "message": "Pulled annotations from Label Studio",
        "defect_names": names_found,
        "reassignments": reassignments,
        "drops": drops,
        "total_labeled_crops": len(label_mapping),
    }

# ── Parameterized Cluster Image Serving ──────────────────────
# This MUST come after all specific /api/clusters/* routes

@app.get("/api/clusters/{cluster_name}/{filename}")
async def serve_cluster_image(cluster_name: str, filename: str):
    """
    Serve a crop image from a cluster folder.
    Label Studio tasks reference these URLs to display images.
    """
    image_path = cfg.CLUSTERS_DIR / cluster_name / filename
    if not image_path.exists():
        raise HTTPException(404, f"Image not found: {cluster_name}/{filename}")
    return FileResponse(
        str(image_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ── Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dashboard.app:app",
        host=cfg.DASHBOARD_HOST,
        port=cfg.DASHBOARD_PORT,
        reload=True,
    )
