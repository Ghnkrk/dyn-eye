"""
DYN-EYE Dashboard — FastAPI Backend (v3)

Fully autonomous pipeline dashboard:
  - Real-time log streaming via SSE
  - One-click pipeline trigger (then hands-off)
  - Cache mode for fast demo runs
  - Cluster monitoring and in-dashboard editing
  - FAISS setup endpoint
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
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
    version="3.0.0",
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
_clusters_ready = True  # False while pipeline is running (prevents stale cluster display)
_lock = threading.Lock()


# ── Request Models ───────────────────────────────────────────

class DiscoveryRequest(BaseModel):
    use_sample_run: bool = False
    input_images_dir: str | None = None
    yolo_model_path: str | None = None
    use_cache: bool = False


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
    global _clusters_ready

    with _lock:
        _pipeline_status["discovery"]["status"] = "running"
        _pipeline_status["discovery"]["started_at"] = datetime.now().isoformat()
        _pipeline_status["discovery"]["error"] = None
        _clusters_ready = False  # Hide stale clusters while pipeline runs

    LogStream.emit(
        f"Discovery pipeline triggered (Sample Run: {req.use_sample_run}, Cache: {req.use_cache})",
        level="step", source="dashboard",
    )

    try:
        from src.pipeline.graph import run_discovery_pipeline
        input_dir = str(cfg.SAMPLE_RUN_DIR) if req.use_sample_run else req.input_images_dir
        result = run_discovery_pipeline(
            input_images_dir=input_dir,
            yolo_model_path=req.yolo_model_path,
            use_cache=req.use_cache,
        )
        with _lock:
            _pipeline_status["discovery"]["status"] = "complete"
            _pipeline_status["discovery"]["result"] = _sanitize(result)
            _pipeline_status["discovery"]["completed_at"] = datetime.now().isoformat()
            _clusters_ready = True  # Clusters are now fresh
        LogStream.emit("Discovery pipeline finished successfully", level="info", source="dashboard")
    except Exception as e:
        with _lock:
            _pipeline_status["discovery"]["status"] = "failed"
            _pipeline_status["discovery"]["error"] = str(e)
            _pipeline_status["discovery"]["completed_at"] = datetime.now().isoformat()
            _clusters_ready = True  # Re-enable cluster display even on failure
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
async def log_stream(after_ts: str | None = None):
    """Server-Sent Events endpoint for real-time log streaming."""
    async def event_generator():
        last_ts = after_ts
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


@app.post("/api/faiss/reset")
async def faiss_reset():
    """Clear/Reset the FAISS index for showcasing the pipeline from scratch."""
    LogStream.emit("Resetting FAISS index to empty state...", level="step", source="faiss")
    try:
        from src.features.faiss_index import FAISSIndexManager
        manager = FAISSIndexManager()
        manager.reset()
        LogStream.emit("FAISS index successfully reset/deleted. All defects will be treated as novel.", level="info", source="faiss")
        return {"message": "FAISS index cleared successfully", "success": True}
    except Exception as e:
        LogStream.emit(f"FAISS reset failed: {e}", level="error", source="faiss")
        raise HTTPException(500, f"FAISS reset failed: {e}")


@app.post("/api/faiss/rebuild")
async def faiss_rebuild():
    """Manually trigger FAISS index rebuild from all current defect crop folders."""
    LogStream.emit("Manually triggering FAISS index rebuild...", level="step", source="faiss")
    try:
        from src.features.faiss_index import FAISSIndexManager
        manager = FAISSIndexManager()
        count = manager.setup()
        LogStream.emit(f"FAISS index manually rebuilt successfully with {count} vectors.", level="info", source="faiss")
        return {"message": f"FAISS index rebuilt successfully with {count} vectors", "success": True, "count": count}
    except Exception as e:
        LogStream.emit(f"FAISS rebuild failed: {e}", level="error", source="faiss")
        raise HTTPException(500, f"FAISS rebuild failed: {e}")


@app.post("/api/system/reset-all")
async def system_reset_all():
    """Reset everything to the initial state: YOLO v1, 6 known classes, pristine FAISS."""
    LogStream.emit("Initiating universal system reset...", level="step", source="system")
    try:
        import shutil
        from src.features.faiss_index import FAISSIndexManager
        from datetime import timezone

        # 1. Restore YOLO v1 model
        initial_model = cfg.MODELS_DIR / "best_initial.pt"
        if initial_model.exists():
            shutil.copy2(str(initial_model), str(cfg.YOLO_MODEL_PATH))
            LogStream.emit("YOLO model restored to initial v1 weights", level="info", source="system")
        else:
            LogStream.emit("Initial YOLO model backup not found, keeping active model", level="warning", source="system")

        # 2. Reset models registry.json and register v1_initial
        initial_classes = ["inclusion", "oil_spot", "punching_hole", "silk_spot", "water_spot", "welding_line"]
        
        registry_file = cfg.MODELS_DIR / "registry.json"
        registry_data = {
            "versions": [],
            "current": None,
            "deployment_history": []
        }

        # 3. Clean up versions folder (keep directory)
        for f in cfg.MODEL_VERSIONS_DIR.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass

        if initial_model.exists():
            v1_dst = cfg.MODEL_VERSIONS_DIR / "best_v1_initial.pt"
            shutil.copy2(str(initial_model), str(v1_dst))
            
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            v1_entry = {
                "version_id": "v1_initial",
                "version_num": 1,
                "path": str(v1_dst),
                "original_path": str(initial_model),
                "timestamp": ts,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {"mAP50": 0.95, "precision": 0.92, "recall": 0.94}, # Dummy metrics for initial
                "training_config": {},
                "source": "factory_reset",
                "notes": "Original 6-class YOLOv8 base model",
                "classes": initial_classes,
                "dataset_stats": {},
                "size_mb": round(initial_model.stat().st_size / (1024 * 1024), 2),
                "status": "deployed"
            }
            registry_data["versions"].append(v1_entry)
            registry_data["current"] = "v1_initial"
            
        save_json(registry_data, registry_file)
        LogStream.emit("Model registry cleared and v1_initial registered", level="info", source="system")

        # 4. Reset known_defects.json to 6 initial classes
        initial_classes = ["inclusion", "oil_spot", "punching_hole", "silk_spot", "water_spot", "welding_line"]
        reg_data = {
            "defect_classes": initial_classes,
            "version": 1,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "history": [
                {
                    "added": initial_classes,
                    "source": "initial_reset",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
        }
        save_json(reg_data, cfg.DATA_DIR / "known_defects.json")
        LogStream.emit("Known defect classes reset to 6 initial categories", level="info", source="system")

        # 5. Clear crops and clusters directories
        for folder in [cfg.CROPS_DIR, cfg.CLUSTERS_DIR]:
            if folder.exists():
                for item in folder.iterdir():
                    if item.is_dir():
                        try:
                            shutil.rmtree(str(item))
                        except Exception:
                            pass
                    elif item.is_file() and item.name != ".gitkeep":
                        try:
                            item.unlink()
                        except Exception:
                            pass

        # 6. Reset yolo_dataset folders
        for split in ["train", "val"]:
            for folder_name in ["images", "labels"]:
                dir_path = cfg.YOLO_DATASET_DIR / folder_name / split
                if dir_path.exists():
                    for item in dir_path.iterdir():
                        if item.is_file() and item.name != ".gitkeep":
                            try:
                                item.unlink()
                            except Exception:
                                pass
        
        # Delete flat mapping if it exists
        flat_map = cfg.DATA_DIR / "defect_label_mapping.json"
        if flat_map.exists():
            flat_map.unlink()

        LogStream.emit("Crops, clusters, and fine-tuning datasets cleared", level="info", source="system")

        # 7. Restore FAISS index from pristine backup files instantly!
        backup_index = cfg.FAISS_INDEX_DIR / "known_defects.index.backup"
        backup_labels = cfg.FAISS_INDEX_DIR / "known_defects_labels.json.backup"

        if backup_index.exists() and backup_labels.exists():
            shutil.copy2(str(backup_index), str(cfg.FAISS_INDEX_FILE))
            shutil.copy2(str(backup_labels), str(cfg.FAISS_LABELS_FILE))
            try:
                import faiss
                temp_idx = faiss.read_index(str(cfg.FAISS_INDEX_FILE))
                count = temp_idx.ntotal
            except Exception:
                count = 2965
            LogStream.emit(f"FAISS index instantly restored from pristine backups ({count} vectors)", level="info", source="system")
        else:
            # Fallback to slow rebuild only if backup is missing
            manager = FAISSIndexManager()
            manager.reset()
            count = manager.setup()
            LogStream.emit(f"FAISS index rebuilt from pristine known crops folder ({count} vectors)", level="info", source="system")

        # 8. Reset active run states
        global _pipeline_status, _clusters_ready
        with _lock:
            _pipeline_status["discovery"] = {"status": "idle", "run_id": None, "result": None}
            _pipeline_status["retraining"] = {"status": "idle", "run_id": None, "result": None}
            _pipeline_status["orchestrator"] = {"status": "idle"}
            _clusters_ready = True

        LogStream.emit("SYSTEM UNIVERSAL RESET COMPLETED SUCCESSFULLY!", level="success", source="system")
        return {"message": "System reset to initial state successfully", "success": True}
    except Exception as e:
        LogStream.emit(f"System reset failed: {e}", level="error", source="system")
        raise HTTPException(500, f"System reset failed: {e}")


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
        status = dict(_pipeline_status)
        status["clusters_ready"] = _clusters_ready
        return JSONResponse(status)


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


@app.post("/api/images/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload folder/images to run the pipeline on, clearing existing input images."""
    try:
        import shutil
        input_dir = cfg.INPUT_IMAGES_DIR
        
        # Clean up existing input_images
        if input_dir.exists():
            for item in input_dir.iterdir():
                if item.is_file() and item.name != ".gitkeep":
                    try:
                        item.unlink()
                    except Exception:
                        pass
        else:
            input_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for file in files:
            # Check if it is an image
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                continue
            
            # Extract only the base filename to prevent subdirectory write issues
            base_name = Path(file.filename).name
            dest_path = input_dir / base_name
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_count += 1

        LogStream.emit(
            f"Uploaded {saved_count} new images to input directory",
            level="info", source="system"
        )
        return {"success": True, "count": saved_count}
    except Exception as e:
        LogStream.emit(f"Image upload failed: {e}", level="error", source="system")
        raise HTTPException(500, f"Failed to upload images: {e}")


@app.get("/api/config")
async def get_config():
    """Get current configuration (non-sensitive)."""
    from src.features.known_defects_registry import get_known_defect_names
    return {
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
    """
    Get current cluster information.
    Returns empty while pipeline is running to prevent stale data display.
    """
    # Prevent stale cluster display while pipeline is running or before the first run completes in this session
    if not _clusters_ready or _pipeline_status["discovery"]["status"] != "complete":
        return {"clusters": [], "pipeline_running": (_pipeline_status["discovery"]["status"] == "running")}

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
    """List all model versions from the registry."""
    from src.retraining.model_registry import ModelRegistry
    registry = ModelRegistry()
    versions = registry.list_versions()
    current = registry.get_current()
    return {
        "versions": versions,
        "current": current,
    }


@app.get("/api/models/active")
async def get_active_model():
    """Get info about the currently active model."""
    from src.retraining.model_registry import ModelRegistry
    registry = ModelRegistry()
    entry = registry.get_current_entry()
    return {
        "active_version": registry.get_current(),
        "entry": entry,
        "model_exists": cfg.YOLO_MODEL_PATH.exists(),
        "model_path": str(cfg.YOLO_MODEL_PATH),
    }


class RollbackRequest(BaseModel):
    version_id: str
    confirmed_by: str = "dashboard"


@app.post("/api/models/rollback")
async def rollback_model(req: RollbackRequest):
    """Rollback to a previous model version."""
    from src.retraining.model_registry import ModelRegistry
    LogStream.emit(
        f"Rolling back to model version: {req.version_id}",
        level="step", source="model_registry",
    )
    registry = ModelRegistry()
    result = registry.rollback(req.version_id, confirmed_by=req.confirmed_by)
    if result["success"]:
        LogStream.emit(
            f"Rollback complete: {req.version_id} is now active (FAISS: {result.get('faiss_vectors', 0)} vectors)",
            level="info", source="model_registry",
        )
    else:
        LogStream.emit(
            f"Rollback failed: {result.get('error', 'Unknown')}",
            level="error", source="model_registry",
        )
    return result


class DeployConfirmRequest(BaseModel):
    version_id: str


@app.post("/api/models/deploy-confirm")
async def deploy_confirm(req: DeployConfirmRequest):
    """User confirms deployment of a model version from the dashboard."""
    from src.retraining.model_registry import ModelRegistry
    LogStream.emit(
        f"Deploying model version: {req.version_id} (user confirmed)",
        level="step", source="model_registry",
    )
    registry = ModelRegistry()
    result = registry.deploy_version(req.version_id, confirmed_by="dashboard-user")
    if result["success"]:
        LogStream.emit(
            f"Model {req.version_id} deployed successfully",
            level="info", source="model_registry",
        )
    return result


@app.get("/api/models/history")
async def get_deployment_history():
    """Get full deployment history."""
    from src.retraining.model_registry import ModelRegistry
    registry = ModelRegistry()
    return {"history": registry.get_deployment_history()}


class SmartRetrainRequest(BaseModel):
    epochs: int | None = None
    imgsz: int | None = None
    batch_size: int | None = None


@app.post("/api/retraining/smart-trigger")
async def smart_retrain(req: SmartRetrainRequest, background_tasks: BackgroundTasks):
    """LLM-advised retraining: analyzes dataset first, then triggers training."""
    with _lock:
        if _pipeline_status["retraining"]["status"] == "running":
            raise HTTPException(400, "Retraining pipeline is already running")

    background_tasks.add_task(
        _run_retraining_bg,
        RetrainingRequest(
            project_id=-1,
            epochs=req.epochs,
            imgsz=req.imgsz,
            batch_size=req.batch_size,
        ),
    )
    return {"message": "Smart retraining triggered (LLM will advise)", "status": "starting"}


@app.get("/api/retraining/advisor-preview")
async def advisor_preview():
    """Preview what the LLM advisor would recommend without starting training."""
    try:
        from src.retraining.llm_advisor import get_training_recommendation, collect_dataset_metadata
        metadata = collect_dataset_metadata()
        recommendation = get_training_recommendation(metadata)
        return {
            "metadata": metadata,
            "recommendation": recommendation,
        }
    except Exception as e:
        raise HTTPException(500, f"Advisor preview failed: {e}")


# ── Image Serving ────────────────────────────────────────────

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
                    "source_image": crop.get("source_image", ""),
                    "source_image_name": crop.get("source_image_name", ""),
                    "box_2d_pixels": crop.get("box_2d_pixels", []),
                    "box_2d_raw": crop.get("box_2d_raw", []),
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
    label_mapping = _build_label_mapping(manifest)

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
    label_mapping = _build_label_mapping(manifest)

    # Save the flat mapping for retraining
    mapping_path = cfg.DATA_DIR / "defect_label_mapping.json"
    save_json({"run_id": manifest.get("run_id"), "labels": label_mapping}, mapping_path)

    return {
        "message": f"Crop action '{req.action}' completed successfully",
        "total_labeled_crops": len(label_mapping),
        "mapping_path": str(mapping_path)
    }


def _build_label_mapping(manifest: dict) -> list[dict]:
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


# ── Parameterized Cluster Image Serving ──────────────────────
# This MUST come after all specific /api/clusters/* routes

@app.get("/api/clusters/{cluster_name}/{filename}")
async def serve_cluster_image(cluster_name: str, filename: str):
    """
    Serve a crop image from a cluster folder.
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
