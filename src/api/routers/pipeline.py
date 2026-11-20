"""Discovery/retraining triggers, orchestrator control, run history."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlmodel import select

from src.api.schemas.pipeline import (
    DiscoveryRequest,
    OrchestratorRequest,
    RetrainingRequest,
    SmartRetrainRequest,
)
from src.api.services import pipeline_service
from src.db.models import PipelineRun, PipelineStepMetric
from src.db.session import get_session
from src.utils.metrics import MetricsTracker

router = APIRouter(prefix="/api/v2/pipeline", tags=["pipeline"])


@router.post("/discovery/trigger")
async def trigger_discovery(req: DiscoveryRequest):
    if pipeline_service.is_running("discovery"):
        raise HTTPException(400, "Discovery pipeline is already running")
    pipeline_service.trigger_discovery(
        use_sample_run=req.use_sample_run,
        input_images_dir=req.input_images_dir,
        yolo_model_path=req.yolo_model_path,
        use_cache=req.use_cache,
    )
    return {"message": "Discovery pipeline triggered", "status": "starting"}


@router.post("/retraining/trigger")
async def trigger_retraining(req: RetrainingRequest):
    if pipeline_service.is_running("retraining"):
        raise HTTPException(400, "Retraining pipeline is already running")
    pipeline_service.trigger_retraining(
        project_id=req.project_id,
        epochs=req.epochs,
        imgsz=req.imgsz,
        batch_size=req.batch_size,
        freeze=req.freeze,
        base_model_path=req.base_model_path,
        upsample=req.upsample,
        upsample_target_ratio=req.upsample_target_ratio,
    )
    return {"message": "Retraining pipeline triggered", "status": "starting"}


@router.post("/retraining/smart-trigger")
async def smart_retrain(req: SmartRetrainRequest):
    """LLM-advised retraining: analyzes dataset first, then triggers training."""
    if pipeline_service.is_running("retraining"):
        raise HTTPException(400, "Retraining pipeline is already running")
    pipeline_service.trigger_retraining(
        project_id=-1,
        epochs=req.epochs,
        imgsz=req.imgsz,
        batch_size=req.batch_size,
        freeze=req.freeze,
        base_model_path=req.base_model_path,
        upsample=req.upsample,
        upsample_target_ratio=req.upsample_target_ratio,
    )
    return {"message": "Smart retraining triggered (LLM will advise)", "status": "starting"}


@router.get("/retraining/advisor-preview")
async def advisor_preview():
    """Preview what the LLM advisor would recommend without starting training."""
    try:
        from src.retraining.llm_advisor import collect_dataset_metadata, get_training_recommendation
        metadata = collect_dataset_metadata()
        recommendation = get_training_recommendation(metadata)
        return {"metadata": metadata, "recommendation": recommendation}
    except Exception as e:
        raise HTTPException(500, f"Advisor preview failed: {e}")


@router.post("/orchestrator/start")
async def start_orchestrator(req: OrchestratorRequest):
    from src.pipeline.orchestrator import orchestrator
    if orchestrator.is_running:
        return {"message": "Orchestrator is already running"}
    orchestrator.start(project_id=req.project_id)
    return {"message": "Orchestrator started"}


@router.post("/orchestrator/stop")
async def stop_orchestrator():
    from src.pipeline.orchestrator import orchestrator
    orchestrator.stop()
    return {"message": "Orchestrator stopped"}


@router.get("/status")
async def get_status():
    return pipeline_service.get_status()


@router.get("/runs")
async def list_runs(limit: int = 50):
    """
    Run history, served from the DB when it's been synced, falling back to
    the raw metrics JSON files (logs/pipeline_runs/*.json) otherwise.
    """
    with get_session() as session:
        rows = session.exec(
            select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(limit)
        ).all()
        if rows:
            results = []
            for run in rows:
                steps = session.exec(
                    select(PipelineStepMetric).where(PipelineStepMetric.run_id == run.id)
                ).all()
                results.append({
                    "run_id": run.run_id,
                    "kind": run.kind,
                    "status": run.status,
                    "steps": [s.step_name for s in steps],
                    "summary": {s.step_name: s.status for s in steps},
                })
            return results

    # Fallback: nothing synced yet (fresh DB) — read straight from disk.
    results = []
    for run_id in MetricsTracker.list_runs()[:limit]:
        try:
            data = MetricsTracker.load_run(run_id)
            statuses = [s.get("status") for s in data.get("steps", {}).values()]
            if statuses and all(s == "success" for s in statuses):
                run_status = "complete"
            elif any(s == "failed" for s in statuses):
                run_status = "failed"
            else:
                run_status = "running"
            results.append({
                "run_id": run_id,
                "kind": "retraining" if run_id.startswith("retrain_") else "discovery",
                "status": run_status,
                "steps": list(data.get("steps", {}).keys()),
                "summary": {k: v.get("status") for k, v in data.get("steps", {}).items()},
            })
        except Exception:
            results.append({"run_id": run_id, "kind": "unknown", "status": "unknown", "steps": [], "summary": {}})
    return results


# Static DAG shapes (mirrors build graphs in src/pipeline/graph.py and
# src/retraining/agent.py). "gate" marks an edge that ends the run early when
# the step produces nothing (or fails), as in the LangGraph conditional edges.
_DAGS = {
    "discovery": {
        "nodes": [
            ("yolo_inference", "YOLO routing"),
            ("dataset_context", "Dataset context"),
            ("vlm_annotation", "VLM annotation"),
            ("crop_extraction", "Crop extraction"),
            ("feature_extraction", "DINOv2 features"),
            ("faiss_search", "FAISS novelty"),
            ("hdbscan_cluster", "HDBSCAN clustering"),
            ("manifest_save", "Save manifest"),
        ],
        "gates": {"yolo_inference": "no unknown images", "vlm_annotation": "no annotations",
                  "crop_extraction": "no crops", "faiss_search": "no novel crops"},
    },
    "retraining": {
        "nodes": [
            ("export_annotations", "Export dataset"),
            ("validate_dataset", "Validate"),
            ("version_dataset", "DVC version"),
            ("llm_advisor", "LLM advisor"),
            ("upsample_dataset", "Upsampling"),
            ("train_yolo", "Train YOLO"),
            ("deploy_model", "Deploy"),
            ("sync_registry", "Sync registry"),
        ],
        "gates": {"validate_dataset": "validation failed", "train_yolo": "below deploy gate"},
    },
}


@router.get("/graph/{kind}")
async def get_graph(kind: str):
    """DAG of a pipeline with each step's state in its latest run (live while running)."""
    if kind not in _DAGS:
        raise HTTPException(404, f"Unknown pipeline {kind}")
    import config as cfg

    run_ids = [r for r in MetricsTracker.list_runs()
               if r.startswith("retrain_") == (kind == "retraining")]
    steps: dict = {}
    run_id = None
    if run_ids:
        # Newest by file time: the run being written right now is touched last
        run_id = max(run_ids, key=lambda r: (cfg.PIPELINE_RUNS_DIR / f"{r}.json").stat().st_mtime)
        try:
            steps = MetricsTracker.load_run(run_id).get("steps", {})
        except Exception:
            steps = {}
    live = pipeline_service.get_status()[kind]
    return {
        "kind": kind,
        "run_id": run_id,
        "run_status": live.get("status", "idle"),
        "started_at": live.get("started_at"),
        "error": live.get("error"),
        "nodes": [{"id": n, "label": label, "gate": _DAGS[kind]["gates"].get(n)}
                  for n, label in _DAGS[kind]["nodes"]],
        "steps": steps,
    }


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    with get_session() as session:
        run = session.exec(select(PipelineRun).where(PipelineRun.run_id == run_id)).first()
        if run is not None:
            steps = session.exec(
                select(PipelineStepMetric).where(PipelineStepMetric.run_id == run.id)
            ).all()
            return {
                "run_id": run.run_id,
                "kind": run.kind,
                "status": run.status,
                "result": run.result,
                "error": run.error,
                "steps": {s.step_name: s.model_dump() for s in steps},
            }

    try:
        return MetricsTracker.load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Run {run_id} not found")
