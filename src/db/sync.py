"""
JSON -> DB mirroring.

The pipeline/retraining code is the source of truth and keeps writing the
JSON files under data/ and models/ exactly as it always has. These
functions read that JSON and upsert it into the SQLite tables in
src/db/models.py so the API can query/filter/paginate instead of
re-parsing JSON on every request.

Call sync_all() on API startup and after any dashboard action that
mutates one of the underlying JSON files (naming a cluster, deploying a
model, etc.) — the HTTP handler still calls the existing JSON-writing
function first, then calls the matching sync_*() here.

Safe to call repeatedly: every sync_*() fully replaces the rows it owns
rather than assuming anything about previous DB state.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

import config as cfg
from src.db.models import (
    Cluster,
    ClusterCrop,
    ClusterFingerprint,
    DeploymentEvent,
    KnownDefectClass,
    ModelVersion,
    OrchestratorState,
    PipelineRun,
    PipelineStepMetric,
)
from src.utils import get_logger, load_json
from src.utils.metrics import MetricsTracker

log = get_logger("db.sync")


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_json_or_empty(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception as e:
        log.warning(f"Failed to read {path} for sync: {e}")
        return {}


# ── Model registry (models/registry.json) ───────────────────

def sync_model_registry(session: Session) -> None:
    data = _load_json_or_empty(cfg.MODELS_DIR / "registry.json")
    versions = data.get("versions", [])
    history = data.get("deployment_history", [])

    for row in session.exec(select(ModelVersion)):
        session.delete(row)
    for row in session.exec(select(DeploymentEvent)):
        session.delete(row)

    for v in versions:
        session.add(
            ModelVersion(
                version_id=v["version_id"],
                version_num=v.get("version_num", 0),
                path=v.get("path", ""),
                original_path=v.get("original_path", ""),
                timestamp=v.get("timestamp", ""),
                created_at=_parse_dt(v.get("created_at")),
                metrics=v.get("metrics", {}),
                training_config=v.get("training_config", {}),
                source=v.get("source", ""),
                notes=v.get("notes", ""),
                classes=v.get("classes", []),
                dataset_stats=v.get("dataset_stats", {}),
                size_mb=v.get("size_mb", 0.0),
                status=v.get("status", "registered"),
            )
        )

    for h in history:
        session.add(
            DeploymentEvent(
                version_id=h.get("version_id", ""),
                previous_version=h.get("previous_version"),
                deployed_at=_parse_dt(h.get("deployed_at")) or datetime.now(),
                confirmed_by=h.get("confirmed_by", "system"),
                action=h.get("action", "deploy"),
            )
        )

    session.commit()
    log.info(f"Synced model registry: {len(versions)} versions, {len(history)} deployment events")


# ── Known defect classes (data/known_defects.json) ──────────

def sync_known_defects(session: Session) -> None:
    data = _load_json_or_empty(cfg.KNOWN_DEFECTS_JSON)
    classes = data.get("defect_classes", [])

    existing = {row.name: row for row in session.exec(select(KnownDefectClass))}
    for name in classes:
        if name not in existing:
            session.add(KnownDefectClass(name=name, source="sync"))
    # Classes are only ever added (never silently removed) by the pipeline,
    # so rows for names no longer present are left as historical record.

    session.commit()
    log.info(f"Synced known defect classes: {len(classes)} active")


# ── Cluster fingerprint registry (cfg.CLUSTER_REGISTRY_PATH) ─

def sync_cluster_fingerprints(session: Session) -> None:
    data = _load_json_or_empty(cfg.CLUSTER_REGISTRY_PATH)
    fingerprints = data.get("fingerprints", [])

    existing = {row.fingerprint_id: row for row in session.exec(select(ClusterFingerprint))}
    seen_ids: set[str] = set()

    for fp in fingerprints:
        fp_id = fp["id"]
        seen_ids.add(fp_id)
        row = existing.get(fp_id) or ClusterFingerprint(fingerprint_id=fp_id)
        row.label = fp.get("label")
        row.confidence = fp.get("confidence")
        row.first_seen = _parse_dt(fp.get("first_seen"))
        row.last_seen = _parse_dt(fp.get("last_seen"))
        row.labeled_at = _parse_dt(fp.get("labeled_at"))
        row.match_count = fp.get("match_count", 0)
        row.cluster_size = fp.get("cluster_size", 0)
        row.centroid = fp.get("centroid", [])
        session.add(row)

    for fp_id, row in existing.items():
        if fp_id not in seen_ids:
            session.delete(row)

    session.commit()
    log.info(f"Synced cluster fingerprints: {len(fingerprints)}")


# ── Cluster manifest (data/clusters/cluster_manifest.json) ──
# This file holds ONE current snapshot (not run history), so every sync
# fully replaces the Cluster/ClusterCrop rows.

def sync_clusters(session: Session) -> None:
    manifest = _load_json_or_empty(cfg.CLUSTERS_DIR / "cluster_manifest.json")
    clusters = manifest.get("clusters", {})
    run_id = manifest.get("run_id", "")

    for row in session.exec(select(ClusterCrop)):
        session.delete(row)
    for row in session.exec(select(Cluster)):
        session.delete(row)
    session.commit()

    for cluster_name, entry in clusters.items():
        cluster = Cluster(
            run_id=run_id,
            cluster_name=cluster_name,
            cluster_id_int=entry.get("cluster_id", -99),
            fingerprint_id=entry.get("fingerprint_id"),
            defect_name=entry.get("defect_name"),
            cohesion=entry.get("cohesion"),
            crop_count=entry.get("crop_count", len(entry.get("crops", []))),
        )
        session.add(cluster)
        session.flush()  # assign cluster.id before attaching crops

        for crop in entry.get("crops", []):
            session.add(
                ClusterCrop(
                    cluster_id=cluster.id,
                    crop_file=crop["crop_file"],
                    crop_path=crop.get("crop_path", ""),
                    source_image=crop.get("source_image", ""),
                    source_image_name=crop.get("source_image_name", ""),
                    box_2d_pixels=crop.get("box_2d_pixels", []),
                    box_2d_raw=crop.get("box_2d_raw", []),
                    physical_traits=crop.get("physical_traits", ""),
                    crop_width=crop.get("crop_width", 0),
                    crop_height=crop.get("crop_height", 0),
                )
            )

    session.commit()
    log.info(f"Synced clusters: {len(clusters)} clusters for run {run_id or '(none)'}")


# ── Orchestrator state (data/orchestrator_state.json) ───────

def sync_orchestrator_state(session: Session) -> None:
    data = _load_json_or_empty(cfg.DATA_DIR / "orchestrator_state.json")

    existing = {row.key: row for row in session.exec(select(OrchestratorState))}
    for key, value in data.items():
        row = existing.get(key) or OrchestratorState(key=key)
        row.value = value if isinstance(value, dict) else {"value": value}
        row.updated_at = datetime.now()
        session.add(row)

    session.commit()
    log.info(f"Synced orchestrator state: {len(data)} keys")


# ── Pipeline run metrics (logs/pipeline_runs/*.json) ─────────

def sync_pipeline_run(session: Session, run_id: str, kind: str | None = None) -> None:
    """Sync a single run (call this right after a discovery/retraining run finishes)."""
    try:
        data = MetricsTracker.load_run(run_id)
    except FileNotFoundError:
        log.warning(f"No metrics file for run {run_id}, skipping sync")
        return

    inferred_kind = kind or ("retraining" if run_id.startswith("retrain_") else "discovery")

    run = session.exec(select(PipelineRun).where(PipelineRun.run_id == run_id)).first()
    if run is None:
        run = PipelineRun(run_id=run_id, kind=inferred_kind)
        session.add(run)
        session.flush()
    else:
        run.kind = inferred_kind

    steps = data.get("steps", {})
    statuses = [s.get("status") for s in steps.values()]
    if statuses and all(s == "success" for s in statuses):
        run.status = "complete"
    elif any(s == "failed" for s in statuses):
        run.status = "failed"
    else:
        run.status = "running"
    run.result = data
    session.add(run)
    session.flush()

    existing_steps = {
        s.step_name: s
        for s in session.exec(select(PipelineStepMetric).where(PipelineStepMetric.run_id == run.id))
    }
    for step_name, metric in steps.items():
        row = existing_steps.get(step_name) or PipelineStepMetric(run_id=run.id, step_name=step_name)
        row.status = metric.get("status", "pending")
        row.start_time = _parse_dt(metric.get("start_time"))
        row.end_time = _parse_dt(metric.get("end_time"))
        row.latency_seconds = metric.get("latency_seconds", 0.0)
        row.items_processed = metric.get("items_processed", 0)
        row.items_failed = metric.get("items_failed", 0)
        row.metadata_json = metric.get("metadata", {})
        row.error = metric.get("error")
        session.add(row)

    session.commit()
    log.info(f"Synced pipeline run {run_id}: {len(steps)} steps, status={run.status}")


def sync_all_pipeline_runs(session: Session) -> None:
    for run_id in MetricsTracker.list_runs():
        sync_pipeline_run(session, run_id)


# ── Full resync (startup / manual trigger) ───────────────────

def sync_all(session: Session) -> None:
    sync_model_registry(session)
    sync_known_defects(session)
    sync_cluster_fingerprints(session)
    sync_clusters(session)
    sync_orchestrator_state(session)
    sync_all_pipeline_runs(session)
    log.info("Full JSON -> DB sync complete")
