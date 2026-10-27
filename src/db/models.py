"""
SQLModel tables for the DYN-EYE dashboard read/audit database.

These tables MIRROR state that the pipeline/retraining code already owns
as JSON files (see config.py path constants). Nothing here is written to
by the pipeline nodes directly — src/db/sync.py keeps it in sync after
each mutation so the API can query/filter/paginate instead of re-parsing
JSON on every request.

No ORM Relationship()s: all joins are done as explicit queries by foreign
key in src/db/sync.py and the API layer. (Bidirectional Relationship()
forward-ref resolution broke under this sqlmodel/sqlalchemy/Python combo —
explicit queries sidestep it entirely and keep sync.py easier to follow.)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Pipeline runs & step metrics ─────────────────────────────
# Mirrors logs/pipeline_runs/{run_id}.json (src/utils/metrics.py)

class PipelineRun(SQLModel, table=True):
    __tablename__ = "pipeline_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, unique=True)
    kind: str = Field(index=True)  # "discovery" | "retraining"
    status: str = Field(default="running", index=True)  # running|complete|failed
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None
    result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error: Optional[str] = None


class PipelineStepMetric(SQLModel, table=True):
    __tablename__ = "pipeline_step_metrics"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="pipeline_runs.id", index=True)
    step_name: str
    status: str = "pending"  # pending|running|success|failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    latency_seconds: float = 0.0
    items_processed: int = 0
    items_failed: int = 0
    metadata_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error: Optional[str] = None


# ── Cluster fingerprints ─────────────────────────────────────
# Mirrors cfg.CLUSTER_REGISTRY_PATH (src/features/cluster_registry.py)

class ClusterFingerprint(SQLModel, table=True):
    __tablename__ = "cluster_fingerprints"

    id: Optional[int] = Field(default=None, primary_key=True)
    fingerprint_id: str = Field(index=True, unique=True)
    label: Optional[str] = None
    confidence: Optional[float] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    labeled_at: Optional[datetime] = None
    match_count: int = 0
    cluster_size: int = 0
    centroid: list[float] = Field(default_factory=list, sa_column=Column(JSON))


# ── Clusters & crops ─────────────────────────────────────────
# Mirrors data/clusters/cluster_manifest.json (dashboard/app.py, manifest_save.py)

class Cluster(SQLModel, table=True):
    __tablename__ = "clusters"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)  # manifest's run_id, not an FK — manifest spans runs
    cluster_name: str = Field(index=True)  # folder name: cluster_000 | noise | unassigned
    cluster_id_int: int  # -1 noise, -2 unassigned, else hdbscan cluster id
    fingerprint_id: Optional[str] = Field(default=None, foreign_key="cluster_fingerprints.fingerprint_id")
    defect_name: Optional[str] = None
    cohesion: Optional[float] = None
    crop_count: int = 0
    updated_at: datetime = Field(default_factory=_utcnow)


class ClusterCrop(SQLModel, table=True):
    __tablename__ = "cluster_crops"

    id: Optional[int] = Field(default=None, primary_key=True)
    cluster_id: int = Field(foreign_key="clusters.id", index=True)
    crop_file: str
    crop_path: str = ""
    source_image: str = ""
    source_image_name: str = ""
    box_2d_pixels: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    box_2d_raw: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    physical_traits: str = ""
    crop_width: int = 0
    crop_height: int = 0


# ── Model registry & deployments ─────────────────────────────
# Mirrors models/registry.json (src/retraining/model_registry.py)

class ModelVersion(SQLModel, table=True):
    __tablename__ = "model_versions"

    id: Optional[int] = Field(default=None, primary_key=True)
    version_id: str = Field(index=True, unique=True)
    version_num: int
    path: str
    original_path: str = ""
    timestamp: str = ""
    created_at: Optional[datetime] = None
    metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    training_config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    source: str = ""
    notes: str = ""
    classes: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    dataset_stats: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    size_mb: float = 0.0
    status: str = "registered"  # registered -> deployed -> retired


class DeploymentEvent(SQLModel, table=True):
    __tablename__ = "deployment_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    version_id: str = Field(index=True)
    previous_version: Optional[str] = None
    deployed_at: datetime = Field(default_factory=_utcnow)
    confirmed_by: str = "system"
    action: str = "deploy"  # deploy | rollback


# ── Known defect classes ─────────────────────────────────────
# Mirrors cfg.KNOWN_DEFECTS_JSON (src/features/known_defects_registry.py)

class KnownDefectClass(SQLModel, table=True):
    __tablename__ = "known_defect_classes"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    added_at: datetime = Field(default_factory=_utcnow)
    source: str = ""


# ── Orchestrator state ───────────────────────────────────────
# Mirrors data/orchestrator_state.json (src/pipeline/orchestrator.py)

class OrchestratorState(SQLModel, table=True):
    __tablename__ = "orchestrator_state"

    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(index=True, unique=True)
    value: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: datetime = Field(default_factory=_utcnow)


# ── Persisted log history ────────────────────────────────────
# The in-memory LogStream ring buffer (src/utils/log_stream.py) stays as the
# live-tail source; this table is the durable history so past runs' logs
# survive a process restart and can be queried by run_id.

class LogEventRecord(SQLModel, table=True):
    __tablename__ = "log_event_records"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: Optional[str] = Field(default=None, index=True)
    ts: datetime = Field(default_factory=_utcnow, index=True)
    level: str = "info"
    source: str = "system"
    message: str = ""
    data: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
