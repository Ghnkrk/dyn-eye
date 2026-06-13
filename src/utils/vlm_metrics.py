"""
VLM Annotation Performance Metrics (Part 3)

Lightweight, non-intrusive instrumentation for Gemini annotation quality.
Metrics go to:
  - data/vlm_metrics.json (append to history)
  - MLflow (via existing MetricsTracker pattern)

Measured:
  - Intra-cluster label consistency (ICC)
  - Confidence distribution (mean, ambiguous fraction, high-conviction fraction)
  - DBCV score (passthrough from clustering)
  - Registry hit rate (from cluster fingerprint matching)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("vlm_metrics")


def log_run_metrics(
    cluster_results: list[dict],
    dbcv_score: float = -1.0,
    registry_hits: int = 0,
    registry_total: int = 0,
    run_id: str = "",
) -> dict:
    """
    Aggregate per-cluster VLM results into run-level metrics.

    Args:
        cluster_results: List of dicts, each with:
            - label: str (final cluster label)
            - confidence: float (final confidence)
            - icc: float (intra-cluster consistency, 0-1)
            - n_samples: int (how many crops were annotated)
            - labels_seen: list[str] (all labels returned by VLM)
        dbcv_score: DBCV validity index from clustering
        registry_hits: Number of clusters that matched registry fingerprints
        registry_total: Total cluster count
        run_id: Pipeline run identifier

    Returns:
        dict with aggregated metrics.
    """
    if not cluster_results:
        return {"error": "no_cluster_results"}

    # ICC stats
    iccs = [r["icc"] for r in cluster_results if "icc" in r]
    mean_icc = sum(iccs) / len(iccs) if iccs else 0.0

    # Confidence stats
    confidences = [r["confidence"] for r in cluster_results if "confidence" in r and r["confidence"] is not None]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    ambiguous_fraction = sum(1 for c in confidences if c < 0.5) / len(confidences) if confidences else 0.0
    high_conviction_fraction = sum(1 for c in confidences if c > 0.85) / len(confidences) if confidences else 0.0

    # Registry hit rate
    hit_rate = registry_hits / registry_total if registry_total > 0 else 0.0

    metrics = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_clusters": len(cluster_results),
        "mean_icc": round(mean_icc, 4),
        "mean_confidence": round(mean_confidence, 4),
        "ambiguous_fraction": round(ambiguous_fraction, 4),
        "high_conviction_fraction": round(high_conviction_fraction, 4),
        "dbcv_score": round(dbcv_score, 4) if dbcv_score > -1 else None,
        "registry_hit_rate": round(hit_rate, 4),
        "registry_hits": registry_hits,
        "registry_total": registry_total,
    }

    # ── Persist to JSON history ─────────────────────────────
    try:
        metrics_path = cfg.VLM_METRICS_PATH
        history = []
        if metrics_path.exists():
            try:
                history = json.loads(metrics_path.read_text(encoding="utf-8"))
                if not isinstance(history, list):
                    history = [history]
            except (json.JSONDecodeError, OSError):
                history = []

        history.append(metrics)
        metrics_path.write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        log.info(f"VLM metrics saved to {metrics_path} (run {run_id})")
    except Exception as e:
        log.warning(f"Failed to persist VLM metrics to JSON: {e}")

    # ── Log to MLflow ───────────────────────────────────────
    try:
        import mlflow

        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"vlm_metrics_{run_id}", nested=True):
            mlflow.log_metrics({
                "vlm_mean_icc": metrics["mean_icc"],
                "vlm_mean_confidence": metrics["mean_confidence"],
                "vlm_ambiguous_fraction": metrics["ambiguous_fraction"],
                "vlm_high_conviction_fraction": metrics["high_conviction_fraction"],
                "vlm_registry_hit_rate": metrics["registry_hit_rate"],
            })
            if metrics["dbcv_score"] is not None:
                mlflow.log_metric("clustering_dbcv", metrics["dbcv_score"])

        log.info(f"VLM metrics logged to MLflow (run {run_id})")
    except Exception as e:
        log.warning(f"MLflow logging failed (non-fatal): {e}")

    # Log summary
    log.info(
        f"VLM metrics summary: ICC={metrics['mean_icc']:.3f}, "
        f"confidence={metrics['mean_confidence']:.3f}, "
        f"ambiguous={metrics['ambiguous_fraction']:.1%}, "
        f"high_conviction={metrics['high_conviction_fraction']:.1%}, "
        f"registry_hit_rate={metrics['registry_hit_rate']:.1%}"
    )

    return metrics
