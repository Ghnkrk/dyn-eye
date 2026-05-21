"""
Retraining Agent — LangGraph-Based

A LangGraph agent that orchestrates the post-labeling workflow:
  1. Export annotations from Label Studio → YOLO format
  2. Validate YOLO dataset format
  3. Version dataset with DVC
  4. Fine-tune YOLO model
  5. Version and deploy model via MLflow

The agent uses tools for each step, allowing flexible execution
and error handling.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.utils import get_logger
from src.utils.metrics import MetricsTracker
# No Label Studio export needed
from src.retraining.tools.dataset_validator import validate_yolo_dataset
from src.retraining.tools.dvc_version import version_dataset
from src.retraining.tools.train_yolo import train_yolo
from src.retraining.tools.mlflow_deploy import deploy_model
from src.features.known_defects_registry import (
    register_from_yolo_model,
    register_from_data_yaml,
)
from src.features.faiss_index import FAISSIndexManager

log = get_logger("retraining_agent")


# ── Agent State ──────────────────────────────────────────────

class RetrainingState(TypedDict, total=False):
    # Input
    project_id: int              # Label Studio project ID
    epochs: int
    imgsz: int
    batch_size: int

    # Export
    export_result: dict

    # Validation
    validation_result: dict

    # Versioning
    dvc_result: dict

    # Training
    training_result: dict

    # Deployment
    deploy_result: dict

    # Post-deploy sync
    sync_result: dict

    # Metadata
    run_id: str
    errors: list[str]


# ── Agent Nodes ──────────────────────────────────────────────

_metrics: MetricsTracker | None = None


def export_node(state: dict) -> dict:
    """Export named dashboard clusters to YOLO dataset format."""
    if _metrics:
        _metrics.start_step("export_annotations")

    try:
        from src.pipeline.orchestrator import ManifestPoller, AnnotationMapper
        import config as cfg

        poller = ManifestPoller()
        named = poller.check_named_clusters()

        if not named:
            error = "No named clusters found in dashboard manifest. Please name at least one cluster in the dashboard before retraining."
            if _metrics:
                _metrics.fail_step("export_annotations", error)
            return {"errors": state.get("errors", []) + [error], "export_result": {}}

        mapper = AnnotationMapper()
        known_classes = cfg.KNOWN_DEFECT_NAMES.copy()
        result = mapper.map_cluster_labels_to_yolo(
            cluster_labels=named,
            label_names=known_classes,
        )

        if result.get("total_mapped", 0) == 0:
            error = "No annotations could be mapped from the dashboard manifest."
            if _metrics:
                _metrics.fail_step("export_annotations", error)
            return {"errors": state.get("errors", []) + [error], "export_result": {}}

        if _metrics:
            _metrics.end_step("export_annotations", items_processed=result.get("total_mapped", 0))

        log.info(f"Export complete: {result}")
        return {"export_result": result}
    except Exception as e:
        if _metrics:
            _metrics.fail_step("export_annotations", str(e))
        return {"errors": state.get("errors", []) + [str(e)], "export_result": {}}


def validate_node(state: dict) -> dict:
    """Validate the exported YOLO dataset."""
    if _metrics:
        _metrics.start_step("validate_dataset")

    try:
        result = validate_yolo_dataset()
        if result["valid"]:
            if _metrics:
                _metrics.end_step("validate_dataset", items_processed=1)
            log.info("Dataset validation PASSED")
        else:
            error_msg = f"Validation failed: {result['errors'][:3]}"
            if _metrics:
                _metrics.fail_step("validate_dataset", error_msg)
            log.error(error_msg)
            return {
                "validation_result": result,
                "errors": state.get("errors", []) + [error_msg],
            }
        return {"validation_result": result}
    except Exception as e:
        if _metrics:
            _metrics.fail_step("validate_dataset", str(e))
        return {"errors": state.get("errors", []) + [str(e)], "validation_result": {}}


def version_dataset_node(state: dict) -> dict:
    """Version dataset using DVC."""
    if _metrics:
        _metrics.start_step("version_dataset")

    # Check if validation passed
    validation = state.get("validation_result", {})
    if not validation.get("valid", False):
        error = "Skipping versioning: dataset validation failed"
        if _metrics:
            _metrics.fail_step("version_dataset", error)
        return {"errors": state.get("errors", []) + [error], "dvc_result": {}}

    try:
        result = version_dataset()
        if result["success"]:
            if _metrics:
                _metrics.end_step("version_dataset", items_processed=1)
        else:
            if _metrics:
                _metrics.fail_step("version_dataset", result.get("error", "Unknown"))
        return {"dvc_result": result}
    except Exception as e:
        if _metrics:
            _metrics.fail_step("version_dataset", str(e))
        return {"errors": state.get("errors", []) + [str(e)], "dvc_result": {}}


def train_node(state: dict) -> dict:
    """Fine-tune YOLO model."""
    if _metrics:
        _metrics.start_step("train_yolo")

    try:
        result = train_yolo(
            epochs=state.get("epochs", cfg.YOLO_TRAIN_EPOCHS),
            imgsz=state.get("imgsz", cfg.YOLO_TRAIN_IMGSZ),
            batch=state.get("batch_size", cfg.YOLO_TRAIN_BATCH),
        )
        if result["success"]:
            if _metrics:
                _metrics.end_step("train_yolo", items_processed=1,
                                  **result.get("metrics", {}))
        else:
            if _metrics:
                _metrics.fail_step("train_yolo", result.get("error", "Unknown"))
        return {"training_result": result}
    except Exception as e:
        if _metrics:
            _metrics.fail_step("train_yolo", str(e))
        return {"errors": state.get("errors", []) + [str(e)], "training_result": {}}


def deploy_node(state: dict) -> dict:
    """Deploy model via MLflow."""
    if _metrics:
        _metrics.start_step("deploy_model")

    training = state.get("training_result", {})
    if not training.get("success", False):
        error = "Skipping deployment: training failed"
        if _metrics:
            _metrics.fail_step("deploy_model", error)
        return {"errors": state.get("errors", []) + [error], "deploy_result": {}}

    try:
        result = deploy_model(
            model_path=training.get("model_path"),
            metrics=training.get("metrics"),
        )
        if result["success"]:
            if _metrics:
                _metrics.end_step("deploy_model", items_processed=1)
        else:
            if _metrics:
                _metrics.fail_step("deploy_model", result.get("error", "Unknown"))
        return {"deploy_result": result}
    except Exception as e:
        if _metrics:
            _metrics.fail_step("deploy_model", str(e))
        return {"errors": state.get("errors", []) + [str(e)], "deploy_result": {}}


def sync_registry_node(state: dict) -> dict:
    """
    Post-deploy: update the known-defects registry and rebuild FAISS.

    This node runs automatically after a model is deployed. It:
      1. Reads class names from the newly deployed YOLO model.
      2. Reads class names from the data.yaml used for training.
      3. Merges both into the known-defects registry.
      4. Rebuilds the FAISS index so the next discovery run
         correctly distinguishes known vs unknown.
    """
    if _metrics:
        _metrics.start_step("sync_registry")

    deploy = state.get("deploy_result", {})
    training = state.get("training_result", {})

    if not deploy.get("success") and not training.get("success"):
        error = "Skipping registry sync: no successful deployment or training"
        if _metrics:
            _metrics.fail_step("sync_registry", error)
        return {"sync_result": {"skipped": True, "reason": error}}

    try:
        # ── 1. Register classes from the deployed YOLO model ─────────
        model_path = training.get("model_path") or str(cfg.YOLO_MODEL_PATH)
        model_added = register_from_yolo_model(model_path)

        # ── 2. Register classes from data.yaml ──────────────────────
        yaml_added = register_from_data_yaml()

        all_added = list(set(model_added + yaml_added))

        # ── 3. Rebuild FAISS index ──────────────────────────────────
        faiss_count = 0
        try:
            manager = FAISSIndexManager()
            faiss_count = manager.setup()
            log.info(f"FAISS index rebuilt: {faiss_count} vectors")
        except Exception as fe:
            log.warning(f"FAISS rebuild skipped (non-fatal): {fe}")

        result = {
            "success": True,
            "new_classes_added": all_added,
            "faiss_vectors": faiss_count,
        }

        if _metrics:
            _metrics.end_step("sync_registry", items_processed=len(all_added))

        log.info(
            f"Registry sync complete: {len(all_added)} new classes, "
            f"{faiss_count} FAISS vectors"
        )
        return {"sync_result": result}

    except Exception as e:
        if _metrics:
            _metrics.fail_step("sync_registry", str(e))
        log.error(f"Registry sync failed: {e}")
        return {
            "errors": state.get("errors", []) + [f"sync_registry: {e}"],
            "sync_result": {"success": False, "error": str(e)},
        }


# ── Conditional Edge ─────────────────────────────────────────

def should_continue_after_validation(state: dict) -> str:
    """Check if we should continue after validation."""
    validation = state.get("validation_result", {})
    if validation.get("valid", False):
        return "version_dataset"
    else:
        return END


def should_continue_after_training(state: dict) -> str:
    """Check if training succeeded before deployment."""
    training = state.get("training_result", {})
    if training.get("success", False):
        return "deploy_model"
    else:
        return END


# ── Graph Assembly ───────────────────────────────────────────

def build_retraining_graph():
    """Build the LangGraph retraining agent."""
    graph = StateGraph(RetrainingState)

    graph.add_node("export_annotations", export_node)
    graph.add_node("validate_dataset", validate_node)
    graph.add_node("version_dataset", version_dataset_node)
    graph.add_node("train_yolo", train_node)
    graph.add_node("deploy_model", deploy_node)
    graph.add_node("sync_registry", sync_registry_node)

    # Linear chain with conditional gates
    graph.add_edge(START, "export_annotations")
    graph.add_edge("export_annotations", "validate_dataset")
    graph.add_conditional_edges(
        "validate_dataset",
        should_continue_after_validation,
        {"version_dataset": "version_dataset", END: END},
    )
    graph.add_edge("version_dataset", "train_yolo")
    graph.add_conditional_edges(
        "train_yolo",
        should_continue_after_training,
        {"deploy_model": "deploy_model", END: END},
    )
    graph.add_edge("deploy_model", "sync_registry")
    graph.add_edge("sync_registry", END)

    return graph.compile()


def run_retraining_pipeline(
    project_id: int | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch_size: int | None = None,
) -> dict:
    """
    Run the complete retraining pipeline.

    Args:
        project_id: Label Studio project ID with completed annotations
        epochs: Training epochs
        imgsz: Training image size
        batch_size: Training batch size

    Returns:
        Final state dict with all step results.
    """
    global _metrics

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _metrics = MetricsTracker(run_id=f"retrain_{run_id}")

    initial_state: RetrainingState = {
        "run_id": run_id,
        "project_id": project_id or -1,
        "epochs": epochs or cfg.YOLO_TRAIN_EPOCHS,
        "imgsz": imgsz or cfg.YOLO_TRAIN_IMGSZ,
        "batch_size": batch_size or cfg.YOLO_TRAIN_BATCH,
        "errors": [],
    }

    log.info(f"═══ Starting retraining pipeline: {run_id} ═══")

    graph = build_retraining_graph()
    final_state = graph.invoke(initial_state)

    log.info(f"═══ Retraining pipeline {run_id} complete ═══")
    final_state["_metrics"] = _metrics.snapshot()

    return final_state
