"""
Tool: MLflow Model Deployment

Registers the fine-tuned YOLO model in MLflow and deploys it
to a local serving endpoint, replacing the existing model.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger, save_json

log = get_logger("mlflow_deploy")


def deploy_model(
    model_path: str | None = None,
    model_name: str = "yolo-defect-detector",
    metrics: dict | None = None,
) -> dict:
    """
    Register and deploy a YOLO model via MLflow.

    Steps:
        1. Log the model artifact to MLflow
        2. Register it under the given model name
        3. Transition it to 'Production' stage
        4. Copy to models/ directory as the active model

    Returns:
        {
            "success": bool,
            "model_name": str,
            "model_version": str,
            "run_id": str,
            "error": str | None,
        }
    """
    import mlflow

    src_model = model_path or str(cfg.YOLO_MODEL_PATH)

    if not Path(src_model).exists():
        return {
            "success": False,
            "model_name": model_name,
            "model_version": "",
            "run_id": "",
            "error": f"Model not found: {src_model}",
        }

    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)

    log.info(f"Deploying model: {src_model}")
    log.info(f"  MLflow URI: {cfg.MLFLOW_TRACKING_URI}")
    log.info(f"  Model name: {model_name}")

    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Log metrics
            if metrics:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)

            # Log parameters
            mlflow.log_param("model_source", src_model)
            mlflow.log_param("timestamp", ts)

            # Log the model artifact
            mlflow.log_artifact(src_model, artifact_path="model")

            # Register model
            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(model_uri, model_name)
            version = result.version

            log.info(f"Registered as {model_name} v{version}")

            # Transition to Production
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )

            # Copy to active model path
            shutil.copy2(src_model, str(cfg.YOLO_MODEL_PATH))

            # Version locally
            version_path = cfg.MODEL_VERSIONS_DIR / f"best_v{ts}_mlflow_v{version}.pt"
            shutil.copy2(src_model, str(version_path))

            # Save deployment metadata
            deploy_meta = {
                "model_name": model_name,
                "model_version": str(version),
                "run_id": run_id,
                "source_path": src_model,
                "deployed_to": str(cfg.YOLO_MODEL_PATH),
                "timestamp": ts,
                "metrics": metrics or {},
            }
            save_json(
                deploy_meta,
                cfg.LOGS_DIR / "deployments" / f"deploy_{ts}.json",
            )

            log.info(f"Model deployed successfully: {model_name} v{version}")

            return {
                "success": True,
                "model_name": model_name,
                "model_version": str(version),
                "run_id": run_id,
                "error": None,
            }

    except Exception as e:
        log.error(f"MLflow deployment failed: {e}")

        # Fallback: still copy model locally
        try:
            shutil.copy2(src_model, str(cfg.YOLO_MODEL_PATH))
            log.info("Fallback: model copied to active path without MLflow")
        except Exception:
            pass

        return {
            "success": False,
            "model_name": model_name,
            "model_version": "",
            "run_id": "",
            "error": str(e),
        }
