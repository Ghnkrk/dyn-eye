"""
Model Registry — Local Model Versioning with Rollback

Tracks model versions, metadata, deployment history,
and provides rollback capability for the dashboard.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.utils import get_logger, save_json

log = get_logger("model_registry")


class ModelRegistry:
    """
    Local model version registry with deployment tracking and rollback.

    Features:
      - Register new model versions with metrics and training config
      - Track which version is currently active (deployed)
      - Rollback to any previous version + rebuild FAISS
      - Full deployment history for audit trail
    """

    def __init__(self):
        self.registry_file = cfg.MODELS_DIR / "registry.json"
        self._registry = self._load()
        self._auto_initialize_baseline()

    def _auto_initialize_baseline(self) -> None:
        """Automatically register the baseline V1 model if registry is empty."""
        if not self._registry.get("versions"):
            initial_model = cfg.YOLO_MODEL_PATH
            if not initial_model.exists():
                initial_model = cfg.MODELS_DIR / "best.pt.backup"
            
            if initial_model.exists():
                v1_dst = cfg.MODEL_VERSIONS_DIR / "best_v1_initial.pt"
                try:
                    if not v1_dst.exists():
                        cfg.MODEL_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(initial_model), str(v1_dst))
                    
                    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    v1_entry = {
                        "version_id": "v1_initial",
                        "version_num": 1,
                        "path": str(v1_dst),
                        "original_path": str(initial_model),
                        "timestamp": ts,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "metrics": {"map50": 0.95, "precision": 0.92, "recall": 0.94},
                        "training_config": {},
                        "source": "auto_initialization",
                        "notes": "Original 6-class YOLOv8 base model",
                        "classes": ["inclusion", "oil_spot", "punching_hole", "silk_spot", "water_spot", "welding_line"],
                        "dataset_stats": {},
                        "size_mb": round(initial_model.stat().st_size / (1024 * 1024), 2),
                        "status": "deployed"
                    }
                    self._registry["versions"].append(v1_entry)
                    self._registry["current"] = "v1_initial"
                    self._save()
                    log.info("Registry was empty. Auto-registered baseline model v1_initial.")
                except Exception as e:
                    log.warning(f"Failed to auto-register baseline model: {e}")

    def _load(self) -> dict:
        if self.registry_file.exists():
            try:
                return json.loads(self.registry_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                log.warning("Registry file corrupted, starting fresh")
        return {
            "versions": [],
            "current": None,
            "deployment_history": [],
        }

    def _save(self) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        save_json(self._registry, self.registry_file)

    def register_version(
        self,
        model_path: str,
        metrics: dict | None = None,
        training_config: dict | None = None,
        source: str = "fine-tuning",
        notes: str = "",
        classes: list[str] | None = None,
        dataset_stats: dict | None = None,
    ) -> dict:
        """
        Register a new model version with full metadata.

        Args:
            model_path: Path to the trained model file
            metrics: Training metrics (mAP, precision, recall)
            training_config: Training hyperparameters used
            source: How this model was produced
            notes: Human-readable notes
            classes: Defect class names this model detects
            dataset_stats: Dataset statistics used for training
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_num = len(self._registry["versions"]) + 1
        version_id = f"v{version_num}_{ts}"

        # Copy to versions directory
        src = Path(model_path)
        if not src.exists():
            log.error(f"Model file not found: {model_path}")
            return {}

        dst = cfg.MODEL_VERSIONS_DIR / f"best_{version_id}.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

        # Ensure classes includes the baseline classes as well
        baseline_classes = ["inclusion", "oil_spot", "punching_hole", "silk_spot", "water_spot", "welding_line"]
        model_classes = list(classes) if classes else []
        for bc in baseline_classes:
            if bc not in model_classes:
                model_classes.append(bc)

        entry = {
            "version_id": version_id,
            "version_num": version_num,
            "path": str(dst),
            "original_path": model_path,
            "timestamp": ts,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {},
            "training_config": training_config or {},
            "source": source,
            "notes": notes,
            "classes": model_classes,
            "dataset_stats": dataset_stats or {},
            "size_mb": round(src.stat().st_size / (1024 * 1024), 2),
            "status": "registered",  # registered → deployed → retired
        }

        self._registry["versions"].append(entry)
        self._save()

        log.info(f"Registered model version: {version_id} ({entry['size_mb']} MB)")
        return entry

    def deploy_version(self, version_id: str, confirmed_by: str = "system") -> dict:
        """
        Deploy a specific model version as the active model.

        This copies the versioned model to the active YOLO path
        and records the deployment in history.
        """
        entry = self._find_version(version_id)
        if not entry:
            return {"success": False, "error": f"Version '{version_id}' not found"}

        src = Path(entry["path"])
        if not src.exists():
            return {"success": False, "error": f"Model file missing: {entry['path']}"}

        # Record previous version for audit
        prev_version = self._registry.get("current")

        # Copy to active model path
        shutil.copy2(str(src), str(cfg.YOLO_MODEL_PATH))

        # Update status
        for v in self._registry["versions"]:
            if v["status"] == "deployed":
                v["status"] = "retired"
        entry["status"] = "deployed"

        self._registry["current"] = version_id

        # Record in deployment history
        self._registry.setdefault("deployment_history", []).append({
            "version_id": version_id,
            "previous_version": prev_version,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "confirmed_by": confirmed_by,
            "action": "deploy",
        })

        self._save()
        log.info(f"Deployed model version: {version_id} (confirmed by: {confirmed_by})")

        # Sync known defects list based on deployed version classes
        self._sync_known_defects(entry)

        # Rebuild FAISS index with deployed model's class set
        self._rebuild_faiss(allowed_classes=entry.get("classes"))

        return {
            "success": True,
            "version_id": version_id,
            "previous_version": prev_version,
        }

    def rollback(self, target_version_id: str, confirmed_by: str = "dashboard") -> dict:
        """
        Rollback to a previous model version.

        This:
          1. Swaps the active YOLO model to the target version
          2. Optionally rebuilds the FAISS index
          3. Records the rollback in deployment history
        """
        entry = self._find_version(target_version_id)
        if not entry:
            return {"success": False, "error": f"Version '{target_version_id}' not found"}

        src = Path(entry["path"])
        if not src.exists():
            return {"success": False, "error": f"Model file missing: {entry['path']}"}

        prev_version = self._registry.get("current")

        # Swap model
        shutil.copy2(str(src), str(cfg.YOLO_MODEL_PATH))

        # Update statuses
        for v in self._registry["versions"]:
            if v["status"] == "deployed":
                v["status"] = "retired"
        entry["status"] = "deployed"

        self._registry["current"] = target_version_id

        # Record rollback
        self._registry.setdefault("deployment_history", []).append({
            "version_id": target_version_id,
            "previous_version": prev_version,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "confirmed_by": confirmed_by,
            "action": "rollback",
        })

        self._save()

        # Sync known defects list based on target version classes
        self._sync_known_defects(entry)

        # Rebuild FAISS index with rolled-back model's class set
        faiss_result = self._rebuild_faiss(allowed_classes=entry.get("classes"))

        log.info(
            f"Rolled back from {prev_version} → {target_version_id} "
            f"(FAISS: {faiss_result.get('vectors', 'skipped')})"
        )

        return {
            "success": True,
            "version_id": target_version_id,
            "previous_version": prev_version,
            "faiss_rebuilt": faiss_result.get("success", False),
            "faiss_vectors": faiss_result.get("vectors", 0),
        }

    def _sync_known_defects(self, entry: dict) -> None:
        """Sync known_defects.json with the deployed/rolled-back model classes."""
        try:
            from src.features.known_defects_registry import load_registry, _save_registry
            reg = load_registry()
            initial_classes = ["inclusion", "oil_spot", "punching_hole", "silk_spot", "water_spot", "welding_line"]
            model_classes = entry.get("classes", [])
            combined = sorted(list(set(initial_classes) | set(model_classes)))
            reg["defect_classes"] = combined
            reg["last_updated"] = datetime.now(timezone.utc).isoformat()
            reg["version"] += 1
            _save_registry(reg)
            log.info(f"Synced known defects registry to classes: {combined}")
        except Exception as e:
            log.warning(f"Failed to sync known defects registry: {e}")

    def set_current(self, version_id: str) -> bool:
        """Set the current active model version (simple alias for deploy)."""
        result = self.deploy_version(version_id)
        return result.get("success", False)

    def list_versions(self) -> list[dict]:
        """Return all registered model versions, newest first."""
        versions = list(self._registry.get("versions", []))
        versions.reverse()
        return versions

    def get_current(self) -> str | None:
        """Get the current active version ID."""
        return self._registry.get("current")

    def get_current_entry(self) -> dict | None:
        """Get the full entry for the current active version."""
        current = self.get_current()
        if current:
            return self._find_version(current)
        return None

    def get_deployment_history(self) -> list[dict]:
        """Get the full deployment history."""
        history = list(self._registry.get("deployment_history", []))
        history.reverse()
        return history

    def _find_version(self, version_id: str) -> dict | None:
        """Find a version entry by ID."""
        for v in self._registry.get("versions", []):
            if v["version_id"] == version_id:
                return v
        return None

    def _rebuild_faiss(self, allowed_classes: list[str] | None = None) -> dict:
        """Rebuild FAISS index after model swap."""
        try:
            from src.features.faiss_index import FAISSIndexManager
            manager = FAISSIndexManager()
            vectors = manager.setup(allowed_classes=allowed_classes)
            return {"success": True, "vectors": vectors}
        except Exception as e:
            log.warning(f"FAISS rebuild failed (non-fatal): {e}")
            return {"success": False, "vectors": 0, "error": str(e)}
