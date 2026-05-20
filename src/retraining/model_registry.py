"""
Model Registry — Local Model Versioning

Tracks model versions, metadata, and deployment history
locally before syncing with MLflow.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.utils import get_logger, save_json

log = get_logger("model_registry")


class ModelRegistry:
    """Local model version registry."""

    def __init__(self):
        self.registry_file = cfg.MODELS_DIR / "registry.json"
        self._registry = self._load()

    def _load(self) -> dict:
        if self.registry_file.exists():
            return json.loads(self.registry_file.read_text(encoding="utf-8"))
        return {"versions": [], "current": None}

    def _save(self) -> None:
        save_json(self._registry, self.registry_file)

    def register_version(
        self,
        model_path: str,
        metrics: dict | None = None,
        source: str = "fine-tuning",
        notes: str = "",
    ) -> dict:
        """Register a new model version."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v{len(self._registry['versions']) + 1}_{ts}"

        # Copy to versions directory
        src = Path(model_path)
        dst = cfg.MODEL_VERSIONS_DIR / f"best_{version_id}.pt"
        shutil.copy2(str(src), str(dst))

        entry = {
            "version_id": version_id,
            "path": str(dst),
            "original_path": model_path,
            "timestamp": ts,
            "metrics": metrics or {},
            "source": source,
            "notes": notes,
            "size_mb": round(src.stat().st_size / (1024 * 1024), 2),
        }

        self._registry["versions"].append(entry)
        self._save()

        log.info(f"Registered model version: {version_id}")
        return entry

    def set_current(self, version_id: str) -> bool:
        """Set the current active model version."""
        for v in self._registry["versions"]:
            if v["version_id"] == version_id:
                src = Path(v["path"])
                if src.exists():
                    shutil.copy2(str(src), str(cfg.YOLO_MODEL_PATH))
                    self._registry["current"] = version_id
                    self._save()
                    log.info(f"Active model set to: {version_id}")
                    return True
        return False

    def list_versions(self) -> list[dict]:
        return self._registry["versions"]

    def get_current(self) -> str | None:
        return self._registry.get("current")
