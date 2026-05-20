"""
Known Defects Registry

Persistent, JSON-backed registry of all known defect class names.
This is the **single source of truth** for what the system considers
"known" — every other module reads from here instead of from a
hardcoded list.

The registry grows automatically:
    - After a YOLO model is fine-tuned and deployed, the new model's
      class names are merged into the registry.
    - When Label Studio exports produce a class map, those names are
      merged too.

File format (data/known_defects.json):
    {
        "defect_classes": ["scratch", "dent", "crack", ...],
        "version": 3,
        "last_updated": "2026-05-19T12:00:00Z",
        "history": [
            {"added": ["scratch", "dent"], "source": "yolo_model", "timestamp": "..."},
            ...
        ]
    }
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("known_defects_registry")

# ── File location ────────────────────────────────────────────
REGISTRY_FILE = cfg.DATA_DIR / "known_defects.json"

# Thread-safe writes
_lock = Lock()


def _empty_registry() -> dict:
    return {
        "defect_classes": [],
        "version": 0,
        "last_updated": None,
        "history": [],
    }


def load_registry() -> dict:
    """Load the known-defects registry from disk."""
    if not REGISTRY_FILE.exists():
        return _empty_registry()
    try:
        data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        # Ensure required keys
        data.setdefault("defect_classes", [])
        data.setdefault("version", 0)
        data.setdefault("last_updated", None)
        data.setdefault("history", [])
        return data
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to load registry, starting fresh: {e}")
        return _empty_registry()


def _save_registry(data: dict) -> None:
    """Persist the registry to disk."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )


def get_known_defect_names() -> list[str]:
    """
    Return the current list of known defect class names.

    This is the function every pipeline node should call instead of
    reading ``cfg.KNOWN_DEFECT_NAMES`` directly.
    """
    reg = load_registry()
    return list(reg["defect_classes"])


def register_defects(
    new_names: list[str],
    source: str = "unknown",
) -> list[str]:
    """
    Merge *new_names* into the registry.

    Args:
        new_names:  Class names to add (duplicates are silently ignored).
        source:     Where the names came from (e.g. "yolo_model",
                    "label_studio_export", "manual").

    Returns:
        List of names that were **actually added** (i.e. were new).
    """
    if not new_names:
        return []

    with _lock:
        reg = load_registry()
        existing = set(reg["defect_classes"])
        added = [n for n in new_names if n not in existing]

        if not added:
            log.info(f"No new defect classes to register (source={source})")
            return []

        reg["defect_classes"] = sorted(existing | set(added))
        reg["version"] += 1
        reg["last_updated"] = datetime.now(timezone.utc).isoformat()
        reg["history"].append({
            "added": added,
            "source": source,
            "timestamp": reg["last_updated"],
        })

        _save_registry(reg)
        log.info(
            f"Registered {len(added)} new defect classes from {source}: {added}  "
            f"(total known: {len(reg['defect_classes'])})"
        )
        return added


def register_from_yolo_model(model_path: str | Path | None = None) -> list[str]:
    """
    Load a YOLO model and register all its class names.

    This is called automatically after a fine-tuned model is deployed.
    """
    from ultralytics import YOLO

    path = str(model_path or cfg.YOLO_MODEL_PATH)
    if not Path(path).exists():
        log.warning(f"YOLO model not found at {path}, skipping class registration")
        return []

    model = YOLO(path)
    class_names = list(model.names.values()) if model.names else []

    if not class_names:
        log.warning("YOLO model has no class names")
        return []

    log.info(f"YOLO model classes ({len(class_names)}): {class_names}")
    return register_defects(class_names, source="yolo_model")


def register_from_data_yaml(yaml_path: str | Path | None = None) -> list[str]:
    """
    Read class names from a YOLO data.yaml and register them.

    This is useful after Label Studio export produces a new data.yaml.
    """
    import yaml

    path = Path(yaml_path or cfg.YOLO_DATASET_DIR / "data.yaml")
    if not path.exists():
        log.warning(f"data.yaml not found at {path}")
        return []

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", [])

    if isinstance(names, dict):
        names = list(names.values())

    return register_defects(names, source="label_studio_export")
