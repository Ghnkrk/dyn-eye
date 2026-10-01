"""Test fixtures: every test runs against a throw-away data dir, never the real one."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect every path the backend reads/writes into tmp_path."""
    paths = {
        "DATA_DIR": tmp_path,
        "INPUT_IMAGES_DIR": tmp_path / "input_images",
        "CROPS_DIR": tmp_path / "crops",
        "CLUSTERS_DIR": tmp_path / "clusters",
        "YOLO_DATASET_DIR": tmp_path / "yolo",
        "KNOWN_DEFECTS_DIR": tmp_path / "known_crops",
        "KNOWN_REPLAY_JSON": tmp_path / "known_replay.json",
        "CLUSTER_REGISTRY_PATH": tmp_path / "clusters" / "cluster_registry.json",
        "CLUSTER_TUNING_CACHE_PATH": tmp_path / "clusters" / "tuning_cache.json",
    }
    for name, p in paths.items():
        monkeypatch.setattr(cfg, name, p)
        if name.endswith("_DIR"):
            p.mkdir(parents=True, exist_ok=True)
    return cfg
