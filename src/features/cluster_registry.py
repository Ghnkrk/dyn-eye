"""
Persistent cluster fingerprint registry.

Cluster folder names (cluster_000, ...) are re-numbered on every run, so they
are NOT a stable identity.  A fingerprint (L2-normalised centroid + id) is:

  - hdbscan_cluster  matches each new cluster to a fingerprint (one-to-one)
                     or creates a new one
  - dashboard naming stores the human defect name ON the fingerprint
  - manifest_save    gives a cluster the label of its fingerprint, so a group
                     that was named once is pre-labelled in every later run

File: cfg.CLUSTER_REGISTRY_PATH  →  {"fingerprints": [{id, centroid, label, ...}]}
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config as cfg
from src.utils.logger import get_logger

log = get_logger("cluster_registry")

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load() -> dict:
    """Load the registry; guarantees every fingerprint has an id."""
    path = cfg.CLUSTER_REGISTRY_PATH
    reg: dict = {"fingerprints": []}
    if path.exists():
        try:
            reg = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Cluster registry corrupted, starting fresh: {e}")
            reg = {"fingerprints": []}
    reg.setdefault("fingerprints", [])
    for fp in reg["fingerprints"]:  # migrate pre-id registries
        fp.setdefault("id", uuid.uuid4().hex[:12])
    return reg


def save(registry: dict) -> None:
    """Atomically persist the registry (write temp file, then replace)."""
    path = cfg.CLUSTER_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    tmp.replace(path)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two L2-normalised vectors."""
    return 1.0 - float(np.dot(a, b))


def match_centroids(
    centroids: dict[int, np.ndarray],
    registry: dict,
    threshold: float,
) -> dict[int, dict]:
    """
    One-to-one assignment of new clusters to existing fingerprints.

    Greedy by ascending distance, each fingerprint used at most once, so two
    different clusters can never inherit the same fingerprint (and label).
    Returns {cluster_id: fingerprint_dict} for matched clusters only.
    """
    fps = registry.get("fingerprints", [])
    pairs: list[tuple[float, int, int]] = []
    for cid, centroid in centroids.items():
        for j, fp in enumerate(fps):
            stored = np.asarray(fp["centroid"], dtype=np.float32)
            d = cosine_distance(centroid, stored)
            if d < threshold:
                pairs.append((d, cid, j))

    pairs.sort()
    matched: dict[int, dict] = {}
    used_fp: set[int] = set()
    for _, cid, j in pairs:
        if cid in matched or j in used_fp:
            continue
        matched[cid] = fps[j]
        used_fp.add(j)
    return matched


def new_fingerprint(centroid: np.ndarray, cluster_size: int) -> dict:
    now = _now()
    return {
        "id": uuid.uuid4().hex[:12],
        "centroid": centroid.tolist(),
        "label": None,          # filled when a human names the cluster
        "confidence": None,
        "first_seen": now,
        "last_seen": now,
        "match_count": 1,
        "cluster_size": int(cluster_size),
    }


def set_label(fingerprint_id: str, label: str | None) -> bool:
    """Store (or clear, with None/"") the human defect name on a fingerprint."""
    if not fingerprint_id:
        return False
    with _lock:
        reg = load()
        for fp in reg["fingerprints"]:
            if fp["id"] == fingerprint_id:
                fp["label"] = (label or "").strip() or None
                fp["labeled_at"] = _now() if fp["label"] else None
                save(reg)
                log.info(f"Fingerprint {fingerprint_id} labelled '{fp['label']}'")
                return True
    log.warning(f"Fingerprint {fingerprint_id} not found; label not stored")
    return False


def get_label(fingerprint_id: str) -> str | None:
    for fp in load()["fingerprints"]:
        if fp["id"] == fingerprint_id:
            return fp.get("label")
    return None
