"""
Node 5 — FAISS Search

Queries the extracted crop features against the pre-built known-defect
FAISS index.  Crops whose nearest-neighbor distance exceeds the
threshold are flagged as novel (truly unknown).
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger
from src.features.faiss_index import FAISSIndexManager

log = get_logger("faiss_search_node")


def faiss_search_node(state: dict) -> dict:
    """
    LangGraph node: FAISS novelty search.

    Reads:
        state["feature_vectors"]
        state["feature_crop_paths"]

    Writes:
        state["faiss_distances"]
        state["faiss_is_novel"]
        state["novel_indices"]
    """
    features = state.get("feature_vectors")
    crop_paths = state.get("feature_crop_paths", [])

    if features is None or len(features) == 0:
        log.warning("No features to query FAISS")
        return {
            "faiss_distances": [],
            "faiss_is_novel": [],
            "novel_indices": [],
        }

    manager = FAISSIndexManager()

    # Try to load existing index; if not found, treat all as novel
    try:
        manager.load()
    except FileNotFoundError:
        log.warning(
            "No FAISS index found. Run FAISS setup first with known defect crops. "
            "Treating ALL crops as novel for now."
        )
        n = len(features)
        return {
            "faiss_distances": [float("inf")] * n,
            "faiss_is_novel": [True] * n,
            "novel_indices": list(range(n)),
        }

    distances, is_novel = manager.infer_batch(features)
    novel_indices = [i for i, v in enumerate(is_novel) if v]

    log.info(
        f"FAISS search: {len(novel_indices)} novel out of {len(features)} crops"
    )

    return {
        "faiss_distances": distances,
        "faiss_is_novel": is_novel,
        "novel_indices": novel_indices,
    }
