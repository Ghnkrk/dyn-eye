"""
Node 4 — Feature Extraction (DINOv2)

Extracts 384-d feature vectors from all crops in batch mode using DINOv2.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger
from src.features.dinov2_extractor import DinoV2Extractor

log = get_logger("feature_extraction_node")


def feature_extraction_node(state: dict) -> dict:
    """
    LangGraph node: DINOv2 feature extraction.

    Reads:
        state["crop_paths"]

    Writes:
        state["feature_vectors"]  (numpy array N×384)
        state["feature_crop_paths"]
    """
    crop_paths = state.get("crop_paths", [])

    if not crop_paths:
        log.warning("No crops to extract features from")
        return {
            "feature_vectors": np.empty((0, cfg.FEATURE_DIM)),
            "feature_crop_paths": [],
        }

    extractor = DinoV2Extractor()
    features = extractor.extract_batch(crop_paths)

    log.info(f"Feature extraction complete: {features.shape[0]} vectors of dim {features.shape[1]}")

    return {
        "feature_vectors": features,
        "feature_crop_paths": crop_paths,
    }
