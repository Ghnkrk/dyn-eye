"""
LangGraph pipeline state schema.

This is the shared state object that flows through all nodes
of the discovery pipeline graph.
"""
from __future__ import annotations
from typing import Any, TypedDict
import operator


class PipelineState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph discovery pipeline.
    Keys accumulate as nodes execute — each node reads what it needs
    and writes its outputs back into state.
    """

    # ── Pipeline Mode ────────────────────────────────────────
    use_cache: bool                      # If True, skip YOLO/VLM/Crop and reuse existing data

    # ── Node 1: YOLO Inference ───────────────────────────────
    # Input: user-provided at graph invocation
    input_images_dir: str                # Path to directory with mixed images
    known_defect_names: list[str]        # List of known defect class names

    # Output
    all_image_paths: list[str]           # All image paths found
    known_image_paths: list[str]         # Images classified as known defects
    unknown_image_paths: list[str]       # Images classified as unknown
    unknown_defects_json: str            # Path to saved JSON file
    yolo_raw_results: list[dict]         # Raw YOLO result metadata per image

    # ── Node 2: VLM Annotation ───────────────────────────────
    vlm_annotations: list[dict]          # Per-image VLM bbox annotations
    # Each dict: {image_path, findings: [{box_2d, physical_traits}], ...}

    # ── Node 3: Crop Extraction ──────────────────────────────
    crop_paths: list[str]                # Paths to all extracted crops
    crop_metadata: list[dict]            # Mapping: crop → source image + bbox

    # ── Node 4: Feature Extraction ───────────────────────────
    feature_vectors: Any                 # numpy array (N, 384)
    feature_crop_paths: list[str]        # Parallel list of crop paths

    # ── Node 5: FAISS Search ─────────────────────────────────
    faiss_distances: list[float]         # Distance to nearest known cluster
    faiss_is_novel: list[bool]           # True if distance > threshold
    novel_indices: list[int]             # Indices of novel crops
    total_detected_crops: int            # Total crop count before novelty filter

    # ── Node 6: HDBSCAN Clustering ───────────────────────────
    cluster_labels: list[int]            # Cluster ID per novel crop (-1 = noise)
    cluster_folders: dict[int, str]      # cluster_id → folder path
    num_clusters: int
    cluster_registry: dict               # Loaded cluster fingerprint registry
    cluster_tuned_params: dict           # Best (min_cluster_size, min_samples) from DBCV grid search
    unassigned_crop_paths: list[str]     # Noise crops too far from any centroid
    dbcv_score: float                    # DBCV validity index of final clustering
    registry_hits: int                   # Clusters matched existing fingerprints
    registry_total: int                  # Total clusters (for hit rate)

    # ── Node 6.5: Dataset Context (Part 2) ───────────────────
    dataset_context: dict                # Run context snapshot for dynamic prompt generation
    vlm_system_prompt: str               # Groq-generated Gemini system prompt

    # ── Node 7: Manifest Save + ICC Metrics ─────────────────
    vlm_cluster_results: list[dict]      # Per-cluster ICC + label results (Part 3)

    # ── Metadata ─────────────────────────────────────────────
    run_id: str                          # Short UUID for this pipeline run
    errors: list[str]                    # Accumulated errors across nodes
