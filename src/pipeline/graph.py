"""
LangGraph Discovery Pipeline — Main Graph Definition

Chains all 7 nodes of the unknown defect discovery pipeline:
  1. YOLO Inference → filter known/unknown
  2. VLM Annotation → bbox detection (one-by-one)
  3. Crop Extraction → cut defect regions
  4. Feature Extraction → DINOv2 embeddings
  5. FAISS Search → novelty detection
  6. HDBSCAN Clustering → group unknowns
  7. Label Studio Sync → upload for naming
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from langgraph.graph import StateGraph, START, END

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.pipeline.state import PipelineState
from src.utils import get_logger, LogStream
from src.utils.metrics import MetricsTracker
from src.features.known_defects_registry import get_known_defect_names

# Import all nodes
from src.pipeline.nodes.yolo_inference import yolo_inference_node
from src.pipeline.nodes.vlm_annotation import vlm_annotation_node
from src.pipeline.nodes.crop_extraction import crop_extraction_node
from src.pipeline.nodes.feature_extraction import feature_extraction_node
from src.pipeline.nodes.faiss_search import faiss_search_node
from src.pipeline.nodes.hdbscan_cluster import hdbscan_cluster_node
from src.pipeline.nodes.label_studio_sync import label_studio_sync_node

log = get_logger("pipeline.graph")


# ── Node display names ──────────────────────────────────────
NODE_LABELS = {
    "yolo_inference": "YOLO Inference",
    "vlm_annotation": "VLM Annotation",
    "crop_extraction": "Crop Extraction",
    "feature_extraction": "DINOv2 Features",
    "faiss_search": "FAISS Novelty Search",
    "hdbscan_cluster": "HDBSCAN Clustering",
    "label_studio_sync": "Label Studio Sync",
}


# ── Wrapped nodes with metrics + live log streaming ─────────

_metrics: MetricsTracker | None = None


def _wrap_node(name: str, fn):
    """Wrap a pipeline node function with metrics tracking and live log streaming."""
    def wrapper(state: dict) -> dict:
        global _metrics
        label = NODE_LABELS.get(name, name)

        # Emit start event
        LogStream.emit(
            f"Starting {label}...",
            level="step",
            source=name,
        )
        if _metrics:
            _metrics.start_step(name)

        try:
            result = fn(state)
            items = 0
            # Heuristic: count items processed based on known output keys
            for key in ["unknown_image_paths", "vlm_annotations", "crop_paths",
                        "feature_crop_paths", "novel_indices", "cluster_folders",
                        "label_studio_task_ids"]:
                if key in result:
                    val = result[key]
                    if isinstance(val, (list, dict)):
                        items = len(val)
                        break

            if _metrics:
                _metrics.end_step(name, items_processed=items)

            LogStream.emit(
                f"{label} complete — {items} items processed",
                level="info",
                source=name,
                data={"items_processed": items},
            )
            return result
        except Exception as e:
            if _metrics:
                _metrics.fail_step(name, str(e))
            LogStream.emit(
                f"{label} FAILED: {e}",
                level="error",
                source=name,
            )
            log.error(f"Node '{name}' failed: {e}")
            return {"errors": state.get("errors", []) + [f"{name}: {e}"]}
    return wrapper


def build_discovery_graph() -> StateGraph:
    """
    Build and compile the LangGraph discovery pipeline.

    Returns a compiled graph ready for invocation.
    """
    graph = StateGraph(PipelineState)

    # Add nodes (wrapped with metrics + log streaming)
    graph.add_node("yolo_inference", _wrap_node("yolo_inference", yolo_inference_node))
    graph.add_node("vlm_annotation", _wrap_node("vlm_annotation", vlm_annotation_node))
    graph.add_node("crop_extraction", _wrap_node("crop_extraction", crop_extraction_node))
    graph.add_node("feature_extraction", _wrap_node("feature_extraction", feature_extraction_node))
    graph.add_node("faiss_search", _wrap_node("faiss_search", faiss_search_node))
    graph.add_node("hdbscan_cluster", _wrap_node("hdbscan_cluster", hdbscan_cluster_node))
    graph.add_node("label_studio_sync", _wrap_node("label_studio_sync", label_studio_sync_node))

    # Define edges (linear chain)
    graph.add_edge(START, "yolo_inference")
    graph.add_edge("yolo_inference", "vlm_annotation")
    graph.add_edge("vlm_annotation", "crop_extraction")
    graph.add_edge("crop_extraction", "feature_extraction")
    graph.add_edge("feature_extraction", "faiss_search")
    graph.add_edge("faiss_search", "hdbscan_cluster")
    graph.add_edge("hdbscan_cluster", "label_studio_sync")
    graph.add_edge("label_studio_sync", END)

    return graph.compile()


def run_discovery_pipeline(
    input_images_dir: str | None = None,
    confidence_threshold: float | None = None,
    yolo_model_path: str | None = None,
    from_vlm_cache: bool = False,
) -> dict:
    """
    Execute the full discovery pipeline.

    Known defect names are loaded automatically from the registry
    (data/known_defects.json) — no manual input needed.
    """
    global _metrics

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _metrics = MetricsTracker(run_id=run_id)

    # Known defects auto-populated from registry
    known_names = get_known_defect_names()

    initial_state: PipelineState = {
        "run_id": run_id,
        "input_images_dir": input_images_dir or str(cfg.INPUT_IMAGES_DIR),
        "known_defect_names": known_names,
        "confidence_threshold": confidence_threshold or cfg.YOLO_CONFIDENCE_THRESHOLD,
        "errors": [],
    }

    if yolo_model_path:
        initial_state["yolo_model_path"] = yolo_model_path

    # Cache skip integration
    if from_vlm_cache:
        cache_path = cfg.DATA_DIR / "vlm_cache.json"
        if cache_path.exists():
            try:
                import json
                with open(cache_path, "r") as f:
                    cached_ann = json.load(f)
                initial_state["vlm_annotations"] = cached_ann
                log.info(f"Loaded {len(cached_ann)} cached VLM annotations from {cache_path}. YOLO and active VLM steps will be bypassed.")
            except Exception as e:
                log.error(f"Failed to load VLM cache: {e}. Executing full active pipeline run.")
        else:
            log.warning(f"VLM cache not found at {cache_path}. Executing full active pipeline run.")

    LogStream.emit(
        f"Discovery pipeline started (run: {run_id})",
        level="step",
        source="pipeline",
        data={
            "run_id": run_id,
            "images_dir": initial_state["input_images_dir"],
            "known_defects": initial_state["known_defect_names"],
        },
    )

    log.info(f"=== Starting discovery pipeline run: {run_id} ===")

    graph = build_discovery_graph()
    final_state = graph.invoke(initial_state)

    LogStream.emit(
        f"Discovery pipeline complete (run: {run_id})",
        level="step",
        source="pipeline",
        data={
            "unknown_count": len(final_state.get("unknown_image_paths", [])),
            "crop_count": len(final_state.get("crop_paths", [])),
            "cluster_count": final_state.get("num_clusters", 0),
        },
    )

    log.info(f"=== Pipeline run {run_id} complete ===")

    # Attach metrics snapshot to final state
    final_state["_metrics"] = _metrics.snapshot()

    return final_state


# ── CLI entry point ──────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the discovery pipeline")
    parser.add_argument("--images-dir", default=None, help="Input images directory")
    parser.add_argument("--confidence", type=float, default=None, help="YOLO confidence threshold")
    parser.add_argument("--model", default=None, help="YOLO model path")
    parser.add_argument("--from-vlm-cache", action="store_true", help="Bypass VLM and run from cached VLM output")
    args = parser.parse_args()

    result = run_discovery_pipeline(
        input_images_dir=args.images_dir,
        confidence_threshold=args.confidence,
        yolo_model_path=args.model,
        from_vlm_cache=args.from_vlm_cache,
    )

    print(f"\nPipeline complete. Run ID: {result.get('run_id')}")
    print(f"  Known defects (from registry): {result.get('known_defect_names', [])}")
    print(f"  Unknown images: {len(result.get('unknown_image_paths', []))}")
    print(f"  Crops extracted: {len(result.get('crop_paths', []))}")
    print(f"  Clusters found: {result.get('num_clusters', 0)}")
    print(f"  Errors: {result.get('errors', [])}")
