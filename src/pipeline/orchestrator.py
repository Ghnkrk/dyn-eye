"""
Autonomous Orchestrator

Runs as a background daemon that:
  1. Monitors cluster folders for new images (hot-sync to Label Studio)
  2. Polls Label Studio for completed cluster naming
  3. Maps cluster names back as labels on original full images (YOLO format)
  4. Uses Gemini to decide when retraining should be triggered
  5. Auto-triggers the retraining pipeline

The only human interaction is naming clusters in Label Studio.
Everything else is fully autonomous.
"""
from __future__ import annotations

import json
import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.utils import get_logger, save_json, load_json, LogStream
from src.utils.io_helpers import list_images

log = get_logger("orchestrator")


# ── Cluster Watcher ──────────────────────────────────────────

class ClusterWatcher:
    """
    Watch data/clusters/ for new images and hot-sync them
    to Label Studio for naming.
    """

    def __init__(self):
        self._known_files: set[str] = set()
        self._scan_clusters()

    def _scan_clusters(self) -> dict[str, list[str]]:
        """Scan cluster folders and return {cluster_name: [image_paths]}."""
        clusters = {}
        if not cfg.CLUSTERS_DIR.exists():
            return clusters
        for d in sorted(cfg.CLUSTERS_DIR.iterdir()):
            if d.is_dir():
                imgs = list_images(d)
                clusters[d.name] = [str(p) for p in imgs]
                for p in imgs:
                    self._known_files.add(str(p))
        return clusters

    def get_new_images(self) -> dict[str, list[str]]:
        """Return only new images since last check."""
        current = self._scan_clusters()
        new_images = {}
        for cluster_name, paths in current.items():
            new = [p for p in paths if p not in self._known_files]
            if new:
                new_images[cluster_name] = new
                for p in new:
                    self._known_files.add(p)
        return new_images

    def get_all_clusters(self) -> dict[str, int]:
        """Return {cluster_name: image_count}."""
        clusters = self._scan_clusters()
        return {k: len(v) for k, v in clusters.items()}


# ── Label Studio Poller ──────────────────────────────────────

class LabelStudioPoller:
    """
    Poll Label Studio for completed annotations (named clusters).
    When clusters are named, map labels back to original images.
    """

    def __init__(self):
        self._api_url = cfg.LABEL_STUDIO_URL
        self._api_key = cfg.LABEL_STUDIO_API_KEY

    def _headers(self):
        return {"Authorization": f"Token {self._api_key}"}

    def get_project_labels(self, project_id: int) -> dict[str, str]:
        """
        Fetch completed annotations from Label Studio project.
        Returns mapping of {image_filename: assigned_label}.
        """
        import requests
        try:
            resp = requests.get(
                f"{self._api_url}/api/projects/{project_id}/tasks",
                headers=self._headers(),
                params={"page_size": 500},
                timeout=10,
            )
            resp.raise_for_status()
            tasks = resp.json().get("results", resp.json()) if isinstance(resp.json(), dict) else resp.json()

            label_map = {}
            for task in tasks:
                annotations = task.get("annotations", [])
                if not annotations:
                    continue
                # Take the latest annotation
                ann = annotations[-1]
                results = ann.get("result", [])
                for r in results:
                    label_name = r.get("value", {}).get("choices", [""])[0] \
                        if r.get("type") == "choices" \
                        else r.get("value", {}).get("rectanglelabels", [""])[0] \
                        if r.get("type") == "rectanglelabels" \
                        else ""
                    if label_name:
                        # Extract filename from task data
                        data = task.get("data", {})
                        image_url = data.get("image", "")
                        filename = Path(image_url).stem if image_url else ""
                        if filename:
                            label_map[filename] = label_name
            return label_map
        except Exception as e:
            log.warning(f"Label Studio poll failed: {e}")
            return {}

    def check_clusters_named(self, project_id: int) -> dict[str, str]:
        """
        Check which cluster folders have been named in Label Studio.
        Returns {old_cluster_name: new_label_name}.
        """
        import requests
        try:
            resp = requests.get(
                f"{self._api_url}/api/projects/{project_id}",
                headers=self._headers(),
                timeout=10,
            )
            resp.raise_for_status()
            project = resp.json()

            # Get task annotations
            label_map = self.get_project_labels(project_id)
            return label_map
        except Exception as e:
            log.warning(f"Cluster naming check failed: {e}")
            return {}


# ── Annotation Back-Mapper ───────────────────────────────────

class AnnotationMapper:
    """
    When clusters are named in Label Studio, maps those names
    back as YOLO-format labels on the ORIGINAL full images
    (not the cropped ones) using the VLM bbox data.
    """

    def __init__(self):
        self._vlm_annotations_file = cfg.DATA_DIR / "vlm_annotations.json"
        self._crop_mapping_file = cfg.DATA_DIR / "crop_to_source.json"

    def load_vlm_annotations(self) -> dict:
        """Load VLM annotation data (image_name -> list of bboxes)."""
        if self._vlm_annotations_file.exists():
            return load_json(self._vlm_annotations_file)
        return {}

    def load_crop_mapping(self) -> dict:
        """Load crop -> source image mapping."""
        if self._crop_mapping_file.exists():
            return load_json(self._crop_mapping_file)
        return {}

    def map_cluster_labels_to_yolo(
        self,
        cluster_labels: dict[str, str],
        label_names: list[str],
    ) -> dict:
        """
        Map cluster names to YOLO labels on original full images.

        Args:
            cluster_labels: {cluster_folder_name: assigned_label}
            label_names: ordered list of all label names for data.yaml

        Returns:
            dict with stats about mapped annotations
        """
        vlm_data = self.load_vlm_annotations()
        crop_map = self.load_crop_mapping()

        yolo_dir = cfg.YOLO_DATASET_DIR
        images_dir = yolo_dir / "images" / "train"
        labels_dir = yolo_dir / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Build label index
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}

        total_mapped = 0
        total_images = 0

        # Walk through cluster directories
        for cluster_name, label_name in cluster_labels.items():
            cluster_dir = cfg.CLUSTERS_DIR / cluster_name
            if not cluster_dir.exists():
                continue

            label_idx = label_to_idx.get(label_name)
            if label_idx is None:
                label_names.append(label_name)
                label_idx = len(label_names) - 1
                label_to_idx[label_name] = label_idx

            # For each crop in the cluster, find the source image
            for crop_path in list_images(cluster_dir):
                crop_name = crop_path.stem
                source_info = crop_map.get(crop_name, {})
                source_image = source_info.get("source_image", "")
                bbox = source_info.get("bbox_normalized")  # [x_center, y_center, w, h]

                if not source_image or not bbox:
                    continue

                # Copy source image to YOLO dataset
                src_img = Path(source_image)
                if src_img.exists():
                    import shutil
                    dst_img = images_dir / src_img.name
                    if not dst_img.exists():
                        shutil.copy2(str(src_img), str(dst_img))
                        total_images += 1

                    # Write/append YOLO label
                    label_file = labels_dir / (src_img.stem + ".txt")
                    line = f"{label_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                    with open(label_file, "a") as f:
                        f.write(line)
                    total_mapped += 1

        # Write data.yaml
        data_yaml = {
            "path": str(yolo_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": len(label_names),
            "names": label_names,
        }
        import yaml
        (yolo_dir / "data.yaml").write_text(
            yaml.dump(data_yaml, default_flow_style=False),
            encoding="utf-8",
        )

        LogStream.emit(
            f"Mapped {total_mapped} annotations across {total_images} images "
            f"with {len(label_names)} classes",
            level="info",
            source="annotation_mapper",
        )

        return {
            "total_mapped": total_mapped,
            "total_images": total_images,
            "label_names": label_names,
        }


# ── Gemini Retraining Decision Agent ────────────────────────

class RetrainingDecisionAgent:
    """
    Uses Gemini API to analyze pipeline state and decide
    whether retraining should be triggered.
    """

    def __init__(self):
        from google import genai
        self._client = genai.Client(api_key=cfg.GEMINI_API_KEY)
        self._model = "gemini-2.0-flash"  # Free tier

    def should_retrain(
        self,
        cluster_stats: dict[str, int],
        named_clusters: dict[str, str],
        current_model_classes: list[str],
        last_training_date: str | None = None,
    ) -> tuple[bool, str]:
        """
        Ask Gemini whether we should trigger retraining now.

        Returns:
            (should_retrain: bool, reasoning: str)
        """
        prompt = f"""You are an autonomous ML pipeline orchestrator for an industrial defect detection system.

Current state:
- Discovered clusters: {json.dumps(cluster_stats)}
- Named clusters (human-labeled): {json.dumps(named_clusters)}
- Current model knows classes: {json.dumps(current_model_classes)}
- Last training date: {last_training_date or 'Never'}
- Current time: {datetime.now().isoformat()}

Decision criteria:
1. Are there enough NAMED clusters (at least 1 new class with 5+ images)?
2. Are the new class names NOT already in the current model?
3. Has enough new data accumulated since last training?

Respond ONLY with a JSON object:
{{"retrain": true/false, "reason": "brief explanation"}}
"""
        try:
            from google.genai import types
            response = self._client.models.generate_content(
                model=self._model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            result = json.loads(response.text)
            should = result.get("retrain", False)
            reason = result.get("reason", "No reason provided")

            LogStream.emit(
                f"Retraining decision: {'YES' if should else 'NO'} — {reason}",
                level="info",
                source="retrain_agent",
            )
            return should, reason

        except Exception as e:
            log.error(f"Gemini decision call failed: {e}")
            LogStream.emit(f"Gemini decision failed: {e}", level="error", source="retrain_agent")
            return False, f"Decision API error: {e}"


# ── Main Orchestrator Loop ───────────────────────────────────

class AutonomousOrchestrator:
    """
    Main background loop that ties everything together.

    Flow:
      1. Discovery pipeline runs (triggered once)
      2. Clusters appear in data/clusters/
      3. Clusters are hot-synced to Label Studio
      4. Human names clusters in Label Studio (ONLY human step)
      5. Orchestrator detects named clusters
      6. Maps labels back to original images in YOLO format
      7. Gemini agent decides if retraining should happen
      8. Retraining pipeline runs autonomously
    """

    def __init__(self):
        self.watcher = ClusterWatcher()
        self.poller = LabelStudioPoller()
        self.mapper = AnnotationMapper()
        self.decision_agent = RetrainingDecisionAgent()
        self._running = False
        self._thread: threading.Thread | None = None
        self._poll_interval = 30  # seconds

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, project_id: int | None = None):
        """Start the autonomous orchestration loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            args=(project_id,),
            daemon=True,
        )
        self._thread.start()
        LogStream.emit("Autonomous orchestrator started", level="info", source="orchestrator")

    def stop(self):
        """Stop the orchestration loop."""
        self._running = False
        LogStream.emit("Autonomous orchestrator stopped", level="info", source="orchestrator")

    def _loop(self, project_id: int | None):
        """Main orchestration loop."""
        while self._running:
            try:
                self._tick(project_id)
            except Exception as e:
                log.error(f"Orchestrator tick failed: {e}")
                LogStream.emit(f"Orchestrator error: {e}", level="error", source="orchestrator")
            time.sleep(self._poll_interval)

    def _tick(self, project_id: int | None):
        """Single orchestration cycle."""
        # 1. Check for new cluster images → hot-sync
        new_images = self.watcher.get_new_images()
        if new_images:
            total_new = sum(len(v) for v in new_images.values())
            LogStream.emit(
                f"Hot-sync: {total_new} new images across {len(new_images)} clusters",
                level="info",
                source="cluster_watcher",
            )
            # TODO: sync new images to Label Studio if project_id available

        # 2. Check if clusters have been named in Label Studio
        if project_id:
            named = self.poller.check_clusters_named(project_id)
            if named:
                LogStream.emit(
                    f"Found {len(named)} named clusters in Label Studio",
                    level="step",
                    source="ls_poller",
                )

                # 3. Map labels back to original images
                cluster_stats = self.watcher.get_all_clusters()
                known_classes = cfg.KNOWN_DEFECT_NAMES.copy()
                mapping_result = self.mapper.map_cluster_labels_to_yolo(
                    cluster_labels=named,
                    label_names=known_classes,
                )

                if mapping_result["total_mapped"] > 0:
                    # 4. Ask Gemini if we should retrain
                    should, reason = self.decision_agent.should_retrain(
                        cluster_stats=cluster_stats,
                        named_clusters=named,
                        current_model_classes=cfg.KNOWN_DEFECT_NAMES,
                    )

                    if should:
                        LogStream.emit(
                            "Auto-triggering retraining pipeline",
                            level="step",
                            source="orchestrator",
                        )
                        self._run_retraining(project_id)

    def _run_retraining(self, project_id: int):
        """Trigger the retraining pipeline."""
        try:
            from src.retraining.agent import run_retraining_pipeline
            result = run_retraining_pipeline(project_id=project_id)
            success = result.get("training_result", {}).get("success", False)
            LogStream.emit(
                f"Retraining {'succeeded' if success else 'failed'}",
                level="info" if success else "error",
                source="retrain_pipeline",
            )
        except Exception as e:
            LogStream.emit(f"Retraining failed: {e}", level="error", source="retrain_pipeline")


# ── Module-level singleton ───────────────────────────────────
orchestrator = AutonomousOrchestrator()
