"""
Autonomous Orchestrator

Runs as a background daemon that:
  1. Monitors cluster folders for new images
  2. Watches for human-assigned defect names in the dashboard manifest
  3. Maps cluster names back as labels on original full images (YOLO format)
  4. Uses LLM to decide when retraining should be triggered
  5. Auto-triggers the retraining pipeline

The only human interaction is naming clusters in the DYN-EYE dashboard.
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
from src.utils.llm import chat_json

log = get_logger("orchestrator")


# ── Cluster Watcher ──────────────────────────────────────────

class ClusterWatcher:
    """
    Watch data/clusters/ for new images and detect changes.
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


# ── Dashboard Manifest Poller ────────────────────────────────

class ManifestPoller:
    """
    Poll the cluster manifest for human-assigned defect names.
    When clusters are named in the dashboard, this picks them up.
    """

    def check_named_clusters(self) -> dict[str, str]:
        """
        Check which clusters have been named in the dashboard manifest.
        Returns {cluster_name: defect_name} for named clusters.
        """
        manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
        if not manifest_path.exists():
            return {}

        try:
            manifest = load_json(manifest_path)
            named = {}
            for cluster_name, entry in manifest.get("clusters", {}).items():
                defect_name = entry.get("defect_name")
                if defect_name:
                    named[cluster_name] = defect_name
            return named
        except Exception as e:
            log.warning(f"Manifest poll failed: {e}")
            return {}


# ── Annotation Back-Mapper ───────────────────────────────────

class AnnotationMapper:
    """
    When clusters are named in the dashboard, maps those names
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

    @staticmethod
    def _add_replay_samples(samples: dict[str, dict], label_to_idx: dict[str, int]) -> int:
        """
        Add known-class images (pseudo-labelled by the current model during the
        discovery run) to `samples` so fine-tuning keeps the old classes.
        Deterministic subset, capped at cfg.REPLAY_MAX_IMAGES.
        """
        import hashlib
        if not cfg.KNOWN_REPLAY_JSON.exists():
            return 0
        try:
            replay = load_json(cfg.KNOWN_REPLAY_JSON)
        except Exception as e:
            log.warning(f"Could not read replay file: {e}")
            return 0

        name_to_idx = {n.lower(): i for n, i in label_to_idx.items()}
        candidates = sorted(replay, key=lambda r: hashlib.md5(r["image_path"].encode()).hexdigest())
        added = 0
        for rec in candidates:
            if added >= cfg.REPLAY_MAX_IMAGES:
                break
            src = Path(rec["image_path"])
            if not src.exists():
                fallback = cfg.INPUT_IMAGES_DIR / src.name
                if not fallback.exists():
                    continue
                src = fallback
            if src.stem in samples:      # never override a newly named label
                continue
            lines = []
            for d in rec["detections"]:
                idx = name_to_idx.get(d["class_name"].lower())
                if idx is None:
                    continue
                cx, cy, w, h = d["bbox_xywhn"]
                lines.append(f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if lines:
                samples[src.stem] = {"src": src, "lines": lines}
                added += 1
        if added:
            log.info(f"Replay: added {added} known-class images as pseudo-labelled samples")
        return added

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
        import hashlib
        import shutil

        crop_map = self.load_crop_mapping()
        label_names = list(label_names)

        yolo_dir = cfg.YOLO_DATASET_DIR
        for split in ("train", "val"):
            (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Build label index
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}

        # ── Pass 1: collect de-duplicated labels per source image ──
        # Label files are REWRITTEN (not appended) from this in-memory view, so
        # re-running the mapper is idempotent instead of piling up duplicate boxes.
        samples: dict[str, dict] = {}
        total_mapped = 0

        for cluster_name, label_name in cluster_labels.items():
            cluster_dir = cfg.CLUSTERS_DIR / cluster_name
            if not cluster_dir.exists():
                continue

            label_idx = label_to_idx.get(label_name)
            if label_idx is None:
                label_names.append(label_name)
                label_idx = len(label_names) - 1
                label_to_idx[label_name] = label_idx

            for crop_path in list_images(cluster_dir):
                source_info = crop_map.get(crop_path.stem, {})
                source_image = source_info.get("source_image", "")
                bbox = source_info.get("bbox_normalized")  # [x_center, y_center, w, h]
                if not source_image or not bbox:
                    continue

                src_img = Path(source_image)
                if not src_img.exists():
                    # Fallback to the current project input images directory
                    fallback_img = cfg.INPUT_IMAGES_DIR / src_img.name
                    if fallback_img.exists():
                        src_img = fallback_img
                if not src_img.exists():
                    continue

                line = f"{label_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                entry = samples.setdefault(src_img.stem, {"src": src_img, "lines": []})
                if line not in entry["lines"]:
                    entry["lines"].append(line)
                    total_mapped += 1

        # ── Replay: known images + current-model detections as pseudo-labels ──
        replay_images = self._add_replay_samples(samples, label_to_idx)

        # ── Pass 2: deterministic train/val split (~20% val) ──
        # Hash-based so an image keeps its split as the dataset grows.
        def _bucket(stem: str) -> int:
            return int(hashlib.md5(stem.encode()).hexdigest(), 16) % 5

        stems = sorted(samples)
        val_stems: set[str] = set()
        if len(stems) >= 5:
            val_stems = {s for s in stems if _bucket(s) == 0}
            if not val_stems:  # guarantee a non-empty val set
                val_stems = {min(stems, key=lambda s: hashlib.md5(s.encode()).hexdigest())}

        total_images = 0
        for stem, entry in samples.items():
            split, other = ("val", "train") if stem in val_stems else ("train", "val")
            src_img = entry["src"]

            # Drop stale copies from the other split (image moved between splits)
            (yolo_dir / "images" / other / src_img.name).unlink(missing_ok=True)
            (yolo_dir / "labels" / other / f"{stem}.txt").unlink(missing_ok=True)

            dst_img = yolo_dir / "images" / split / src_img.name
            if not dst_img.exists():
                shutil.copy2(str(src_img), str(dst_img))
                total_images += 1
            (yolo_dir / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(entry["lines"]) + "\n", encoding="utf-8"
            )

        # Write data.yaml (tiny datasets have no val split → validate on train)
        data_yaml = {
            "path": str(yolo_dir),
            "train": "images/train",
            "val": "images/val" if val_stems else "images/train",
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
            "replay_images": replay_images,
            "label_names": label_names,
        }


# ── LLM Retraining Decision Agent ────────────────────────

class RetrainingDecisionAgent:
    """
    Uses the configured LLM (cfg.LLM_MODEL_ID via src.utils.llm) to analyze
    pipeline state and decide whether retraining should be triggered.
    """

    def should_retrain(
        self,
        cluster_stats: dict[str, int],
        named_clusters: dict[str, str],
        current_model_classes: list[str],
        last_training_date: str | None = None,
    ) -> tuple[bool, str]:
        """
        Ask the LLM whether we should trigger retraining now.

        Returns:
            (should_retrain: bool, reasoning: str)
            A reason starting with "Decision API error" means the call failed
            (transient) and the caller should retry later.
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
            result = chat_json(prompt, temperature=0.1)
            should = bool(result.get("retrain", False))
            reason = result.get("reason", "No reason provided")

            LogStream.emit(
                f"Retraining decision: {'YES' if should else 'NO'} — {reason}",
                level="info",
                source="retrain_agent",
            )
            return should, reason

        except Exception as e:
            log.error(f"LLM decision call failed: {e}")
            LogStream.emit(f"LLM decision failed: {e}", level="error", source="retrain_agent")
            return False, f"Decision API error: {e}"


# ── Main Orchestrator Loop ───────────────────────────────────

class AutonomousOrchestrator:
    """
    Main background loop that ties everything together.

    Flow:
      1. Discovery pipeline runs (triggered once)
      2. Clusters appear in data/clusters/
      3. Human names clusters in the DYN-EYE dashboard (ONLY human step)
      4. Orchestrator detects named clusters from manifest
      5. Maps labels back to original images in YOLO format
      6. LLM agent decides if retraining should happen
      7. Retraining pipeline runs autonomously
    """

    def __init__(self):
        self.watcher = ClusterWatcher()
        self.manifest_poller = ManifestPoller()
        self.mapper = AnnotationMapper()
        self._decision_agent: RetrainingDecisionAgent | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._poll_interval = 30  # seconds
        self._retry_after = 0.0   # epoch seconds; backoff after a failed decision call
        self._state_path = cfg.DATA_DIR / "orchestrator_state.json"

    @property
    def decision_agent(self) -> "RetrainingDecisionAgent":
        """Built lazily so importing this module never requires a LLM key."""
        if self._decision_agent is None:
            self._decision_agent = RetrainingDecisionAgent()
        return self._decision_agent

    # ── Persisted state (survives restarts, prevents re-processing) ──
    def _load_state(self) -> dict:
        try:
            return load_json(self._state_path)
        except Exception:
            return {}

    def _save_state(self, **updates) -> None:
        state = self._load_state()
        state.update(updates)
        save_json(state, self._state_path)

    @staticmethod
    def _signature(named: dict[str, str], cluster_stats: dict[str, int]) -> str:
        """Fingerprint of (names, cluster sizes): changes only when there is something new to act on."""
        import hashlib
        canon = json.dumps({"named": named, "stats": cluster_stats}, sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()[:16]

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
        # 1. Check for new cluster images
        new_images = self.watcher.get_new_images()
        if new_images:
            total_new = sum(len(v) for v in new_images.values())
            LogStream.emit(
                f"Detected {total_new} new images across {len(new_images)} clusters",
                level="info",
                source="cluster_watcher",
            )

        # 2. Check if clusters have been named in the dashboard manifest
        named = self.manifest_poller.check_named_clusters()
        if not named:
            return

        # Nothing new since the last time we acted on this exact state → no-op.
        cluster_stats = self.watcher.get_all_clusters()
        signature = self._signature(named, cluster_stats)
        state = self._load_state()
        if state.get("processed_signature") == signature:
            return
        if time.time() < self._retry_after:
            return

        LogStream.emit(
            f"Found {len(named)} named clusters in manifest",
            level="step",
            source="manifest_poller",
        )

        # 3. Map labels back to original images (idempotent rewrite)
        from src.features.known_defects_registry import get_known_defect_names
        known_classes = get_known_defect_names()
        mapping_result = self.mapper.map_cluster_labels_to_yolo(
            cluster_labels=named,
            label_names=list(known_classes),
        )

        if mapping_result["total_mapped"] == 0:
            self._save_state(processed_signature=signature)
            return

        # 4. Ask LLM if we should retrain
        should, reason = self.decision_agent.should_retrain(
            cluster_stats=cluster_stats,
            named_clusters=named,
            current_model_classes=known_classes,
            last_training_date=state.get("last_trained_at"),
        )
        if reason.startswith("Decision API error"):
            # Transient failure: don't mark processed, but back off 5 min
            self._retry_after = time.time() + 300
            return

        if should:
            LogStream.emit(
                "Auto-triggering retraining pipeline",
                level="step",
                source="orchestrator",
            )
            if not self._run_retraining(project_id):
                # Busy or failed to start: leave unprocessed so we retry later
                self._retry_after = time.time() + 300
                return

        # Decision made (retrained or declined): don't re-ask until state changes
        self._save_state(processed_signature=signature)

    def _run_retraining(self, project_id: int | None) -> bool:
        """Trigger the retraining pipeline. Returns False if it could not run."""
        try:
            from src.retraining.agent import run_retraining_pipeline
            result = run_retraining_pipeline(project_id=project_id or -1)
            success = result.get("training_result", {}).get("success", False)
            if success:
                self._save_state(last_trained_at=datetime.now(timezone.utc).isoformat())
            LogStream.emit(
                f"Retraining {'succeeded' if success else 'failed'}",
                level="info" if success else "error",
                source="retrain_pipeline",
            )
            return True
        except Exception as e:
            LogStream.emit(f"Retraining failed: {e}", level="error", source="retrain_pipeline")
            return False


# ── Module-level singleton ───────────────────────────────────
orchestrator = AutonomousOrchestrator()
