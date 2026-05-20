"""
FAISS Index Manager

Two modes:
  - setup: Build a FAISS index from known-class defect crops
  - infer: Query new crops against the known index to determine novelty
"""
from __future__ import annotations

import json
import numpy as np
import faiss
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.utils import get_logger
from src.features.dinov2_extractor import DinoV2Extractor

log = get_logger("faiss_index")


class FAISSIndexManager:
    """
    Manages the FAISS index for known-defect feature matching.

    - setup(): Build and save an index from known-class crop images.
    - infer(): Query individual/batch features to get distance to
      nearest known cluster → determine novelty.
    """

    def __init__(self, extractor: DinoV2Extractor | None = None):
        self.extractor = extractor or DinoV2Extractor()
        self.index: faiss.IndexFlatL2 | None = None
        self.labels: list[str] = []  # Class label per vector in index
        self.index_path = str(cfg.FAISS_INDEX_FILE)
        self.labels_path = str(cfg.FAISS_LABELS_FILE)

    # ── Setup Mode ───────────────────────────────────────────
    def setup(
        self,
        known_crops_dir: str | Path | None = None,
        class_subdirs: bool = True,
    ) -> int:
        """
        Build the FAISS index from known defect crops.

        If class_subdirs=True, expects structure:
            known_crops_dir/
                scratch/
                    img1.jpg
                    img2.jpg
                dent/
                    img3.jpg
                ...

        Returns the number of vectors indexed.
        """
        known_dir = Path(known_crops_dir or cfg.KNOWN_DEFECTS_DIR)
        log.info(f"Building FAISS index from {known_dir}")

        all_paths: list[str] = []
        all_labels: list[str] = []

        if class_subdirs:
            for subdir in sorted(known_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                class_name = subdir.name
                from src.utils.io_helpers import list_images
                imgs = list_images(subdir)
                all_paths.extend([str(p) for p in imgs])
                all_labels.extend([class_name] * len(imgs))
                log.info(f"  Class '{class_name}': {len(imgs)} images")
        else:
            from src.utils.io_helpers import list_images
            imgs = list_images(known_dir)
            all_paths = [str(p) for p in imgs]
            all_labels = ["known"] * len(imgs)

        if not all_paths:
            log.warning("No known-class images found. FAISS index will be empty.")
            self.index = faiss.IndexFlatL2(cfg.FEATURE_DIM)
            self.labels = []
            self._save()
            return 0

        # Extract features
        features = self.extractor.extract_batch(all_paths)

        # Build index
        self.index = faiss.IndexFlatL2(cfg.FEATURE_DIM)
        self.index.add(features.astype(np.float32))
        self.labels = all_labels

        self._save()

        # Auto-register discovered class names in the known-defects registry
        unique_classes = sorted(set(all_labels))
        if unique_classes:
            try:
                from src.features.known_defects_registry import register_defects
                register_defects(unique_classes, source="faiss_setup")
            except Exception as e:
                log.warning(f"Could not register classes in registry: {e}")

        log.info(f"FAISS index built: {self.index.ntotal} vectors, {len(set(all_labels))} classes")
        return self.index.ntotal

    # ── Infer Mode ───────────────────────────────────────────
    def load(self) -> None:
        """Load a previously saved FAISS index."""
        if not Path(self.index_path).exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run setup() first with known defect crops."
            )
        self.index = faiss.read_index(self.index_path)
        self.labels = json.loads(
            Path(self.labels_path).read_text(encoding="utf-8")
        )
        log.info(f"FAISS index loaded: {self.index.ntotal} vectors")

    def infer_batch(
        self,
        features: np.ndarray,
        threshold: float = cfg.FAISS_NOVELTY_THRESHOLD,
    ) -> tuple[list[float], list[bool]]:
        """
        Query a batch of feature vectors against the known index.

        Returns:
            distances: List of L2 distances to nearest known vector
            is_novel: List of bools (True = novel / unknown)
        """
        if self.index is None:
            self.load()

        if self.index.ntotal == 0:
            log.warning("FAISS index is empty — treating ALL as novel")
            return [float("inf")] * len(features), [True] * len(features)

        D, I = self.index.search(features.astype(np.float32), k=1)
        distances = D[:, 0].tolist()
        is_novel = [d > threshold for d in distances]

        novel_count = sum(is_novel)
        log.info(
            f"FAISS infer: {novel_count}/{len(features)} crops are novel "
            f"(threshold={threshold:.1f})"
        )

        return distances, is_novel

    def get_nearest_label(self, feature: np.ndarray) -> tuple[str, float]:
        """Get the nearest known class label and its distance."""
        if self.index is None:
            self.load()
        if self.index.ntotal == 0:
            return "unknown", float("inf")

        D, I = self.index.search(feature.reshape(1, -1).astype(np.float32), k=1)
        idx = int(I[0, 0])
        dist = float(D[0, 0])
        label = self.labels[idx] if idx < len(self.labels) else "unknown"
        return label, dist

    # ── Persistence ──────────────────────────────────────────
    def _save(self) -> None:
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        Path(self.labels_path).write_text(
            json.dumps(self.labels, indent=2), encoding="utf-8"
        )
        log.info(f"FAISS index saved to {self.index_path}")
