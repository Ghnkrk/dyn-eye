"""
Upsampling page backend: prepare the training set, preview/generate class-
balancing augmentations, and list originals next to their augmented copies.

All work happens on cfg.YOLO_DATASET_DIR, the same folder the fine-tune trains
on. The fine-tune re-exports that folder and regenerates the augmentations
from the saved settings, so what is reviewed here is what gets trained on.
"""
from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path

import config as cfg
from src.retraining.tools import upsample as up
from src.utils import get_logger

log = get_logger("upsampling_service")

_lock = threading.Lock()  # one prepare/generate/clear at a time


class Busy(RuntimeError):
    """Another dataset operation (or a fine-tune) is using the dataset folder."""


def _guard():
    from src.retraining import agent
    if agent._run_lock.locked():
        raise Busy("A fine-tune is running; wait for it to finish before changing the training set.")
    if not _lock.acquire(blocking=False):
        raise Busy("Another upsampling operation is already running.")


def _dataset() -> Path:
    return Path(cfg.YOLO_DATASET_DIR)


def _boxes(label_path: Path, names: list[str]) -> list[dict]:
    if not label_path.exists():
        return []
    return [
        {"class_id": c, "class_name": names[c] if c < len(names) else str(c), "xywhn": list(b)}
        for c, b in up._parse_labels(label_path)
    ]


def _origin_kind(origin: str) -> str:
    return origin.split(":", 1)[0] if origin else "unknown"


# ── Status / balance ─────────────────────────────────────────

def status() -> dict:
    ds = _dataset()
    names = up._class_names(ds)
    settings = up.load_settings()
    samples = up._train_samples(ds)
    sources = up.load_sources(ds)
    origins = Counter(_origin_kind(sources.get(s["image"].stem, "unknown")) for s in samples)
    val_dir = ds / "images" / "val"
    val_images = [p for p in val_dir.iterdir() if p.suffix.lower() in up.IMAGE_SUFFIXES] if val_dir.exists() else []

    before = up.count_instances(ds)
    current = up.count_instances(ds, include_augmented=True)
    balance = up.class_balance(before, settings["imbalance_threshold"])
    manifest = up.load_manifest(ds)
    return {
        "dataset_ready": bool(names) and bool(samples),
        "class_names": names,
        "train_images": len(samples),
        "val_images": len(val_images),
        "origins": dict(origins),
        "counts": before,
        "counts_with_augmented": current,
        "augmented_images": len(manifest.get("images", [])),
        "last_run": {k: v for k, v in manifest.items() if k != "images"} or None,
        "balance": balance,
        "recommendation": _recommendation(before, balance, settings["imbalance_threshold"]),
        "settings": settings,
        "defaults": up.default_settings(),
    }


def _recommendation(counts: dict[str, int], balance: dict, threshold: float) -> dict:
    minority = balance.get("minority_classes", [])
    if not minority:
        return {"recommend": False, "reason": "Classes are reasonably balanced; upsampling is not needed."}
    largest = balance.get("largest_class")
    parts = ", ".join(f"{c} ({counts.get(c, 0)})" for c in minority)
    return {
        "recommend": True,
        "reason": (
            f"{parts} {'has' if len(minority) == 1 else 'have'} at most 1/{threshold:g} of the boxes of "
            f"{largest} ({counts.get(largest, 0)}), a {balance.get('imbalance_ratio')}× imbalance. "
            "Upsampling them reduces the model's bias towards the larger classes."
        ),
    }


# ── Actions ──────────────────────────────────────────────────

def prepare_dataset() -> dict:
    """Rebuild the training set from named clusters + replay + external datasets."""
    from src.features.known_defects_registry import get_known_defect_names
    from src.pipeline.orchestrator import AnnotationMapper, ManifestPoller

    _guard()
    try:
        up.remove_augmented(_dataset())
        named = ManifestPoller().check_named_clusters()
        if not named:
            raise ValueError("No named clusters yet. Name at least one cluster on the Clusters page first.")
        result = AnnotationMapper().map_cluster_labels_to_yolo(
            cluster_labels=named, label_names=list(get_known_defect_names())
        )
        log.info(f"Upsampling page prepared the training set: {result}")
        return result
    finally:
        _lock.release()


def preview(settings: dict) -> dict:
    s = up.validate_settings(settings)
    return up.upsample_dataset(
        target_ratio=s["target_ratio"], max_factor=s["max_factor"],
        imbalance_threshold=s["imbalance_threshold"], transforms=s["transforms"],
        seed=s["seed"], dataset_dir=_dataset(), dry_run=True,
    )


def generate(settings: dict) -> dict:
    s = up.validate_settings(settings)
    _guard()
    try:
        result = up.upsample_dataset(
            target_ratio=s["target_ratio"], max_factor=s["max_factor"],
            imbalance_threshold=s["imbalance_threshold"], transforms=s["transforms"],
            seed=s["seed"], dataset_dir=_dataset(),
        )
        # Generating means "use this for the next fine-tune"
        up.save_settings({**s, "enabled": True})
        return result
    finally:
        _lock.release()


def clear() -> dict:
    _guard()
    try:
        return {"removed_files": up.remove_augmented(_dataset())}
    finally:
        _lock.release()


# ── Viewer ───────────────────────────────────────────────────

def groups() -> dict:
    """Each source image with its augmented copies and every box, per class."""
    ds = _dataset()
    names = up._class_names(ds)
    labels = ds / "labels" / "train"
    manifest = up.load_manifest(ds)
    by_source: dict[str, dict] = {}
    for e in manifest.get("images", []):
        src = e["source_image"]
        g = by_source.get(src)
        if g is None:
            g = by_source[src] = {
                "source_image": src,
                "origin": e.get("origin", "unknown"),
                "boxes": _boxes(labels / f"{Path(src).stem}.txt", names),
                "for_classes": [],
                "augmented": [],
            }
        if e["for_class"] not in g["for_classes"]:
            g["for_classes"].append(e["for_class"])
        g["augmented"].append({
            "image": e["image"],
            "geometric": e["geometric"],
            "photometric": e.get("photometric", []),
            "for_class": e["for_class"],
            "boxes": _boxes(labels / f"{Path(e['image']).stem}.txt", names),
        })

    classes: dict[str, dict] = {}
    for g in by_source.values():
        for c in g["for_classes"]:
            info = classes.setdefault(c, {"class_name": c, "originals": 0, "augmented": 0})
            info["originals"] += 1
            info["augmented"] += sum(1 for a in g["augmented"] if a["for_class"] == c)
    counts = up.count_instances(ds)
    after = up.count_instances(ds, include_augmented=True)
    for c, info in classes.items():
        info["boxes_before"] = counts.get(c, 0)
        info["boxes_after"] = after.get(c, 0)
    return {
        "classes": sorted(classes.values(), key=lambda c: c["class_name"]),
        "groups": sorted(by_source.values(), key=lambda g: g["source_image"]),
    }


def image_path(name: str) -> Path | None:
    """A train/val image of the dataset, by bare file name only."""
    if not name or Path(name).name != name or name.startswith("."):
        return None
    for split in ("train", "val"):
        p = _dataset() / "images" / split / name
        if p.is_file() and p.suffix.lower() in up.IMAGE_SUFFIXES:
            return p
    return None
