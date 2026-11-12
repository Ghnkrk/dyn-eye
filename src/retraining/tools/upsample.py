"""
Class-balancing upsampler for the YOLO training split.

Adds augmented copies of training images that contain under-represented
classes, using label-preserving OpenCV transforms: flips (and optionally 90°
rotations), plus brightness/contrast, gamma, blur and noise. Box coordinates
are transformed exactly, never estimated.

Only the train split is touched — val stays real, so validation metrics are
not inflated by near-duplicates of training images.

Augmented files are named "<stem>__aug<k>.<ext>", so each run can remove the
previous run's copies before the dataset is exported and versioned.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("upsample")

AUG_MARKER = "__aug"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

Box = tuple[float, float, float, float]  # YOLO normalized (x_center, y_center, w, h)


# ── Geometric transforms (image op + exact box mapping) ──────

def _hflip(img):
    return cv2.flip(img, 1)


def _vflip(img):
    return cv2.flip(img, 0)


def _rot180(img):
    return cv2.rotate(img, cv2.ROTATE_180)


def _rot90cw(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def _rot90ccw(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


FLIP_TRANSFORMS = [
    ("hflip", _hflip, lambda x, y, w, h: (1 - x, y, w, h)),
    ("vflip", _vflip, lambda x, y, w, h: (x, 1 - y, w, h)),
    ("rot180", _rot180, lambda x, y, w, h: (1 - x, 1 - y, w, h)),
]
ROTATE_TRANSFORMS = [
    ("rot90cw", _rot90cw, lambda x, y, w, h: (1 - y, x, h, w)),
    ("rot90ccw", _rot90ccw, lambda x, y, w, h: (y, 1 - x, h, w)),
]
IDENTITY = ("none", lambda img: img, lambda x, y, w, h: (x, y, w, h))


def geometric_transforms(allow_rotate90: bool) -> list:
    """Order matters: copy k of an image uses transform k mod len, so the first
    copies of any image always differ geometrically from the original."""
    return FLIP_TRANSFORMS + (ROTATE_TRANSFORMS if allow_rotate90 else []) + [IDENTITY]


# ── Photometric transforms (labels unchanged) ────────────────

def _photometric(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    applied = []

    alpha = rng.uniform(0.85, 1.15)
    beta = rng.uniform(-18, 18)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    applied.append("contrast")

    if rng.random() < 0.5:
        gamma = rng.uniform(0.8, 1.25)
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, lut)
        applied.append("gamma")

    roll = rng.random()
    if roll < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        applied.append("blur")
    elif roll < 0.7:
        noise = rng.normal(0, rng.uniform(3, 8), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        applied.append("noise")

    return img, applied


# ── Dataset I/O ──────────────────────────────────────────────

def _read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def _write_image(path: Path, img) -> bool:
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))
    return ok


def _parse_labels(label_path: Path) -> list[tuple[int, Box]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), tuple(float(v) for v in parts[1:])))
    return boxes


def _format_labels(boxes: list[tuple[int, Box]]) -> str:
    lines = []
    for cls, box in boxes:
        x, y, w, h = (min(1.0, max(0.0, v)) for v in box)
        lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + "\n"


def is_augmented(path: Path) -> bool:
    return AUG_MARKER in path.stem


def _class_names(dataset_dir: Path) -> list[str]:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        return []
    return list((yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}).get("names", []))


def _train_samples(dataset_dir: Path) -> list[dict]:
    """Original (non-augmented) train images that have a non-empty label file."""
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    samples = []
    if not images_dir.exists():
        return samples
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_SUFFIXES or is_augmented(img):
            continue
        label = labels_dir / f"{img.stem}.txt"
        if not label.exists():
            continue
        boxes = _parse_labels(label)
        if boxes:
            samples.append({"image": img, "boxes": boxes, "counts": Counter(c for c, _ in boxes)})
    return samples


def remove_augmented(dataset_dir: Path | None = None) -> int:
    """Delete every augmented image/label from previous runs. Returns files removed."""
    dataset_dir = Path(dataset_dir or cfg.YOLO_DATASET_DIR)
    removed = 0
    for sub in ("images", "labels"):
        for split in ("train", "val"):
            d = dataset_dir / sub / split
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.is_file() and is_augmented(f):
                    f.unlink()
                    removed += 1
    if removed:
        log.info(f"Removed {removed} augmented files from previous runs")
    return removed


def count_instances(dataset_dir: Path | None = None, include_augmented: bool = False) -> dict[str, int]:
    """Box count per class name in the train split."""
    dataset_dir = Path(dataset_dir or cfg.YOLO_DATASET_DIR)
    names = _class_names(dataset_dir)
    labels_dir = dataset_dir / "labels" / "train"
    counts: Counter = Counter()
    if labels_dir.exists():
        for f in labels_dir.glob("*.txt"):
            if not include_augmented and is_augmented(f):
                continue
            counts.update(c for c, _ in _parse_labels(f))
    return {(names[c] if c < len(names) else str(c)): n for c, n in sorted(counts.items())}


# ── Balancing ────────────────────────────────────────────────

def upsample_dataset(
    target_ratio: float | None = None,
    max_factor: float | None = None,
    allow_rotate90: bool | None = None,
    seed: int | None = None,
    dataset_dir: Path | None = None,
) -> dict:
    """
    Bring every class in the train split up to `target_ratio` × the largest
    class's box count, by writing augmented copies of images containing it.

    A class never grows past `max_factor` × its original count, so a class with
    3 boxes is not cloned 30 times to match a class with 90 (that would just
    teach the model to memorise those 3 defects).
    """
    target_ratio = cfg.UPSAMPLE_TARGET_RATIO if target_ratio is None else target_ratio
    max_factor = cfg.UPSAMPLE_MAX_FACTOR if max_factor is None else max_factor
    allow_rotate90 = cfg.UPSAMPLE_ALLOW_ROTATE90 if allow_rotate90 is None else allow_rotate90
    seed = cfg.UPSAMPLE_SEED if seed is None else seed
    if not 0 < target_ratio <= 1:
        raise ValueError(f"target_ratio must be in (0, 1], got {target_ratio}")
    if max_factor < 1:
        raise ValueError(f"max_factor must be >= 1, got {max_factor}")

    dataset_dir = Path(dataset_dir or cfg.YOLO_DATASET_DIR)
    remove_augmented(dataset_dir)

    names = _class_names(dataset_dir)
    samples = _train_samples(dataset_dir)
    original: Counter = Counter()
    for s in samples:
        original.update(s["counts"])

    def named(counter) -> dict[str, int]:
        return {(names[c] if c < len(names) else str(c)): int(n) for c, n in sorted(counter.items())}

    result = {
        "target_ratio": target_ratio,
        "max_factor": max_factor,
        "allow_rotate90": allow_rotate90,
        "before": named(original),
        "after": named(original),
        "generated_images": 0,
        "classes_upsampled": [],
        "capped_classes": [],
    }
    if len(original) < 2:
        result["note"] = "Need at least 2 classes in the train split to balance"
        return result

    wanted = math.ceil(max(original.values()) * target_ratio)
    target = {c: min(wanted, math.floor(n * max_factor)) for c, n in original.items()}
    result["capped_classes"] = sorted(n for n, c in zip(named(original), sorted(original)) if target[c] < wanted)

    geoms = geometric_transforms(allow_rotate90)
    rng = np.random.default_rng(seed)
    counts = Counter(original)
    copies: Counter = Counter()
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"

    # One copy per distinct geometric variant; the identity transform (photometric
    # change only) is too close to the original to be worth a second copy.
    per_image_cap = len(geoms) - 1

    for cls in sorted(original, key=lambda c: (original[c], c)):
        if counts[cls] >= target[cls]:
            continue
        # Purest images first, each reused up to per_image_cap times: an image
        # that is mostly this class adds the fewest boxes of other (often
        # already large) classes that share images with it.
        candidates = sorted(
            (s for s in samples if cls in s["counts"]),
            key=lambda s: (-s["counts"][cls] / sum(s["counts"].values()), s["image"].name),
        )
        while counts[cls] < target[cls]:
            available = [s for s in candidates if copies[s["image"].name] < per_image_cap]
            if not available:
                result.setdefault("limited_classes", []).append(names[cls] if cls < len(names) else str(cls))
                break
            sample = available[0]
            k = copies[sample["image"].name]
            copies[sample["image"].name] += 1

            img = _read_image(sample["image"])
            geom_name, geom_img, geom_box = geoms[k % len(geoms)]
            stem = f"{sample['image'].stem}{AUG_MARKER}{k:03d}"
            written = False
            if img is not None:
                out, photo = _photometric(geom_img(img), rng)
                written = _write_image(images_dir / f"{stem}{sample['image'].suffix}", out)
            if not written:
                log.warning(f"Could not read/write {sample['image'].name}; leaving it out of upsampling")
                candidates.remove(sample)
                continue

            boxes = [(c, geom_box(*b)) for c, b in sample["boxes"]]
            (labels_dir / f"{stem}.txt").write_text(_format_labels(boxes), encoding="utf-8")

            counts.update(sample["counts"])
            result["generated_images"] += 1
            log.debug(f"{stem}: {geom_name} + {'+'.join(photo)}")

        if counts[cls] > original[cls]:
            result["classes_upsampled"].append(names[cls] if cls < len(names) else str(cls))

    result["after"] = named(counts)
    log.info(
        f"Upsampling generated {result['generated_images']} images: "
        f"{result['before']} -> {result['after']}"
    )
    return result


def class_balance(class_counts: dict[str, int]) -> dict:
    """Imbalance summary used by the advisor (no dataset access)."""
    counts = {k: v for k, v in class_counts.items() if v > 0}
    if len(counts) < 2:
        return {"imbalance_ratio": 1.0, "minority_classes": [], "suggest_upsampling": False,
                "max_factor": cfg.UPSAMPLE_MAX_FACTOR}
    largest, smallest = max(counts.values()), min(counts.values())
    ratio = round(largest / smallest, 2)
    minority = sorted(k for k, v in counts.items() if v * cfg.UPSAMPLE_IMBALANCE_THRESHOLD <= largest)
    return {
        "imbalance_ratio": ratio,
        "largest_class": max(counts, key=counts.get),
        "minority_classes": minority,
        "suggest_upsampling": bool(minority),
        "max_factor": cfg.UPSAMPLE_MAX_FACTOR,
    }
