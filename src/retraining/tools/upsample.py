"""
Class-balancing upsampler for the YOLO training split.

Adds augmented copies of full training images that contain under-represented
classes, using label-preserving OpenCV transforms: flips (and optionally 90°
rotations), plus brightness/contrast, gamma, blur and noise. Box coordinates
are transformed exactly, never estimated.

Only the train split is touched — val stays real, so validation metrics are
not inflated by near-duplicates of training images.

Augmented files are named "<stem>__aug<k>.<ext>", so each run can remove the
previous run's copies before the dataset is exported and versioned. Every
generated image is recorded in <dataset>/upsample_manifest.json (source image,
where that image came from, transforms) for the Upsampling page viewer.

Settings chosen on the Upsampling page are saved to
data/upsampling_settings.json and reused by the fine-tune; with the same
dataset and seed the fine-tune regenerates exactly the images that were
reviewed.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger, save_json

log = get_logger("upsample")

AUG_MARKER = "__aug"
MANIFEST_NAME = "upsample_manifest.json"
SOURCES_NAME = "sources.json"
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

PHOTOMETRIC_KEYS = ("contrast", "gamma", "blur", "noise")


def default_transforms() -> dict[str, bool]:
    return {
        "hflip": True, "vflip": True, "rot180": True, "rot90": bool(cfg.UPSAMPLE_ALLOW_ROTATE90),
        "contrast": True, "gamma": True, "blur": True, "noise": True,
    }


def geometric_transforms(allow_rotate90: bool, transforms: dict[str, bool] | None = None) -> list:
    """Order matters: copy k of an image uses transform k mod len, so the first
    copies of any image always differ geometrically from the original."""
    t = transforms or {}
    flips = [g for g in FLIP_TRANSFORMS if t.get(g[0], True)]
    return flips + (ROTATE_TRANSFORMS if allow_rotate90 else []) + [IDENTITY]


# ── Photometric transforms (labels unchanged) ────────────────

def _photometric(img: np.ndarray, rng: np.random.Generator,
                 enabled: dict[str, bool] | None = None) -> tuple[np.ndarray, list[str]]:
    on = enabled or {}
    applied = []
    # Every random draw happens whether or not its step is enabled, so turning
    # one transform off does not change the parameters of the others.
    alpha, beta = rng.uniform(0.85, 1.15), rng.uniform(-18, 18)
    if on.get("contrast", True):
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        applied.append("contrast")

    gamma_roll, gamma = rng.random(), rng.uniform(0.8, 1.25)
    if on.get("gamma", True) and gamma_roll < 0.5:
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, lut)
        applied.append("gamma")

    roll, sigma = rng.random(), rng.uniform(3, 8)
    if roll < 0.3 and on.get("blur", True):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        applied.append("blur")
    elif 0.3 <= roll < 0.7 and on.get("noise", True):
        noise = rng.normal(0, sigma, img.shape)
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


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_manifest(dataset_dir: Path | None = None) -> dict:
    """What the last (real) upsampling run generated, or {} if none."""
    return _read_json(Path(dataset_dir or cfg.YOLO_DATASET_DIR) / MANIFEST_NAME)


def load_sources(dataset_dir: Path | None = None) -> dict[str, str]:
    """{image stem: "cluster" | "replay" | "external:<name>"}, written by the dataset export."""
    return _read_json(Path(dataset_dir or cfg.YOLO_DATASET_DIR) / SOURCES_NAME)


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
    (dataset_dir / MANIFEST_NAME).unlink(missing_ok=True)
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


# ── Saved settings (Upsampling page → fine-tune) ─────────────

def settings_path() -> Path:
    return cfg.DATA_DIR / "upsampling_settings.json"


def default_settings() -> dict:
    return {
        "enabled": False,
        "target_ratio": cfg.UPSAMPLE_TARGET_RATIO,
        "max_factor": cfg.UPSAMPLE_MAX_FACTOR,
        "imbalance_threshold": cfg.UPSAMPLE_IMBALANCE_THRESHOLD,
        "seed": cfg.UPSAMPLE_SEED,
        "transforms": default_transforms(),
    }


def validate_settings(settings: dict | None) -> dict:
    """Fill gaps with defaults and reject impossible values (ValueError)."""
    out = default_settings()
    for k, v in (settings or {}).items():
        if k == "transforms" and isinstance(v, dict):
            out["transforms"].update({t: bool(on) for t, on in v.items() if t in out["transforms"]})
        elif k in out and k != "transforms" and v is not None:
            out[k] = v
    out["enabled"] = bool(out["enabled"])
    out["target_ratio"] = float(out["target_ratio"])
    out["max_factor"] = float(out["max_factor"])
    out["imbalance_threshold"] = float(out["imbalance_threshold"])
    out["seed"] = int(out["seed"])
    if not 0 < out["target_ratio"] <= 1:
        raise ValueError(f"target_ratio must be in (0, 1], got {out['target_ratio']}")
    if out["max_factor"] < 1:
        raise ValueError(f"max_factor must be >= 1, got {out['max_factor']}")
    if out["imbalance_threshold"] < 1:
        raise ValueError(f"imbalance_threshold must be >= 1, got {out['imbalance_threshold']}")
    if not any(out["transforms"].values()):
        raise ValueError("Turn on at least one transform")
    return out


def load_settings() -> dict:
    try:
        return validate_settings(_read_json(settings_path()))
    except ValueError:
        return default_settings()


def save_settings(settings: dict) -> dict:
    clean = validate_settings(settings)
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return clean


# ── Balancing ────────────────────────────────────────────────

def upsample_dataset(
    target_ratio: float | None = None,
    max_factor: float | None = None,
    allow_rotate90: bool | None = None,
    seed: int | None = None,
    dataset_dir: Path | None = None,
    imbalance_threshold: float | None = None,
    transforms: dict[str, bool] | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Bring every minority class in the train split up to `target_ratio` × the
    largest class's box count, by writing augmented copies of the full
    training images that contain it.

    A class is a minority when it has at most 1/`imbalance_threshold` of the
    largest class's boxes; other classes are never targeted. A class never
    grows past `max_factor` × its original count, so a class with 3 boxes is
    not cloned 30 times to match a class with 90.

    dry_run=True returns the before/after counts without touching the dataset.
    """
    target_ratio = cfg.UPSAMPLE_TARGET_RATIO if target_ratio is None else target_ratio
    max_factor = cfg.UPSAMPLE_MAX_FACTOR if max_factor is None else max_factor
    seed = cfg.UPSAMPLE_SEED if seed is None else seed
    imbalance_threshold = cfg.UPSAMPLE_IMBALANCE_THRESHOLD if imbalance_threshold is None else imbalance_threshold
    tf = default_transforms()
    if transforms:
        tf.update({k: bool(v) for k, v in transforms.items() if k in tf})
    if allow_rotate90 is not None:
        tf["rot90"] = bool(allow_rotate90)
    if not 0 < target_ratio <= 1:
        raise ValueError(f"target_ratio must be in (0, 1], got {target_ratio}")
    if max_factor < 1:
        raise ValueError(f"max_factor must be >= 1, got {max_factor}")
    if imbalance_threshold < 1:
        raise ValueError(f"imbalance_threshold must be >= 1, got {imbalance_threshold}")
    if not any(tf.values()):
        raise ValueError("Turn on at least one transform")

    dataset_dir = Path(dataset_dir or cfg.YOLO_DATASET_DIR)
    if not dry_run:
        remove_augmented(dataset_dir)

    names = _class_names(dataset_dir)
    samples = _train_samples(dataset_dir)
    sources = load_sources(dataset_dir)
    original: Counter = Counter()
    for s in samples:
        original.update(s["counts"])

    def cname(c: int) -> str:
        return names[c] if c < len(names) else str(c)

    def named(counter) -> dict[str, int]:
        return {cname(c): int(n) for c, n in sorted(counter.items())}

    result = {
        "target_ratio": target_ratio,
        "max_factor": max_factor,
        "imbalance_threshold": imbalance_threshold,
        "allow_rotate90": tf["rot90"],
        "transforms": tf,
        "seed": seed,
        "dry_run": dry_run,
        "before": named(original),
        "after": named(original),
        "generated_images": 0,
        "minority_classes": [],
        "classes_upsampled": [],
        "capped_classes": [],
        "limited_classes": [],
    }
    if len(original) < 2:
        result["note"] = "Need at least 2 classes in the train split to balance"
        return result

    largest = max(original.values())
    wanted = math.ceil(largest * target_ratio)
    minority = {c for c, n in original.items() if n * imbalance_threshold <= largest}
    result["minority_classes"] = sorted(cname(c) for c in minority)
    target = {c: (min(wanted, math.floor(n * max_factor)) if c in minority else n) for c, n in original.items()}
    result["capped_classes"] = sorted(cname(c) for c in minority if target[c] < wanted)

    geoms = geometric_transforms(tf["rot90"], tf)
    photo_on = {k: tf[k] for k in PHOTOMETRIC_KEYS}
    any_photo = any(photo_on.values())
    rng = np.random.default_rng(seed)
    counts = Counter(original)
    copies: Counter = Counter()
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    entries: list[dict] = []

    # One copy per distinct geometric variant. The identity transform
    # (photometric change only) is too close to the original to be worth a
    # second copy — unless it is the only option left (all flips turned off).
    usable = [g for g in geoms if g[0] != "none" or any_photo]
    per_image_cap = max(len(geoms) - 1, 1) if usable else 0

    for cls in sorted(original, key=lambda c: (original[c], c)):
        if counts[cls] >= target[cls]:
            continue
        # Purest images first: an image that is mostly this class adds the
        # fewest boxes of other (often already large) classes sharing it.
        candidates = sorted(
            (s for s in samples if cls in s["counts"]),
            key=lambda s: (-s["counts"][cls] / sum(s["counts"].values()), s["image"].name),
        )
        while counts[cls] < target[cls]:
            available = [s for s in candidates if copies[s["image"].name] < per_image_cap]
            if not available:
                result["limited_classes"].append(cname(cls))
                break
            sample = available[0]
            k = copies[sample["image"].name]
            copies[sample["image"].name] += 1

            geom_name, geom_img, geom_box = usable[k % len(usable)]
            stem = f"{sample['image'].stem}{AUG_MARKER}{k:03d}"
            aug_name = f"{stem}{sample['image'].suffix}"
            if dry_run:
                photo = _photometric(np.zeros((2, 2), np.uint8), rng, photo_on)[1]  # keeps rng in step
            else:
                img = _read_image(sample["image"])
                written = False
                photo = []
                if img is not None:
                    out, photo = _photometric(geom_img(img), rng, photo_on)
                    written = _write_image(images_dir / aug_name, out)
                if not written:
                    log.warning(f"Could not read/write {sample['image'].name}; leaving it out of upsampling")
                    candidates.remove(sample)
                    continue
                boxes = [(c, geom_box(*b)) for c, b in sample["boxes"]]
                (labels_dir / f"{stem}.txt").write_text(_format_labels(boxes), encoding="utf-8")

            entries.append({
                "image": aug_name,
                "source_image": sample["image"].name,
                "origin": sources.get(sample["image"].stem, "unknown"),
                "for_class": cname(cls),
                "geometric": geom_name,
                "photometric": photo,
            })
            counts.update(sample["counts"])
            result["generated_images"] += 1

        if counts[cls] > original[cls]:
            result["classes_upsampled"].append(cname(cls))

    result["after"] = named(counts)
    if not dry_run:
        save_json({**result, "images": entries}, dataset_dir / MANIFEST_NAME)
    log.info(
        f"Upsampling {'(dry run) would generate' if dry_run else 'generated'} "
        f"{result['generated_images']} images: {result['before']} -> {result['after']}"
    )
    return result


def class_balance(class_counts: dict[str, int], imbalance_threshold: float | None = None) -> dict:
    """Imbalance summary used by the advisor (no dataset access)."""
    threshold = cfg.UPSAMPLE_IMBALANCE_THRESHOLD if imbalance_threshold is None else imbalance_threshold
    counts = {k: v for k, v in class_counts.items() if v > 0}
    if len(counts) < 2:
        return {"imbalance_ratio": 1.0, "minority_classes": [], "suggest_upsampling": False,
                "max_factor": cfg.UPSAMPLE_MAX_FACTOR}
    largest, smallest = max(counts.values()), min(counts.values())
    ratio = round(largest / smallest, 2)
    minority = sorted(k for k, v in counts.items() if v * threshold <= largest)
    return {
        "imbalance_ratio": ratio,
        "largest_class": max(counts, key=counts.get),
        "minority_classes": minority,
        "suggest_upsampling": bool(minority),
        "max_factor": cfg.UPSAMPLE_MAX_FACTOR,
    }
