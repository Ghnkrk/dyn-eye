"""
VLM Annotation Reliability Score (VARS)
========================================

Ground-truth-free proxy metric for VLM annotation quality.

VARS = 0.50 × CDS  +  0.30 × BQS  +  0.20 × DRS

Sub-scores
----------
CDS — Crop Discriminability Score
    Are VLM-cropped regions visually distinct from random non-defect patches
    sampled from the same source images (in DINOv2 embedding space)?

BQS — Box Quality Score
    Are the bounding boxes geometrically reasonable?
    (tight coverage, not blurry, not overzoomed, sensible aspect ratio)

DRS — Detection Rate Score
    Is the fraction of images with at least one detection plausible?
    (not 0%, not 100% — real defects are neither universal nor absent)

Device
------
Runs DINOv2 on CUDA if available, falls back to CPU. The model is loaded
once and cached in the module scope so repeated calls within the same run
are fast.
"""
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("vlm_score")

# Module-level cached DINOv2 model (lazy-loaded on first call)
_dino_model = None
_dino_device = None
_dino_preprocess = None


# ── DINOv2 helpers ──────────────────────────────────────────────

def _get_dino():
    """Lazy-load DINOv2 model. Cached after first call."""
    global _dino_model, _dino_device, _dino_preprocess
    if _dino_model is not None:
        return _dino_model, _dino_device, _dino_preprocess

    import torch
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[VARS] Loading DINOv2 on {device}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _dino_model = model
    _dino_device = device
    _dino_preprocess = preprocess
    return model, device, preprocess


def _embed_bgr_patches(patches: list[np.ndarray], batch_size: int = 32) -> Optional[np.ndarray]:
    """
    Embed a list of OpenCV BGR numpy arrays via DINOv2.
    Returns L2-normalised (N, 384) float32 array, or None if no patches.
    """
    if not patches:
        return None

    import torch
    import torch.nn.functional as F

    model, device, preprocess = _get_dino()
    tensors = []
    for p in patches:
        if p is None or p.size == 0:
            continue
        rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        tensors.append(preprocess(rgb))

    if not tensors:
        return None

    all_feats = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size]).to(device)
            feats = model(batch)
            feats = F.normalize(feats, p=2, dim=-1)
            all_feats.append(feats.cpu().float().numpy())

    return np.concatenate(all_feats, axis=0)


# ── Background patch sampler ─────────────────────────────────

def _sample_background_patches(
    source_image: str,
    annotated_boxes_pixels: list[list[int]],  # [[y1,x1,y2,x2], ...]
    n_patches: int,
    patch_size: int,
    rng: random.Random,
    iou_threshold: float = 0.10,
) -> tuple[list[np.ndarray], bool]:
    """
    Sample n_patches from source_image avoiding annotated bounding boxes.

    Returns:
        (patches_list, is_clean)
        is_clean = True  → patches do NOT overlap any annotated region
                 = False → image was too heavily annotated; no usable patches
                           (caller should use zero-detection fallback images)
    """
    img = cv2.imread(str(source_image))
    if img is None:
        return [], False

    img_h, img_w = img.shape[:2]
    ps = min(patch_size, img_h, img_w)

    # Estimate how much of the image is covered by annotation boxes
    if annotated_boxes_pixels:
        covered_mask = np.zeros((img_h, img_w), dtype=bool)
        for box in annotated_boxes_pixels:
            y1, x1, y2, x2 = (int(v) for v in box)
            covered_mask[
                max(0, y1) : min(img_h, y2),
                max(0, x1) : min(img_w, x2),
            ] = True
        coverage = float(covered_mask.sum()) / max(1, img_h * img_w)
    else:
        coverage = 0.0

    # If image is more than 70% covered by defect boxes, we cannot reliably
    # find non-defect patches — signal to the caller to use the fallback pool.
    if coverage > 0.70:
        return [], False

    patches: list[np.ndarray] = []
    max_attempts = n_patches * 30
    attempts = 0

    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        px = rng.randint(0, max(1, img_w - ps))
        py = rng.randint(0, max(1, img_h - ps))

        # Check overlap fraction between candidate patch and every annotation box
        overlaps = False
        if annotated_boxes_pixels:
            patch_area = ps * ps
            for box in annotated_boxes_pixels:
                y1, x1, y2, x2 = (int(v) for v in box)
                iy1 = max(py, y1)
                ix1 = max(px, x1)
                iy2 = min(py + ps, y2)
                ix2 = min(px + ps, x2)
                if iy1 < iy2 and ix1 < ix2:
                    inter = (iy2 - iy1) * (ix2 - ix1)
                    if inter / max(1, patch_area) > iou_threshold:
                        overlaps = True
                        break

        if not overlaps:
            patch = img[py : py + ps, px : px + ps]
            if patch.size > 0:
                patches.append(patch)

    # If we couldn't collect enough patches after many tries, still flag OK
    # as long as we got at least 1 — the coverage check above is the hard gate.
    return patches, True


# ── Sub-score 1: Crop Discriminability Score (CDS) ──────────

def compute_cds(
    crop_metadata: list[dict],
    vlm_annotations: list[dict],
    crop_feature_vectors: Optional[np.ndarray],
    n_bg_patches: int = 5,
    patch_size: int = 96,
    rng_seed: int = 42,
) -> float:
    """
    Crop Discriminability Score — how visually distinct are VLM-detected
    crops from random non-defect regions of the same images?

    Algorithm
    ---------
    1. Group crop_metadata by source_image.
    2. For each source image, sample n_bg_patches background patches that
       do NOT overlap the annotated bounding boxes.
    3. For images that are too heavily annotated (>70% covered), pull
       background patches from zero-detection images in the same run
       (the "clean fallback pool").
    4. Embed background patches with DINOv2.
    5. Compute cosine-space silhouette score between VLM-crop embeddings
       (label=1) and background patch embeddings (label=0).
    6. Normalise to [0, 1] via  score = (sil + 1) / 2.

    Returns 0.5 (neutral) if fewer than 4 total samples or only 1 class.
    """
    rng = random.Random(rng_seed)

    # Need crop embeddings — prefer the pipeline's already-computed vectors
    if crop_feature_vectors is not None and len(crop_feature_vectors) > 0:
        crop_vecs = np.asarray(crop_feature_vectors, dtype=np.float32)
    else:
        # Fall back: re-embed crop images
        crop_imgs = []
        for m in crop_metadata:
            img = cv2.imread(str(m.get("crop_path", "")))
            if img is not None:
                crop_imgs.append(img)
        if not crop_imgs:
            log.warning("[CDS] No crop images available — returning neutral 0.5")
            return 0.5
        crop_vecs = _embed_bgr_patches(crop_imgs)
        if crop_vecs is None:
            return 0.5

    if len(crop_vecs) < 2:
        return 0.5

    # ── Build zero-detection fallback image pool ──────────────
    zero_detection_paths: list[str] = []
    for ann in vlm_annotations:
        if not ann.get("findings") and not ann.get("error"):
            p = ann.get("image_path", "")
            if p and Path(p).exists():
                zero_detection_paths.append(p)

    # Group annotation boxes by source image for efficient lookup
    boxes_by_image: dict[str, list[list[int]]] = {}
    for m in crop_metadata:
        src = str(m.get("source_image", ""))
        box = m.get("box_2d_pixels", [])
        if src and box:
            boxes_by_image.setdefault(src, []).append(box)

    annotated_sources = list(boxes_by_image.keys())

    # ── Pre-compute visual signatures for zero-detection images ──
    # Embed a center patch of each clean image as its visual background signature
    zero_detection_signatures: list[dict] = []
    if zero_detection_paths:
        log.info(f"[CDS] Pre-embedding {len(zero_detection_paths)} zero-detection background signatures...")
        patches = []
        valid_paths = []
        for path in zero_detection_paths:
            img = cv2.imread(str(path))
            if img is not None:
                ih, iw = img.shape[:2]
                cs = min(224, ih, iw)
                cy, cx = ih // 2, iw // 2
                patch = img[cy - cs // 2 : cy + cs // 2, cx - cs // 2 : cx + cs // 2]
                if patch.size > 0:
                    patches.append(patch)
                    valid_paths.append(path)
        if patches:
            bg_sigs = _embed_bgr_patches(patches)
            if bg_sigs is not None:
                for pth, sig in zip(valid_paths, bg_sigs):
                    zero_detection_signatures.append({
                        "path": pth,
                        "signature": sig
                    })

    # ── Pre-compute visual signatures for annotated source images ──
    annotated_signatures: dict[str, np.ndarray] = {}
    if annotated_sources:
        log.info(f"[CDS] Pre-embedding {len(annotated_sources)} annotated source signatures...")
        patches = []
        valid_srcs = []
        for src in annotated_sources:
            img = cv2.imread(str(src))
            if img is not None:
                ih, iw = img.shape[:2]
                cs = min(224, ih, iw)
                cy, cx = ih // 2, iw // 2
                patch = img[cy - cs // 2 : cy + cs // 2, cx - cs // 2 : cx + cs // 2]
                if patch.size > 0:
                    patches.append(patch)
                    valid_srcs.append(src)
        if patches:
            src_sigs = _embed_bgr_patches(patches)
            if src_sigs is not None:
                for src, sig in zip(valid_srcs, src_sigs):
                    annotated_signatures[src] = sig

    # Collect background patches
    all_bg_patches: list[np.ndarray] = []

    for src in annotated_sources:
        # 1. Try to sample clean background patches from the same image first
        patches, is_clean = _sample_background_patches(
            source_image=src,
            annotated_boxes_pixels=boxes_by_image[src],
            n_patches=n_bg_patches,
            patch_size=patch_size,
            rng=rng,
        )

        if is_clean and patches:
            all_bg_patches.extend(patches)
        else:
            # Fallback: image is over-annotated (>70% coverage).
            # Pull clean background patches from visually similar zero-detection images.
            src_sig = annotated_signatures.get(src)
            matched_paths: list[str] = []

            if src_sig is not None and zero_detection_signatures:
                # Rank zero-detection images by cosine similarity to the target background
                similarities = []
                for item in zero_detection_signatures:
                    sim = float(np.dot(src_sig, item["signature"]))
                    similarities.append((sim, item["path"]))
                
                # Sort descending
                similarities.sort(key=lambda x: x[0], reverse=True)

                # Filter by threshold (>=0.80 similarity) or fall back to top 3
                threshold = 0.80
                candidates = [path for sim, path in similarities if sim >= threshold]
                if not candidates:
                    candidates = [path for sim, path in similarities[:3]]
                matched_paths = candidates
                log.debug(
                    f"[CDS] Selected {len(matched_paths)} visually matching background fallback images for {Path(src).name}"
                )
            else:
                # No signatures: fallback to all zero detections
                matched_paths = zero_detection_paths

            # Extract patches from selected visually matching images
            prefix_patches = []
            if matched_paths:
                chosen_paths = rng.sample(matched_paths, min(5, len(matched_paths)))
                for path in chosen_paths:
                    img = cv2.imread(str(path))
                    if img is None:
                        continue
                    ih, iw = img.shape[:2]
                    ps = min(patch_size, ih, iw)
                    for _ in range(n_bg_patches):
                        px = rng.randint(0, max(1, iw - ps))
                        py = rng.randint(0, max(1, ih - ps))
                        p = img[py : py + ps, px : px + ps]
                        if p.size > 0:
                            prefix_patches.append(p)

            if prefix_patches:
                chosen = rng.sample(prefix_patches, min(n_bg_patches, len(prefix_patches)))
                all_bg_patches.extend(chosen)
            else:
                log.debug(f"[CDS] No zero-detection fallback available at all for {src}, skipping")

    if not all_bg_patches:
        log.warning("[CDS] Could not sample any background patches — returning neutral 0.5")
        return 0.5

    # ── Embed background patches ──────────────────────────────
    try:
        bg_vecs = _embed_bgr_patches(all_bg_patches)
    except Exception as e:
        log.warning(f"[CDS] DINOv2 embedding of background patches failed: {e}")
        return 0.5

    if bg_vecs is None or len(bg_vecs) == 0:
        return 0.5

    # ── Cosine silhouette score ───────────────────────────────
    try:
        from sklearn.metrics import silhouette_score

        n_crop = len(crop_vecs)
        n_bg = len(bg_vecs)

        if n_crop + n_bg < 4:
            return 0.5

        # L2-normalise both sets (DINOv2 outputs are already normalised,
        # but normalise again defensively in case crop_vecs came from state)
        def _l2(v):
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            return v / np.where(norms == 0, 1.0, norms)

        crop_vecs_n = _l2(crop_vecs)
        bg_vecs_n   = _l2(bg_vecs)

        X = np.concatenate([crop_vecs_n, bg_vecs_n], axis=0)
        y = np.array([1] * n_crop + [0] * n_bg)

        sil = float(silhouette_score(X, y, metric="cosine"))
        # sil ∈ [-1, 1] → normalise to [0, 1]
        cds = (sil + 1.0) / 2.0
        log.info(
            f"[CDS] Silhouette={sil:.4f} -> CDS={cds:.4f}  "
            f"(crops={n_crop}, bg_patches={n_bg})"
        )
        return round(float(cds), 4)

    except Exception as e:
        log.warning(f"[CDS] Silhouette computation failed: {e}")
        return 0.5


# ── Sub-score 2: Box Quality Score (BQS) ────────────────────

def compute_bqs(
    crop_metadata: list[dict],
    vlm_annotations: list[dict],
) -> float:
    """
    Box Quality Score — geometric sanity check on VLM bounding boxes.

    Factors (all averaged with equal weight):
      tight_ratio         crop area is 2–40% of source image area
      non_blurry_ratio    Laplacian variance of crop > 25.0
      non_overzoomed      crop doesn't cover > 85% of source image
      aspect_ok           aspect ratio between 0.15 and 6.0
    """
    if not crop_metadata:
        return 0.5  # neutral

    tight_count = 0
    non_blurry_count = 0
    non_overzoomed_count = 0
    aspect_ok_count = 0
    n = len(crop_metadata)

    for m in crop_metadata:
        crop_path = str(m.get("crop_path", ""))
        box = m.get("box_2d_pixels", [])   # [y1, x1, y2, x2]
        crop_w = int(m.get("crop_width", 0))
        crop_h = int(m.get("crop_height", 0))

        # ── Aspect ratio check ─────────────────────────────────
        if crop_w > 0 and crop_h > 0:
            ar = max(crop_w, crop_h) / max(min(crop_w, crop_h), 1)
            if 0.15 <= 1 / ar <= 6.0 or ar <= 6.0:
                aspect_ok_count += 1

        # ── Source image dimensions (needed for ratios) ─────────
        src_img_path = m.get("source_image", "")
        src_img = None
        if src_img_path and Path(str(src_img_path)).exists():
            src_img = cv2.imread(str(src_img_path))
        
        if src_img is not None and crop_w > 0 and crop_h > 0:
            src_h, src_w = src_img.shape[:2]
            crop_area = crop_w * crop_h
            src_area = src_w * src_h

            # ── Tight ratio ────────────────────────────────────
            area_ratio = crop_area / max(1, src_area)
            if 0.02 <= area_ratio <= 0.40:
                tight_count += 1

            # ── Non-overzoomed ─────────────────────────────────
            w_ratio = crop_w / max(1, src_w)
            h_ratio = crop_h / max(1, src_h)
            if not (w_ratio > 0.85 and h_ratio > 0.85):
                non_overzoomed_count += 1
        else:
            # No source image info — give partial credit
            tight_count += 0
            non_overzoomed_count += 1  # assume OK if source not available

        # ── Blur check ─────────────────────────────────────────
        crop_img = cv2.imread(crop_path) if crop_path else None
        if crop_img is not None:
            try:
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if lap_var > 25.0:
                    non_blurry_count += 1
            except Exception:
                non_blurry_count += 1  # assume OK if computation fails
        else:
            non_blurry_count += 1  # missing file: give benefit of the doubt

    bqs = (
        (tight_count / n)
        + (non_blurry_count / n)
        + (non_overzoomed_count / n)
        + (aspect_ok_count / n)
    ) / 4.0

    log.info(
        f"[BQS] tight={tight_count}/{n}, non_blurry={non_blurry_count}/{n}, "
        f"non_overzoomed={non_overzoomed_count}/{n}, aspect_ok={aspect_ok_count}/{n} "
        f"-> BQS={bqs:.4f}"
    )
    return round(float(bqs), 4)


# ── Sub-score 3: Detection Rate Score (DRS) ─────────────────

def compute_drs(vlm_annotations: list[dict]) -> float:
    """
    Detection Rate Score — penalises extreme detection rates.

    VLM finding 100% of images anomalous → likely hallucinating.
    VLM finding 0% of images anomalous  → completely missing defects.
    Score peaks at 50% detection rate and falls symmetrically.

    DRS = 1 − |detection_rate − 0.5| × 2
    """
    if not vlm_annotations:
        return 0.5  # no data → neutral

    total = len(vlm_annotations)
    # An annotation entry has a defect if findings is non-empty
    with_detection = sum(
        1 for ann in vlm_annotations
        if ann.get("findings") and len(ann["findings"]) > 0
    )
    rate = with_detection / max(1, total)
    drs = max(0.0, 1.0 - abs(rate - 0.5) * 2.0)

    log.info(
        f"[DRS] {with_detection}/{total} images with detections "
        f"({rate:.1%}) -> DRS={drs:.4f}"
    )
    return round(float(drs), 4)


# -- Composite VARS -------------------------------------------

def compute_vars(
    crop_metadata: list[dict],
    vlm_annotations: list[dict],
    crop_feature_vectors: Optional[np.ndarray] = None,
    n_bg_patches: int = 5,
    patch_size: int = 96,
) -> dict:
    """
    Compute the VLM Annotation Reliability Score (VARS).

    VARS = 0.50 × CDS  +  0.30 × BQS  +  0.20 × DRS

    Returns a dict with:
        vars_score (float 0–1)
        cds (float 0–1)
        bqs (float 0–1)
        drs (float 0–1)
        vars_pct (int 0–100, for display)
        total_crops (int)
        total_annotated_images (int)
        detection_rate (float 0–1)
        interpretation (str)
    """
    log.info("[VARS] Computing VLM Annotation Reliability Score...")

    # ── CDS ────────────────────────────────────────────────────
    try:
        cds = compute_cds(
            crop_metadata=crop_metadata,
            vlm_annotations=vlm_annotations,
            crop_feature_vectors=crop_feature_vectors,
            n_bg_patches=n_bg_patches,
            patch_size=patch_size,
        )
    except Exception as e:
        log.warning(f"[VARS] CDS failed: {e} — using neutral 0.5")
        cds = 0.5

    # ── BQS ────────────────────────────────────────────────────
    try:
        bqs = compute_bqs(crop_metadata=crop_metadata, vlm_annotations=vlm_annotations)
    except Exception as e:
        log.warning(f"[VARS] BQS failed: {e} — using neutral 0.5")
        bqs = 0.5

    # ── DRS ────────────────────────────────────────────────────
    try:
        drs = compute_drs(vlm_annotations=vlm_annotations)
    except Exception as e:
        log.warning(f"[VARS] DRS failed: {e} — using neutral 0.5")
        drs = 0.5

    # ── Weighted composite ─────────────────────────────────────
    vars_score = round(0.50 * cds + 0.30 * bqs + 0.20 * drs, 4)
    vars_pct   = int(round(vars_score * 100))

    # Detection rate for report
    total = len(vlm_annotations)
    with_det = sum(1 for a in vlm_annotations if a.get("findings"))
    detection_rate = round(with_det / max(1, total), 4)

    # Human-readable interpretation
    if vars_score >= 0.80:
        interpretation = "High reliability — VLM crops are visually distinct and geometrically clean."
    elif vars_score >= 0.60:
        interpretation = "Moderate reliability — VLM is producing plausible annotations."
    elif vars_score >= 0.40:
        interpretation = "Low-moderate reliability — some crops may overlap background regions."
    else:
        interpretation = "Low reliability — crops are not clearly distinct from background."

    result = {
        "vars_score": vars_score,
        "vars_pct": vars_pct,
        "cds": cds,
        "bqs": bqs,
        "drs": drs,
        "total_crops": len(crop_metadata),
        "total_annotated_images": total,
        "detection_rate": detection_rate,
        "interpretation": interpretation,
    }

    log.info(
        f"[VARS] Final -> VARS={vars_score:.4f} ({vars_pct}%)  "
        f"CDS={cds:.4f}  BQS={bqs:.4f}  DRS={drs:.4f}  "
        f"Interpretation: {interpretation}"
    )
    return result
