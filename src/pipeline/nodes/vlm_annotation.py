"""
Node 2 / 7 — VLM Annotation (Sequential, one image at a time)

Sends each unknown defect image to Gemma 4-31b-it via Google GenAI
for bounding-box detection.

Part 2 enhancement: reads dynamic VLM prompt from state if available.
Part 3 enhancement: multi-sample ICC scoring per cluster + metrics logging.
"""
from __future__ import annotations

import hashlib
import time
import json
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("vlm_annotation")


# ── Schema (identical to reference script) ───────────────────
class Anomaly(BaseModel):
    physical_traits: str = Field(
        description="Description of visual geometry/texture (e.g., 'jagged dark sliver')."
    )
    box_2d: list[int] = Field(
        description="[ymin, xmin, ymax, xmax] 0-1000."
    )


class InspectionReport(BaseModel):
    anomalies_found: bool
    findings: list[Anomaly]


# ── Static prompt (fallback if dynamic prompt is not available) ──
SYSTEM_PROMPT = """
# MISSION
You are a High-Precision Industrial Metrology AI. Your sole purpose is to detect ANY manufacturing violation or deviation from the ideal geometry and surface of a mechanical component for YOLO training data.

# CRITICAL DETECTION RULES
- Detect **all anomalies** that should not be present on a properly manufactured industrial part.
- Be extremely sensitive to both obvious and subtle defects.
- Prioritize anything that breaks expected symmetry, continuity, or surface uniformity.

# VIOLATION CATEGORIES (Detect ALL of these)
1. **STRUCTURAL / GEOMETRIC VIOLATIONS** (Highest Priority)
   - Holes, voids, missing material, gaps, or perforations
   - Chipped edges, nicks, crescent gaps, broken teeth (on gears)
   - Any deviation from expected outer/inner silhouette or circularity
   - Extra or missing features compared to ideal design

2. **SURFACE & TEXTURE VIOLATIONS**
   - Scratches, gouges, dents, pits, burrs, chips
   - Streaks, chatter marks, waviness, or linear disruptions
   - Rough patches or texture changes
   - sharp difference in the surface or edges

3. **TONAL & MATERIAL VIOLATIONS**
   - Discoloration, haze, burn marks, rust, stains, or cloudy areas
   - Any region that breaks surface homogeneity

# BOXING INSTRUCTIONS
- Draw tight but complete bounding boxes that fully enclose the violation.
- For edge/hole defects: Include 5-8% of surrounding material for context.
- Use standard COCO format: [y_min, x_min, y_max, x_max] in 0-1000 scale.

# OUTPUT
Return ONLY the structured JSON payload.
"""


def _annotate_single_image(
    client: genai.Client,
    config: types.GenerateContentConfig,
    image_path: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    """
    Send a single image to VLM and return annotation dict.
    Includes retry logic with exponential backoff.
    """
    f_name = Path(image_path).name
    img = Image.open(image_path)

    for attempt in range(cfg.VLM_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=cfg.VLM_MODEL_ID,
                contents=[system_prompt, img],
                config=config,
            )
            data = json.loads(response.text)

            findings = []
            if data.get("anomalies_found"):
                for f in data.get("findings", []):
                    box = f.get("box_2d", [])
                    if len(box) == 4:
                        findings.append({
                            "box_2d": box,
                            "physical_traits": f.get("physical_traits", ""),
                        })

            log.info(
                f"[VLM] {f_name}: {len(findings)} anomalies found "
                f"(tokens: {response.usage_metadata.prompt_token_count})"
            )
            return {
                "image_path": image_path,
                "image_name": f_name,
                "anomalies_found": data.get("anomalies_found", False),
                "findings": findings,
                "prompt_tokens": response.usage_metadata.prompt_token_count,
            }

        except Exception as e:
            err_str = str(e)
            is_503 = "503" in err_str or "high demand" in err_str.lower() or "unavailable" in err_str.lower() or "getaddrinfo failed" in err_str.lower()
            
            sleep_sec = cfg.VLM_BACKOFF_FACTOR ** (attempt + 1)
            if is_503:
                sleep_sec = max(sleep_sec, 8 + attempt * 4)
                
            log.warning(f"[VLM] {f_name} attempt {attempt+1} failed: {e}. Retrying in {sleep_sec}s...")
            if attempt < cfg.VLM_MAX_RETRIES - 1:
                time.sleep(sleep_sec)

    log.error(f"[VLM] {f_name}: all {cfg.VLM_MAX_RETRIES} attempts failed")
    return {
        "image_path": image_path,
        "image_name": f_name,
        "anomalies_found": False,
        "findings": [],
        "error": "All VLM retries exhausted",
    }


# ── Part 3: Multi-sample ICC per cluster ────────────────────

def _deterministic_sample(paths: list[str], n: int) -> list[str]:
    """
    Deterministically sample n items from paths using a seed
    derived from the sorted path content (reproducible across runs).
    """
    if len(paths) <= n:
        return paths
    seed_str = "|".join(sorted(paths))
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
    import random
    rng = random.Random(seed)
    return rng.sample(paths, n)


def _annotate_cluster_for_icc(
    cluster_folder: str,
    client: genai.Client,
    config: types.GenerateContentConfig,
    system_prompt: str,
    n_samples: int = 3,
) -> dict:
    """
    Annotate 2-3 crops from a cluster independently for ICC scoring.
    Returns a dict with label, confidence, icc, and per-sample results.
    """
    from src.utils.io_helpers import list_images
    folder = Path(cluster_folder)
    all_crops = [str(p) for p in list_images(folder)]

    if not all_crops:
        return {
            "label": "empty",
            "confidence": 0.0,
            "icc": 1.0,
            "n_samples": 0,
            "labels_seen": [],
        }

    # Deterministic sampling
    sample_paths = _deterministic_sample(all_crops, n_samples)

    from src.utils import LogStream

    labels_seen = []
    confidences = []
    all_sample_results = []

    for idx, path in enumerate(sample_paths):
        msg = f"ICC: Evaluating cluster '{folder.name}' (crop {idx+1}/{len(sample_paths)})"
        log.info(msg)
        LogStream.emit(msg, level="progress", source="manifest_save")

        result = _annotate_single_image(client, config, path, system_prompt)
        all_sample_results.append(result)

        # Extract the most prominent label from findings
        findings = result.get("findings", [])
        if findings:
            # Use physical_traits as the label proxy
            traits = [f.get("physical_traits", "").strip().lower() for f in findings]
            primary = traits[0] if traits else "unknown"
            labels_seen.append(primary)
            confidences.append(1.0)  # Binary confidence: found anomaly
        else:
            labels_seen.append("no_defect")
            confidences.append(0.0)

        # Respect rate limiting (at least 4 seconds per image)
        sleep_duration = max(4.0, getattr(cfg, "VLM_SLEEP_BETWEEN", 4.5))
        time.sleep(sleep_duration)

    # ICC: fraction of samples with the same plurality label
    if labels_seen:
        from collections import Counter
        label_counts = Counter(labels_seen)
        plurality_label, plurality_count = label_counts.most_common(1)[0]
        icc = plurality_count / len(labels_seen)
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    else:
        plurality_label = "unknown"
        icc = 1.0
        mean_confidence = 0.0

    return {
        "label": plurality_label,
        "confidence": mean_confidence,
        "icc": icc,
        "n_samples": len(sample_paths),
        "labels_seen": labels_seen,
        "sample_results": all_sample_results,
    }


# ── Main node ───────────────────────────────────────────────

def vlm_annotation_node(state: dict) -> dict:
    """
    LangGraph node: VLM annotation (sequential, one image per call).

    Runs AFTER dataset_context_node, which may have placed a Groq-generated
    domain-specific detection prompt in state["vlm_system_prompt"].

    If no dynamic prompt is available, falls back to the static SYSTEM_PROMPT.

    Reads:
        state["unknown_image_paths"]
        state["use_cache"]
        state["vlm_system_prompt"]   (optional — from dataset_context node)

    Writes:
        state["vlm_annotations"]
    """
    if state.get("use_cache"):
        log.info("Cache mode — skipping VLM annotation (reusing cached annotations).")
        return {"vlm_annotations": state.get("vlm_annotations", [])}

    unknown_paths = state.get("unknown_image_paths", [])
    if not unknown_paths:
        log.warning("No unknown images to annotate")
        return {"vlm_annotations": []}

    # Use dynamic prompt from dataset_context if available, else static
    system_prompt = state.get("vlm_system_prompt") or SYSTEM_PROMPT
    if system_prompt is not SYSTEM_PROMPT and system_prompt != SYSTEM_PROMPT:
        log.info("Using dynamic VLM detection prompt from Groq")
    else:
        log.info("Using static VLM detection prompt (fallback)")

    client = genai.Client()
    gen_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=InspectionReport,
        temperature=cfg.VLM_TEMPERATURE,
    )

    from src.utils import LogStream

    # ── Per-image bbox annotation ────────────────────────────
    annotations: list[dict] = []
    for idx, img_path in enumerate(unknown_paths):
        msg = f"[VLM] Processing image {idx+1}/{len(unknown_paths)}: {Path(img_path).name}"
        log.info(msg)
        LogStream.emit(msg, level="progress", source="vlm_annotation")
        result = _annotate_single_image(client, gen_config, img_path, system_prompt)
        annotations.append(result)
        time.sleep(cfg.VLM_SLEEP_BETWEEN)

    total_findings = sum(len(a.get("findings", [])) for a in annotations)
    msg_end = f"VLM annotation complete: {len(annotations)} images, {total_findings} total findings"
    log.info(msg_end)
    LogStream.emit(msg_end, level="info", source="vlm_annotation")

    # Save to data/vlm_cache.json automatically
    cache_path = cfg.DATA_DIR / "vlm_cache.json"
    try:
        from src.utils import save_json
        cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
        save_json(annotations, str(cache_path))
        log.info(f"Saved {len(annotations)} VLM annotations to cache at {cache_path}")
    except Exception as e:
        log.warning(f"Failed to write VLM cache: {e}")

    return {"vlm_annotations": annotations}
