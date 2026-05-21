"""
Node 2 — VLM Annotation (Sequential, one image at a time)

Sends each unknown defect image to Gemma 4-31b-it via Google GenAI
for bounding-box detection.  Uses the exact same model, params, and
prompt from single_vlm_3 copy 2.py.
"""
from __future__ import annotations

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


# ── Prompt (identical to reference script) ───────────────────
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
                contents=[SYSTEM_PROMPT, img],
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
            log.warning(f"[VLM] {f_name} attempt {attempt+1} failed: {e}")
            if attempt < cfg.VLM_MAX_RETRIES - 1:
                time.sleep(cfg.VLM_BACKOFF_FACTOR ** (attempt + 1))

    log.error(f"[VLM] {f_name}: all {cfg.VLM_MAX_RETRIES} attempts failed")
    return {
        "image_path": image_path,
        "image_name": f_name,
        "anomalies_found": False,
        "findings": [],
        "error": "All VLM retries exhausted",
    }


def vlm_annotation_node(state: dict) -> dict:
    """
    LangGraph node: VLM annotation (sequential, one image per call).

    Reads:
        state["unknown_image_paths"]

    Writes:
        state["vlm_annotations"]
    """
    if state.get("use_cache"):
        log.info("Cache mode — skipping VLM annotation (reusing cached annotations).")
        return {"vlm_annotations": state.get("vlm_annotations", []), "_cached": True}

    unknown_paths = state.get("unknown_image_paths", [])
    if not unknown_paths:
        log.warning("No unknown images to annotate")
        return {"vlm_annotations": []}

    client = genai.Client()
    gen_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=InspectionReport,
        temperature=cfg.VLM_TEMPERATURE,
    )

    annotations: list[dict] = []
    for idx, img_path in enumerate(unknown_paths):
        log.info(f"[VLM] Processing {idx+1}/{len(unknown_paths)}: {Path(img_path).name}")
        result = _annotate_single_image(client, gen_config, img_path)
        annotations.append(result)
        time.sleep(cfg.VLM_SLEEP_BETWEEN)

    total_findings = sum(len(a.get("findings", [])) for a in annotations)
    log.info(
        f"VLM annotation complete: {len(annotations)} images, "
        f"{total_findings} total findings"
    )

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
