"""
Label Studio ML SDK Backend Model

Serves VLM-based predictions as pre-annotations for Label Studio.
When Label Studio sends an image task, this backend calls the VLM
to generate bounding-box annotations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from label_studio_ml.model import LabelStudioMLBase

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("ls_ml_backend")


# ── VLM Schema ───────────────────────────────────────────────
class Anomaly(BaseModel):
    physical_traits: str = Field(
        description="Description of visual geometry/texture."
    )
    box_2d: list[int] = Field(
        description="[ymin, xmin, ymax, xmax] 0-1000."
    )


class InspectionReport(BaseModel):
    anomalies_found: bool
    findings: list[Anomaly]


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


class VLMBackend(LabelStudioMLBase):
    """
    Label Studio ML Backend that uses VLM (Gemma 4-31b-it)
    to generate bounding-box pre-annotations for defect detection.
    """

    def setup(self):
        """Initialize VLM client."""
        self.client = genai.Client()
        self.gen_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=InspectionReport,
            temperature=cfg.VLM_TEMPERATURE,
        )
        log.info("VLM Backend initialized")

    def predict(self, tasks, context=None, **kwargs):
        """
        Generate VLM predictions for Label Studio tasks.

        Each task has an image URL. We download/load it, send to VLM,
        and return Label Studio-compatible annotation format.
        """
        predictions = []

        for task in tasks:
            image_url = task.get("data", {}).get("image", "")
            log.info(f"Predicting for task: {image_url}")

            try:
                # Resolve image path
                img_path = self._resolve_image_path(image_url)
                if not img_path:
                    predictions.append({"result": [], "score": 0.0})
                    continue

                img = Image.open(img_path)
                original_width, original_height = img.size

                # Call VLM
                response = self.client.models.generate_content(
                    model=cfg.VLM_MODEL_ID,
                    contents=[SYSTEM_PROMPT, img],
                    config=self.gen_config,
                )

                data = json.loads(response.text)
                results = []

                if data.get("anomalies_found"):
                    for idx, finding in enumerate(data.get("findings", [])):
                        box = finding.get("box_2d", [])
                        if len(box) != 4:
                            continue

                        # Convert from 0-1000 to percentage
                        y1, x1, y2, x2 = box
                        x_pct = x1 / 10.0
                        y_pct = y1 / 10.0
                        w_pct = (x2 - x1) / 10.0
                        h_pct = (y2 - y1) / 10.0

                        # Skip tiny boxes
                        if w_pct * h_pct < 0.1:
                            continue

                        traits = finding.get("physical_traits", f"defect_{idx}")

                        results.append({
                            "id": f"vlm_{idx}",
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image",
                            "original_width": original_width,
                            "original_height": original_height,
                            "value": {
                                "x": x_pct,
                                "y": y_pct,
                                "width": w_pct,
                                "height": h_pct,
                                "rotation": 0,
                                "rectanglelabels": [traits[:50]],  # Truncate for label
                            },
                        })

                score = len(results) / max(1, len(data.get("findings", [])))
                predictions.append({
                    "result": results,
                    "score": round(score, 2),
                })

                time.sleep(cfg.VLM_SLEEP_BETWEEN)

            except Exception as e:
                log.error(f"VLM prediction failed: {e}")
                predictions.append({"result": [], "score": 0.0})

        return predictions

    def fit(self, event, data, **kwargs):
        """
        Called when annotations are created/updated in Label Studio.
        We use this hook to trigger downstream processing if needed.
        """
        log.info(f"Fit called with event: {event}")
        return {}

    def _resolve_image_path(self, url: str) -> str | None:
        """Resolve image URL to a local file path."""
        if url.startswith("/data/local-files/"):
            # Local file storage
            relative = url.replace("/data/local-files/?d=", "")
            local_path = cfg.DATA_DIR / relative
            if local_path.exists():
                return str(local_path)

        # Try as absolute path
        if Path(url).exists():
            return url

        # Try relative to input images
        candidate = cfg.INPUT_IMAGES_DIR / Path(url).name
        if candidate.exists():
            return str(candidate)

        log.warning(f"Could not resolve image path: {url}")
        return None
