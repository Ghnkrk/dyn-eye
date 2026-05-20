import os
import time
import json
import glob
import cv2
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ==========================================
# 1. CONFIGURATION
# ==========================================
os.environ["GEMINI_API_KEY"] = "AIzaSyCOTORQ_xn-j-OffOrNibKtEGEGMf7_Zm0"
client = genai.Client()
MODEL_ID = "gemma-4-31b-it"

INPUT_FOLDER = "./input_images_2"
OUTPUT_FOLDER = "./output_2" 
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SLEEP_BETWEEN_IMAGES = 4.5
MAX_RETRIES = 5
BACKOFF_FACTOR = 2

# ==========================================
# 2. SCHEMA (Pure Discovery - No Labels)
# ==========================================
class Anomaly(BaseModel):
    physical_traits: str = Field(description="Description of visual geometry/texture (e.g., 'jagged dark sliver').")
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax] 0-1000.")

class InspectionReport(BaseModel):
    anomalies_found: bool
    findings: list[Anomaly]

# ==========================================
# 3. IMPROVISED "INTELLIGENT FILTER" PROMPT
# ==========================================
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

# ==========================================
# 4. EXECUTION
# ==========================================
def run_inspection():
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.[jp][pn]*"))
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=InspectionReport,
        temperature=0.1,

    )
    
    for f_path in files:
        f_name = os.path.basename(f_path)
        print(f"\n[SCANNING]: {f_name}")    
        img = Image.open(f_path)
        
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[SYSTEM_PROMPT, img],
                    config=config
                )
                print(f"prompt tokens: {response.usage_metadata.prompt_token_count}")
                data = json.loads(response.text)
                cv_img = cv2.imread(f_path)
                h, w = cv_img.shape[:2]

                if data.get("anomalies_found"):
                    for idx, finding in enumerate(data.get("findings", [])):
                        box = finding.get("box_2d")
                        traits = finding.get("physical_traits")
                        
                        y1, x1, y2, x2 = int(box[0]*h/1000), int(box[1]*w/1000), int(box[2]*h/1000), int(box[3]*w/1000)
                        
                        # Geometric Area Guard (Ignore boxes smaller than 10x10 pixels total)
                        if (x2 - x1) * (y2 - y1) < 100:
                            continue

                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(cv_img, f"ID_{idx+1}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        print(f"    -> Finding {idx+1}: {traits}")

                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f_name), cv_img)
                print(f"  [SUCCESS] Metadata processed.")
                break 

            except Exception as e:
                print(f"  [!] Attempt {attempt + 1} Failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_FACTOR ** (attempt + 1))

        time.sleep(SLEEP_BETWEEN_IMAGES)

if __name__ == "__main__":
    run_inspection()