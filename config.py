"""
Global configuration for DYN-EYE Unknown Defect Discovery Pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
INPUT_IMAGES_DIR = DATA_DIR / "input_images"
CROPS_DIR = DATA_DIR / "crops"
CLUSTERS_DIR = DATA_DIR / "clusters"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
YOLO_DATASET_DIR = DATA_DIR / "yolo_dataset"
UNKNOWN_DEFECTS_JSON = DATA_DIR / "unknown_defects.json"
KNOWN_DEFECTS_DIR = DATA_DIR / "known_defect_crops"  # For FAISS setup
SAMPLE_RUN_DIR = DATA_DIR / "samplerun"

MODELS_DIR = PROJECT_ROOT / "models"
YOLO_MODEL_PATH = MODELS_DIR / "best.pt"
MODEL_VERSIONS_DIR = MODELS_DIR / "versions"

LOGS_DIR = PROJECT_ROOT / "logs"
PIPELINE_RUNS_DIR = LOGS_DIR / "pipeline_runs"

# ============================================================
# ENSURE DIRECTORIES EXIST
# ============================================================
for d in [
    DATA_DIR, INPUT_IMAGES_DIR, CROPS_DIR, CLUSTERS_DIR,
    FAISS_INDEX_DIR, YOLO_DATASET_DIR, KNOWN_DEFECTS_DIR,
    YOLO_DATASET_DIR / "images" / "train",
    YOLO_DATASET_DIR / "images" / "val",
    YOLO_DATASET_DIR / "labels" / "train",
    YOLO_DATASET_DIR / "labels" / "val",
    MODELS_DIR, MODEL_VERSIONS_DIR,
    LOGS_DIR, PIPELINE_RUNS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# YOLO SETTINGS
# ============================================================
YOLO_CONFIDENCE_THRESHOLD = 0.30
KNOWN_DEFECTS_JSON = DATA_DIR / "known_defects.json"

# Dynamic — reads from data/known_defects.json at access time.
# Populated automatically when fine-tuned models are deployed.
# Prefer importing get_known_defect_names() from
# src.features.known_defects_registry for hot-reload behaviour.
def _load_known_names() -> list[str]:
    """Read the known-defects registry (fast, no heavy imports)."""
    import json as _json
    try:
        data = _json.loads(KNOWN_DEFECTS_JSON.read_text(encoding="utf-8"))
        return data.get("defect_classes", [])
    except (FileNotFoundError, _json.JSONDecodeError):
        return []

KNOWN_DEFECT_NAMES: list[str] = _load_known_names()

# ============================================================
# VLM SETTINGS (Gemma 4-31b-it via Google GenAI)
# ============================================================
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "AIzaSyCOTORQ_xn-j-OffOrNibKtEGEGMf7_Zm0",
)
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
VLM_MODEL_ID = "gemma-4-31b-it"
VLM_TEMPERATURE = 0.1
VLM_SLEEP_BETWEEN = 4.5
VLM_MAX_RETRIES = 5
VLM_BACKOFF_FACTOR = 2

# ============================================================
# RESNET / FEATURE EXTRACTION
# ============================================================
FEATURE_DIM = 384  # DINOv2 ViT-S/14 output dimension
FEATURE_BATCH_SIZE = 32

# ============================================================
# FAISS SETTINGS
# ============================================================
FAISS_INDEX_FILE = FAISS_INDEX_DIR / "known_defects.index"
FAISS_LABELS_FILE = FAISS_INDEX_DIR / "known_defects_labels.json"
FAISS_NOVELTY_THRESHOLD = 0.35  # Squared L2 distance above which = novel/unknown (using L2 normalized DINOv2 space)

# ============================================================
# HDBSCAN SETTINGS
# ============================================================
HDBSCAN_MIN_CLUSTER_SIZE = 4
HDBSCAN_MIN_SAMPLES = 2
HDBSCAN_METRIC = "euclidean"

# ============================================================
# (Label Studio removed — cluster editing is done in the dashboard)
# ============================================================

# ============================================================
# MLFLOW SETTINGS
# ============================================================
_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
if _tracking_uri and not _tracking_uri.startswith(("http://", "https://", "file://", "sqlite://")):
    # It's a raw disk path. Convert to standard file:// URI (e.g. file:///E:/path)
    MLFLOW_TRACKING_URI = Path(_tracking_uri).resolve().as_uri()
else:
    MLFLOW_TRACKING_URI = _tracking_uri
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT_NAME = "dyneye-yolo-defect-detection"

# ============================================================
# DASHBOARD SETTINGS
# ============================================================
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8501

# ============================================================
# YOLO TRAINING DEFAULTS
# ============================================================
YOLO_TRAIN_EPOCHS = 50
YOLO_TRAIN_IMGSZ = 640
YOLO_TRAIN_BATCH = 16
