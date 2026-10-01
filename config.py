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

# Replay: known-class images + the CURRENT model's detections are mixed into the
# fine-tuning set as pseudo-labels so old classes are not forgotten when the
# exported dataset otherwise contains only newly named defects.
KNOWN_REPLAY_JSON = DATA_DIR / "known_replay.json"
REPLAY_MIN_CONF = 0.50        # only confident detections become pseudo-labels
REPLAY_MAX_IMAGES = 150       # cap so replay never drowns the new classes
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
# LLM / VLM SETTINGS
# ============================================================
<<<<<<< HEAD
# Secrets come ONLY from the environment / .env (never hardcode defaults here).
def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip().strip("'\" ")
=======
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "=",
).strip("'\" ")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
>>>>>>> 8131959fae1c5eaeac2a047c4d746d33d3dc573d

GEMINI_API_KEY = _env("GEMINI_API_KEY")   # VLM only (Google GenAI)
GROQ_API_KEY = _env("GROQ_API_KEY")       # every text-LLM call (Groq)

# ── All model IDs live here (each overridable via env var) ──
# When a provider decommissions a model, change it HERE ONLY.
#   VLM_MODEL_ID   Google GenAI — bbox annotation of unknown images (the only Gemini use)
#   LLM_MODEL_ID   Groq         — training advisor, auto-retrain decision, dynamic prompt writer
VLM_MODEL_ID = _env("VLM_MODEL_ID", "gemma-4-31b-it")
LLM_MODEL_ID = _env("LLM_MODEL_ID", "qwen/qwen3.8-27b")
LLM_TEMPERATURE = 0.2
VLM_TEMPERATURE = 0.1
# Per-image result cache: skip the VLM for images already annotated with the same prompt+model
VLM_IMAGE_CACHE = _env("VLM_IMAGE_CACHE", "1").lower() in ("1", "true", "yes")
VLM_IMAGE_CACHE_PATH = DATA_DIR / "vlm_image_cache.json"
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
HDBSCAN_MIN_CLUSTER_SIZE = 3   # Lowered from 4 — allows smaller clusters to form
HDBSCAN_MIN_SAMPLES = 1        # Lowered from 2 — reduces noise points aggressively
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"  # Changed from leaf — better for varied density

# ── Cluster fingerprint registry (Part 1) ──
CLUSTER_REGISTRY_PATH = CLUSTERS_DIR / "cluster_registry.json"
CLUSTER_TUNING_CACHE_PATH = CLUSTERS_DIR / "tuning_cache.json"
CLUSTER_MATCH_THRESHOLD = 0.30       # Cosine distance below which a new cluster inherits an existing label
CLUSTER_NOISE_REASSIGN_THRESHOLD = 0.50  # Cosine dist threshold for reassigning noise points to nearest centroid

# ── UMAP / dimensionality reduction ──
UMAP_N_COMPONENTS = 10       # More components = more separable space for 4+ classes
UMAP_N_NEIGHBORS = 20        # Higher = more global structure (better for 200+ crops)
UMAP_MIN_DIST = 0.05         # Tighter clusters (was 0.1)
UMAP_RANDOM_STATE = 42
UMAP_BATCH_THRESHOLD = 30    # Activate UMAP earlier (was 50)

# ============================================================
# MLFLOW SETTINGS
# ============================================================
_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not _tracking_uri:
    # Use serverless local file store
    MLFLOW_TRACKING_URI = (DATA_DIR / "mlruns").resolve().as_uri()
elif not _tracking_uri.startswith(("http://", "https://", "file://", "sqlite://")):
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
# LLM TRAINING ADVISOR (model + temperature: see LLM_* above)
# ============================================================
LLM_MIN_CROPS_PER_CLASS = 10   # Minimum crops per class before LLM considers training

# ============================================================
# YOLO TRAINING DEFAULTS
# Demo default: 1 epoch. Use 30-100 for a real fine-tune.
YOLO_TRAIN_EPOCHS = 1
# Auto-deploy gate: a trained model below this mAP50 is registered but NOT deployed.
# 0.0 = always deploy (legacy behaviour). Set e.g. 0.5 in production.
YOLO_MIN_DEPLOY_MAP50 = float(os.environ.get("YOLO_MIN_DEPLOY_MAP50", "0.0"))
# Classes the original base model was trained on (used to seed the model registry)
BASELINE_CLASSES = ["inclusion", "oil_spot", "punching_hole", "silk_spot", "water_spot", "welding_line"]
YOLO_TRAIN_IMGSZ = 640
YOLO_TRAIN_BATCH = 16

YOLO_TRAIN_LR0 = 0.01
YOLO_TRAIN_LRF = 0.01
YOLO_TRAIN_MOMENTUM = 0.937
YOLO_TRAIN_WEIGHT_DECAY = 0.0005
YOLO_TRAIN_WARMUP_EPOCHS = 3.0
YOLO_TRAIN_PATIENCE = 20
YOLO_TRAIN_OPTIMIZER = "auto"
YOLO_TRAIN_COS_LR = False
YOLO_TRAIN_FREEZE = None

# ============================================================
# VLM DYNAMIC PROMPT SETTINGS (Part 2)
# ============================================================
VLM_PROMPT_CACHE_PATH = DATA_DIR / "vlm_prompt_cache.json"
INSPECTION_DOMAIN = os.environ.get("INSPECTION_DOMAIN", "metallic")
# Off by default: single-domain (metallic) datasets are served well by the static
# prompt, and an LLM-rewritten prompt adds run-to-run variance + a Groq dependency.
# Set USE_DYNAMIC_PROMPT=1 when onboarding a new inspection domain.
USE_DYNAMIC_PROMPT = os.environ.get("USE_DYNAMIC_PROMPT", "0").strip().lower() in ("1", "true", "yes")

# ============================================================
# VLM METRICS (Part 3)
# ============================================================
VLM_METRICS_PATH = DATA_DIR / "vlm_metrics.json"
VLM_ICC_SAMPLES = 5   # Number of crops to sample per cluster for ICC scoring
