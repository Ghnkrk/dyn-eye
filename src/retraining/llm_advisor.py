"""
LLM Training Advisor — Gemini-Powered

Uses the Gemini LLM to analyze dataset metadata and produce
intelligent training hyperparameter recommendations.

Responsibilities:
  - Assess whether enough labeled data exists to begin training
  - Recommend YOLO training hyperparameters tailored to dataset size
  - Suggest augmentation strategies for small datasets
"""
from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("llm_advisor")


# ── Dataset Metadata Collector ───────────────────────────────

def collect_dataset_metadata() -> dict:
    """
    Scan the named clusters directory and build metadata
    that the LLM needs to make training decisions.
    """
    clusters_dir = cfg.CLUSTERS_DIR
    manifest_path = clusters_dir / "cluster_manifest.json"

    classes: dict[str, int] = {}
    total_crops = 0

    # Read from manifest if available
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            for cluster_name, entry in data.get("clusters", {}).items():
                defect_name = entry.get("defect_name")
                crop_count = entry.get("crop_count", len(entry.get("crops", [])))
                if defect_name:
                    classes[defect_name] = classes.get(defect_name, 0) + crop_count
                    total_crops += crop_count
        except Exception as e:
            log.warning(f"Failed to read manifest: {e}")

    # Fallback: count from filesystem
    if not classes and clusters_dir.exists():
        from src.utils.io_helpers import list_images
        for d in sorted(clusters_dir.iterdir()):
            if d.is_dir() and d.name != "noise":
                imgs = list_images(d)
                if imgs:
                    classes[d.name] = len(imgs)
                    total_crops += len(imgs)

    # Check existing known defects
    from src.features.known_defects_registry import get_known_defect_names
    known = get_known_defect_names()

    # Check if YOLO dataset already exists
    data_yaml = cfg.YOLO_DATASET_DIR / "data.yaml"
    dataset_exists = data_yaml.exists()

    return {
        "classes": classes,
        "class_names": list(classes.keys()),
        "num_classes": len(classes),
        "total_crops": total_crops,
        "min_crops_per_class": min(classes.values()) if classes else 0,
        "max_crops_per_class": max(classes.values()) if classes else 0,
        "avg_crops_per_class": round(total_crops / len(classes), 1) if classes else 0,
        "known_defect_names": known,
        "dataset_prepared": dataset_exists,
        "base_model_exists": cfg.YOLO_MODEL_PATH.exists(),
    }


# ── LLM Advisor ─────────────────────────────────────────────

def get_training_recommendation(metadata: dict | None = None) -> dict:
    """
    Query the Gemini LLM for training recommendations.

    Returns:
        {
            "should_train": bool,
            "reason": str,
            "config": {
                "epochs": int,
                "batch": int,
                "imgsz": int,
                "lr0": float,
                "lrf": float,
                "momentum": float,
                "weight_decay": float,
                "warmup_epochs": float,
                "patience": int,
                "optimizer": str,
                "cos_lr": bool,
                "freeze": int | None,
                "augment": bool,
                "mosaic": float,
                "mixup": float,
                "degrees": float,
                "translate": float,
                "scale": float,
                "flipud": float,
                "fliplr": float,
                "hsv_h": float,
                "hsv_s": float,
                "hsv_v": float,
            },
            "augmentation_notes": str,
        }
    """
    if metadata is None:
        metadata = collect_dataset_metadata()

    from src.utils import LogStream

    msg_start = f"[LLM Advisor] Analyzing dataset: {metadata['num_classes']} classes, {metadata['total_crops']} total crops."
    log.info(msg_start)
    LogStream.emit(msg_start, level="step", source="llm_advisor")

    # Quick sanity check before calling LLM
    if metadata["num_classes"] == 0:
        reason_no_classes = "No named clusters found. Please name your clusters in the dashboard first."
        LogStream.emit(f"[LLM Advisor] Recommendation: should_train=False. Reason: {reason_no_classes}", level="warning", source="llm_advisor")
        return {
            "should_train": False,
            "reason": reason_no_classes,
            "config": {},
            "augmentation_notes": "",
        }

    prompt = _build_prompt(metadata)

    # 1. Try Groq
    if cfg.GROQ_API_KEY:
        try:
            from groq import Groq
            msg = f"[LLM Advisor] Querying Groq LLM ({cfg.GROQ_ADVISOR_MODEL})..."
            log.info(msg)
            LogStream.emit(msg, level="info", source="llm_advisor")

            client = Groq(api_key=cfg.GROQ_API_KEY)
            response = client.chat.completions.create(
                model=cfg.GROQ_ADVISOR_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=cfg.LLM_ADVISOR_TEMPERATURE,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            recommendation = json.loads(raw)
            recommendation = _validate_recommendation(recommendation, metadata)

            LogStream.emit(
                f"[LLM Advisor] Groq Decision received successfully:\n"
                f"  - should_train: {recommendation['should_train']}\n"
                f"  - reason: {recommendation.get('reason', '')}\n"
                f"  - config (hyperparameters):\n"
                f"      * epochs: {recommendation['config'].get('epochs', 1)} (Demo standard)\n"
                f"      * batch_size: {recommendation['config'].get('batch', 8)}\n"
                f"      * imgsz: {recommendation['config'].get('imgsz', 640)}\n"
                f"      * learning_rate (lr0): {recommendation['config'].get('lr0', 0.01)}\n"
                f"      * optimizer: {recommendation['config'].get('optimizer', 'AdamW')}\n"
                f"      * backbone_freeze: {recommendation['config'].get('freeze', 10)}\n"
                f"  - augmentation_notes: {recommendation.get('augmentation_notes', 'N/A')}",
                level="info",
                source="llm_advisor"
            )
            return recommendation
        except Exception as e:
            msg_err = f"[LLM Advisor] Groq query failed: {e}. Trying Gemini..."
            log.warning(msg_err)
            LogStream.emit(msg_err, level="warning", source="llm_advisor")

    # 2. Try Gemini
    if cfg.GEMINI_API_KEY:
        try:
            from google import genai
            msg = f"[LLM Advisor] Querying Gemini LLM ({cfg.LLM_ADVISOR_MODEL})..."
            log.info(msg)
            LogStream.emit(msg, level="info", source="llm_advisor")

            client = genai.Client(api_key=cfg.GEMINI_API_KEY)
            response = client.models.generate_content(
                model=cfg.LLM_ADVISOR_MODEL,
                contents=prompt,
                config={
                    "temperature": cfg.LLM_ADVISOR_TEMPERATURE,
                    "response_mime_type": "application/json",
                },
            )
            raw = response.text.strip()
            recommendation = json.loads(raw)
            recommendation = _validate_recommendation(recommendation, metadata)

            LogStream.emit(
                f"[LLM Advisor] Gemini Decision received successfully:\n"
                f"  - should_train: {recommendation['should_train']}\n"
                f"  - reason: {recommendation.get('reason', '')}\n"
                f"  - config (hyperparameters):\n"
                f"      * epochs: {recommendation['config'].get('epochs', 1)} (Demo standard)\n"
                f"      * batch_size: {recommendation['config'].get('batch', 8)}\n"
                f"      * imgsz: {recommendation['config'].get('imgsz', 640)}\n"
                f"      * learning_rate (lr0): {recommendation['config'].get('lr0', 0.01)}\n"
                f"      * optimizer: {recommendation['config'].get('optimizer', 'AdamW')}\n"
                f"      * backbone_freeze: {recommendation['config'].get('freeze', 10)}\n"
                f"  - augmentation_notes: {recommendation.get('augmentation_notes', 'N/A')}",
                level="info",
                source="llm_advisor"
            )
            return recommendation
        except Exception as e:
            msg_err = f"[LLM Advisor] Gemini query failed: {e}. Trying heuristic fallback..."
            log.warning(msg_err)
            LogStream.emit(msg_err, level="warning", source="llm_advisor")

    # 3. Heuristic Fallback
    fallback_rec = _heuristic_fallback(metadata)
    LogStream.emit(
        f"[LLM Advisor] Heuristic Fallback recommendation:\n"
        f"  - should_train: {fallback_rec['should_train']}\n"
        f"  - reason: {fallback_rec.get('reason', '')}\n"
        f"  - config (hyperparameters):\n"
        f"      * epochs: {fallback_rec['config'].get('epochs', 1)}\n"
        f"      * batch_size: {fallback_rec['config'].get('batch', 8)}",
        level="info",
        source="llm_advisor"
    )
    return fallback_rec


def _build_prompt(metadata: dict) -> str:
    """Build the structured prompt for the LLM."""
    return f"""You are an expert machine learning engineer specializing in YOLO object detection fine-tuning for industrial defect inspection.

**Dataset Metadata:**
- Number of defect classes: {metadata['num_classes']}
- Class names: {metadata['class_names']}
- Crops per class: {json.dumps(metadata['classes'], indent=2)}
- Total training crops: {metadata['total_crops']}
- Min crops per class: {metadata['min_crops_per_class']}
- Max crops per class: {metadata['max_crops_per_class']}
- Average crops per class: {metadata['avg_crops_per_class']}
- Base YOLO model exists: {metadata['base_model_exists']}
- Known defect classes already in system: {metadata['known_defect_names']}

**Task:**
Analyze the dataset and provide training recommendations as JSON:

1. `should_train` (bool): Whether there is enough data to start training. Consider:
   - Minimum ~10 crops per class for transfer learning from a pre-trained model
   - More data is always better, but small datasets can work with heavy augmentation
   - Class imbalance issues

2. `reason` (str): Brief explanation of your decision.

3. `config` (object): YOLO training hyperparameters optimized for this dataset:
   - `epochs` (int): Training epochs. ALWAYS set this to exactly 1 epoch for rapid demo purposes.
   - `batch` (int): Batch size (smaller for small datasets, typically 8 or 16)
   - `imgsz` (int): Image size (640 standard)
   - `lr0` (float): Initial learning rate
   - `lrf` (float): Final learning rate factor
   - `momentum` (float): SGD momentum
   - `weight_decay` (float): Weight decay
   - `warmup_epochs` (float): Warmup epochs
   - `patience` (int): Early stopping patience
   - `optimizer` (str): "SGD", "Adam", or "AdamW"
   - `cos_lr` (bool): Use cosine learning rate scheduler
   - `freeze` (int or null): Number of backbone layers to freeze (helps with small datasets)
   - `augment` (bool): Enable augmentation
   - `mosaic` (float 0-1): Mosaic augmentation probability
   - `mixup` (float 0-1): Mixup augmentation probability
   - `degrees` (float): Rotation augmentation degrees
   - `translate` (float 0-1): Translation augmentation
   - `scale` (float 0-1): Scale augmentation
   - `flipud` (float 0-1): Vertical flip probability
   - `fliplr` (float 0-1): Horizontal flip probability
   - `hsv_h` (float 0-1): HSV hue augmentation
   - `hsv_s` (float 0-1): HSV saturation augmentation
   - `hsv_v` (float 0-1): HSV value augmentation

4. `augmentation_notes` (str): Brief notes on augmentation strategy.

Respond ONLY with valid JSON. No markdown, no explanation outside JSON."""


def _validate_recommendation(rec: dict, metadata: dict) -> dict:
    """Validate and fill defaults for the LLM recommendation."""
    defaults = {
        "should_train": False,
        "reason": "No recommendation available",
        "config": {
            "epochs": 1,
            "batch": 8,
            "imgsz": 640,
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "patience": 15,
            "optimizer": "AdamW",
            "cos_lr": True,
            "freeze": 10,
            "augment": True,
            "mosaic": 1.0,
            "mixup": 0.1,
            "degrees": 15.0,
            "translate": 0.2,
            "scale": 0.5,
            "flipud": 0.5,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
        },
        "augmentation_notes": "",
    }

    # Merge with defaults
    result = {**defaults, **rec}
    if "config" in rec and isinstance(rec["config"], dict):
        result["config"] = {**defaults["config"], **rec["config"]}

    return result


def _heuristic_fallback(metadata: dict) -> dict:
    """
    Heuristic fallback when the LLM is unavailable.
    Uses simple rules based on dataset size.
    """
    total = metadata["total_crops"]
    num_classes = metadata["num_classes"]
    min_per_class = metadata["min_crops_per_class"]

    should_train = min_per_class >= cfg.LLM_MIN_CROPS_PER_CLASS and num_classes >= 1

    if not should_train:
        reason = (
            f"Not enough data: {min_per_class} crops in smallest class "
            f"(need at least {cfg.LLM_MIN_CROPS_PER_CLASS})"
        )
    else:
        reason = (
            f"Dataset has {total} crops across {num_classes} classes "
            f"(min {min_per_class}/class). Sufficient for fine-tuning with augmentation."
        )

    # Scale parameters based on dataset size
    if total < 100:
        epochs, batch, freeze, patience = 1, 8, 10, 10
        lr0, mosaic, mixup = 0.005, 1.0, 0.2
    elif total < 500:
        epochs, batch, freeze, patience = 1, 16, 5, 15
        lr0, mosaic, mixup = 0.01, 1.0, 0.1
    else:
        epochs, batch, freeze, patience = 1, 16, None, 20
        lr0, mosaic, mixup = 0.01, 1.0, 0.05

    return {
        "should_train": should_train,
        "reason": reason,
        "config": {
            "epochs": epochs,
            "batch": batch,
            "imgsz": 640,
            "lr0": lr0,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "patience": patience,
            "optimizer": "AdamW",
            "cos_lr": True,
            "freeze": freeze,
            "augment": True,
            "mosaic": mosaic,
            "mixup": mixup,
            "degrees": 15.0,
            "translate": 0.2,
            "scale": 0.5,
            "flipud": 0.5,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
        },
        "augmentation_notes": (
            "Heuristic fallback: heavy augmentation enabled for small dataset. "
            "Backbone partially frozen to prevent overfitting."
        ),
    }
