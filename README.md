# DYN-EYE — Autonomous Defect Discovery & Self-Learning Pipeline

> **An end-to-end MLOps system** for industrial visual inspection.  
> YOLO detects known defects; unknown/novel anomalies are extracted, clustered with HDBSCAN, annotated with Gemma VLM (using domain-aware dynamic prompting), and automatically fed back to fine-tune YOLO — closing the loop with interactive dashboard curation.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Environment Setup](#environment-setup)
6. [Folder & File Placement Guide](#folder--file-placement-guide)
7. [Configuration Reference](#configuration-reference)
8. [Running the Project](#running-the-project)
9. [Dashboard Guide](#dashboard-guide)
10. [Pipeline Deep-Dive](#pipeline-deep-dive)
11. [Retraining & Model Versioning](#retraining--model-versioning)
12. [Troubleshooting](#troubleshooting)

---

## System Overview

```
Input Images ──► YOLO Inference ──► Known? ──► SKIP (already tracked)
                                 └─► Unknown crop ──► DINOv2 Feature Extraction
                                                    ──► FAISS Novelty Filter
                                                    ──► HDBSCAN Clustering
                                                    ──► Gemma VLM Annotation
                                                    ──► Dashboard Review & Labelling
                                                    ──► YOLO Dataset Generation
                                                    ──► Fine-tune YOLO (full/selective)
                                                    ──► Deploy & version new model
```

---

## Architecture

| Layer | Technology | Role |
|---|---|---|
| **Object Detection** | YOLOv8/v10/v11 (Ultralytics) | Detects known defects using active model version |
| **Feature Extraction** | DINOv2 ViT-S/14 (384-dim) | Embeds candidate crops into high-dimensional space |
| **Novelty Detection** | FAISS IndexFlatL2 | Filters out known-looking crops based on registry embeddings |
| **Clustering** | HDBSCAN | Groups remaining novel anomaly crops into clusters |
| **VLM Annotation** | Google Gemini (`gemma-4-31b-it`) | Detects visual traits and annotates anomalies inside crops |
| **LLM Advisor & Prompting** | Groq (`llama-3.3-70b-versatile`) | Generates dynamic VLM prompts & advises on retraining |
| **Pipeline Orchestration** | LangGraph (stateful DAG) | Coordinates discovery and retraining runs |
| **Experiment Tracking** | MLflow | Tracks model performance metrics and VLM runs |
| **Dashboard** | FastAPI + Vanilla JS + CSS | Custom UI for cluster management, labeling, and training monitoring |
| **Packaging** | `uv` (PEP 517, fast resolver) | Manages python environment and dependencies |

---

## Repository Structure

```
Protosem2/
├── config.py                   # Central config — paths, thresholds, and VLM settings
├── main.py                     # CLI entry point (dashboard / discover / retrain / setup-faiss)
├── pyproject.toml              # uv-managed dependencies
├── README.md                   # System documentation
│
├── src/
│   ├── pipeline/
│   │   ├── graph.py            # LangGraph DAG definition
│   │   ├── orchestrator.py     # FastAPI backend + SSE log streaming
│   │   ├── state.py            # Typed PipelineState dataclass
│   │   └── nodes/
│   │       ├── yolo_inference.py       # YOLO detection + known/unknown split
│   │       ├── faiss_search.py         # Novelty filter vs. FAISS index
│   │       ├── feature_extraction.py   # DINOv2 embedding
│   │       ├── crop_extraction.py      # Bounding-box crop saver
│   │       ├── hdbscan_cluster.py      # HDBSCAN clustering
│   │       └── vlm_annotation.py       # Gemma VLM annotation & ICC scoring
│   │
│   ├── retraining/
│   │   ├── agent.py            # LangGraph retraining agent
│   │   ├── llm_advisor.py      # Groq LLM advisor (when/how to retrain)
│   │   └── model_registry.py   # Model versioning, rollback, and FAISS sync
│   │
│   ├── features/
│   │   ├── known_defects_registry.py   # Hot-reload known class list
│   │   └── faiss_index.py              # FAISS index construction/query
│   │
│   └── utils/
│       ├── __init__.py         # Logger, LogStream, get_logger exports
│       ├── logger.py           # Structured SSE log emitter
│       ├── metrics.py          # MLflow metric tracker
│       └── vlm_metrics.py      # VLM consistency & performance metrics logging
│
├── dashboard/
│   ├── app.py                  # FastAPI app factory & routes
│   └── static/
│       ├── index.html          # Dashboard page (HTML5)
│       ├── style.css           # Vanilla dark theme styling
│       └── app.js              # SSE client + all UI interactions & stats
│
├── data/                       # Runtime data (git-ignored, created automatically)
│   ├── input_images/           # Drop inspection images here
│   ├── crops/                  # YOLO-detected bounding-box crops
│   ├── clusters/               # HDBSCAN cluster folders and cluster_manifest.json
│   ├── faiss_index/            # FAISS index + label JSON
│   ├── known_defect_crops/     # Per-class seed crops for FAISS bootstrapping
│   ├── yolo_dataset/           # Auto-generated YOLO fine-tuning dataset
│   ├── known_defects.json      # Active list of known defect class names
│   ├── vlm_cache.json          # VLM response cache (avoids duplicate API calls)
│   └── vlm_icc_cache.json      # VLM consistency (ICC) cache for fast demo runs
│
├── models/
│   ├── best.pt                 # Active YOLO model (symlink-like — replaced on deploy)
│   ├── best_initial.pt         # Original baseline model (never overwritten)
│   ├── registry.json           # Model version registry
│   └── versions/               # Archived fine-tuned model checkpoints
│
├── logs/                       # Runtime logs (git-ignored)
└── runs/                       # YOLO training run artifacts (git-ignored)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.12 | 3.12 recommended |
| CUDA | 13.0 | For PyTorch GPU (`torch==2.11.0+cu130`) |
| `uv` | latest | `pip install uv` |
| Git | any | For version control |

---

## Environment Setup

### 1. Clone and enter the repo

```bash
git clone <your-repo-url>
cd Protosem2
```

### 2. Create virtual environment and install dependencies

```bash
# Install uv if not already installed
pip install uv

# Create venv and sync all dependencies (reads pyproject.toml)
uv sync
```

### 3. Create your `.env` file

```bash
# .env  (never commit this file — it is in .gitignore)
GEMINI_API_KEY=your_google_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

**API keys needed:**
- **Gemini** → VLM anomaly annotation
- **Groq** → LLM dynamic prompt generator and training advisor

### 4. Activate the environment

```powershell
# Windows PowerShell
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

---

## Folder & File Placement Guide

> All directories are **auto-created** by `config.py` on first import.

### Required: YOLO Model Weights

Place your baseline YOLO weights under `models/`:

```
models/
├── best.pt           ← active model (used by pipeline)
└── best_initial.pt   ← backup baseline (used by Factory Reset)
```

### Optional: Seed images for FAISS index

To bootstrap the FAISS novelty filter with known-class examples, place crop images under:

```
data/known_defect_crops/
├── inclusion/
│   ├── crop_001.jpg
│   └── crop_002.jpg
└── <class_name>/
    └── ...
```

The folder name becomes the class label. The FAISS index is built automatically when the pipeline first runs (or on Factory Reset).

---

## Configuration Reference

All parameters are configured in `config.py`. Key settings:

| Variable | Default | Purpose |
|---|---|---|
| `YOLO_CONFIDENCE_THRESHOLD` | `0.30` | Min YOLO confidence to count a detection |
| `FAISS_NOVELTY_THRESHOLD` | `0.35` | L2 distance threshold above which a crop is "unknown/novel" |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `4` | Min crops required to form a cluster |
| `HDBSCAN_MIN_SAMPLES` | `2` | HDBSCAN core-point density parameter |
| `VLM_ICC_SAMPLES` | `5` | Number of representative crops evaluated for cluster consistency |
| `VLM_SLEEP_BETWEEN` | `4.5` | Sleep delay (seconds) between VLM calls to avoid RPM limits |
| `YOLO_TRAIN_EPOCHS` | `1` | Default epochs for YOLO retraining |
| `FEATURE_DIM` | `384` | DINOv2 ViT-S/14 output feature dimension |

---

## Running the Project

### Start the Dashboard (recommended)

```bash
uv run python main.py dashboard
```

Open → **http://localhost:8501**

The interactive dashboard serves as the central control panel to:
- Upload new datasets or use default files
- Trigger the discovery pipeline
- Curate clusters, move/drop crops, and define defect names (labels)
- Monitor YOLO retraining logs and deploy model versions
- Reset the environment to its initial state

### Run Pipeline Headlessly

```bash
uv run python main.py discover
```

Runs the full pipeline on `data/input_images/` without launching the UI.

### Run with Cache (Fast Demo Mode)

```bash
uv run python main.py discover --use-cache
```

Skips actual YOLO inferences and VLM anomaly annotations by loading cached outputs from `data/vlm_cache.json` and `data/vlm_icc_cache.json`. This provides an instant pipeline demo.

---

## Dashboard Guide

### Pipeline Panel (Left)
- **Known Defects**: List of classes registered in the current active model.
- **Upload Zone**: Drag & drop images or use standard upload.
- **Run Pipeline**: Executes the full LangGraph discovery sequence.
- **VLM Prompt Viewer**: Non-intrusive collapsible panel displaying the dynamic prompt generated for the run.

### Cluster Management (Center)
- Displays grouped crops representing novel anomalies.
- **Rename & Label**: Click on a cluster card to name the anomaly class (labeling).
- **Move / Drop**: Re-assign crops to other clusters or drop outliers/noise to prune the training dataset.

### Execution Log & Retraining Monitor (Right)
- **Live Logs**: Real-time SSE logs detailing pipeline steps and processing status.
- **Retraining Panel**: Fine-tuning triggers, custom training parameter edits, and LLM advice.

---

## Pipeline Deep-Dive

### 1. YOLO Inference (`yolo_inference.py`)
- Runs the active YOLO model version (`models/best.pt`) on input images.
- Detections matching known defects are skipped. Bounding boxes with unknown traits or low registry confidence are sent forward as **unknown defect candidates**.

### 2. Crop Extraction (`crop_extraction.py`)
- Extracts candidate bounding box crops and saves them to `data/crops/`.
- Tracks crop mapping in `data/crop_to_source.json` to preserve spatial metadata.

### 3. Feature Extraction (`feature_extraction.py`)
- Generates 384-dimensional normalized feature embeddings using a pre-trained DINOv2 ViT-S/14 model.

### 4. FAISS Novelty Filter (`faiss_search.py`)
- Compares embeddings against the FAISS index of known defects.
- Crops with an L2 distance greater than `FAISS_NOVELTY_THRESHOLD` are flagged as truly novel. Known-looking crops are filtered out to prevent redundant VLM cost.

### 5. HDBSCAN Clustering (`hdbscan_cluster.py`)
- Clusters remaining novel crops based on their embedding vectors.
- Assigns crops to distinct cluster folders representing unique defect classes.

### 6. VLM Annotation (`vlm_annotation.py`)
- **Dynamic Prompting**: Groq (`llama-3.3-70b-versatile`) uses the dataset context (novelty ratio, known classes) to generate a custom, domain-aware VLM system prompt. If unavailable, it falls back to a static default.
- **Anomaly Annotation**: Gemma VLM (`gemma-4-31b-it`) detects visual traits and draws bounding boxes on representative crops. It does **not** perform final labeling.
- **Intra-Cluster Consistency (ICC)**: Measures consistency across a sample of 5 crops per cluster. Uses a smart retry backoff with linear scaling (up to 24s) to survive API rate-limits, and caches results to `vlm_icc_cache.json` for fast demo runs.

### 7. Dataset Generation & Fine-tuning
- Users review and approve defect names (final labels) in the dashboard.
- The system generates YOLO annotation labels (`data/yolo_dataset/`).
- The retraining agent trains YOLO on the augmented dataset and rebuilds the FAISS index with the new class seed embeddings.

---

## Retraining & Model Versioning

1. **Fine-Tuning**: Trains the model on the expanded dataset.
2. **Registry Tracking**: Registers model versions in `models/registry.json` along with training metadata, classes, and performance metrics.
3. **Deployment**: Deploys the model as the active `models/best.pt` and updates the active FAISS index.
4. **Rollback**: Restores any previous model checkpoint and rebuilds the FAISS index to align with that version's classes.

---

## Troubleshooting

### VLM Rate Limits (503 / High Demand)
- The pipeline handles Gemini API 503 limits automatically via linear-exponential backoff (up to 24s).
- You can manually clear the cache file `data/vlm_cache.json` to force fresh VLM calls.
- If you hit rate limits frequently, verify `VLM_SLEEP_BETWEEN` is set to `4.5` or higher in `config.py`.

### FAISS Index Out of Sync
- Run `main.py setup-faiss` to manually rebuild the index from the seed crops.
- Or click **Factory Reset** in the dashboard settings to restore clean defaults.

---
*DYN-EYE — Autonomous Anomaly Detection & Self-Learning Pipeline*
