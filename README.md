# DYN-EYE — Autonomous Defect Discovery & Self-Learning Pipeline

> **An end-to-end MLOps system** for industrial visual inspection.  
> YOLO detects known defects; unknown/novel anomalies are extracted, clustered with HDBSCAN, labelled with Gemini VLM, and automatically fed back to fine-tune YOLO — closing the loop without human annotation.

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
12. [.gitignore Notes](#gitignore-notes)
13. [Troubleshooting](#troubleshooting)

---

## System Overview

```
Input Images ──► YOLO Inference ──► Known? ──► SKIP (already tracked)
                                 └─► Unknown crop ──► DINOv2 Feature Extraction
                                                    ──► FAISS Novelty Filter
                                                    ──► HDBSCAN Clustering
                                                    ──► Gemini VLM Labelling
                                                    ──► YOLO Dataset Generation
                                                    ──► Fine-tune YOLO (LoRA/full)
                                                    ──► Deploy & version new model
```

---

## Architecture

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8/v10/v11 (Ultralytics) |
| Feature Extraction | DINOv2 ViT-S/14 (384-dim) |
| Novelty Detection | FAISS IndexFlatL2 |
| Clustering | HDBSCAN |
| VLM Annotation | Google Gemini (`gemma-4-31b-it`) |
| LLM Training Advisor | Groq (`llama-3.3-70b-versatile`) |
| Pipeline Orchestration | LangGraph (stateful DAG) |
| Experiment Tracking | MLflow |
| Dashboard | FastAPI + Vanilla JS (SSE streaming) |
| Packaging | `uv` (PEP 517, fast resolver) |

---

## Repository Structure

```
dyn-eye/
├── config.py                   # Central config — all paths, thresholds, API keys
├── main.py                     # CLI entry point  (dashboard / pipeline / reset)
├── pyproject.toml              # uv-managed dependencies
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
│   │       ├── hdbscan_cluster.py      # HDBSCAN + cluster management
│   │       ├── vlm_annotation.py       # Gemini VLM labelling + caching
│   │       └── label_studio_sync.py    # (legacy, unused)
│   │
│   ├── retraining/
│   │   ├── agent.py            # LangGraph retraining agent
│   │   ├── llm_advisor.py      # Groq LLM advisor (when/how to retrain)
│   │   ├── model_registry.py   # Model versioning, rollback, FAISS sync
│   │   └── tools/
│   │       └── dvc_version.py  # DVC-backed dataset versioning
│   │
│   ├── features/
│   │   └── known_defects_registry.py   # Hot-reload known class list
│   │
│   └── utils/
│       ├── __init__.py         # Logger, LogStream, get_logger exports
│       ├── logger.py           # Structured SSE log emitter
│       └── metrics.py          # MLflow metric tracker
│
├── dashboard/
│   ├── app.py                  # FastAPI app factory
│   └── static/
│       ├── index.html          # Single-page dashboard
│       ├── style.css           # Pitch-black dark theme
│       └── app.js              # SSE client + all UI interactions
│
├── data/                       # Runtime data (git-ignored, created automatically)
│   ├── input_images/           # Drop inspection images here
│   ├── crops/                  # YOLO-detected bounding-box crops
│   ├── clusters/               # HDBSCAN cluster folders (one per cluster)
│   ├── faiss_index/            # FAISS index + label JSON
│   ├── known_defect_crops/     # Per-class seed crops for FAISS bootstrapping
│   ├── yolo_dataset/           # Auto-generated YOLO fine-tuning dataset
│   ├── known_defects.json      # Active list of known defect class names
│   ├── unknown_defects.json    # Detected unknown defect metadata
│   └── vlm_cache.json          # VLM response cache (avoids duplicate API calls)
│
├── models/
│   ├── best.pt                 # Active YOLO model (symlink-like — replaced on deploy)
│   ├── best_initial.pt         # Original baseline model (never overwritten)
│   ├── registry.json           # Model version registry
│   └── versions/               # Archived fine-tuned model checkpoints
│
├── logs/                       # Runtime logs (git-ignored)
├── runs/                       # YOLO training run artifacts (git-ignored)
│
├── Dockerfile
├── docker-compose.yml
├── .env                        # API keys (git-ignored — see below)
└── .gitignore
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.12 | 3.12 recommended |
| CUDA | 13.0 | For PyTorch GPU (`torch==2.11.0+cu130`) |
| `uv` | latest | `pip install uv` |
| Git | any | For version control |

> **CPU-only?** Replace `torch==2.11.0+cu130` and `torchvision==0.26.0+cu130` in `pyproject.toml` with the standard CPU wheels and remove the `[[tool.uv.index]]` block.

---

## Environment Setup

### 1. Clone and enter the repo

```bash
git clone https://github.com/Ghnkrk/dyn-eye
cd dyn-eye
```

### 2. Create virtual environment and install dependencies

```bash
# Install uv if not already installed
pip install uv

# Create venv and sync all dependencies (reads pyproject.toml)
uv sync
```

> This installs PyTorch 2.11.0 + CUDA 13.0 from the PyTorch index automatically.

### 3. Create your `.env` file

```bash
# .env  (never commit this file — it is in .gitignore)
GEMINI_API_KEY=your_google_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

**API keys needed:**
- **Gemini** → [aistudio.google.com](https://aistudio.google.com) — used for VLM annotation
- **Groq** → [console.groq.com](https://console.groq.com) — used for LLM retraining advisor

### 4. Activate the environment

```powershell
# Windows PowerShell
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

---

## Folder & File Placement Guide

> All directories are **auto-created** by `config.py` on first import. You only need to manually place the model weights.

### Required: YOLO Model Weights

Place your trained YOLOv8/v10/v11 model as:

```
models/
├── best.pt           ← active model (used by pipeline)
└── best_initial.pt   ← backup baseline (used by Factory Reset)
```

Both files must be present. If you only have one checkpoint, copy it to both names:

```bash
cp models/best.pt models/best_initial.pt
```

### Optional: Seed images for FAISS index

To bootstrap the FAISS novelty filter with known-class examples, place crop images under:

```
data/known_defect_crops/
├── inclusion/
│   ├── crop_001.jpg
│   └── crop_002.jpg
├── scratch/
│   └── crop_001.jpg
└── <class_name>/
    └── ...
```

The folder name becomes the class label. The FAISS index is built automatically when the pipeline first runs (or on Factory Reset).

### Optional: Input Images

Drop the images you want to inspect into:

```
data/input_images/
├── frame_0001.jpg
├── frame_0002.jpg
└── ...
```

Or use the **dashboard drag-and-drop zone** to upload a folder at runtime.

### Known Defects Registry

On first run, `data/known_defects.json` is created automatically from the YOLO model's class names. Format:

```json
{
  "defect_classes": ["inclusion", "scratch", "oil_spot", "..."]
}
```

You can edit this file manually to add/remove classes before training.

---

## Configuration Reference

All knobs live in `config.py`. Key settings:

| Variable | Default | Purpose |
|---|---|---|
| `YOLO_CONFIDENCE_THRESHOLD` | `0.30` | Min YOLO confidence to count a detection |
| `FAISS_NOVELTY_THRESHOLD` | `0.35` | L2 dist above which a crop is "unknown" |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `4` | Min crops to form a cluster |
| `HDBSCAN_MIN_SAMPLES` | `2` | HDBSCAN core-point threshold |
| `VLM_SLEEP_BETWEEN` | `4.5s` | Rate-limit pause between Gemini calls |
| `YOLO_TRAIN_EPOCHS` | `1` | Default training epochs (overridable via dashboard) |
| `FEATURE_DIM` | `384` | DINOv2 ViT-S/14 output dimension |
| `LLM_MIN_CROPS_PER_CLASS` | `10` | Min crops before LLM advisor suggests retraining |
| `DASHBOARD_PORT` | `8501` | Dashboard web server port |

---

## Running the Project

### Start the Dashboard (recommended)

```bash
uv run python main.py dashboard
```

Open → **http://localhost:8501**

This starts the FastAPI server and serves the interactive dashboard where you can:
- Upload images or point to the default dataset
- Run the full discovery pipeline
- Review clusters and confirm defect labels
- Trigger YOLO fine-tuning
- Manage model versions (deploy / rollback)
- Factory reset the entire system

### Run Pipeline Headlessly

```bash
uv run python main.py pipeline
```

Runs the full pipeline on `data/input_images/` without the dashboard.

### Factory Reset

```bash
uv run python main.py reset
```

Or click **Factory Reset** in the dashboard Settings. This:
- Restores `models/best.pt` ← `models/best_initial.pt`
- Resets `data/known_defects.json` to baseline classes
- Clears all crops, clusters, YOLO dataset
- Rebuilds FAISS index from `data/known_defect_crops/`
- Clears the model version registry

---

## Dashboard Guide

### Pipeline Control Panel (Left)

| Section | What it does |
|---|---|
| **Known Defects** | Auto-synced list of current model classes |
| **Upload Zone** | Drag & drop images/folder → sets `data/input_images/` |
| **Run Pipeline** | Triggers the full LangGraph DAG |
| **Collapse ▬** | Minimises pipeline into a slim progress tray |

### Discovery Pipeline (Steps)

The pipeline tray shows real-time progress through nodes:

1. YOLO Inference
2. Crop Extraction
3. Feature Extraction (DINOv2)
4. FAISS Novelty Filter
5. HDBSCAN Clustering
6. VLM Annotation (Gemini)
7. YOLO Dataset Generation
8. Fine-tuning Ready

### Cluster Editor

After pipeline completes, each cluster appears as a card. Click a cluster to:
- View all crops in that cluster
- Set a defect label (name the class)
- Move crops between clusters
- Drop noisy/wrong crops
- Delete empty clusters

Confirming a cluster name adds it to the known defects registry and queues it for retraining.

### Right Panel — Live Logs & Fine-tuning

- **Live Pipeline Events** → SSE-streamed log of every pipeline step
- **Fine-tuning Monitor** → Epoch-by-epoch YOLO training log
- Click **Expand ↗** on either to open a full-screen terminal with pure-black background

### Model Registry

- Compact view shows the active model version
- Click **Expand ✚** to open the full registry modal
- Hover **🏷️** icon → see class list; hover **📊** → see metrics
- Use **Deploy** / **Rollback** to switch active model versions

---

## Pipeline Deep-Dive

### 1. YOLO Inference (`yolo_inference.py`)
- Runs `best.pt` on every image in `data/input_images/`
- Detections with confidence ≥ `YOLO_CONFIDENCE_THRESHOLD` and class in known defects → logged, skipped
- All other bounding boxes → passed forward as **unknown candidates**

### 2. Crop Extraction (`crop_extraction.py`)
- Saves each unknown bounding box as a cropped JPG to `data/crops/`
- Maintains `data/crop_to_source.json` for traceability

### 3. Feature Extraction (`feature_extraction.py`)
- Loads DINOv2 ViT-S/14 from `torch.hub`
- Embeds each crop to a 384-dim L2-normalised vector

### 4. FAISS Novelty Filter (`faiss_search.py`)
- Queries the FAISS flat index of known-class embeddings
- Squared L2 distance > `FAISS_NOVELTY_THRESHOLD` → truly novel
- Known-looking crops are discarded here (reduces VLM cost)

### 5. HDBSCAN Clustering (`hdbscan_cluster.py`)
- Groups novel crops into clusters using HDBSCAN
- Creates `data/clusters/<cluster_N>/` folders
- Empty clusters (noise points) are pruned automatically

### 6. VLM Annotation (`vlm_annotation.py`)
- Sends each cluster's representative crops to Gemini
- Receives structured defect label + confidence
- Results cached in `data/vlm_cache.json` to avoid re-billing

### 7. Dataset Generation & Fine-tuning
- Confirmed clusters → YOLO format labels written to `data/yolo_dataset/`
- `src/retraining/agent.py` runs the LangGraph retraining DAG
- Groq LLM advisor recommends epochs/imgsz/batch or defers training
- Fine-tuned model saved to `models/versions/` and deployed to `models/best.pt`
- FAISS index rebuilt to include new class embeddings

---

## Retraining & Model Versioning

Every fine-tuning run:
1. Trains YOLO on the augmented dataset (base classes + new classes)
2. Saves checkpoint → `models/versions/<version_id>/best.pt`
3. Updates `models/registry.json` with metrics, class list, timestamp
4. Rebuilds FAISS index for the new class set
5. Deploys new model as `models/best.pt`

**Rollback** swaps `best.pt` back to any previous version and rebuilds FAISS to match that version's class list.

---

## .gitignore Notes

The `.gitignore` is correctly structured. Key observations:

| Pattern | Reason |
|---|---|
| `data/` + `!data/.gitkeep` | Runtime data excluded; placeholder keeps folder in git |
| `models/*.pt` | Model weights (large binary) excluded |
| `models/versions/` | Fine-tuned checkpoints excluded |
| `logs/`, `runs/` | Training artifacts excluded |
| `.env` | API keys never committed |
| `.venv/` | Virtual environment excluded |
| `*.dvc`, `dvc.lock` | DVC metadata excluded (use DVC remote for data) |
| `mlruns/` | MLflow local store excluded |

### ⚠️ Issues to fix before pushing

1. **`models/best_initial.pt` is NOT ignored** — at 51 MB this will bloat the repo. Add to `.gitignore`:
   ```
   models/*.pt
   ```
   This already covers it — but ensure `best_initial.pt` hasn't been tracked yet:
   ```bash
   git rm --cached models/best_initial.pt models/best.pt 2>/dev/null
   ```

2. **`data/label_studio_sync_pending.json`** and **`data/manifest_save_pending.json`** are data-dir files that get generated at runtime — already covered by `data/` ignore.

3. **`uv.lock`** (1.4 MB) — currently **not ignored** and should be committed (it's the lockfile equivalent of `poetry.lock`). This is correct — keep it.

4. **`yolo26n.pt`** in project root — this 5.5 MB file is **NOT ignored**. Add it:
   ```
   # Add to .gitignore
   *.pt
   ```
   Or move it into `models/` where it's already covered.

5. **`__pycache__/`** in root — already in `.gitignore`, but run:
   ```bash
   git rm -r --cached __pycache__/ 2>/dev/null
   ```

---

## Troubleshooting

### Dashboard doesn't start
```bash
# Ensure port 8501 is free
netstat -ano | findstr :8501
# Then run
uv run python main.py dashboard
```

### FAISS index empty / novelty filter broken
```bash
# Rebuild FAISS from scratch using known_defect_crops/
uv run python -c "from src.retraining.model_registry import rebuild_faiss_index; rebuild_faiss_index()"
```
Or click **Factory Reset** in the dashboard.

### VLM annotation stuck / skipping
- Check `GEMINI_API_KEY` is set in `.env`
- Check `data/vlm_cache.json` — delete it to force fresh annotations
- Increase `VLM_SLEEP_BETWEEN` in `config.py` if hitting rate limits

### Groq LLM advisor not triggering
- Check `GROQ_API_KEY` is set in `.env`
- Ensure clusters have ≥ `LLM_MIN_CROPS_PER_CLASS` (default: 10) crops

### YOLO training fails
- Verify `data/yolo_dataset/images/train/` has images
- Check `runs/` for the YOLO training run log for the specific error
- Reduce `YOLO_TRAIN_BATCH` in `config.py` if OOM

### `torch` CUDA version mismatch
```bash
python -c "import torch; print(torch.version.cuda)"
# Should print 13.0
# If not, reinstall:
uv sync --reinstall-package torch torchvision
```

---

## Quick-Start Checklist

```
✅ Clone repo
✅ uv sync
✅ Create .env with GEMINI_API_KEY and GROQ_API_KEY
✅ Place models/best.pt  (your trained YOLOv8 checkpoint)
✅ Place models/best_initial.pt  (copy of best.pt for factory reset)
✅ (Optional) Seed data/known_defect_crops/<class>/ with example crops
✅ Drop inspection images into data/input_images/
✅ uv run python main.py dashboard
✅ Open http://localhost:8501
✅ Click Run Pipeline
```

---

*DYN-EYE — Autonomous Anomaly Detection & Self-Learning Pipeline*
