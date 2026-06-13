# DYN-EYE — Autonomous Defect Discovery & Self-Learning Pipeline

> **An end-to-end MLOps system** for industrial visual inspection.  
> YOLO detects known defects; unknown/novel anomalies are extracted, clustered with HDBSCAN, annotated with Gemma VLM (using domain-aware dynamic prompting), and automatically fed back to fine-tune YOLO — closing the loop with interactive dashboard curation.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Benchmark Results](#benchmark-results)
4. [Repository Structure](#repository-structure)
5. [Prerequisites](#prerequisites)
6. [Environment Setup](#environment-setup)
7. [Folder & File Placement Guide](#folder--file-placement-guide)
8. [Configuration Reference](#configuration-reference)
9. [Running the Project](#running-the-project)
10. [Dashboard Guide](#dashboard-guide)
11. [Pipeline Deep-Dive](#pipeline-deep-dive)
12. [Retraining & Model Versioning](#retraining--model-versioning)
13. [Troubleshooting](#troubleshooting)

---

## System Overview

```
Input Images ──► YOLO Inference ──► Known? ──► SKIP (already tracked)
                                 └─► Unknown crop ──► DINOv2 Feature Extraction
                                                    ──► FAISS Novelty Filter
                                                    ──► HDBSCAN Clustering
                                                    ──► Gemma VLM Annotation
                                                    ──► Statistical ICC Quality Score
                                                    ──► Dashboard Review & Labelling
                                                    ──► YOLO Dataset Generation
                                                    ──► Fine-tune YOLO (full/selective)
                                                    ──► Deploy & version new model
```

---

## Architecture

| Layer | Technology | Role |
|---|---|---|
| **Object Detection** | YOLOv8/v10/v11 (Ultralytics) | Detects known defects using the active model version |
| **Feature Extraction** | DINOv2 ViT-S/14 (384-dim) | Embeds candidate crops into a high-dimensional visual feature space |
| **Novelty Detection** | FAISS IndexFlatL2 | Filters known-looking crops based on registry embeddings |
| **Clustering** | HDBSCAN + DBCV grid search | Groups remaining novel anomaly crops into coherent clusters |
| **VLM Annotation** | Google Gemini (`gemma-4-31b-it`) | Detects and annotates visual anomaly traits within crops |
| **LLM Advisor & Prompting** | Groq (`llama-3.3-70b-versatile`) | Generates dynamic VLM prompts and advises on retraining readiness |
| **Cluster Quality Metric** | Statistical ICC (one-way ANOVA) | Measures cluster cohesion and separation using DINOv2 embeddings |
| **Pipeline Orchestration** | LangGraph (stateful DAG) | Coordinates discovery and retraining runs |
| **Experiment Tracking** | MLflow | Tracks model performance, cluster quality, and VLM run metrics |
| **Dashboard** | FastAPI + Vanilla JS | Custom UI for cluster management, labeling, and training monitoring |
| **Packaging** | `uv` (PEP 517, fast resolver) | Manages the Python environment and dependencies |

---

## Benchmark Results

Measured on a single representative run against a real industrial inspection dataset of **50 unknown images** (64 crops detected), with 6 known defect classes registered in FAISS.

### Pipeline Run — `run_id: 20260613_071448`

| Stage | Result |
|---|---|
| Input images | 50 |
| Crops extracted | 64 |
| Novel crops (post-FAISS filter) | 50 / 64 (78%) |
| HDBSCAN clusters formed | **2** |
| Crops per cluster | cluster_000: 22, cluster_001: 24 |
| Noise / unassigned crops | 4 noise + 4 unassigned |
| Dimensionality reduction | PCA 384D → 5D (UMAP fallback) |
| DBCV score | 0.0000 (flat — expected with PCA at small scale) |
| Registry hit rate | 100% (2/2 clusters matched fingerprints) |

### Cluster Quality — Statistical ICC (DINOv2 Embedding Space)

> The Intraclass Correlation Coefficient is computed purely from DINOv2 feature embeddings — no VLM API calls required. Two metrics are produced per run:

| Metric | Value | Interpretation |
|---|---|---|
| **Cluster 0 cohesion** | 0.7140 | Crops tightly grouped around centroid in feature space |
| **Cluster 1 cohesion** | 0.7591 | Slightly tighter — more visually homogeneous cluster |
| **Mean per-cluster cohesion** | **0.7366** | Both clusters are well-concentrated (>0.70 is good) |
| **Global ICC (one-way ANOVA)** | **0.1741** | Moderate between-cluster separation relative to within-cluster variance |

> **Cohesion** (per-cluster) = mean cosine similarity of each crop's embedding to its cluster centroid. Range [0, 1]. Above 0.70 indicates a visually consistent defect group.  
> **Global ICC** = standard one-way ANOVA decomposition across all clusters: `(MS_between − MS_within) / (MS_between + (k₀−1) × MS_within)`. Range [0, 1]. Higher values indicate cleaner cluster separation.

### Cache Mode vs Full Run

| Mode | VLM API calls | ICC computation | Runtime |
|---|---|---|---|
| Full run | ~50–80 calls (VLM annotation) | Instant (NumPy, embeddings) | ~3–5 min |
| `--use-cache` | 0 | Instant (NumPy, embeddings) | **< 30 seconds** |

---

## Repository Structure

```
Protosem2/
├── config.py                   # Central config — paths, thresholds, and VLM settings
├── main.py                     # CLI entry point (dashboard / discover / retrain / setup-faiss)
├── pyproject.toml              # uv-managed dependencies
│
├── src/
│   ├── pipeline/
│   │   ├── graph.py            # LangGraph DAG definition
│   │   ├── orchestrator.py     # FastAPI backend + SSE log streaming
│   │   ├── state.py            # Typed PipelineState dataclass
│   │   └── nodes/
│   │       ├── yolo_inference.py       # YOLO detection + known/unknown split
│   │       ├── dataset_context.py      # Dataset context builder + Groq dynamic prompt
│   │       ├── faiss_search.py         # Novelty filter vs. FAISS index
│   │       ├── feature_extraction.py   # DINOv2 embedding
│   │       ├── crop_extraction.py      # Bounding-box crop saver
│   │       ├── hdbscan_cluster.py      # HDBSCAN + DBCV grid search + registry
│   │       ├── vlm_annotation.py       # Gemma VLM annotation + retry logic
│   │       └── manifest_save.py        # Manifest save + statistical ICC + metrics
│   │
│   ├── retraining/
│   │   ├── agent.py            # LangGraph retraining agent
│   │   ├── llm_advisor.py      # Groq LLM advisor (when/how to retrain)
│   │   └── model_registry.py   # Model versioning, rollback, and FAISS sync
│   │
│   ├── features/
│   │   ├── known_defects_registry.py   # Hot-reload known class list
│   │   └── faiss_index.py              # FAISS index construction and query
│   │
│   └── utils/
│       ├── __init__.py         # Logger, LogStream, get_logger, save_json exports
│       ├── logger.py           # Structured SSE log emitter
│       ├── metrics.py          # MLflow metric tracker
│       └── vlm_metrics.py      # ICC + cluster quality metrics logging
│
├── dashboard/
│   ├── app.py                  # FastAPI app factory & all API routes
│   └── static/
│       ├── index.html          # Single-page dashboard (HTML5)
│       ├── style.css           # Vanilla dark theme
│       └── app.js              # SSE client + all UI interactions
│
├── data/                       # Runtime data (git-ignored, auto-created)
│   ├── input_images/           # Drop inspection images here
│   ├── crops/                  # YOLO-detected bounding-box crops
│   ├── clusters/               # HDBSCAN cluster folders + cluster_manifest.json
│   ├── faiss_index/            # FAISS index + label JSON
│   ├── known_defect_crops/     # Per-class seed crops for FAISS bootstrapping
│   ├── yolo_dataset/           # Auto-generated YOLO fine-tuning dataset
│   ├── known_defects.json      # Active list of known defect class names
│   ├── vlm_cache.json          # VLM response cache (avoids duplicate API calls)
│   └── vlm_metrics.json        # Historical ICC and quality metrics per run
│
├── models/
│   ├── best.pt                 # Active YOLO model (replaced on deploy)
│   ├── best_initial.pt         # Baseline model (never overwritten — used by Factory Reset)
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

> **CPU-only?** Replace the CUDA torch wheels in `pyproject.toml` with standard CPU wheels.

---

## Environment Setup

### 1. Clone and enter the repo

```bash
git clone <your-repo-url>
cd Protosem2
```

### 2. Install dependencies

```bash
pip install uv
uv sync
```

### 3. Create your `.env` file

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

**Keys needed:**
- **Gemini** → VLM anomaly annotation (`gemma-4-31b-it`)
- **Groq** → Dynamic VLM prompt generation + retraining advisor

### 4. Activate the environment

```powershell
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

---

## Folder & File Placement Guide

> All directories are **auto-created** by `config.py` on first import.

### Required: YOLO Model Weights

```
models/
├── best.pt           ← active model (used by pipeline)
└── best_initial.pt   ← backup baseline (used by Factory Reset)
```

### Optional: Seed images for FAISS index

```
data/known_defect_crops/
├── inclusion/
│   └── crop_001.jpg
└── <class_name>/
    └── ...
```

The folder name becomes the defect class label. The FAISS index is built automatically on first run or Factory Reset.

---

## Configuration Reference

| Variable | Default | Purpose |
|---|---|---|
| `YOLO_CONFIDENCE_THRESHOLD` | `0.30` | Min YOLO confidence to count a detection |
| `FAISS_NOVELTY_THRESHOLD` | `0.35` | L2 distance above which a crop is flagged as novel |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `4` | Min crops required to form a cluster |
| `HDBSCAN_MIN_SAMPLES` | `2` | HDBSCAN core-point density parameter |
| `VLM_SLEEP_BETWEEN` | `4.5` | Minimum sleep (seconds) between Gemma VLM API calls |
| `VLM_MAX_RETRIES` | `5` | Max retry attempts per VLM call |
| `YOLO_TRAIN_EPOCHS` | `1` | Default training epochs (overridable via dashboard) |
| `FEATURE_DIM` | `384` | DINOv2 ViT-S/14 output feature dimension |

---

## Running the Project

### Start the Dashboard (recommended)

```bash
uv run python main.py dashboard
```

Open → **http://localhost:8501**

### Run Pipeline Headlessly (full run)

```bash
uv run python main.py discover
```

### Fast Demo / Cache Mode

```bash
uv run python main.py discover --use-cache
```

Reuses cached YOLO detections and VLM annotations from `data/vlm_cache.json`. The statistical ICC computation still runs in real-time (it uses embeddings already in memory — no API calls). Completes in **< 30 seconds**.

---

## Dashboard Guide

### Pipeline Panel (Left)
- **Known Defects**: Current model class list.
- **Upload Zone**: Drag & drop images.
- **Run Pipeline**: Executes the full LangGraph discovery sequence.
- **VLM Prompt Viewer**: Non-intrusive collapsible section showing the dynamic prompt used for the current run.

### Cluster Management (Center)
- **Rename & Label**: Click any cluster card to assign a defect name (final labeling — done by the user, not the VLM).
- **Move / Drop**: Re-assign crops to other clusters or remove noisy detections.

### Execution Log & Retraining Monitor (Right)
- **Live Logs**: Real-time SSE-streamed pipeline step events.
- **Retraining Panel**: LLM advisor output, fine-tuning triggers, and custom parameter controls.

---

## Pipeline Deep-Dive

### 1. YOLO Inference (`yolo_inference.py`)
Runs `models/best.pt` on all input images. Detections matching known classes are skipped; the rest move forward as **unknown candidates**.

### 2. Crop Extraction (`crop_extraction.py`)
Saves candidate bounding boxes to `data/crops/` and records spatial metadata in `data/crop_to_source.json`.

### 3. Feature Extraction (`feature_extraction.py`)
Encodes each crop as a 384-dimensional L2-normalized vector using DINOv2 ViT-S/14.

### 4. FAISS Novelty Filter (`faiss_search.py`)
Compares embeddings against the indexed known-defect vectors. Crops closer than `FAISS_NOVELTY_THRESHOLD` are discarded as already-known, reducing unnecessary downstream processing.

### 5. HDBSCAN Clustering (`hdbscan_cluster.py`)
Groups the remaining novel crops using HDBSCAN, with an automated DBCV grid search to select the optimal `min_cluster_size` and `min_samples` per run. Produces named cluster folders and a fingerprint registry for run-to-run cluster identity tracking.

### 6. Dynamic VLM Prompting & Annotation (`dataset_context.py` + `vlm_annotation.py`)
- **Dynamic Prompting**: Groq (`llama-3.3-70b-versatile`) generates a domain-aware system prompt from the run context (novelty ratio, known classes, batch size). Falls back to a static default prompt if Groq is unavailable.
- **VLM Annotation**: Gemma (`gemma-4-31b-it`) receives individual crops and the system prompt, returning structured bounding box annotations and physical trait descriptions. The VLM performs **detection and annotation only** — it does not assign final defect names.
- **Rate-limit resilience**: 4.5 s sleep between calls; 503/overload errors trigger a linear-exponential backoff scaling up to 24 s across retries.

### 7. Statistical ICC Quality Scoring (`manifest_save.py`)
After clustering, the pipeline computes the Intraclass Correlation Coefficient (ICC) directly from the DINOv2 embeddings already in memory — **zero extra API calls**.

**Per-cluster cohesion** (mean cosine similarity of each crop to its cluster centroid):
```
cohesion_i = mean( embed_j · centroid_i )   for all crops j in cluster i
```

**Global one-way ANOVA ICC**:
```
ICC = (MS_between − MS_within) / (MS_between + (k₀ − 1) × MS_within)
```
where `k₀` is the unequal-group-size correction factor. This is the same ICC(1) formula from Shrout & Fleiss (1979), applied across the 384-dimensional embedding space.

Results and interpretation from the benchmark run:

| Metric | Value |
|---|---|
| Cluster 0 cohesion | 0.7140 |
| Cluster 1 cohesion | 0.7591 |
| Mean cohesion | **0.7366** |
| Global ICC | **0.1741** |

Both clusters show high cohesion (> 0.70), meaning the HDBSCAN groupings are visually consistent in DINOv2's feature space. The moderate global ICC reflects the fact that both clusters are compact but occupy overlapping regions of the broader embedding space — expected for subtle surface-level defect variations.

### 8. Dashboard Review & YOLO Dataset Generation
Users review clusters, name the defect classes, and optionally drop noisy crops. The system generates YOLO-format annotation files and queues retraining.

---

## Retraining & Model Versioning

1. **Fine-Tuning**: Trains YOLO on the expanded dataset including newly discovered defect classes.
2. **Registry Tracking**: Stores model versions in `models/registry.json` with metrics, class list, and timestamp.
3. **Deployment**: Promotes the new model to `models/best.pt` and rebuilds the FAISS index.
4. **Rollback**: Restores any previous checkpoint and realigns FAISS to that version's class set.

---

## Troubleshooting

### VLM Rate Limits (503 / High Demand)
The pipeline applies automatic backoff — up to 24 s for 503 errors. To bypass VLM entirely, run with `--use-cache`.

### FAISS Index Out of Sync
```bash
uv run python main.py setup-faiss
```
Or use **Factory Reset** in the dashboard.

### Dashboard Port In Use
```bash
netstat -ano | findstr :8501
uv run python main.py dashboard
```

---
*DYN-EYE — Autonomous Anomaly Detection & Self-Learning Pipeline*
