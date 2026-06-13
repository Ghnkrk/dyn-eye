# dyn-eye — Agent Implementation Instructions

You are working on the **dyn-eye** repo at `https://github.com/Ghnkrk/dyn-eye`.
Read the entire codebase before making any changes. Understand the LangGraph DAG in
`src/pipeline/graph.py`, the state shape in `src/pipeline/state.py`, and how nodes
communicate before touching anything.

Implement the three improvements below in order. Each section tells you what to build
and why — figure out the how from the existing code patterns.

---

## Part 1 — Fix clustering determinism and stability

### Problem
HDBSCAN produces different cluster assignments across runs on the same input. This makes
the entire downstream labelling pipeline unreliable. Additionally, the same visual defect
group gets re-annotated by Gemini on every run even though it was already labelled before.

### What to implement

**Determinism fix in `hdbscan_cluster.py`:**
- If UMAP is used for dimensionality reduction, set `random_state=42` and `n_jobs=1`.
- If UMAP is not used, add it as a reduction step before HDBSCAN for batches of 50+ crops.
  For fewer than 50 crops, use PCA instead — it's deterministic and needs no tuning.
- On the HDBSCAN call itself, always set `core_dist_n_jobs=1`. This is the primary source
  of non-deterministic tie-breaking.
- Set `cluster_selection_method="leaf"` for more granular, stable clusters.

**Adaptive parameter tuning:**
- Before each clustering run, do a small grid search over a few `(min_cluster_size,
  min_samples)` combinations and pick the one with the best DBCV score
  (`hdbscan.validity.validity_index`). Scale the search range to the current batch size —
  hardcoded constants don't work well across different input volumes.
- Cache the winning params in `data/clusters/tuning_cache.json` so you don't re-tune
  if the batch size hasn't changed significantly. Store the DBCV score alongside the params.
- If the validity index import fails or scoring errors out, fall back to the existing
  config defaults — don't break the pipeline.

**Noise point handling:**
- After clustering, don't silently drop noise points (label == -1). Instead, reassign each
  one to its nearest cluster centroid if the cosine distance is below a reasonable threshold.
  Points that are still too far from any cluster should be stored separately in state as
  "unassigned" crops for potential manual review — don't force them into a cluster.

**Persistent cluster identity (cluster fingerprint registry):**
- After clustering, compute an L2-normalised centroid for each cluster and store it as a
  fingerprint in `data/clusters/cluster_registry.json`. Include the label (once known),
  confidence, first-seen and last-seen timestamps, and how many times this cluster has
  been matched across runs.
- At the start of each clustering run, load this registry and compare new cluster centroids
  against existing ones using cosine distance. If a new cluster is close enough to an
  existing fingerprint, it inherits that label — mark it as a registry hit in state and
  skip Gemini annotation for it entirely.
- This achieves the "one label per group, labelled once" goal without any human intervention.
- Add the registry path and the match distance threshold to `config.py`.

**State changes needed in `state.py`:**
Add fields for: the loaded cluster registry, the tuned params from the grid search,
and a list of unassigned crop paths.

---

## Part 2 — LLM-curated dynamic VLM prompts

### Problem
The Gemini prompt for VLM annotation is static. A steel casting run and a PCB run get
identical prompts, which wastes Gemini's ability to reason domain-specifically. The known
class list, novelty signal strength, and cluster characteristics all contain information
that should shape how Gemini describes what it sees.

### What to implement

**New pipeline node — `dataset_context.py` inside `src/pipeline/nodes/`:**
This node runs after `hdbscan_cluster` and before `vlm_annotation`. It collects a
context snapshot of the current run — things like: total novel crop count, known class
names and their frequencies, novelty ratio (novel crops vs all detected crops), cluster
count and size distribution, mean and p95 of FAISS distances for novel crops, and any
inspection domain hint from config. Package this into a dict stored in state.

Then call the Groq LLM (already wired in `src/retraining/llm_advisor.py` — follow that
pattern) with a system prompt that instructs it to write a Gemini annotation prompt based
on that context. The Groq call should produce only the Gemini prompt text — nothing else.

Key things the Groq-generated prompt must always include:
- The known class names, so Gemini can distinguish novel defects from near-misses
- Instruction to return structured JSON with at minimum: label, confidence (0–1),
  description, severity
- If FAISS distances are high (crops are very different from anything known), tell Gemini
  to coin a new descriptive label rather than approximate a known one
- Appropriate vocabulary for the inferred domain

Cache the generated prompt in `data/vlm_prompt_cache.json` keyed by a hash of the context
dict. If the same context hash appears again, skip the Groq call and reuse the cached prompt.

If `GROQ_API_KEY` is missing or the call fails, fall back silently to the existing static
prompt — log a warning but don't crash.

**Wire FAISS distances into state:**
In `faiss_search.py`, after the FAISS query, store the per-crop L2 distances into state.
Also store the total crop count before the novelty filter — this is needed to compute the
novelty ratio in the context builder.

**Update `vlm_annotation.py`:**
Read `state.vlm_system_prompt` and use it as the Gemini system prompt. Keep the current
hardcoded prompt as a module-level fallback constant so nothing breaks if the new node
didn't run.

**DAG wiring in `graph.py`:**
Insert the new node between `hdbscan_cluster` and `vlm_annotation`. Remove the direct edge
between those two.

**Config additions:**
Add the VLM prompt cache path and an `INSPECTION_DOMAIN` string (default `"unknown"`) that
users can set to something like `"steel_casting"` or `"pcb"` to give the LLM a domain hint.

**State additions:**
Fields for `vlm_system_prompt`, `dataset_context`, `faiss_distances`, and
`total_detected_crops` (before the novelty filter).

---

## Part 3 — VLM annotation performance metrics

### Goal
Lightweight, non-intrusive instrumentation to understand how well Gemini is annotating.
No dashboard changes. No new API dependencies. Metrics go to MLflow and a JSON file.

### What to measure

**Intra-cluster label consistency (ICC):**
For each cluster, instead of annotating only one representative crop, sample 2–3 crops
independently (deterministically — use a seed derived from the crop paths so sampling
is reproducible). Annotate each one with Gemini. ICC for that cluster is the fraction
of samples that return the same top-level label. Use the plurality label as the final
cluster label. Use the mean confidence as the final confidence.

Respect `VLM_SLEEP_BETWEEN` between samples. If a sample fails due to rate limiting,
skip it — an ICC computed from fewer samples is still valid. If only one crop is
available, ICC is 1.0 trivially.

**Confidence distribution:**
Track mean confidence across all clusters per run, plus the fraction of clusters with
confidence below 0.5 (ambiguous) and above 0.85 (high conviction).

**DBCV score (free — already computed in Part 1):**
Just pass it through to the metrics log. No extra work.

**Registry hit rate:**
The fraction of clusters that matched an existing fingerprint and skipped Gemini.
This naturally increases over time as the registry grows — a rising hit rate means
the system is converging on a stable defect vocabulary.

### Where to put this

Create `src/utils/vlm_metrics.py`. It should expose a single `log_run_metrics()` function
that takes the per-cluster results accumulated during `vlm_annotation.py`, aggregates them
into run-level stats, writes to `data/vlm_metrics.json` (append to history), and logs the
flat metrics to MLflow using the existing tracker pattern in `src/utils/metrics.py`.

Call `log_run_metrics()` at the end of the `vlm_annotation` node. If MLflow logging fails,
catch it and warn — don't crash.

Add a `run_id` field to `PipelineState` (short UUID, set at pipeline start) so metrics
from different runs are distinguishable in the JSON history.

### Constraints

- No dashboard file changes.
- No changes to `src/retraining/` except what's explicitly needed.
- Don't remove or rename any existing config constants — only add new ones.
- The pipeline must complete end-to-end even if every new component in all three parts
  fails — every new code path needs a try/except with fallback and a warning log.