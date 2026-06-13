"""
Node 6 — HDBSCAN Clustering (Deterministic, with fingerprint registry)

Clusters the novel (unknown) defect crops by their feature embeddings.
Improvements over the baseline:
  - Deterministic: UMAP (≥50 crops) or PCA (<50) with fixed seeds
  - Adaptive: DBCV-scored grid search over (min_cluster_size, min_samples)
  - Noise handling: reassign near-centroid noise; isolate true outliers
  - Persistent identity: cluster fingerprint registry survives across runs
"""
from __future__ import annotations

import hashlib
import json
import shutil
import numpy as np
import hdbscan
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("hdbscan_cluster_node")


# ── Cluster fingerprint registry ────────────────────────────

def _load_cluster_registry() -> dict:
    """Load the persistent cluster fingerprint registry."""
    path = cfg.CLUSTER_REGISTRY_PATH
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Cluster registry corrupted, starting fresh: {e}")
    return {"fingerprints": []}


def _save_cluster_registry(registry: dict) -> None:
    """Persist the cluster fingerprint registry."""
    path = cfg.CLUSTER_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two L2-normalised vectors."""
    dot = float(np.dot(a, b))
    return 1.0 - dot


def _match_centroid_to_registry(
    centroid: np.ndarray,
    registry: dict,
    threshold: float,
) -> dict | None:
    """Find the closest matching fingerprint in the registry, if within threshold."""
    best_match = None
    best_dist = threshold  # only match if strictly below
    for fp in registry.get("fingerprints", []):
        stored = np.array(fp["centroid"], dtype=np.float32)
        dist = _cosine_distance(centroid, stored)
        if dist < best_dist:
            best_dist = dist
            best_match = fp
    return best_match


# ── DBCV grid search with tuning cache ──────────────────────

def _load_tuning_cache() -> dict:
    path = cfg.CLUSTER_TUNING_CACHE_PATH
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_tuning_cache(cache: dict) -> None:
    path = cfg.CLUSTER_TUNING_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _dbcv_grid_search(
    features: np.ndarray,
    n_samples: int,
) -> tuple[int, int, float]:
    """
    Grid search over (min_cluster_size, min_samples) scored by DBCV.
    Allows dynamic scaling of candidates up to half the sample size to prevent over-segmentation.
    """
    cache_path = cfg.CLUSTER_TUNING_CACHE_PATH
    
    # Invalidate cache if it contains old hardcoded or bad values
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            # If the cache has invalid or old entries, clear it to recalculate with new candidates
            if any(v.get("dbcv_score") == -np.inf or "min_cluster_size" not in v for v in cache.values()):
                cache = {}
                log.info("Old/invalid tuning cache cleared.")
        except Exception:
            pass

    # Cache key: batch size bucket (±20%)
    bucket = round(n_samples / 10) * 10
    cache_key = str(bucket)
    if cache_key in cache:
        cached = cache[cache_key]
        log.info(
            f"Tuning cache hit for batch ~{bucket}: "
            f"mcs={cached['min_cluster_size']}, ms={cached['min_samples']}, "
            f"dbcv={cached['dbcv_score']:.4f}"
        )
        return cached["min_cluster_size"], cached["min_samples"], cached["dbcv_score"]

    try:
        from hdbscan.validity import validity_index
    except ImportError:
        log.warning("hdbscan.validity not available — using config defaults")
        return cfg.HDBSCAN_MIN_CLUSTER_SIZE, cfg.HDBSCAN_MIN_SAMPLES, -1.0

    base = cfg.HDBSCAN_MIN_CLUSTER_SIZE
    
    # Scale candidates dynamically up to 50% of the dataset size to prevent over-segmentation
    mcs_candidates = set([
        max(2, base - 1),
        base,
        max(2, base + 2),
        max(2, int(np.sqrt(n_samples))),
        max(2, n_samples // 10),
        max(2, n_samples // 8),
        max(2, n_samples // 6),
        max(2, n_samples // 5),
        max(2, n_samples // 4),
        max(2, n_samples // 3),
        max(2, n_samples // 2),
    ])
    mcs_candidates = sorted([m for m in mcs_candidates if m < n_samples and m > 1])
    
    ms_candidates = [1, 2, 3, 4, 5]

    best_score = -np.inf
    best_mcs, best_ms = base, cfg.HDBSCAN_MIN_SAMPLES

    for mcs in mcs_candidates:
        for ms in ms_candidates:
            if ms > mcs:
                continue
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    metric=cfg.HDBSCAN_METRIC,
                    core_dist_n_jobs=1,
                    cluster_selection_method=cfg.HDBSCAN_CLUSTER_SELECTION_METHOD,
                    allow_single_cluster=True,
                )
                labels = clusterer.fit_predict(features)
                n_clusters = len(set(labels) - {-1})
                if n_clusters < 1:
                    continue
                score = validity_index(features, labels)
                # We want a valid finite DBCV score
                if np.isnan(score) or np.isinf(score):
                    continue
                if score > best_score:
                    best_score = score
                    best_mcs = mcs
                    best_ms = ms
            except Exception:
                continue

    # Fallback if no valid score was found during grid search
    if best_score == -np.inf:
        log.warning("Grid search yielded no finite DBCV scores — trying largest mcs as fallback")
        for mcs in reversed(mcs_candidates):
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=min(2, mcs),
                    metric=cfg.HDBSCAN_METRIC,
                    core_dist_n_jobs=1,
                    cluster_selection_method=cfg.HDBSCAN_CLUSTER_SELECTION_METHOD,
                    allow_single_cluster=True,
                )
                labels = clusterer.fit_predict(features)
                n_clusters = len(set(labels) - {-1})
                if 1 <= n_clusters <= 3:
                    best_mcs = mcs
                    best_ms = min(2, mcs)
                    best_score = 0.0
                    log.info(f"Fallback selected mcs={mcs} to target {n_clusters} clusters.")
                    break
            except Exception:
                continue

    log.info(
        f"DBCV grid search complete: best mcs={best_mcs}, ms={best_ms}, "
        f"score={best_score:.4f} (searched {len(mcs_candidates)*len(ms_candidates)} combos)"
    )

    # Cache the result
    cache[cache_key] = {
        "min_cluster_size": best_mcs,
        "min_samples": best_ms,
        "dbcv_score": float(best_score),
        "n_samples": n_samples,
    }
    try:
        _save_tuning_cache(cache)
    except Exception as e:
        log.warning(f"Failed to save tuning cache: {e}")

    return best_mcs, best_ms, float(best_score)


# ── Dimensionality reduction ────────────────────────────────

def _reduce_dimensions(features: np.ndarray) -> np.ndarray:
    """
    Deterministic dimensionality reduction.
    UMAP for ≥50 crops (with fixed random_state, n_jobs=1).
    PCA for <50 crops (fully deterministic).
    """
    n = len(features)
    if n <= 10:
        return features

    if n >= cfg.UMAP_BATCH_THRESHOLD:
        try:
            import umap
            n_comp = min(cfg.UMAP_N_COMPONENTS, n - 2)
            reducer = umap.UMAP(
                n_components=n_comp,
                n_neighbors=min(cfg.UMAP_N_NEIGHBORS, n - 1),
                min_dist=cfg.UMAP_MIN_DIST,
                random_state=cfg.UMAP_RANDOM_STATE,
                n_jobs=1,
                metric="cosine",
            )
            reduced = reducer.fit_transform(features)
            log.info(f"UMAP reduction: {features.shape[1]}D → {n_comp}D ({n} crops)")
            return reduced
        except ImportError:
            log.warning("umap-learn not installed — falling back to PCA")
        except Exception as e:
            log.warning(f"UMAP failed — falling back to PCA: {e}")

    # PCA fallback (always deterministic)
    from sklearn.decomposition import PCA
    n_comp = min(cfg.UMAP_N_COMPONENTS, n - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    reduced = pca.fit_transform(features)
    log.info(f"PCA reduction: {features.shape[1]}D → {n_comp}D ({n} crops)")
    return reduced


# ── Noise point reassignment ────────────────────────────────

def _reassign_noise_points(
    features_norm: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[int]]:
    """
    Reassign noise points (label == -1) to nearest cluster centroid
    if cosine distance is below threshold. Returns updated labels
    and list of indices that remain unassigned.
    """
    cluster_ids = sorted(set(labels) - {-1})
    if not cluster_ids:
        # No clusters at all — everything is unassigned
        return labels, [i for i, lbl in enumerate(labels) if lbl == -1]

    # Compute centroids (L2-normalised)
    centroids = {}
    for cid in cluster_ids:
        mask = labels == cid
        centroid = features_norm[mask].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        centroids[cid] = centroid

    noise_indices = [i for i, lbl in enumerate(labels) if lbl == -1]
    unassigned = []
    reassigned_count = 0

    for idx in noise_indices:
        vec = features_norm[idx]
        best_cid = None
        best_dist = threshold
        for cid, centroid in centroids.items():
            dist = _cosine_distance(vec, centroid)
            if dist < best_dist:
                best_dist = dist
                best_cid = cid
        if best_cid is not None:
            labels[idx] = best_cid
            reassigned_count += 1
        else:
            unassigned.append(idx)

    if reassigned_count > 0:
        log.info(
            f"Noise reassignment: {reassigned_count} reassigned, "
            f"{len(unassigned)} remain unassigned"
        )

    return labels, unassigned


# ── Main node ───────────────────────────────────────────────

def hdbscan_cluster_node(state: dict) -> dict:
    """
    LangGraph node: HDBSCAN clustering on novel crops with deterministic
    dimensionality reduction, adaptive parameter tuning, noise reassignment,
    and persistent cluster fingerprint registry.

    Reads:
        state["feature_vectors"]
        state["feature_crop_paths"]
        state["novel_indices"]

    Writes:
        state["cluster_labels"]
        state["cluster_folders"]
        state["num_clusters"]
        state["cluster_registry"]
        state["cluster_tuned_params"]
        state["unassigned_crop_paths"]
        state["dbcv_score"]
        state["registry_hits"]
        state["registry_total"]
    """
    features = state.get("feature_vectors")
    crop_paths = state.get("feature_crop_paths", [])
    novel_indices = state.get("novel_indices", [])

    if not novel_indices or features is None:
        log.warning("No novel crops to cluster")
        return {
            "cluster_labels": [],
            "cluster_folders": {},
            "num_clusters": 0,
            "cluster_registry": {},
            "cluster_tuned_params": {},
            "unassigned_crop_paths": [],
            "dbcv_score": -1.0,
            "registry_hits": 0,
            "registry_total": 0,
        }

    # Select only novel features
    novel_features = features[novel_indices]
    novel_paths = [crop_paths[i] for i in novel_indices]
    n_samples = len(novel_features)

    # L2 normalize embeddings
    norms = np.linalg.norm(novel_features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    novel_features_norm = novel_features / norms

    # ── Dimensionality reduction (deterministic) ────────────
    novel_features_projected = _reduce_dimensions(novel_features_norm)

    # ── Adaptive parameter tuning via DBCV grid search ──────
    dbcv_score = -1.0
    tuned_params = {}

    if n_samples < 3:
        # Too few for clustering — put everything in one cluster
        cluster_labels = np.zeros(n_samples, dtype=int)
        tuned_params = {"min_cluster_size": 2, "min_samples": 1, "method": "single_cluster_fallback"}
    else:
        try:
            best_mcs, best_ms, dbcv_score = _dbcv_grid_search(
                novel_features_projected, n_samples
            )
            tuned_params = {
                "min_cluster_size": best_mcs,
                "min_samples": best_ms,
                "dbcv_score": dbcv_score,
            }
        except Exception as e:
            log.warning(f"DBCV grid search failed — using defaults: {e}")
            best_mcs = cfg.HDBSCAN_MIN_CLUSTER_SIZE
            best_ms = cfg.HDBSCAN_MIN_SAMPLES
            tuned_params = {"min_cluster_size": best_mcs, "min_samples": best_ms, "method": "default_fallback"}

        log.info(
            f"Clustering {n_samples} novel crops with "
            f"min_cluster_size={best_mcs}, min_samples={best_ms}"
        )

        if n_samples < best_mcs:
            # Bypass HDBSCAN — use Agglomerative
            log.info(f"Only {n_samples} crops (< mcs={best_mcs}). Using AgglomerativeClustering.")
            if n_samples == 1:
                cluster_labels = np.array([0])
            else:
                from sklearn.cluster import AgglomerativeClustering
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=0.65,
                    metric='euclidean',
                    linkage='average'
                )
                cluster_labels = clustering.fit_predict(novel_features_projected)
        else:
            # Run HDBSCAN with tuned parameters + deterministic settings
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=best_mcs,
                min_samples=best_ms,
                metric=cfg.HDBSCAN_METRIC,
                core_dist_n_jobs=1,
                cluster_selection_method=cfg.HDBSCAN_CLUSTER_SELECTION_METHOD,
                allow_single_cluster=True,
            )
            cluster_labels = clusterer.fit_predict(novel_features_projected)

            # Fallback if high noise
            noise_ratio = sum(1 for l in cluster_labels if l == -1) / len(cluster_labels)
            if len(set(cluster_labels) - {-1}) == 0 or noise_ratio > 0.85:
                log.info(f"HDBSCAN high noise ({noise_ratio:.1%}). Falling back to AgglomerativeClustering.")
                from sklearn.cluster import AgglomerativeClustering
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=0.65,
                    metric='euclidean',
                    linkage='average'
                )
                cluster_labels = clustering.fit_predict(novel_features_projected)

    # ── Silhouette-optimized K-Means fallback for single cluster ──
    unique_labels = set(cluster_labels)
    num_clusters = len(unique_labels - {-1})

    if num_clusters <= 1 and n_samples >= 6:
        log.info("Single cluster detected. Running Silhouette-optimized K-Means.")
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        best_k = 1
        best_score = -1
        best_labels = None
        max_k = min(8, n_samples // 2)

        if max_k >= 2:
            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(novel_features_norm)
                score = silhouette_score(novel_features_norm, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels

            if best_labels is not None:
                cluster_labels = best_labels
                log.info(f"K-Means selected K={best_k}, silhouette={best_score:.4f}")

    # ── Noise reassignment ──────────────────────────────────
    cluster_labels = np.array(cluster_labels)
    cluster_labels, unassigned_indices = _reassign_noise_points(
        novel_features_norm, cluster_labels, cfg.CLUSTER_NOISE_REASSIGN_THRESHOLD
    )
    unassigned_paths = [novel_paths[i] for i in unassigned_indices]

    # ── Recount after reassignment ──────────────────────────
    unique_labels = set(int(l) for l in cluster_labels if l != -1)
    num_clusters = len(unique_labels)
    noise_count = sum(1 for l in cluster_labels if l == -1)

    log.info(
        f"Clustering result: {num_clusters} clusters, "
        f"{noise_count} noise, {len(unassigned_paths)} unassigned"
    )

    # ── Cluster fingerprint registry ────────────────────────
    registry = _load_cluster_registry()
    registry_hits = 0

    # Compute centroids per cluster and match against registry
    cluster_centroids: dict[int, np.ndarray] = {}
    for cid in sorted(unique_labels):
        mask = cluster_labels == cid
        centroid = novel_features_norm[mask].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        cluster_centroids[cid] = centroid

        # Try to match against existing fingerprints
        match = _match_centroid_to_registry(
            centroid, registry, cfg.CLUSTER_MATCH_THRESHOLD
        )
        if match:
            registry_hits += 1
            match["last_seen"] = datetime.now(timezone.utc).isoformat()
            match["match_count"] = match.get("match_count", 0) + 1
            log.info(
                f"  Cluster {cid}: matched registry fingerprint "
                f"'{match.get('label', 'unknown')}' (matched {match['match_count']}x)"
            )
        else:
            # Register new fingerprint
            new_fp = {
                "centroid": centroid.tolist(),
                "label": None,  # Will be filled once VLM labels it
                "confidence": None,
                "first_seen": datetime.now(timezone.utc).isoformat(),
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "match_count": 1,
                "cluster_size": int(sum(mask)),
            }
            registry["fingerprints"].append(new_fp)

    try:
        _save_cluster_registry(registry)
    except Exception as e:
        log.warning(f"Failed to save cluster registry: {e}")

    # ── Create cluster directories ──────────────────────────
    clusters_dir = cfg.CLUSTERS_DIR
    if clusters_dir.exists():
        # Preserve registry files, remove cluster folders
        for item in clusters_dir.iterdir():
            if item.is_dir() and item.name not in ("__pycache__",):
                shutil.rmtree(item)
    clusters_dir.mkdir(parents=True, exist_ok=True)

    cluster_folders: dict[int, str] = {}
    all_labels_set = set(int(l) for l in cluster_labels)

    for label in sorted(all_labels_set):
        if label == -1:
            folder_name = "noise"
        else:
            folder_name = f"cluster_{label:03d}"
        folder_path = clusters_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        cluster_folders[int(label)] = str(folder_path)

    # Unassigned folder
    if unassigned_paths:
        unassigned_dir = clusters_dir / "unassigned"
        unassigned_dir.mkdir(parents=True, exist_ok=True)

    # Copy crops into cluster folders
    for idx, (path, label) in enumerate(zip(novel_paths, cluster_labels)):
        src = Path(path)
        label_int = int(label)
        if label_int == -1 and idx in unassigned_indices:
            dst_dir = clusters_dir / "unassigned"
        elif label_int in cluster_folders:
            dst_dir = Path(cluster_folders[label_int])
        else:
            continue
        dst = dst_dir / src.name
        shutil.copy2(str(src), str(dst))

    # Log cluster sizes
    for label in sorted(all_labels_set):
        count = sum(1 for l in cluster_labels if l == label)
        name = "noise" if label == -1 else f"cluster_{label:03d}"
        log.info(f"  {name}: {count} crops")
    if unassigned_paths:
        log.info(f"  unassigned: {len(unassigned_paths)} crops")

    return {
        "cluster_labels": [int(l) for l in cluster_labels],
        "cluster_folders": cluster_folders,
        "num_clusters": num_clusters,
        "cluster_registry": registry,
        "cluster_tuned_params": tuned_params,
        "unassigned_crop_paths": unassigned_paths,
        "dbcv_score": dbcv_score,
        "registry_hits": registry_hits,
        "registry_total": num_clusters,
    }
