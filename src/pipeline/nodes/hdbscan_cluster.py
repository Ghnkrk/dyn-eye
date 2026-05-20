"""
Node 6 — HDBSCAN Clustering

Clusters the novel (unknown) defect crops by their feature embeddings
using HDBSCAN.  Each cluster is saved as a separate folder under
data/clusters/cluster_<id>/.
"""
from __future__ import annotations

import shutil
import numpy as np
import hdbscan
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger

log = get_logger("hdbscan_cluster_node")


def get_dynamic_hdbscan_params(n_samples: int) -> tuple[int, int]:
    """
    Dynamically estimate optimal HDBSCAN min_cluster_size and min_samples
    based on the number of training points and user settings.
    """
    # Use the user's configured default min size as a scaling anchor
    base_min = getattr(cfg, "HDBSCAN_MIN_CLUSTER_SIZE", 4)
    
    if n_samples < 5:
        # Extremely small collections (e.g. sample runs)
        min_cluster_size = 2
        min_samples = 1
    elif n_samples < 20:
        # Small sets
        min_cluster_size = max(2, min(base_min, 3))
        min_samples = max(1, min_cluster_size // 2)
    elif n_samples < 60:
        # Medium sets (use the user's customized base size!)
        min_cluster_size = base_min
        min_samples = max(2, min_cluster_size // 2)
    elif n_samples < 150:
        # Large sets
        min_cluster_size = int(base_min * 1.5)
        min_samples = max(2, min_cluster_size // 2)
    else:
        # Huge datasets (e.g. 200+ images)
        min_cluster_size = max(base_min * 2, int(np.sqrt(n_samples)))
        min_samples = max(3, min_cluster_size // 2)
        
    return min_cluster_size, min_samples


def hdbscan_cluster_node(state: dict) -> dict:
    """
    LangGraph node: HDBSCAN clustering on novel crops with dynamic param sizing.

    Reads:
        state["feature_vectors"]
        state["feature_crop_paths"]
        state["novel_indices"]

    Writes:
        state["cluster_labels"]
        state["cluster_folders"]
        state["num_clusters"]
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
        }

    # Select only novel features
    novel_features = features[novel_indices]
    novel_paths = [crop_paths[i] for i in novel_indices]

    # Dynamically estimate cluster parameters
    min_cluster_size, min_samples = get_dynamic_hdbscan_params(len(novel_features))

    log.info(
        f"Clustering {len(novel_features)} novel crops. "
        f"Dynamic Parameters chosen: min_cluster_size={min_cluster_size}, min_samples={min_samples}"
    )

    # L2 normalize embeddings to ensure clean metric distances
    norms = np.linalg.norm(novel_features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    novel_features_norm = novel_features / norms

    # Dimensionality reduction via PCA (highly recommended for high-dimensional density clustering)
    if len(novel_features_norm) > 10:
        from sklearn.decomposition import PCA
        n_comp = min(5, len(novel_features_norm) - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        novel_features_projected = pca.fit_transform(novel_features_norm)
        log.info(f"Reduced feature space from {novel_features_norm.shape[1]}D to {n_comp}D via PCA")
    else:
        novel_features_projected = novel_features_norm

    # Add distance verification layer for small sample sizes to avoid HDBSCAN crash
    if len(novel_features) < min_cluster_size:
        log.info(
            f"Only {len(novel_features)} novel crops. Bypassing HDBSCAN and verifying "
            "distances to determine if a single unknown class is available."
        )
        if len(novel_features) == 1:
            cluster_labels = np.array([0])
        else:
            from sklearn.cluster import AgglomerativeClustering
            # Group crops if they are close enough (threshold ~ 0.65 in DINOv2 space)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.65,
                metric='euclidean',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(novel_features_projected)
    else:
        # Run HDBSCAN with dynamic parameters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=cfg.HDBSCAN_METRIC,
            allow_single_cluster=True
        )
        cluster_labels = clusterer.fit_predict(novel_features_projected)

        # Fallback to AgglomerativeClustering if HDBSCAN labels everything or almost everything as noise
        unique_labels = set(cluster_labels)
        noise_ratio = sum(1 for l in cluster_labels if l == -1) / len(cluster_labels)
        if len(unique_labels - {-1}) == 0 or noise_ratio > 0.85:
            log.info(f"HDBSCAN returned high noise ({noise_ratio:.1%}). Falling back to AgglomerativeClustering.")
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.65,
                metric='euclidean',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(novel_features_projected)

    unique_labels = set(cluster_labels)
    num_clusters = len(unique_labels - {-1})  # -1 = noise

    # Fallback to Silhouette-optimized K-Means if we only got 1 cluster and have enough samples
    if num_clusters <= 1 and len(novel_features_norm) >= 6:
        log.info("Clustering returned 1 cluster. Running Silhouette-optimized K-Means to find natural subdivisions.")
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        best_k = 4
        best_score = -1
        best_labels = None

        max_k = min(8, len(novel_features_norm) // 2)
        if max_k >= 2:
            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(novel_features_norm)
                score = silhouette_score(novel_features_norm, labels)
                log.info(f"K-Means K={k} Silhouette Score: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels

            if best_labels is not None:
                cluster_labels = best_labels
                unique_labels = set(cluster_labels)
                num_clusters = len(unique_labels - {-1})
                log.info(f"Selected K-Means K={best_k} as the best partitioning with score {best_score:.4f}")

    noise_count = sum(1 for l in cluster_labels if l == -1)

    log.info(
        f"HDBSCAN found {num_clusters} clusters, "
        f"{noise_count} noise points"
    )

    # Clean and create cluster directories
    clusters_dir = cfg.CLUSTERS_DIR
    if clusters_dir.exists():
        shutil.rmtree(clusters_dir)
    clusters_dir.mkdir(parents=True, exist_ok=True)

    cluster_folders: dict[int, str] = {}

    for label in sorted(unique_labels):
        if label == -1:
            folder_name = "noise"
        else:
            folder_name = f"cluster_{label:03d}"

        folder_path = clusters_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        cluster_folders[int(label)] = str(folder_path)

    # Copy crops into their cluster folders
    for idx, (path, label) in enumerate(zip(novel_paths, cluster_labels)):
        src = Path(path)
        label_int = int(label)
        dst_dir = Path(cluster_folders[label_int])
        dst = dst_dir / src.name
        shutil.copy2(str(src), str(dst))

    # Log cluster sizes
    for label in sorted(unique_labels):
        count = sum(1 for l in cluster_labels if l == label)
        folder_name = "noise" if label == -1 else f"cluster_{label:03d}"
        log.info(f"  {folder_name}: {count} crops")

    return {
        "cluster_labels": [int(l) for l in cluster_labels],
        "cluster_folders": cluster_folders,
        "num_clusters": num_clusters,
    }
