"""
Node 7 — Label Studio Sync

Syncs HDBSCAN cluster folders with Label Studio.
- Creates a project per pipeline run.
- Uploads each cluster's crop images directly via file upload API.
- Each crop is tagged with its cluster ID and source image metadata.
- Cluster names assigned in Label Studio map back to defect names
  via a saved manifest (cluster → crop → source image + bbox).
"""
from __future__ import annotations

import os
import json
import base64
import requests
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger, save_json

log = get_logger("label_studio_sync")


_cached_access_token = None

def _get_headers(content_type: str = "application/json") -> dict:
    """
    Get API headers for Label Studio. Dynamically supports both
    legacy Token auth and new JWT-based PAT auth with automatic refresh.
    """
    global _cached_access_token
    token = cfg.LABEL_STUDIO_API_KEY.strip()

    headers = {}

    if token.startswith("eyJ"):
        if _cached_access_token:
            headers["Authorization"] = f"Bearer {_cached_access_token}"
        else:
            try:
                log.info("JWT refresh token detected. Requesting short-lived Access Token...")
                resp = requests.post(
                    f"{cfg.LABEL_STUDIO_URL}/api/token/refresh",
                    json={"refresh": token},
                    timeout=15
                )
                if resp.status_code == 200:
                    _cached_access_token = resp.json()["access"]
                    log.info("Successfully acquired JWT short-lived Access Token.")
                    headers["Authorization"] = f"Bearer {_cached_access_token}"
                else:
                    log.warning(
                        f"JWT refresh failed (status {resp.status_code}). "
                        "Falling back to legacy Token header."
                    )
                    headers["Authorization"] = f"Token {token}"
            except Exception as e:
                log.warning(f"Error refreshing JWT token: {e}. Falling back to legacy Token header.")
                headers["Authorization"] = f"Token {token}"
    else:
        headers["Authorization"] = f"Token {token}"

    if content_type:
        headers["Content-Type"] = content_type

    return headers


def _get_auth_header_only() -> dict:
    """Get just the Authorization header (for multipart uploads where Content-Type is auto-set)."""
    return _get_headers(content_type=None)


def _create_project(run_id: str, cluster_names: list[str]) -> int | None:
    """
    Create a Label Studio project for cluster review & naming.

    Labeling interface lets the user:
    1. View the crop image and its auto-assigned cluster
    2. Reassign to a different cluster via Choices
    3. Type a custom defect name (free-text)
    4. Flag bad crops for removal
    """
    # Build cluster reassignment choices
    cluster_choices = "\n".join(
        f'    <Choice value="{name}" />' for name in cluster_names
    )

    label_config = f"""<View>
  <View style="display:flex; gap:16px;">
    <View style="flex:1;">
      <Image name="image" value="$image" zoom="true" />
    </View>
    <View style="flex:1; padding:12px;">
      <Header value="Current Cluster: $cluster_name" size="3" />
      <Header value="Source: $source_image" size="5" />
      <Header value="Traits: $physical_traits" size="5" />

      <View style="margin-top:16px;">
        <Header value="1. Assign Defect Name" size="4" />
        <TextArea name="defect_name" toName="image"
                  placeholder="Type defect name (e.g. scratch, dent, stain)..."
                  maxSubmissions="1" editable="true"
                  rows="1" />
      </View>

      <View style="margin-top:12px;">
        <Header value="2. Reassign Cluster (optional)" size="4" />
        <Choices name="cluster_assignment" toName="image"
                 choice="single" showInline="true" required="false">
{cluster_choices}
        </Choices>
      </View>

      <View style="margin-top:12px;">
        <Header value="3. Quality" size="4" />
        <Choices name="quality" toName="image"
                 choice="single" showInline="true" required="false">
          <Choice value="keep" selected="true" />
          <Choice value="drop" />
        </Choices>
      </View>
    </View>
  </View>
</View>"""

    payload = {
        "title": f"DYN-EYE Cluster Review — {run_id}",
        "description": (
            f"Review clustered novel defect crops from run {run_id}. "
            "For each crop: name the defect, optionally reassign its cluster, "
            "and flag bad crops for removal. Use 'Label All Tasks' for bulk labeling."
        ),
        "label_config": label_config,
    }

    try:
        resp = requests.post(
            f"{cfg.LABEL_STUDIO_URL}/api/projects",
            headers=_get_headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        project_id = resp.json()["id"]
        log.info(f"Created Label Studio project {project_id}: '{payload['title']}'")
        return project_id
    except Exception as e:
        log.error(f"Failed to create Label Studio project: {e}")
        return None


def _upload_image_file(project_id: int, image_path: Path) -> str | None:
    """
    Upload a single image file directly to Label Studio via the file upload API.
    Returns the LS-accessible URL for the uploaded file, or None on failure.
    """
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{cfg.LABEL_STUDIO_URL}/api/projects/{project_id}/import",
                headers=_get_auth_header_only(),
                files={"file": (image_path.name, f, "image/jpeg")},
                timeout=30,
            )
        resp.raise_for_status()
        result = resp.json()

        # The response contains the file upload ID and generated URL
        if "file_upload_ids" in result:
            upload_id = result["file_upload_ids"][0]
            return f"/data/upload/{upload_id}"
        elif "task_ids" in result:
            # If LS auto-created a task, we need its ID
            return result["task_ids"]
        return None
    except Exception as e:
        log.warning(f"Failed to upload image {image_path.name}: {e}")
        return None


def _upload_tasks_with_images(
    project_id: int,
    cluster_folders: dict[int, str],
    crop_metadata: list[dict],
    run_id: str,
) -> list[int]:
    """
    Upload cluster images as tasks to Label Studio.

    Strategy: Upload each image file directly to Label Studio via the
    /api/projects/{id}/import multipart endpoint. This auto-creates a task
    with an internal /data/upload/ URL (no CORS issues). Then PATCH each
    task to add cluster metadata (cluster_name, source_image, bbox, etc.).
    Uses parallel execution for high-speed synchronization.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Build a lookup: crop filename → metadata dict
    meta_lookup: dict[str, dict] = {}
    for meta in crop_metadata:
        crop_name = Path(meta.get("crop_path", "")).name
        meta_lookup[crop_name] = meta

    upload_items = []
    for cluster_id, folder_path in cluster_folders.items():
        folder = Path(folder_path)
        if not folder.exists():
            continue

        cluster_name = folder.name
        if cluster_name == "noise":
            continue  # Skip noise crops

        from src.utils.io_helpers import list_images
        images = list_images(folder)
        for img_path in images:
            meta = meta_lookup.get(img_path.name, {})
            upload_items.append((img_path, cluster_id, cluster_name, meta))

    log.info(f"Uploading {len(upload_items)} crops from all clusters in parallel to project {project_id}...")

    def upload_and_patch_single(item):
        img_path, cluster_id, cluster_name, meta = item
        # 1) Upload file → LS auto-creates a task with internal image URL
        created_task_id = _upload_and_create_task(project_id, img_path)
        if not created_task_id:
            return None

        # 2) PATCH the task to add our cluster + source metadata
        _patch_task_metadata(
            task_id=created_task_id,
            cluster_id=int(cluster_id),
            cluster_name=cluster_name,
            source_image=meta.get("source_image_name", "unknown"),
            source_file=img_path.name,
            box_pixels=meta.get("box_2d_pixels", []),
            box_raw=meta.get("box_2d_raw", []),
            traits=meta.get("physical_traits", ""),
        )
        return created_task_id

    task_ids: list[int] = []
    # Use 16 parallel workers to dramatically speed up round-trips
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(upload_and_patch_single, item): item for item in upload_items}
        for future in as_completed(futures):
            try:
                tid = future.result()
                if tid:
                    task_ids.append(tid)
            except Exception as e:
                log.error(f"Failed parallel task upload/patch task: {e}")

    log.info(f"Uploaded {len(task_ids)} total tasks to project {project_id}")
    return task_ids


def _upload_and_create_task(project_id: int, image_path: Path) -> int | None:
    """
    Upload a single image file to Label Studio via the import endpoint.
    LS auto-creates a task with an internal /data/upload/ URL.
    Returns the created task ID, or None on failure.
    """
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{cfg.LABEL_STUDIO_URL}/api/projects/{project_id}/import",
                headers=_get_auth_header_only(),
                files={"file": (image_path.name, f, "image/jpeg")},
                timeout=30,
            )
        resp.raise_for_status()
        result = resp.json()
        # Response: {"task_count": 1, "file_upload_ids": [N], ...}
        # We need the task ID — get it from the task list for this upload
        file_ids = result.get("file_upload_ids", [])
        if not file_ids:
            log.warning(f"No file_upload_ids in response for {image_path.name}")
            return None

        # Fetch the task created from this file upload
        tasks_resp = requests.get(
            f"{cfg.LABEL_STUDIO_URL}/api/projects/{project_id}/tasks",
            headers=_get_headers(),
            params={"fields": "all", "page_size": 1, "page": 1,
                    "filters": json.dumps({"conjunction": "and", "items": [
                        {"filter": "filter:tasks:file_upload", "operator": "equal",
                         "type": "Number", "value": file_ids[0]}
                    ]})},
            timeout=15,
        )
        if tasks_resp.status_code == 200:
            tasks_data = tasks_resp.json()
            tasks_list = tasks_data if isinstance(tasks_data, list) else tasks_data.get("tasks", [])
            if tasks_list:
                return tasks_list[0]["id"]

        # Fallback: get the most recently created task
        latest_resp = requests.get(
            f"{cfg.LABEL_STUDIO_URL}/api/projects/{project_id}/tasks",
            headers=_get_headers(),
            params={"page_size": 1},
            timeout=15,
        )
        if latest_resp.status_code == 200:
            latest_data = latest_resp.json()
            latest_list = latest_data if isinstance(latest_data, list) else latest_data.get("tasks", [])
            if latest_list:
                return latest_list[-1]["id"]

        log.warning(f"Could not find task for uploaded file {image_path.name}")
        return None
    except Exception as e:
        log.warning(f"Failed to upload {image_path.name}: {e}")
        return None


def _patch_task_metadata(
    task_id: int,
    cluster_id: int,
    cluster_name: str,
    source_image: str,
    source_file: str,
    box_pixels: list,
    box_raw: list,
    traits: str,
):
    """PATCH a task to add cluster and source metadata to its data dict."""
    try:
        # First GET the existing task data (to preserve the image URL)
        get_resp = requests.get(
            f"{cfg.LABEL_STUDIO_URL}/api/tasks/{task_id}",
            headers=_get_headers(),
            timeout=10,
        )
        if get_resp.status_code != 200:
            return

        existing_data = get_resp.json().get("data", {})
        existing_data.update({
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "source_image": source_image,
            "source_file": source_file,
            "box_pixels": json.dumps(box_pixels) if box_pixels else "",
            "box_raw": json.dumps(box_raw) if box_raw else "",
            "physical_traits": traits,
        })

        requests.patch(
            f"{cfg.LABEL_STUDIO_URL}/api/tasks/{task_id}",
            headers=_get_headers(),
            json={"data": existing_data},
            timeout=10,
        )
    except Exception:
        pass  # Non-critical — task still exists with image


def _save_cluster_manifest(
    run_id: str,
    cluster_folders: dict[int, str],
    crop_metadata: list[dict],
    project_id: int | None,
) -> Path:
    """
    Save a manifest JSON that maps:
      cluster_name → [{crop_file, source_image, bbox_pixels, bbox_raw, traits}]

    This is the master mapping used later to propagate human-assigned
    cluster names back to full-frame source images with bounding boxes.
    """
    meta_lookup: dict[str, dict] = {}
    for meta in crop_metadata:
        crop_name = Path(meta.get("crop_path", "")).name
        meta_lookup[crop_name] = meta

    manifest = {
        "run_id": run_id,
        "project_id": project_id,
        "clusters": {},
    }

    for cluster_id, folder_path in cluster_folders.items():
        folder = Path(folder_path)
        cluster_name = folder.name

        if not folder.exists():
            continue

        from src.utils.io_helpers import list_images
        images = list_images(folder)

        cluster_entry = {
            "cluster_id": int(cluster_id),
            "crop_count": len(images),
            "defect_name": None,  # Will be filled after human naming
            "crops": [],
        }

        for img_path in images:
            meta = meta_lookup.get(img_path.name, {})
            cluster_entry["crops"].append({
                "crop_file": img_path.name,
                "crop_path": str(img_path),
                "source_image": meta.get("source_image", ""),
                "source_image_name": meta.get("source_image_name", ""),
                "box_2d_pixels": meta.get("box_2d_pixels", []),
                "box_2d_raw": meta.get("box_2d_raw", []),
                "physical_traits": meta.get("physical_traits", ""),
                "crop_width": meta.get("crop_width", 0),
                "crop_height": meta.get("crop_height", 0),
            })

        manifest["clusters"][cluster_name] = cluster_entry

    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    save_json(manifest, manifest_path)
    log.info(f"Cluster manifest saved to {manifest_path}")
    return manifest_path


def label_studio_sync_node(state: dict) -> dict:
    """
    LangGraph node: sync clusters with Label Studio.

    Reads:
        state["cluster_folders"]
        state["crop_metadata"]
        state["run_id"]

    Writes:
        state["label_studio_project_id"]
        state["label_studio_task_ids"]
    """
    cluster_folders = state.get("cluster_folders", {})
    crop_metadata = state.get("crop_metadata", [])
    run_id = state.get("run_id", "unknown_run")

    if not cluster_folders:
        log.warning("No clusters to sync with Label Studio")
        return {
            "label_studio_project_id": None,
            "label_studio_task_ids": [],
        }

    # Always save the manifest regardless of LS availability
    _save_cluster_manifest(run_id, cluster_folders, crop_metadata, None)

    if not cfg.LABEL_STUDIO_API_KEY:
        log.warning(
            "LABEL_STUDIO_API_KEY not set. Skipping Label Studio sync. "
            "Set it via environment variable or config.py."
        )
        sync_info = {
            "run_id": run_id,
            "cluster_folders": {str(k): v for k, v in cluster_folders.items()},
            "instruction": "Import these cluster folders manually into Label Studio.",
        }
        save_json(sync_info, cfg.DATA_DIR / "label_studio_sync_pending.json")
        return {
            "label_studio_project_id": None,
            "label_studio_task_ids": [],
        }

    # Get cluster names (excluding noise)
    cluster_names = []
    for cid, folder in cluster_folders.items():
        name = Path(folder).name
        if name != "noise":
            cluster_names.append(name)

    # Create project
    project_id = _create_project(run_id, cluster_names or ["unknown_defect"])
    if project_id is None:
        return {
            "label_studio_project_id": None,
            "label_studio_task_ids": [],
            "errors": state.get("errors", []) + ["Failed to create Label Studio project"],
        }

    # Update manifest with project ID
    _save_cluster_manifest(run_id, cluster_folders, crop_metadata, project_id)

    # Upload tasks with image files
    task_ids = _upload_tasks_with_images(
        project_id, cluster_folders, crop_metadata, run_id,
    )

    return {
        "label_studio_project_id": project_id,
        "label_studio_task_ids": task_ids,
    }
