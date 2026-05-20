"""
Tool: DVC Dataset Versioning

Versions the YOLO dataset directory using DVC (Data Version Control).
Creates a DVC-tracked snapshot and git tags.
"""
from __future__ import annotations

import subprocess
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import config as cfg
from src.utils import get_logger, save_json

log = get_logger("dvc_version")


def _run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    """Run a subprocess command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd,
        cwd=cwd or str(cfg.PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def version_dataset(
    dataset_dir: str | None = None,
    version_tag: str | None = None,
    message: str | None = None,
) -> dict:
    """
    Version the YOLO dataset using DVC.

    Steps:
        1. dvc add <dataset_dir>
        2. git add <dataset_dir>.dvc .gitignore
        3. git commit -m "<message>"
        4. git tag <version_tag>

    Returns:
        {
            "success": bool,
            "version_tag": str,
            "dvc_file": str,
            "error": str | None,
        }
    """
    ds_dir = dataset_dir or str(cfg.YOLO_DATASET_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = version_tag or f"dataset-v{ts}"
    msg = message or f"Dataset version {tag}"
    cwd = str(cfg.PROJECT_ROOT)

    log.info(f"Versioning dataset: {ds_dir}")
    log.info(f"  Version tag: {tag}")

    # Check DVC is initialized
    dvc_dir = cfg.PROJECT_ROOT / ".dvc"
    if not dvc_dir.exists():
        log.info("Initializing DVC...")
        rc, out, err = _run_cmd(["dvc", "init"], cwd=cwd)
        if rc != 0:
            return {
                "success": False,
                "version_tag": tag,
                "dvc_file": "",
                "error": f"dvc init failed: {err}",
            }

    # DVC add
    log.info(f"Running: dvc add {ds_dir}")
    rc, out, err = _run_cmd(["dvc", "add", ds_dir], cwd=cwd)
    if rc != 0:
        return {
            "success": False,
            "version_tag": tag,
            "dvc_file": "",
            "error": f"dvc add failed: {err}",
        }

    dvc_file = f"{ds_dir}.dvc"
    log.info(f"DVC file: {dvc_file}")

    # Git add
    rc, out, err = _run_cmd(
        ["git", "add", dvc_file, ".gitignore"],
        cwd=cwd,
    )
    if rc != 0:
        log.warning(f"git add warning: {err}")

    # Git commit
    rc, out, err = _run_cmd(
        ["git", "commit", "-m", msg, "--allow-empty"],
        cwd=cwd,
    )
    if rc != 0:
        log.warning(f"git commit warning: {err}")

    # Git tag
    rc, out, err = _run_cmd(
        ["git", "tag", "-a", tag, "-m", msg],
        cwd=cwd,
    )
    if rc != 0:
        log.warning(f"git tag warning: {err}")

    # Save version metadata
    version_meta = {
        "tag": tag,
        "timestamp": ts,
        "dataset_dir": ds_dir,
        "dvc_file": dvc_file,
        "message": msg,
    }
    save_json(
        version_meta,
        cfg.LOGS_DIR / "dataset_versions" / f"{tag}.json",
    )

    log.info(f"Dataset versioned successfully: {tag}")
    return {
        "success": True,
        "version_tag": tag,
        "dvc_file": dvc_file,
        "error": None,
    }
