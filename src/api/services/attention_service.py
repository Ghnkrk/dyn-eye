"""
"Your input is needed" items for the dashboard bell, toasts, sidebar dots and
page banners.

Items are derived from the current system state on every request, so they can
never drift out of sync: an item disappears as soon as its situation is
resolved. Each key embeds the state it was raised for (a run id, a file time,
a version id), so dismissing an item silences that occurrence only; the next
discovery run or trained model raises a fresh one.

Severity: "blocking" = the system cannot move forward without you;
"recommended" = it can, but you probably want to act.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config as cfg
from src.utils import get_logger

log = get_logger("attention")

SPECIAL_CLUSTERS = {"noise", "unassigned"}


def _state_path() -> Path:
    return cfg.DATA_DIR / "attention_state.json"


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts else None


def _item(key, severity, title, detail, href, action, since=None, page=None) -> dict:
    return {
        "key": key, "severity": severity, "title": title, "detail": detail,
        "href": href, "action": action, "since": since, "page": page or href,
    }


def _latest_run(retraining: bool) -> tuple[str | None, dict]:
    runs_dir = cfg.PIPELINE_RUNS_DIR
    if not runs_dir.exists():
        return None, {}
    files = [p for p in runs_dir.glob("*.json") if p.stem.startswith("retrain_") == retraining]
    if not files:
        return None, {}
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return latest.stem, (_read(latest) or {})


# ── Detectors ────────────────────────────────────────────────

def _unnamed_clusters() -> list[dict]:
    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    manifest = _read(manifest_path) or {}
    clusters = manifest.get("clusters", {})
    unnamed = [n for n, c in clusters.items()
               if n not in SPECIAL_CLUSTERS and not (c or {}).get("defect_name")
               and (cfg.CLUSTERS_DIR / n).exists()]
    if not unnamed:
        return []
    n = len(unnamed)
    return [_item(
        f"clusters-unnamed:{manifest.get('run_id', '')}", "blocking",
        f"{n} defect cluster{'s' if n != 1 else ''} need{'' if n != 1 else 's'} a name",
        "Discovery found new defect groups. Name them (or merge/drop them) so they can be used for fine-tuning.",
        "/clusters", "Name clusters", _iso(_mtime(manifest_path)),
    )]


def _ready_to_finetune() -> list[dict]:
    manifest_path = cfg.CLUSTERS_DIR / "cluster_manifest.json"
    manifest = _read(manifest_path) or {}
    named = [c for n, c in manifest.get("clusters", {}).items()
             if n not in SPECIAL_CLUSTERS and (c or {}).get("defect_name")]
    labels_time = _mtime(manifest_path)
    if not named or not labels_time:
        return []
    run_id, _ = _latest_run(retraining=True)
    last_train = _mtime(cfg.PIPELINE_RUNS_DIR / f"{run_id}.json") if run_id else None
    if last_train and last_train >= labels_time:
        return []
    names = sorted({c["defect_name"] for c in named})
    return [_item(
        f"ready-to-finetune:{int(labels_time)}", "recommended",
        "New labels are ready for fine-tuning",
        f"Named clusters ({', '.join(names[:4])}{'…' if len(names) > 4 else ''}) have changed since the last fine-tune.",
        "/finetuning", "Start a fine-tune", _iso(labels_time),
    )]


def _model_awaiting_deploy() -> list[dict]:
    reg = _read(cfg.MODELS_DIR / "registry.json") or {}
    pending = [v for v in reg.get("versions", []) if v.get("status") == "registered"]
    if not pending:
        return []
    v = max(pending, key=lambda e: e.get("version_num", 0))
    map50 = (v.get("metrics") or {}).get("map50")
    score = f" (mAP50 {map50:.3f})" if isinstance(map50, (int, float)) else ""
    return [_item(
        f"model-awaiting-deploy:{v['version_id']}", "blocking",
        f"Model {v['version_id']} is waiting for your deploy decision",
        f"It was trained{score} but not deployed automatically. Compare it on the Inference page, then deploy or delete it.",
        "/models", "Review model", v.get("created_at"),
    )]


def _advisor_not_ready() -> list[dict]:
    run_id, run = _latest_run(retraining=True)
    adv = (run.get("steps") or {}).get("llm_advisor") or {}
    meta = adv.get("metadata") or {}
    if adv.get("status") != "success" or meta.get("should_train", True):
        return []
    return [_item(
        f"advisor-not-ready:{run_id}", "recommended",
        "Not enough data to fine-tune yet",
        meta.get("reason") or "The advisor decided the labelled data is too small to train on.",
        "/", "Run discovery on more images", adv.get("end_time"),
    )]


def _failed_runs() -> list[dict]:
    from src.api.services import pipeline_service

    items = []
    live = pipeline_service.get_status()
    for retraining, kind, label, href in ((False, "discovery", "Discovery", "/"), (True, "retraining", "Fine-tuning", "/finetuning")):
        run_id, run = _latest_run(retraining)
        if not run_id:
            continue
        steps = run.get("steps") or {}
        failed = [s for s in steps.values() if s.get("status") == "failed"]
        interrupted = [s for s in steps.values() if s.get("status") == "running"] \
            if live.get(kind, {}).get("status") != "running" else []
        bad = failed or interrupted
        if not bad:
            continue
        step = bad[0]
        what = "failed" if failed else "was interrupted"
        detail = step.get("error") or ("The backend stopped while this step was running." if not failed else "")
        items.append(_item(
            f"run-failed:{run_id}", "recommended",
            f"{label} {what} at {step.get('step_name', '?').replace('_', ' ')}",
            (detail or "See the run for details.")[:300],
            f"/runs/{run_id}", "Open run", step.get("end_time") or step.get("start_time"), page=href,
        ))
    return items


def _upsampling() -> list[dict]:
    from src.retraining.tools import upsample as up

    ds = Path(cfg.YOLO_DATASET_DIR)
    items = []
    settings = up.load_settings()
    counts = up.count_instances(ds)
    balance = up.class_balance(counts, settings["imbalance_threshold"])
    if balance.get("suggest_upsampling") and not settings["enabled"]:
        minority = balance["minority_classes"]
        digest = hashlib.md5(json.dumps(counts, sort_keys=True).encode()).hexdigest()[:10]
        items.append(_item(
            f"upsampling-recommended:{digest}", "recommended",
            f"Upsampling recommended for {len(minority)} minority class{'es' if len(minority) != 1 else ''}",
            f"{', '.join(minority[:4])}{'…' if len(minority) > 4 else ''} are under-represented "
            f"({balance['imbalance_ratio']}× imbalance). Turn upsampling on before the next fine-tune.",
            "/upsampling", "Set up upsampling", None,
        ))
    manifest_path = ds / up.MANIFEST_NAME
    manifest = up.load_manifest(ds)
    if manifest.get("images"):
        t = _mtime(manifest_path)
        n = len(manifest["images"])
        items.append(_item(
            f"augmented-review:{int(t or 0)}", "recommended",
            f"{n} augmented images ready to review",
            "Check the copies next to their originals before they are used for fine-tuning.",
            "/upsampling", "Review images", _iso(t),
        ))
    return items


DETECTORS = [_unnamed_clusters, _model_awaiting_deploy, _ready_to_finetune,
             _advisor_not_ready, _upsampling, _failed_runs]


# ── Dismiss / snooze ─────────────────────────────────────────

def _load_state() -> dict:
    return _read(_state_path()) or {}


def _save_state(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def dismiss(key: str, snooze_minutes: int | None = None) -> None:
    """Hide an item: for snooze_minutes, or until its situation changes (new key)."""
    state = _load_state()
    until = (datetime.now(timezone.utc) + timedelta(minutes=snooze_minutes)).isoformat() if snooze_minutes else None
    state[key] = {"until": until}
    _save_state(state)


def items() -> list[dict]:
    found: list[dict] = []
    for detect in DETECTORS:
        try:
            found.extend(detect())
        except Exception as e:  # one broken detector must not hide the others
            log.warning(f"Attention detector {detect.__name__} failed: {e}")

    state = _load_state()
    now = datetime.now(timezone.utc)
    visible = []
    for it in found:
        hidden = state.get(it["key"])
        if hidden is not None:
            until = hidden.get("until")
            if until is None or datetime.fromisoformat(until) > now:
                continue
        visible.append(it)

    # Forget dismissals for items that no longer exist, so the file stays small
    live_keys = {it["key"] for it in found}
    if any(k not in live_keys for k in state):
        _save_state({k: v for k, v in state.items() if k in live_keys})

    order = {"blocking": 0, "recommended": 1}
    return sorted(visible, key=lambda it: (order.get(it["severity"], 2), it["since"] or ""))
