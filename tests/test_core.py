import json

import numpy as np
import pytest
from langgraph.graph import END


# ── cluster fingerprint registry ─────────────────────────────

def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_registry_match_is_one_to_one(sandbox):
    from src.features import cluster_registry as creg

    reg = {"fingerprints": [creg.new_fingerprint(_unit([1, 0, 0]), 5)]}
    # two new clusters both close to the single fingerprint: only the closer one may inherit it
    centroids = {0: _unit([1, 0.05, 0]), 1: _unit([1, 0.2, 0])}
    matched = creg.match_centroids(centroids, reg, threshold=0.3)
    assert list(matched) == [0]


def test_registry_threshold_and_label_roundtrip(sandbox):
    from src.features import cluster_registry as creg

    fp = creg.new_fingerprint(_unit([1, 0, 0]), 3)
    creg.save({"fingerprints": [fp]})
    assert creg.match_centroids({0: _unit([0, 1, 0])}, creg.load(), 0.3) == {}  # too far

    assert creg.set_label(fp["id"], "  scratch ")
    assert creg.get_label(fp["id"]) == "scratch"
    assert creg.set_label(fp["id"], "")  # clearing
    assert creg.get_label(fp["id"]) is None
    assert not creg.set_label("nope", "x")


def test_registry_migrates_missing_ids(sandbox):
    from src.features import cluster_registry as creg

    sandbox.CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    sandbox.CLUSTER_REGISTRY_PATH.write_text(json.dumps({"fingerprints": [{"centroid": [1, 0]}]}))
    assert creg.load()["fingerprints"][0]["id"]


# ── manifest: stable identity + valid boxes ──────────────────

def test_normalize_box():
    from src.pipeline.nodes.manifest_save import _normalize_box

    assert _normalize_box([100, 200, 300, 600]) == pytest.approx([0.4, 0.2, 0.4, 0.2])
    assert _normalize_box([-50, -50, 2000, 2000]) == pytest.approx([0.5, 0.5, 1.0, 1.0])  # clamped
    assert _normalize_box([500, 500, 500, 900]) is None  # zero height
    assert _normalize_box([900, 100, 100, 200]) is None  # inverted
    assert _normalize_box([]) is None


def test_manifest_prelabels_from_fingerprint_not_folder_name(sandbox):
    from src.pipeline.nodes.manifest_save import _save_cluster_manifest

    d = sandbox.CLUSTERS_DIR / "cluster_000"
    d.mkdir(parents=True)
    (d / "a.jpg").write_bytes(b"x")
    # an OLD manifest names cluster_000 "wrong_name" — it must not leak into the new run
    (sandbox.CLUSTERS_DIR / "cluster_manifest.json").write_text(
        json.dumps({"clusters": {"cluster_000": {"defect_name": "wrong_name"}}}))
    _save_cluster_manifest(
        "r1", {0: str(d)}, [], cluster_fingerprints={0: "fp1"}, inherited_labels={},
    )
    m = json.loads((sandbox.CLUSTERS_DIR / "cluster_manifest.json").read_text())
    assert m["clusters"]["cluster_000"]["defect_name"] is None
    assert m["clusters"]["cluster_000"]["fingerprint_id"] == "fp1"

    _save_cluster_manifest(
        "r2", {0: str(d)}, [], cluster_fingerprints={0: "fp1"}, inherited_labels={0: "scratch"},
    )
    m = json.loads((sandbox.CLUSTERS_DIR / "cluster_manifest.json").read_text())
    assert m["clusters"]["cluster_000"]["defect_name"] == "scratch"


# ── graph: gating + cleanup ──────────────────────────────────

def test_gate_routing():
    from src.pipeline.graph import _gate

    g = _gate("next", "items", "items")
    assert g({"items": [1]}) == "next"
    assert g({"items": []}) == END
    assert g({}) == END
    assert g({"items": [1], "errors": ["boom"]}) == END


def test_clean_previous_run_keeps_registry(sandbox):
    from src.pipeline.graph import _clean_previous_run

    (sandbox.CROPS_DIR / "c.jpg").write_bytes(b"x")
    cl = sandbox.CLUSTERS_DIR
    (cl / "cluster_000").mkdir(parents=True)
    (cl / "cluster_000" / "a.jpg").write_bytes(b"x")
    (cl / "cluster_manifest.json").write_text("{}")
    sandbox.CLUSTER_REGISTRY_PATH.write_text('{"fingerprints": []}')
    sandbox.CLUSTER_TUNING_CACHE_PATH.write_text("{}")

    _clean_previous_run()

    assert sandbox.CLUSTER_REGISTRY_PATH.exists() and sandbox.CLUSTER_TUNING_CACHE_PATH.exists()
    assert not (cl / "cluster_000").exists() and not (cl / "cluster_manifest.json").exists()
    assert not any(sandbox.CROPS_DIR.iterdir())


def test_run_lock_rejects_concurrent_runs():
    import src.pipeline.graph as g

    assert g._run_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError):
            g.run_discovery_pipeline()
    finally:
        g._run_lock.release()


# ── yolo fallback matching ───────────────────────────────────

def test_known_crop_match_is_exact_on_stem():
    from src.pipeline.nodes.yolo_inference import _match_known_crop

    idx = [("rust", "img1_crop_0000"), ("dent", "img10")]
    assert _match_known_crop("img1", idx) == "rust"
    assert _match_known_crop("img10", idx) == "dent"
    assert _match_known_crop("img", idx) is None
    assert _match_known_crop("img100", idx) is None


# ── YOLO label export ────────────────────────────────────────

def _make_clusters(sandbox, n=10, cluster="c_000"):
    (sandbox.CLUSTERS_DIR / cluster).mkdir(parents=True, exist_ok=True)
    cmap = {}
    for i in range(n):
        img = sandbox.INPUT_IMAGES_DIR / f"im{i}.jpg"
        img.write_bytes(b"x")
        (sandbox.CLUSTERS_DIR / cluster / f"im{i}_crop_0000.jpg").write_bytes(b"x")
        cmap[f"im{i}_crop_0000"] = {"source_image": str(img), "bbox_normalized": [.5, .5, .2, .2]}
    (sandbox.DATA_DIR / "crop_to_source.json").write_text(json.dumps(cmap))


def _label_lines(sandbox):
    return [line for p in (sandbox.YOLO_DATASET_DIR / "labels").rglob("*.txt")
            for line in p.read_text().splitlines()]


def test_export_is_idempotent_and_splits(sandbox):
    import yaml
    from src.pipeline.orchestrator import AnnotationMapper

    _make_clusters(sandbox)
    m = AnnotationMapper()
    for _ in range(3):
        r = m.map_cluster_labels_to_yolo({"c_000": "scratch"}, ["rust"])

    assert r["total_mapped"] == 10
    assert len(_label_lines(sandbox)) == 10  # not 30
    assert len(list((sandbox.YOLO_DATASET_DIR / "images" / "val").iterdir())) >= 1
    y = yaml.safe_load((sandbox.YOLO_DATASET_DIR / "data.yaml").read_text())
    assert y["val"] == "images/val" and y["names"] == ["rust", "scratch"] and y["nc"] == 2


def test_tiny_dataset_validates_on_train(sandbox):
    import yaml
    from src.pipeline.orchestrator import AnnotationMapper

    _make_clusters(sandbox, n=3)
    AnnotationMapper().map_cluster_labels_to_yolo({"c_000": "scratch"}, ["rust"])
    y = yaml.safe_load((sandbox.YOLO_DATASET_DIR / "data.yaml").read_text())
    assert y["val"] == "images/train"


def test_replay_adds_known_classes_without_overriding_new_labels(sandbox):
    from src.pipeline.orchestrator import AnnotationMapper

    _make_clusters(sandbox, n=6)
    known = sandbox.INPUT_IMAGES_DIR / "known0.jpg"
    known.write_bytes(b"x")
    clash = sandbox.INPUT_IMAGES_DIR / "im0.jpg"  # also in replay: the new label must win
    sandbox.KNOWN_REPLAY_JSON.write_text(json.dumps([
        {"image_path": str(known), "detections": [{"class_name": "Rust", "bbox_xywhn": [.3, .3, .1, .1]}]},
        {"image_path": str(clash), "detections": [{"class_name": "rust", "bbox_xywhn": [.9, .9, .1, .1]}]},
    ]))
    r = AnnotationMapper().map_cluster_labels_to_yolo({"c_000": "scratch"}, ["rust"])

    assert r["replay_images"] == 1
    lines = _label_lines(sandbox)
    assert any(line.startswith("0 0.300000") for line in lines)  # replayed old class (idx 0 = rust)
    assert not any(line.startswith("0 0.900000") for line in lines)  # clash ignored
    assert r["total_mapped"] == 6  # replay is not counted as new labels


# ── orchestrator state ───────────────────────────────────────

def test_signature_changes_only_when_state_changes():
    from src.pipeline.orchestrator import AutonomousOrchestrator as O

    a = O._signature({"c0": "x"}, {"c0": 5})
    assert a == O._signature({"c0": "x"}, {"c0": 5})
    assert a != O._signature({"c0": "y"}, {"c0": 5})
    assert a != O._signature({"c0": "x"}, {"c0": 6})


# ── LLM reply parsing ────────────────────────────────────────

def test_llm_parse_json_handles_reasoning_and_fences():
    from src.utils.llm import parse_json

    assert parse_json('<think>maybe {"a": 0}</think>\n```json\n{"retrain": true}\n```') == {"retrain": True}
    assert parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    with pytest.raises(Exception):
        parse_json("no json here")
