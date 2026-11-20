"""'Your input is needed' items: detection, clearing, dismiss and snooze."""
import json
import os
import time

import pytest


@pytest.fixture
def att(sandbox, tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(sandbox, "PIPELINE_RUNS_DIR", tmp_path / "runs")
    (tmp_path / "models").mkdir()
    (tmp_path / "runs").mkdir()
    from src.api.services import attention_service
    return attention_service


def _manifest(sandbox, clusters, run_id="r1"):
    for name in clusters:
        (sandbox.CLUSTERS_DIR / name).mkdir(parents=True, exist_ok=True)
    (sandbox.CLUSTERS_DIR / "cluster_manifest.json").write_text(json.dumps(
        {"run_id": run_id, "clusters": {n: {"defect_name": d} for n, d in clusters.items()}}))


def _keys(att):
    return [i["key"].split(":")[0] for i in att.items()]


def test_nothing_needed_on_an_empty_system(att):
    assert att.items() == []


def test_unnamed_clusters_block_until_named(att, sandbox):
    _manifest(sandbox, {"cluster_000": None, "cluster_001": "scratch", "noise": None})
    items = att.items()
    assert items[0]["key"] == "clusters-unnamed:r1" and items[0]["severity"] == "blocking"
    assert items[0]["title"].startswith("1 defect cluster")  # noise is never counted

    _manifest(sandbox, {"cluster_000": "dent", "cluster_001": "scratch"})
    assert "clusters-unnamed" not in _keys(att)
    assert "ready-to-finetune" in _keys(att)  # named, never trained on


def test_ready_to_finetune_clears_after_a_newer_finetune(att, sandbox):
    _manifest(sandbox, {"cluster_000": "dent"})
    assert "ready-to-finetune" in _keys(att)
    run = sandbox.PIPELINE_RUNS_DIR / "retrain_x.json"
    run.write_text(json.dumps({"steps": {}}))
    later = time.time() + 5
    os.utime(run, (later, later))
    assert "ready-to-finetune" not in _keys(att)


def test_registered_model_waits_for_deploy_decision(att, sandbox):
    reg = {"current": "v1", "versions": [
        {"version_id": "v1", "version_num": 1, "status": "deployed"},
        {"version_id": "v2_x", "version_num": 2, "status": "registered", "metrics": {"map50": 0.12}},
    ]}
    (sandbox.MODELS_DIR / "registry.json").write_text(json.dumps(reg))
    item = att.items()[0]
    assert item["key"] == "model-awaiting-deploy:v2_x" and item["severity"] == "blocking"
    assert "0.120" in item["detail"]


def test_failed_and_advisor_not_ready_runs(att, sandbox):
    (sandbox.PIPELINE_RUNS_DIR / "20260101_000000.json").write_text(json.dumps({"steps": {
        "vlm_annotation": {"step_name": "vlm_annotation", "status": "failed", "error": "429"}}}))
    (sandbox.PIPELINE_RUNS_DIR / "retrain_1.json").write_text(json.dumps({"steps": {
        "llm_advisor": {"status": "success", "metadata": {"should_train": False, "reason": "too few crops"}}}}))
    by_key = {i["key"]: i for i in att.items()}
    assert by_key["run-failed:20260101_000000"]["detail"] == "429"
    assert by_key["advisor-not-ready:retrain_1"]["detail"] == "too few crops"


def test_dismiss_hides_until_situation_changes_and_snooze_expires(att, sandbox):
    _manifest(sandbox, {"cluster_000": None}, run_id="r1")
    att.dismiss("clusters-unnamed:r1")
    assert "clusters-unnamed" not in _keys(att)

    _manifest(sandbox, {"cluster_000": None}, run_id="r2")  # a new discovery run raises it again
    assert "clusters-unnamed" in _keys(att)

    att.dismiss("clusters-unnamed:r2", snooze_minutes=60)
    assert "clusters-unnamed" not in _keys(att)
    state = json.loads((sandbox.DATA_DIR / "attention_state.json").read_text())
    state["clusters-unnamed:r2"]["until"] = "2000-01-01T00:00:00+00:00"  # snooze over
    (sandbox.DATA_DIR / "attention_state.json").write_text(json.dumps(state))
    assert "clusters-unnamed" in _keys(att)
    assert "clusters-unnamed:r1" not in json.loads((sandbox.DATA_DIR / "attention_state.json").read_text())
