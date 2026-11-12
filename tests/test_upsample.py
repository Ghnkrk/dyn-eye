"""Class-balancing upsampler: exact box transforms, balancing, cleanup, advisor flag."""
import hashlib

import cv2
import numpy as np
import pytest
import yaml

from src.retraining.tools import upsample as up


def _write_sample(root, split, stem, boxes, size=(120, 200)):
    """Black image with a white rectangle per box; YOLO label file alongside."""
    h, w = size
    img = np.zeros((h, w, 3), np.uint8)
    for _, (x, y, bw, bh) in boxes:
        x1, y1 = int((x - bw / 2) * w), int((y - bh / 2) * h)
        x2, y2 = int((x + bw / 2) * w), int((y + bh / 2) * h)
        img[y1:y2, x1:x2] = 255
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), img)
    (root / "labels" / split / f"{stem}.txt").write_text(
        "".join(f"{c} {x} {y} {bw} {bh}\n" for c, (x, y, bw, bh) in boxes)
    )


def _dataset(root, names, train, val=()):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.dump({"names": names, "nc": len(names)}))
    for stem, boxes in train:
        _write_sample(root, "train", stem, boxes)
    for stem, boxes in val:
        _write_sample(root, "val", stem, boxes)


@pytest.mark.parametrize("name,img_op,box_op", up.FLIP_TRANSFORMS + up.ROTATE_TRANSFORMS)
def test_geometric_transform_maps_box_exactly(name, img_op, box_op):
    h, w = 120, 200
    box = (0.30, 0.25, 0.20, 0.30)  # off-centre so every transform moves it
    img = np.zeros((h, w), np.uint8)
    x, y, bw, bh = box
    img[int((y - bh / 2) * h):int((y + bh / 2) * h), int((x - bw / 2) * w):int((x + bw / 2) * w)] = 255

    out = img_op(img)
    ys, xs = np.nonzero(out)
    oh, ow = out.shape
    actual = ((xs.min() + xs.max() + 1) / 2 / ow, (ys.min() + ys.max() + 1) / 2 / oh,
              (xs.max() - xs.min() + 1) / ow, (ys.max() - ys.min() + 1) / oh)
    assert actual == pytest.approx(box_op(*box), abs=0.01), name


def test_balances_minority_up_to_cap_and_leaves_val_alone(tmp_path):
    root = tmp_path / "yolo"
    big = [(f"a{i}", [(0, (0.5, 0.5, 0.2, 0.2))]) for i in range(10)]
    small = [(f"b{i}", [(1, (0.3, 0.3, 0.2, 0.2))]) for i in range(2)]
    _dataset(root, ["big", "small"], big + small, val=[("v0", [(1, (0.5, 0.5, 0.2, 0.2))])])
    val_before = sorted(p.name for p in (root / "images" / "val").iterdir())

    r = up.upsample_dataset(target_ratio=1.0, max_factor=3.0, dataset_dir=root)

    assert r["before"] == {"big": 10, "small": 2}
    assert r["after"] == {"big": 10, "small": 6}  # wants 10, capped at 3 × 2
    assert r["capped_classes"] == ["small"]
    assert r["generated_images"] == 4
    assert up.count_instances(root, include_augmented=True) == {"big": 10, "small": 6}
    aug = sorted(p for p in (root / "images" / "train").iterdir() if up.is_augmented(p))
    assert len(aug) == 4
    assert all((root / "labels" / "train" / f"{p.stem}.txt").exists() for p in aug)
    assert sorted(p.name for p in (root / "images" / "val").iterdir()) == val_before


def test_rerun_replaces_previous_copies_deterministically(tmp_path):
    root = tmp_path / "yolo"
    _dataset(root, ["big", "small"],
             [(f"a{i}", [(0, (0.5, 0.5, 0.2, 0.2))]) for i in range(6)] + [("b0", [(1, (0.3, 0.3, 0.2, 0.2))])])

    def digest():
        files = sorted(p for p in root.rglob("*") if p.is_file() and up.is_augmented(p))
        return [(p.name, hashlib.md5(p.read_bytes()).hexdigest()) for p in files]

    up.upsample_dataset(target_ratio=1.0, max_factor=3.0, seed=7, dataset_dir=root)
    first = digest()
    up.upsample_dataset(target_ratio=1.0, max_factor=3.0, seed=7, dataset_dir=root)
    assert digest() == first and len(first) == 2 * 2  # 2 images + 2 labels, not 4 + 4

    assert up.remove_augmented(root) == 4
    assert digest() == []


def test_prefers_images_where_the_minority_class_is_alone(tmp_path):
    root = tmp_path / "yolo"
    _dataset(root, ["big", "small"], [
        *[(f"a{i}", [(0, (0.5, 0.5, 0.2, 0.2))]) for i in range(8)],
        ("mixed", [(0, (0.2, 0.2, 0.1, 0.1)), (0, (0.8, 0.8, 0.1, 0.1)), (1, (0.5, 0.5, 0.1, 0.1))]),
        ("pure", [(1, (0.4, 0.4, 0.2, 0.2))]),
    ])
    r = up.upsample_dataset(target_ratio=0.5, max_factor=3.0, dataset_dir=root)  # small: 2 → 5
    aug = [p.stem for p in (root / "images" / "train").iterdir() if up.is_augmented(p)]
    # The pure image covers all 3 copies, so no extra "big" boxes are added
    assert sorted(aug) == ["pure__aug000", "pure__aug001", "pure__aug002"]
    assert r["after"] == {"big": 10, "small": 5}


def test_stops_and_reports_when_images_run_out(tmp_path):
    root = tmp_path / "yolo"
    _dataset(root, ["big", "small"],
             [(f"a{i}", [(0, (0.5, 0.5, 0.2, 0.2))]) for i in range(20)] + [("b0", [(1, (0.3, 0.3, 0.2, 0.2))])])
    r = up.upsample_dataset(target_ratio=1.0, max_factor=10.0, dataset_dir=root)
    assert r["after"]["small"] == 4  # 1 original + one copy per flip variant
    assert r["limited_classes"] == ["small"]


def test_balanced_or_single_class_dataset_is_left_unchanged(tmp_path):
    root = tmp_path / "yolo"
    _dataset(root, ["only"], [(f"a{i}", [(0, (0.5, 0.5, 0.2, 0.2))]) for i in range(3)])
    r = up.upsample_dataset(dataset_dir=root)
    assert r["generated_images"] == 0 and "note" in r


def test_class_balance_flags_minorities():
    b = up.class_balance({"Streak marks": 80, "Dark spots": 40, "Oil spill": 25})
    assert b["suggest_upsampling"] is True
    assert b["minority_classes"] == ["Dark spots", "Oil spill"]  # ≤ 80 / 2.0
    assert b["imbalance_ratio"] == 3.2
    assert up.class_balance({"a": 30, "b": 25})["suggest_upsampling"] is False


def test_advisor_fallback_recommends_upsampling_for_imbalance():
    from src.retraining.llm_advisor import _heuristic_fallback, _validate_recommendation

    classes = {"Streak marks": 80, "Oil spill": 25}
    meta = {
        "classes": classes, "num_classes": 2, "total_crops": 105,
        "min_crops_per_class": 25, "max_crops_per_class": 80,
        "class_balance": up.class_balance(classes),
    }
    rec = _heuristic_fallback(meta)
    assert rec["recommend_upsampling"] is True
    assert "Oil spill (25)" in rec["upsampling_reason"]

    # LLM answers without the new fields still get a sensible default
    filled = _validate_recommendation({"should_train": True, "config": {}}, meta)
    assert filled["recommend_upsampling"] is True and filled["upsampling_reason"]


def test_upsample_stage_sits_between_advisor_and_training():
    from src.retraining.agent import build_retraining_graph

    edges = {(e.source, e.target) for e in build_retraining_graph().get_graph().edges}
    assert ("llm_advisor", "upsample_dataset") in edges
    assert ("upsample_dataset", "train_yolo") in edges
    assert ("llm_advisor", "train_yolo") not in edges
