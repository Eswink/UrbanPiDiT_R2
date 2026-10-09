"""CI-safe checks for the wide-region (129x129) single-factor rollout-dose driver.

The wide single-factor round applies the registered 800-update long-rollout recipe
unchanged to the wide store and scores the frozen central 65x65 box through
``boundary_margins=(32,)``; the fidelity-critical parts are that the boundary
margin is exactly the target-box margin, that the instance paths point at the
wide store whose test manifest is ``test.jsonl``, that the driver never opens the
sealed test manifest, and that it refuses an existing output or a missing store.
Counterproofs keep the checks honest: each assertion fails if the constant is
wrong.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wide = _load("study_r7_s3_wide_single_factor")


def test_frozen_constants():
    assert wide.SEEDS == (41, 42, 43)
    assert (wide.UPDATES, wide.LR, wide.WARMUP) == (800, 2e-5, 10)
    assert wide.PHYSICAL_WEIGHTS == (1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5)
    assert wide.BOUNDARY_MARGINS == (32,)
    assert wide.EVALUATION_LEADS == (6, 12, 24, 48, 72)
    assert wide.REASONING_STEPS == 4
    assert (wide.STEPS, wide.WEIGHT_DECAY, wide.BATCH_SIZE, wide.CLIP) == (4, 1e-4, 1, 1.)
    assert (wide.CHECKPOINT_EVERY, wide.BF16) == (20, False)
    assert wide.PLANNED_SECONDS == 9000.0
    assert wide.HARD_CAP_SECONDS == 18000.0
    assert wide.PER_SEED_SECONDS == 4200.0
    assert wide.PLANNED_SECONDS < wide.HARD_CAP_SECONDS
    assert wide.PER_SEED_SECONDS <= wide.HARD_CAP_SECONDS


def test_boundary_margin_is_the_frozen_target_box_margin():
    from data.download import read_plan_wide
    from training.r7_boundary_metrics import boundary_masks

    assert wide.BOUNDARY_MARGINS == (read_plan_wide.TARGET_MARGIN_CELLS,)
    masks = {name: mask for name, _, mask in boundary_masks(129, 129, wide.BOUNDARY_MARGINS)}
    # the interior_32 region is exactly the frozen 65x65 box, not the full grid
    assert int(masks["interior_32"].sum()) == 65 * 65
    assert int(masks["interior_32"].sum()) == read_plan_wide.TARGET_POINTS ** 2
    assert int(masks["full"].sum()) == 129 * 129


def test_instance_paths_point_at_the_wide_instance():
    assert wide.INSTANCE.name == "r7_s3_wide_instance_v4_20261009_attempt01"
    assert wide.STORE.parent.name == "store"
    assert wide.STORE.name == "cache.zarr"
    assert wide.TRAIN_MANIFEST.name == "train.jsonl"
    assert wide.VAL_MANIFEST.name == "val.jsonl"
    assert wide.TEST_MANIFEST.name == "test.jsonl"
    assert wide.TEST_MANIFEST.parent.name == "manifests"


def test_wide_val_cohorts_match_the_registered_v3_cohorts():
    from training import r7_s3_v3_screen as screen

    assert wide.WIDE_VAL_COHORTS == screen.V3_VAL_COHORTS
    assert wide.WIDE_VAL_COHORTS == {"6": 472, "12": 468, "24": 460, "48": 444, "72": 428}


def test_protocol_constants_pin_region_recipe_and_no_scientific_claim():
    body = wide.protocol_constants()
    assert body["format"] == "r7-s3-wide-single-factor-protocol-v1"
    assert body["kind"] == "screen"
    assert body["region"]["points"] == [129, 129]
    assert body["region"]["frozen_box_points"] == [65, 65]
    assert body["boundary_margins"] == [32]
    assert body["recipe"] == {
        "mode": "long_rollout", "updates": 800, "lr": 2e-5, "warmup": 10,
        "physical_weights": [1., .5, 0., .5, 0., 0., 0., .5, 0., 0., 0., .5],
        "steps": 4, "weight_decay": 1e-4, "batch_size": 1, "clip": 1.,
        "checkpoint_every": 20, "bf16": False}
    assert body["seeds"] == [41, 42, 43]
    assert body["evaluation_leads_hours"] == [6, 12, 24, 48, 72]
    assert body["evaluation"]["boundary_margins"] == [32]
    assert body["evaluation"]["device"] == "cuda:0"
    assert body["scientific_claim"] is False
    assert body["test_read"] is False
    assert body["test_manifest_never_read"].endswith("test.jsonl")
    assert "never opened" in body["test_read_policy"]
    # counterproof: the pure constants must not carry a store-derived identity
    assert "source_sha256" not in body
    assert "train_data_identity" not in body


def test_driver_refuses_an_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        wide.main(["--out", str(tmp_path)])


def test_driver_refuses_before_touching_the_store(tmp_path, monkeypatch):
    # With no built store the driver must fail on the missing store, not run. The
    # store path is monkeypatched so this holds whether or not the wide instance
    # happens to be built on this machine (otherwise the test would run the study).
    monkeypatch.setattr(wide, "STORE", tmp_path / "absent" / "cache.zarr")
    out = tmp_path / "fresh"
    with pytest.raises(FileNotFoundError):
        wide.main(["--out", str(out)])
    assert not out.exists()
