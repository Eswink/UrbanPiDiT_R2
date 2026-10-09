"""CI-safe checks for the wide-region (S3 lateral-context) D2 baselines driver.

The wide D2 round scores the parameter-free references on the 129x129 instance and
must read out the frozen central 65x65 box through ``boundary_margins=(32,)``; the
fidelity-critical parts are that the boundary margin is exactly the target-box
margin, that the driver never opens the sealed test manifest, and that it refuses
to overwrite an existing output. Counterproofs keep the checks honest.
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


wide_d2 = _load("study_r7_s3_wide_d2_baselines")


def test_frozen_constants():
    assert wide_d2.EVALUATION_LEADS == (6, 12, 24, 48, 72)
    assert wide_d2.BOUNDARY_MARGINS == (32,)
    assert wide_d2.PLANNED_SECONDS_ROUND < wide_d2.HARD_CAP_SECONDS_ROUND


def test_boundary_margin_is_the_frozen_target_box_margin():
    from data.download import read_plan_wide
    from training.r7_boundary_metrics import boundary_masks

    assert wide_d2.BOUNDARY_MARGINS == (read_plan_wide.TARGET_MARGIN_CELLS,)
    masks = {name: mask for name, _, mask in boundary_masks(129, 129, wide_d2.BOUNDARY_MARGINS)}
    assert int(masks["interior_32"].sum()) == read_plan_wide.TARGET_POINTS ** 2


def test_instance_paths_point_at_the_wide_instance():
    assert wide_d2.INSTANCE.name == "r7_s3_wide_instance_v4_20261009_attempt01"
    assert wide_d2.STORE.parent.name == "store"
    assert wide_d2.TRAIN_MANIFEST.name == "train.jsonl"
    assert wide_d2.VAL_MANIFEST.name == "val.jsonl"
    assert wide_d2.TEST_MANIFEST.name == "test.jsonl"


def test_protocol_constants_pin_the_region_and_never_read_test():
    body = wide_d2.protocol_constants()
    assert body["region"]["points"] == [129, 129]
    assert body["evaluation"]["boundary_margins"] == [32]
    assert body["evaluation"]["split"] == "val"
    assert body["evaluation"]["lead_hours"] == [6, 12, 24, 48, 72]
    assert body["scientific_claim"] is False
    assert "never opened" in body["test_read_policy"]
    assert body["test_manifest_never_read"].endswith("test.jsonl")
    # The constant body must not carry a source/data identity: those come only from
    # the built store, so a clean checkout can never pin a stale one.
    assert "source_sha256" not in body
    assert "data_identity" not in body


def test_driver_refuses_an_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        wide_d2.main(["--out", str(tmp_path)])


def test_driver_refuses_before_touching_the_store(tmp_path, monkeypatch):
    # With no built store the driver must fail on the missing store, not run. The
    # store path is monkeypatched so this holds whether or not the wide instance
    # happens to be built on this machine (otherwise the test would run the study).
    monkeypatch.setattr(wide_d2, "STORE", tmp_path / "absent" / "cache.zarr")
    out = tmp_path / "fresh"
    with pytest.raises(FileNotFoundError):
        wide_d2.main(["--out", str(out)])
    assert not out.exists()
