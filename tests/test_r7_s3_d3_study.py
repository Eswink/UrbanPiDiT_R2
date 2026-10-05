"""Structural checks for the S3-D3 incumbent study driver.

The round itself needs the archived actual-C protocol and the v2 store, which
are run-time artifacts and not in the repository; these tests pin the parts
that must not silently drift: the sealed-test refusal, the frozen round
constants, and the fine_tune contract the driver assembles (mode l6, no
inherited process supervision, one frozen endpoint). A counterproof shows the
contract check catches a changed mode instead of declaring victory on the
labels alone.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_driver():
    path = ROOT / "scripts" / "study_r7_s3_d3_incumbent.py"
    spec = importlib.util.spec_from_file_location("study_r7_s3_d3_incumbent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d3 = _load_driver()


def test_frozen_round_constants_are_the_registered_values():
    assert d3.SEEDS == (41, 42, 43)
    assert (d3.MODE, d3.UPDATES, d3.STEPS) == ("l6", 400, 4)
    assert d3.LAMBDA12 == 0.5 and d3.EVALUATION_LEADS == (6, 12, 24, 48, 72)
    assert d3.PLANNED_SECONDS_ROUND < d3.HARD_CAP_SECONDS_ROUND
    assert d3.DEADLINE_SECONDS_PER_SEED <= d3.HARD_CAP_SECONDS_ROUND
    assert d3.EXPECTED_COHORTS == {"6": 472, "12": 468, "24": 460, "48": 444, "72": 428}


def test_test_split_is_refused_by_name():
    with pytest.raises(ValueError, match="sealed"):
        d3.refused_test_manifest(ROOT / "outputs/whatever/test.jsonl")
    d3.refused_test_manifest(ROOT / "outputs/whatever/train.jsonl")
    d3.refused_test_manifest(ROOT / "outputs/whatever/val.jsonl")


def _toy_contract():
    spec = {"architecture": "window", "in_channels": 17, "out_channels": 17,
            "history_steps": 2, "dim": 8, "depth": 1, "heads": 2, "window_size": 4,
            "patch_size": 2, "anchored_processes": 8, "free_processes": 8}
    initialization = {"format": "test-fixture"}
    windows = {"excluded_sample_ids": [], "window_sha256": "0" * 64}
    return d3.contract_for(41, spec, initialization, "ab" * 32, "cd" * 32, windows)


def test_contract_matches_the_archived_actual_c_recipe():
    contract = _toy_contract()
    assert contract["kind"] == "process"
    assert contract["autoregression"]["mode"] == "l6"
    # child_contract zeroes lambda12 for l6; fine_tune later binds physical_steps=1.
    assert contract["autoregression"]["lambda12"] == 0.0
    assert contract["autoregression"]["physical_rollout_bptt"] is True
    assert contract["autoregression"]["internal_reasoning_bptt"] is True
    assert "process_supervision" not in contract
    assert contract["initialization"] == {"format": "test-fixture"}


def test_contract_mode_is_load_bearing(monkeypatch):
    monkeypatch.setattr(d3, "MODE", "two_step")
    contract = _toy_contract()
    assert contract["autoregression"]["mode"] == "two_step"
    assert contract["autoregression"]["lambda12"] == d3.LAMBDA12
