"""Offline tests for the ERA5 v2 read plan and the normalization floor audit (#63).

No network, no data writes: the read plan is pure computation over recorded
chunk geometry, and the floor audit is a read-only bookkeeping pass.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data" / "download" / "read_plan_frozen.py"
AUDIT = ROOT / "data" / "preprocess" / "normalization_audit.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_channel_plan_matches_the_published_17_channel_order():
    plan = _load("read_plan", PLAN)
    rows = plan.channel_plan()
    assert len(rows) == 17
    assert [row["channel"] for row in rows] == [
        "t2m", "u10", "v10", "mslp", "z850", "t850", "q850", "u850", "v850",
        "z500", "t500", "q500", "u500", "v500", "z250", "u250", "v250"]
    assert {row["level_hpa"] for row in rows if row["level_hpa"]} == {250, 500, 850}
    surface = [row for row in rows if row["level_hpa"] is None]
    assert [row["channel"] for row in surface] == ["t2m", "u10", "v10", "mslp"]


def test_window_availability_and_gap_exclusion():
    plan = _load("read_plan", PLAN)
    # 120 six-hour steps (30 days), history 2 frames, +72h = 12 lead frames.
    count, starts = plan.complete_windows(120, stride_steps=1)
    assert count == 120 - (2 - 1) - 12
    assert starts[0] == 0 and starts[-1] == 106
    # A gap forbids any window crossing it; windows before it survive.
    count_gap, starts_gap = plan.complete_windows(120, stride_steps=1, gaps=(50,))
    assert all(not (start <= 50 <= start + 13) for start in starts_gap)
    assert count_gap == count - 14  # windows starting at 38..51 crossed the gap
    # 13 frames cannot host history(2) + 12 lead frames; 14 frames host exactly one.
    assert plan.complete_windows(13)[0] == 0
    assert plan.complete_windows(14)[0] == 1
    with pytest.raises(ValueError):
        plan.complete_windows(120, stride_steps=0)


def test_init_hour_coverage_rotates_over_four_utc_hours():
    plan = _load("read_plan", PLAN)
    _, starts = plan.complete_windows(120, stride_steps=4)
    coverage = plan.init_hour_coverage(starts)
    assert sorted(coverage) == [0, 6, 12, 18]
    assert sum(coverage.values()) == len(starts)


def test_cost_models_count_chunk_unions_and_never_claim_http_bytes():
    plan = _load("read_plan", PLAN)
    arco_layout = plan.SourceLayout(
        name="temperature", dims=("time", "level", "lat", "lon"),
        shape=(100000, 37, 721, 1440), chunks=(1, 37, 721, 1440), dtype_bytes=2)
    cost = plan.arco_global_per_time_cost([arco_layout], times=121)
    field_cells = 37 * 721 * 1440
    assert cost["chunk_union_touched"] == 121
    assert cost["decoded_bytes"] == 2 * field_cells * 121
    assert cost["cropped_bytes"] == 2 * 65 * 65 * 121
    assert cost["http_bytes"] is None
    assert plan.arco_global_per_time_cost([arco_layout], times=242)["decoded_bytes"] \
        == 2 * cost["decoded_bytes"]
    tile_layout = plan.SourceLayout(
        name="temperature", dims=("time", "level", "lat", "lon"),
        shape=(100000, 13, 181, 360), chunks=(24, 13, 181, 360), dtype_bytes=2)
    tile_cost = plan.earthmover_temporal_tile_cost([tile_layout], times=121)
    assert tile_cost["chunk_union_touched"] == math.ceil(121 / 24)
    assert tile_cost["decoded_bytes"] == math.ceil(121 / 24) * (13 * 181 * 360) * 24 * 2
    assert tile_cost["http_bytes"] is None
    with pytest.raises(ValueError):
        plan.SourceLayout(name="x", dims=("time",), shape=(10,), chunks=(1,),
                          dtype_bytes=0)


def test_frozen_protocol_seals_scope_and_excludes_2020_from_test():
    plan = _load("read_plan", PLAN)
    protocol = plan.frozen_protocol()
    assert protocol["splits"]["train_years"] == [2016, 2017, 2018]
    assert protocol["splits"]["val_years"] == [2019]
    assert protocol["splits"]["test_candidate_years"] == [2021]
    assert protocol["splits"]["dev_accessed_years_excluded_from_test"] == [2020]
    assert 2020 not in protocol["splits"]["test_candidate_years"]
    assert protocol["channel_count"] == 17
    assert protocol["d1"]["days"] == 30 and protocol["d1"]["init_times"] == 120
    assert "future ERA5 is never used as boundary forcing" in protocol["halo_rule"]
    assert "train years only" in protocol["normalization_rule"]
    assert protocol["budget_caps"]["new_artifacts_bytes"] == 16 * 2**30
    assert protocol["budget_caps"]["decoded_source_bytes"] == 64 * 2**30
    assert protocol["scientific_claim"] is False
    assert protocol["protocol_sha256"] == _load("read_plan", PLAN).canonical_digest(
        {key: value for key, value in protocol.items() if key != "protocol_sha256"})


def test_normalization_floor_audit_reports_crushed_small_magnitude_channels():
    audit = _load("norm_audit", AUDIT)
    # A moisture-flux-convergence-like label: values ~1e-9, std below the floor.
    tiny = np.linspace(-2e-9, 2e-9, 100)
    record = audit.audit_channel(tiny)
    assert record["floor_active"] is True
    assert record["std"] < 1e-6 <= record["floored_std"]
    assert record["near_zero_fraction_normalized"] > 0.9, \
        "a 1e-9-scale channel normalized by the 1e-6 floor is nearly zeroed"
    # A normal-magnitude channel (q850 ~ 1e-3): the floor must stay inactive.
    normal = np.linspace(1e-4, 2e-3, 100)
    healthy = audit.audit_channel(normal)
    assert healthy["floor_active"] is False
    assert healthy["near_zero_fraction_normalized"] == 0.0
    report = audit.audit_label_set({"moisture_flux_convergence": tiny,
                                    "q_advection": normal})
    assert report["floor_active_channels"] == ["moisture_flux_convergence"]
    assert report["scientific_claim"] is False
    with pytest.raises(ValueError):
        audit.audit_channel(tiny, eps=-1.0)
    with pytest.raises(ValueError):
        audit.audit_channel(np.array([np.nan]))
