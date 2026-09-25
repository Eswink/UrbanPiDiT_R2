"""Offline tests for the adaptive-halting gate audit (#7).

The audit decides whether #7's gate is satisfiable. These tests pin the two
mechanisms that decision rests on: per-seed depth-benefit counting, and the
per-variable tolerance feasibility that mirrors the policy search.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "training" / "r7_halting_gate_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("r7_halting_gate_audit_under_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHANNELS = ["t2m", "z250"]


def _summary_row(arm, depth, variable, values):
    return {
        "arm": arm, "depth": depth, "variable": variable, "lead_hours": 6.0,
        "seeds": len(values), "rmse_mean": float(np.mean(values)),
        "rmse_sd": None, "values": list(values),
    }


def _result(tmp_path, rows):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({
        "protocol_sha256": "p" * 64, "summary": rows,
    }), encoding="utf-8")
    return path


def _store(tmp_path, std=(1.0, 10.0)):
    import zarr

    path = tmp_path / "cache.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.attrs["channels"] = list(CHANNELS)
    root.create_array("normalization_std", data=np.asarray(std, dtype=np.float32))
    return path


def test_deeper_consistently_worse_is_detected(tmp_path):
    module = _module()
    rows = []
    for seed in range(3):
        # Both channels and both depths get worse with depth, for every seed.
        rows.append(_summary_row("generic", 0, "t2m", [1.0 + seed] * 1))
        rows.append(_summary_row("generic", 3, "t2m", [2.0 + seed] * 1))
        rows.append(_summary_row("generic", 0, "z250", [10.0 + seed] * 1))
        rows.append(_summary_row("generic", 3, "z250", [20.0 + seed] * 1))
    # One row per arm/depth/variable carrying three seed values.
    rows = []
    for depth, factor in ((0, 1.0), (3, 2.0)):
        rows.append(_summary_row("generic", depth, "t2m",
                                 [factor * (1.0 + s) for s in range(3)]))
        rows.append(_summary_row("generic", depth, "z250",
                                 [factor * (10.0 + s) for s in range(3)]))
    _, table = module.load_depth_table(_result(tmp_path, rows))
    names, std = module.normalization_std(_store(tmp_path))
    curve = module.depth_curve(table, names, std, arm="generic", seeds=(41, 42, 43))
    benefit = module.gauge_depth_benefit([curve])
    assert benefit[0]["seeds_deeper_better"] == 0
    assert benefit[0]["seeds_deeper_worse"] == 3
    assert benefit[0]["deeper_is_consistently_worse"] is True
    assert benefit[0]["relative_change"] > 0


def test_mixed_depth_benefit_is_not_flagged_as_consistent(tmp_path):
    """One seed improving must stop the audit calling depth harmful."""
    module = _module()
    rows = []
    for depth, factors in ((0, [1.0, 1.0, 1.0]), (3, [0.5, 1.0, 2.0])):
        rows.append(_summary_row("generic", depth, "t2m", factors))
        rows.append(_summary_row("generic", depth, "z250", factors))
    _, table = module.load_depth_table(_result(tmp_path, rows))
    names, std = module.normalization_std(_store(tmp_path))
    benefit = module.gauge_depth_benefit(
        [module.depth_curve(table, names, std, arm="generic", seeds=(41, 42, 43))])
    assert benefit[0]["deeper_is_consistently_worse"] is False


def test_tolerance_feasibility_mirrors_the_policy_search(tmp_path):
    """A single blocking variable must make every tolerance infeasible."""
    module = _module()
    rows = []
    for depth, t2m, z250 in ((0, 1.0, 1.5), (3, 1.0, 1.0)):
        rows.append(_summary_row("process_feedback", depth, "t2m", [t2m] * 3))
        rows.append(_summary_row("process_feedback", depth, "z250", [z250] * 3))
    _, table = module.load_depth_table(_result(tmp_path, rows))
    feasibility = module.tolerance_feasibility(
        table, reference_depth=3, tolerances=(0.0, 0.01, 0.10),
        baseline_arm="process_feedback", candidate_arm="process_feedback")
    assert all(row["feasible"] is False for row in feasibility)
    assert all(row["any_depth_feasible"] is False for row in feasibility)
    # z250 is 1.5x the reference, so even a 10% tolerance must fail on it.
    worst = [row for row in feasibility if row["tolerance"] == 0.10][0]
    assert worst["blocking_variable"] == "z250"
    assert worst["blocking_ratio"] == pytest.approx(1.5)
    assert worst["variables_within_tolerance"] == 1


def test_tolerance_feasibility_finds_a_qualifying_depth(tmp_path):
    module = _module()
    rows = []
    for depth, factor in ((0, 1.2), (1, 1.005), (3, 1.0)):
        rows.append(_summary_row("process_feedback", depth, "t2m", [factor] * 3))
        rows.append(_summary_row("process_feedback", depth, "z250", [factor] * 3))
    _, table = module.load_depth_table(_result(tmp_path, rows))
    tight = module.tolerance_feasibility(
        table, reference_depth=3, tolerances=(0.01,),
        baseline_arm="process_feedback", candidate_arm="process_feedback")
    by_depth = {row["depth"]: row for row in tight}
    # K=0 is 1.2x the reference and must fail; K=1 at 1.005x must pass.
    assert by_depth[0]["feasible"] is False
    assert by_depth[1]["feasible"] is True
    assert all(row["any_depth_feasible"] is True for row in tight)


def test_score_weights_channels_equally_after_normalization(tmp_path):
    """The objective must be equal-channel normalized, as the training loss is."""
    module = _module()
    rows = []
    for depth, t2m, z250 in ((0, 2.0, 20.0), (3, 1.0, 10.0)):
        rows.append(_summary_row("generic", depth, "t2m", [t2m] * 3))
        rows.append(_summary_row("generic", depth, "z250", [z250] * 3))
    _, table = module.load_depth_table(_result(tmp_path, rows))
    names, std = module.normalization_std(_store(tmp_path, std=(1.0, 10.0)))
    shallow = module.score(table, names, std, arm="generic", depth=0, seed_index=0)
    deep = module.score(table, names, std, arm="generic", depth=3, seed_index=0)
    # 2.0/1.0 and 20.0/10.0 both equal 2.0; the deep row is 1.0 for both.
    assert shallow == pytest.approx(2.0)
    assert deep == pytest.approx(1.0)


def test_audit_marks_the_gate_unsatisfiable_when_the_reference_is_not_the_ceiling(tmp_path):
    module = _module()
    rows = []
    for factor in (1.0, 2.0):
        for depth in (0, 3):
            value = factor if depth == 0 else 2.0 * factor
            rows.append(_summary_row("process_feedback", depth, "t2m", [value] * 3))
            rows.append(_summary_row("process_feedback", depth, "z250", [value] * 3))
    report = module.audit(_result(tmp_path, rows), _store(tmp_path))
    assert report["reference_is_accuracy_ceiling"] is False
    assert report["gate_satisfiable"] is False
    assert report["verdict"]["adaptive_benefit_demonstrated"] is False
    # Depth hurts on both channels here, so the tolerance IS satisfiable while the
    # reference is not the ceiling: the summary must name the confounding cause.
    assert "confounded" in report["verdict"]["summary"]
    assert "fixed-Kmax" in report["verdict"]["summary"]
    assert report["scientific_claim"] is False
    assert any("three seeds" in line for line in report["limitations"])


def test_audit_requires_at_least_two_measured_depths(tmp_path):
    module = _module()
    rows = [_summary_row("generic", 3, "t2m", [1.0] * 3),
            _summary_row("generic", 3, "z250", [1.0] * 3)]
    _, table = module.load_depth_table(_result(tmp_path, rows))
    names, std = module.normalization_std(_store(tmp_path))
    with pytest.raises(ValueError, match="fewer than two measured depths"):
        module.depth_curve(table, names, std, arm="generic", seeds=(41,))


def test_report_is_exclusive_and_self_describing(tmp_path):
    module = _module()
    report = {"format": "r7-halting-gate-audit-v1", "scientific_claim": False}
    path = tmp_path / "nested" / "report.json"
    module.write_report(path, report)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        module.write_report(path, report)


def test_module_states_the_tolerance_rule_it_mirrors():
    source = MODULE.read_text(encoding="utf-8")
    assert "every variable must meet the tolerance" in source.lower()
    assert "no aggregation" in source.lower() or "without any aggregation" in source.lower()
    assert "re-tunes nothing" in source
