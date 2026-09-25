"""Offline tests for the fair-budget co-reasoning comparison (#6).

These cover the reporting logic that decides what the comparison may claim:
per-variable pairing, the win/loss count, and the refusal to average across
physical units. Training is not exercised here.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "training" / "r7_coreasoning_compare.py"


def _module():
    spec = importlib.util.spec_from_file_location("r7_coreasoning_under_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rmse(directory: Path, rows) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "rmse.csv"
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lead_hours", "variable", "rmse", "unit", "n_initializations"])
        for lead, variable, rmse, unit in rows:
            writer.writerow([lead, variable, rmse, unit, 8])
    return directory


def _record(arm, seed, depth, directory):
    return {"arm": arm, "seed": seed, "depth": depth, "evaluation_dir": str(directory)}


def test_arm_configs_keep_the_budget_parity_settings():
    module = _module()
    generic = module.arm_config("generic", {"latent_tokens": 16}, 17)
    process = module.arm_config(
        "process", {"anchored_processes": 8, "free_processes": 8,
                    "use_forecast_feedback": True}, 17)
    assert generic["in_channels"] == process["in_channels"] == 17
    assert generic["out_channels"] == process["out_channels"] == 17
    for key in ("dim", "depth", "heads", "window_size", "patch_size", "history_steps"):
        assert generic[key] == process[key], key
    # Feedback is the only difference between the two process arms.
    off = module.arm_config(
        "process", {"anchored_processes": 8, "free_processes": 8,
                    "use_forecast_feedback": False}, 17)
    assert off != process
    assert {k for k in off if off[k] != process[k]} == {"use_forecast_feedback"}


def test_protocol_freezes_before_any_step_and_declares_its_limits():
    from training.r7_experiment import canonical_digest

    module = _module()
    protocol = module.protocol_payload("a" * 64, 11, 200, (0, 1, 3), (41, 42, 43))
    assert protocol["frozen_before_any_step"] is True
    assert protocol["scientific_claim"] is False
    assert protocol["test_evaluated"] is False
    assert protocol["seeds"] == [41, 42, 43]
    assert protocol["optimizer_updates"] == 200
    assert protocol["ablation_depths"] == [0, 1, 3]
    assert protocol["all_seeds_retained"] is True
    assert len(protocol["arms"]) == 3
    assert {arm["name"] for arm in protocol["arms"]} == {
        "generic", "process_no_feedback", "process_feedback"}
    assert protocol["protocol_sha256"] == canonical_digest(
        {k: v for k, v in protocol.items() if k != "protocol_sha256"})
    assert any("no significance test" in line for line in protocol["limitations"])
    assert any("not extension until a metric improves" in line
               for line in protocol["limitations"])


def test_read_rmse_rows_keeps_units_and_rejects_an_empty_table(tmp_path):
    module = _module()
    directory = _write_rmse(tmp_path / "e", [
        (6, "t2m", 5.5, "K"), (6, "mslp", 210.0, "Pa")])
    rows = module.read_rmse_rows(directory)
    assert [row["variable"] for row in rows] == ["t2m", "mslp"]
    assert {row["unit"] for row in rows} == {"K", "Pa"}
    assert all(row["n_initializations"] == 8 for row in rows)

    empty = tmp_path / "empty"
    empty.mkdir()
    with (empty / "rmse.csv").open("x", encoding="utf-8", newline="") as handle:
        handle.write("lead_hours,variable,rmse,unit,n_initializations\n")
    with pytest.raises(ValueError, match="empty RMSE table"):
        module.read_rmse_rows(empty)
    with pytest.raises(FileNotFoundError):
        module.read_rmse_rows(tmp_path / "missing")


def test_summarize_reports_seed_spread_per_variable_and_depth(tmp_path):
    module = _module()
    records = []
    for seed, value in ((41, 5.0), (42, 6.0), (43, 7.0)):
        directory = _write_rmse(tmp_path / f"g_{seed}", [(6, "t2m", value, "K")])
        records.append(_record("generic", seed, 3, directory))
    table = module.summarize(records)
    assert len(table) == 1
    row = table[0]
    assert row["arm"] == "generic" and row["depth"] == 3
    assert row["variable"] == "t2m" and row["unit"] == "K"
    assert row["seeds"] == 3
    assert row["rmse_mean"] == pytest.approx(6.0)
    assert row["rmse_sd"] == pytest.approx(1.0)
    assert sorted(row["values"]) == [5.0, 6.0, 7.0]


def test_compare_counts_wins_and_losses_per_variable(tmp_path):
    module = _module()
    records = []
    for arm, t2m, mslp in (("generic", 5.0, 200.0),
                           ("process_feedback", 4.0, 210.0)):
        directory = _write_rmse(tmp_path / arm, [
            (6, "t2m", t2m, "K"), (6, "mslp", mslp, "Pa")])
        records.append(_record(arm, 41, 3, directory))
    table = module.summarize(records)
    blocks = module.compare(table)
    assert len(blocks) == 1
    block = blocks[0]
    assert block["arm"] == "process_feedback"
    assert block["variables_improved"] == 1
    assert block["variables_worsened"] == 1
    # A mixed outcome must not be reported as a win.
    assert block["beats_baseline_everywhere"] is False
    deltas = {entry["variable"]: entry["delta"] for entry in block["deltas"]}
    assert deltas["t2m"] == pytest.approx(-1.0)
    assert deltas["mslp"] == pytest.approx(10.0)
    # Deltas must keep their units: -1 K and +10 Pa are not comparable.
    units = {entry["variable"]: entry["unit"] for entry in block["deltas"]}
    assert units == {"t2m": "K", "mslp": "Pa"}


def test_compare_flags_a_uniform_win_only_when_no_variable_worsens(tmp_path):
    module = _module()
    records = []
    for arm, values in (("generic", [5.0, 200.0]), ("process_feedback", [4.0, 190.0])):
        directory = _write_rmse(tmp_path / f"u_{arm}", [
            (6, "t2m", values[0], "K"), (6, "mslp", values[1], "Pa")])
        records.append(_record(arm, 41, 3, directory))
    block = module.compare(module.summarize(records))[0]
    assert block["variables_improved"] == 2
    assert block["variables_worsened"] == 0
    assert block["beats_baseline_everywhere"] is True


def test_compare_is_paired_per_depth_and_lead_time(tmp_path):
    """Different depths and leads must not be silently mixed together."""
    module = _module()
    records = []
    for arm, value in (("generic", 5.0), ("process_feedback", 4.0)):
        for depth in (0, 3):
            directory = _write_rmse(tmp_path / f"{arm}_K{depth}", [
                (6, "t2m", value + depth, "K")])
            records.append(_record(arm, 41, depth, directory))
    table = module.summarize(records)
    at_depth_3 = module.compare(table, depth=3)[0]
    at_depth_0 = module.compare(table, depth=0)[0]
    assert at_depth_3["depth"] == 3 and at_depth_0["depth"] == 0
    # The depth offset cancels inside each depth, so both show the same gap.
    for block in (at_depth_3, at_depth_0):
        assert block["deltas"][0]["delta"] == pytest.approx(-1.0)
        assert block["deltas"][0]["lead_hours"] == 6


def test_no_aggregate_score_is_computed():
    """An aggregate across K/Pa/(m s-1) would be meaningless and must not exist."""
    module = _module()
    assert not hasattr(module, "aggregate_score")
    assert not hasattr(module, "mean_delta_overall")
    source = MODULE.read_text(encoding="utf-8")
    assert "would average across physical units" in source


def test_comparison_refuses_a_receipt_that_is_not_a_real_acquisition(tmp_path):
    module = _module()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "status": "failed-no-fallback", "synthetic_fallback": False,
        "local_artifact": {"sha256": "0" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="real acquisition receipt"):
        module.run_comparison(tmp_path / "s.zarr", receipt, tmp_path / "out")

    receipt.write_text(json.dumps({
        "status": "downloaded-real-source", "synthetic_fallback": True,
        "local_artifact": {"sha256": "0" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic_fallback=False"):
        module.run_comparison(tmp_path / "s.zarr", receipt, tmp_path / "out")


def test_module_keeps_every_seed_and_variable_in_its_report():
    source = MODULE.read_text(encoding="utf-8")
    assert "all_seeds_retained" in source
    assert "Every variable and depth is kept" in source
    assert "no synthetic fallback" in source

def test_seed_sign_disagreement_blocks_a_win_claim(tmp_path):
    """Opposite per-seed deltas must be reported as unresolved, not as a win.

    This is the #6 situation: an arm can improve on more variables than it
    worsens while every one of those deltas flips sign between seeds, which
    means the effect is below seed noise.
    """
    module = _module()
    records = []
    # The arm is better on average (-0.25 K) but every per-seed delta flips sign.
    for seed, generic_value, arm_value in ((41, 5.0, 4.0), (42, 6.0, 6.5)):
        base_dir = _write_rmse(tmp_path / f"g{seed}", [(6, "t2m", generic_value, "K")])
        arm_dir = _write_rmse(tmp_path / f"a{seed}", [(6, "t2m", arm_value, "K")])
        records.append(_record("generic", seed, 3, base_dir))
        records.append(_record("process_feedback", seed, 3, arm_dir))
    block = module.compare(module.summarize(records))[0]
    assert block["variables_improved"] == 1
    assert block["variables_sign_consistent"] == 0
    assert block["gate_met"] is False
    paired = block["seed_paired"][0]
    assert paired["sign_consistent"] is False
    assert paired["direction"] == "unresolved"
    # The deltas genuinely have opposite signs, which is why this is unresolved.
    assert paired["seed_deltas"][0] < 0 < paired["seed_deltas"][1]
    assert block["deltas"][0]["delta"] < 0, "mean delta must look like a win"


def test_gate_is_met_only_when_every_consistent_variable_improves(tmp_path):
    module = _module()
    records = []
    for seed, base, arm in ((41, 5.0, 4.0), (42, 6.0, 5.0)):
        records.append(_record("generic", seed, 3,
                               _write_rmse(tmp_path / f"gg{seed}", [(6, "t2m", base, "K")])))
        records.append(_record("process_feedback", seed, 3,
                               _write_rmse(tmp_path / f"aa{seed}", [(6, "t2m", arm, "K")])))
    block = module.compare(module.summarize(records))[0]
    assert block["variables_sign_consistent"] == 1
    assert block["established_improved"] == 1
    assert block["established_worsened"] == 0
    assert block["gate_met"] is True
