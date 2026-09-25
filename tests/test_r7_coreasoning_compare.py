"""Offline tests for the seed-identity-keyed co-reasoning comparison (#6/#60).

These cover the reporting logic that decides what the comparison may claim:
full-key seed pairing, fail-closed refusal of incomplete or inconsistent
inputs, the separate naming of the seed-mean and cross-case pooled statistics,
the frozen-criteria gate, and the historical re-aggregation audit. Training is
not exercised here.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import statistics
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
    """Write a bare rmse.csv (no provenance): case identity stays count-only."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "rmse.csv"
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lead_hours", "variable", "rmse", "unit", "n_initializations"])
        for lead, variable, rmse, unit in rows:
            writer.writerow([lead, variable, rmse, unit, 8])
    return directory


def _write_evaluation(directory: Path, rows, *,
                      inits=("2019-01-01T00:00:00", "2019-01-02T00:00:00"),
                      units_override=None) -> Path:
    """Write rmse.csv plus provenance.json whose per-case MSE squares to the RMSE.

    Per-case MSE entries are exactly representable for power-of-two RMSEs, so
    sqrt(mean(case_mse)) reproduces the rmse.csv value bit-exactly.
    """
    directory.mkdir(parents=True, exist_ok=True)
    n = len(inits)
    with (directory / "rmse.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lead_hours", "variable", "rmse", "unit", "n_initializations"])
        for lead, variable, rmse, unit in rows:
            writer.writerow([lead, variable, rmse, unit, n])
    leads = sorted({float(row[0]) for row in rows})
    channels = sorted({row[1] for row in rows})
    rmse_by_key = {(float(row[0]), row[1]): float(row[2]) for row in rows}
    units_by_channel = units_override or {row[1]: row[3] for row in rows}
    mse = [[rmse_by_key[(lead, channel)] ** 2 for channel in channels]
           for lead in leads]
    provenance = {
        "n_evaluated": n,
        "initializations": [{"init_time": init, "valid_times": [init], "mse": mse}
                            for init in inits],
        "channels": channels,
        "units": [units_by_channel[channel] for channel in channels],
        "lead_hours": leads,
        "split": "val",
        "evaluation_manifest_sha256": "0" * 64,
    }
    with (directory / "provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle)
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


def test_read_rmse_rows_fails_closed_on_corrupt_values(tmp_path):
    """NaN, Inf, negative RMSE and duplicate table entries must all be errors."""
    module = _module()
    cases = [
        ("nan", [(6, "t2m", float("nan"), "K")], "non-finite or negative RMSE"),
        ("inf", [(6, "t2m", float("inf"), "K")], "non-finite or negative RMSE"),
        ("negative", [(6, "t2m", -1.0, "K")], "non-finite or negative RMSE"),
        ("duplicate", [(6, "t2m", 1.0, "K"), (6.0, "t2m", 2.0, "K")],
         "duplicate table entry"),
        ("empty_variable", [(6, "", 1.0, "K")], "empty variable or unit"),
        ("empty_unit", [(6, "t2m", 1.0, "")], "empty variable or unit"),
    ]
    for name, rows, pattern in cases:
        with pytest.raises(ValueError, match=pattern):
            module.read_rmse_rows(_write_rmse(tmp_path / name, rows))


def test_read_rmse_rows_rejects_malformed_fields(tmp_path):
    module = _module()
    directory = tmp_path / "bad"
    directory.mkdir()
    with (directory / "rmse.csv").open("x", encoding="utf-8", newline="") as handle:
        handle.write("lead_hours,variable,rmse,unit,n_initializations\n6,t2m,not-a-float,K,8\n")
    with pytest.raises(ValueError, match="malformed RMSE row"):
        module.read_rmse_rows(directory)


def test_summarize_keeps_explicit_seed_ids_and_names_the_seed_mean(tmp_path):
    module = _module()
    records = []
    for seed, value in ((41, 4.0), (42, 6.0), (43, 8.0)):
        directory = _write_evaluation(tmp_path / f"g_{seed}", [(6, "t2m", value, "K")])
        records.append(_record("generic", seed, 3, directory))
    table = module.summarize(records)
    assert len(table) == 1
    row = table[0]
    assert row["arm"] == "generic" and row["depth"] == 3
    assert row["variable"] == "t2m" and row["unit"] == "K"
    assert row["seeds"] == [41, 42, 43]
    assert row["seed_rmse"] == {"41": 4.0, "42": 6.0, "43": 8.0}
    assert row["rmse_seed_mean"] == pytest.approx(6.0)
    assert row["rmse_seed_sd"] == pytest.approx(2.0)
    # The cross-case pooled statistic is computed separately from the per-case
    # MSE and must not be confused with the seed mean: pooling every case of
    # every seed weights the high-RMSE seeds by their squared errors, so it is
    # genuinely larger here.
    all_squared = [value ** 2 for value in (4.0, 4.0, 6.0, 6.0, 8.0, 8.0)]
    assert row["rmse_pooled_cases"] == pytest.approx(
        math.sqrt(statistics.fmean(all_squared)))
    assert row["rmse_pooled_cases"] > row["rmse_seed_mean"]
    assert row["case_identity"] == "exact"
    assert row["per_seed_n_initializations"] == {"41": 2, "42": 2, "43": 2}


def test_summarize_marks_count_only_identity_when_provenance_is_absent(tmp_path):
    module = _module()
    records = [_record("generic", 41, 3,
                       _write_rmse(tmp_path / "bare", [(6, "t2m", 5.0, "K")]))]
    row = module.summarize(records)[0]
    assert row["case_identity"] == "count-only"
    assert row["rmse_pooled_cases"] is None
    assert set(row["case_set_digests"].values()) == {None}


def test_summarize_rejects_a_duplicate_seed_for_one_key(tmp_path):
    module = _module()
    directory = _write_rmse(tmp_path / "g", [(6, "t2m", 5.0, "K")])
    records = [_record("generic", 41, 3, directory), _record("generic", 41, 3, directory)]
    with pytest.raises(ValueError, match="duplicate seed 41"):
        module.summarize(records)


def test_summarize_rejects_partial_record_identity(tmp_path):
    module = _module()
    first = dict(_record("generic", 41, 3,
                         _write_rmse(tmp_path / "a", [(6, "t2m", 5.0, "K")])),
                 identity={"dataset_identity": "x", "model_code_sha256": "y",
                           "optimizer_updates": 200})
    second = dict(_record("generic", 42, 3,
                          _write_rmse(tmp_path / "b", [(6, "t2m", 6.0, "K")])))
    with pytest.raises(ValueError, match="some records but not all"):
        module.summarize([first, second])
    second["identity"] = {"dataset_identity": "other", "model_code_sha256": "y",
                          "optimizer_updates": 200}
    with pytest.raises(ValueError, match="one dataset/model-code/updates identity"):
        module.summarize([first, second])


def test_seed_deltas_are_paired_by_seed_id_not_position(tmp_path):
    """The #60 isolation repro: identical means, reordered records.

    baseline seeds 41/42/43 carry [1, 10, 100]; process carries the same seeds
    with [0.9, 9.9, 99.9]. Pairing by seed id gives every seed a -0.1 delta and
    direction=improved; the earlier positional zip turned a mere reordering of
    the input records into direction=unresolved with the same means.
    """
    module = _module()
    base_values = {41: 1.0, 42: 10.0, 43: 100.0}
    arm_values = {41: 0.9, 42: 9.9, 43: 99.9}

    def build():
        records = []
        for seed, value in base_values.items():
            records.append(_record("generic", seed, 3, _write_rmse(
                tmp_path / f"p_g{seed}", [(6, "t2m", value, "K")])))
        for seed, value in arm_values.items():
            records.append(_record("process_feedback", seed, 3, _write_rmse(
                tmp_path / f"p_a{seed}", [(6, "t2m", value, "K")])))
        return records

    records = build()
    forward = module.compare(module.summarize(records))
    # Reordering the same records must not change anything.
    shuffled = module.compare(module.summarize(list(reversed(records))))
    assert forward == shuffled, "reordering the input records must not change the output"
    paired = forward[0]["seed_paired"][0]
    assert {entry["seed"]: pytest.approx(entry["delta"]) for entry in paired["seed_deltas"]} \
        == {41: -0.1, 42: -0.1, 43: -0.1}
    assert paired["direction"] == "improved" and paired["sign_consistent"] is True


def test_compare_counts_wins_and_losses_per_variable(tmp_path):
    module = _module()
    records = []
    for arm, t2m, mslp in (("generic", 4.0, 128.0),
                           ("process_feedback", 2.0, 256.0)):
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
    assert deltas["t2m"] == pytest.approx(-2.0)
    assert deltas["mslp"] == pytest.approx(128.0)
    # Deltas must keep their units: -2 K and +128 Pa are not comparable.
    units = {entry["variable"]: entry["unit"] for entry in block["deltas"]}
    assert units == {"t2m": "K", "mslp": "Pa"}


def test_compare_flags_a_uniform_win_only_when_no_variable_worsens(tmp_path):
    module = _module()
    records = []
    for arm, values in (("generic", [4.0, 256.0]), ("process_feedback", [2.0, 128.0])):
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


def test_compare_fails_closed_on_a_missing_baseline_row(tmp_path):
    module = _module()
    records = []
    records.append(_record("generic", 41, 3,
                           _write_rmse(tmp_path / "g", [(6, "t2m", 5.0, "K")])))
    records.append(_record("process_feedback", 41, 3, _write_rmse(
        tmp_path / "a", [(6, "t2m", 4.0, "K"), (6, "mslp", 200.0, "Pa")])))
    with pytest.raises(ValueError, match="missing baseline row"):
        module.compare(module.summarize(records))
    blocked = module.compare(module.summarize(records), on_incomplete="blocked")
    assert blocked[0]["blocked_keys"][0]["variable"] == "mslp"
    assert blocked[0]["beats_baseline_everywhere"] is False


def test_compare_fails_closed_on_differing_seed_sets(tmp_path):
    module = _module()
    records = []
    for seed in (41, 42, 43):
        records.append(_record("generic", seed, 3, _write_rmse(
            tmp_path / f"g{seed}", [(6, "t2m", 5.0, "K")])))
    for seed in (41, 42):
        records.append(_record("process_feedback", seed, 3, _write_rmse(
            tmp_path / f"a{seed}", [(6, "t2m", 4.0, "K")])))
    with pytest.raises(ValueError, match="seed sets differ"):
        module.compare(module.summarize(records))


def test_compare_fails_closed_on_a_unit_conflict(tmp_path):
    """The same variable must keep one unit across arms at the same lead."""
    module = _module()
    records = []
    records.append(_record("generic", 41, 3,
                           _write_rmse(tmp_path / "g", [(6, "t2m", 5.0, "K")])))
    records.append(_record("process_feedback", 41, 3,
                           _write_rmse(tmp_path / "a", [(6, "t2m", 278.15, "degC")])))
    with pytest.raises(ValueError, match="unit mismatch"):
        module.compare(module.summarize(records))


def test_compare_fails_closed_when_equal_counts_hide_different_cases(tmp_path):
    """Equal n_initializations never implies the same cases (the #58 lesson)."""
    module = _module()
    shared = ("2019-01-01T00:00:00", "2019-01-02T00:00:00")
    other = ("2019-01-01T00:00:00", "2019-03-04T00:00:00")
    records = [
        _record("generic", 41, 3, _write_evaluation(
            tmp_path / "g", [(6, "t2m", 4.0, "K")], inits=shared)),
        _record("process_feedback", 41, 3, _write_evaluation(
            tmp_path / "a", [(6, "t2m", 2.0, "K")], inits=other)),
    ]
    with pytest.raises(ValueError, match="initialization sets differ"):
        module.compare(module.summarize(records))


def test_compare_binds_exact_case_identity_when_provenance_matches(tmp_path):
    module = _module()
    shared = ("2019-01-01T00:00:00", "2019-01-02T00:00:00")
    records = [
        _record("generic", 41, 3, _write_evaluation(
            tmp_path / "g", [(6, "t2m", 4.0, "K")], inits=shared)),
        _record("process_feedback", 41, 3, _write_evaluation(
            tmp_path / "a", [(6, "t2m", 2.0, "K")], inits=shared)),
    ]
    table = module.summarize(records)
    assert table[0]["case_identity"] == "exact"
    block = module.compare(table)[0]
    assert block["case_identity"] == "exact"
    assert block["seed_paired"][0]["seed_deltas"][0]["seed"] == 41


def test_summarize_fails_closed_when_rmse_disagrees_with_per_case_mse(tmp_path):
    """A tampered rmse.csv must not silently survive the provenance cross-check."""
    module = _module()
    directory = _write_evaluation(tmp_path / "tampered", [(6, "t2m", 2.0, "K")])
    provenance = json.loads((directory / "provenance.json").read_text())
    provenance["initializations"][0]["mse"] = [[9.0]]
    (directory / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="does not match the pooled per-case"):
        module.summarize([_record("generic", 41, 3, directory)])


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


def test_seed_sign_disagreement_is_unresolved_and_gate_is_separate(tmp_path):
    """Opposite per-seed deltas must be reported as unresolved, not as a win.

    This is the #6 situation: an arm can improve on more variables than it
    worsens while every one of those deltas flips sign between seeds, which
    means the effect is below seed noise. The research gate is no longer a
    by-product of the comparison blocks: it reads pre-frozen criteria only.
    """
    module = _module()
    records = []
    # The arm is better on average (-0.25 K) but every per-seed delta flips sign.
    for seed, generic_value, arm_value in ((41, 4.0, 2.0), (42, 6.0, 6.5)):
        base_dir = _write_rmse(tmp_path / f"g{seed}", [(6, "t2m", generic_value, "K")])
        arm_dir = _write_rmse(tmp_path / f"a{seed}", [(6, "t2m", arm_value, "K")])
        records.append(_record("generic", seed, 3, base_dir))
        records.append(_record("process_feedback", seed, 3, arm_dir))
    block = module.compare(module.summarize(records))[0]
    assert block["variables_improved"] == 1
    assert block["variables_sign_consistent"] == 0
    assert block["variables_unresolved"] == 1
    assert "gate_met" not in block, "the gate must not be a descriptive by-product"
    paired = block["seed_paired"][0]
    assert paired["sign_consistent"] is False
    assert paired["direction"] == "unresolved"
    assert {entry["seed"] for entry in paired["seed_deltas"]} == {41, 42}
    # The deltas genuinely have opposite signs, which is why this is unresolved.
    assert paired["seed_deltas"][0]["delta"] < 0 < paired["seed_deltas"][1]["delta"]
    assert block["deltas"][0]["delta"] < 0, "mean delta must look like a win"


def test_gate_reads_frozen_criteria_only(tmp_path):
    """One improved variable must not carry a gate while others stay unresolved.

    t2m improves consistently at both seeds; mslp improves on average but its
    per-seed deltas flip sign (unresolved); t850 worsens consistently. The
    frozen criteria decide which of these the gate may claim.
    """
    module = _module()
    # ((generic seed values), (arm seed values)) per variable, seed order (41, 42)
    plan = {"t2m": ((4.0, 6.0), (2.0, 4.0)),            # improved, consistent
            "mslp": ((128.0, 256.0), (100.0, 280.0)),   # mean -2 but sign flips
            "t850": ((4.0, 6.0), (6.0, 8.0))}           # worsened, consistent
    records = []
    for arm, offset in (("generic", 0), ("process_feedback", 1)):
        for seed_index, seed in enumerate((41, 42)):
            rows = [(6, variable, values[offset][seed_index],
                     "Pa" if variable == "mslp" else "K")
                    for variable, values in plan.items()]
            records.append(_record(arm, seed, 3,
                                   _write_rmse(tmp_path / f"{arm}_{seed}", rows)))
    comparison = module.compare(module.summarize(records))
    block = comparison[0]
    assert block["variables_improved"] == 2    # t2m consistent; mslp by mean only
    assert block["variables_worsened"] == 1    # t850 by mean delta
    assert block["variables_unresolved"] == 1  # mslp (its mean delta is negative)
    directions = {entry["variable"]: entry["direction"] for entry in block["seed_paired"]}
    assert directions == {"t2m": "improved", "mslp": "unresolved", "t850": "worsened"}

    criteria = {"arm": "process_feedback", "required_direction": "improved",
                "allow_unresolved_required": 0,
                "required_variables": [
                    {"variable": "t2m", "unit": "K", "lead_hours": 6.0}]}
    assert module.evaluate_gate(comparison, criteria)["gate_met"] is True

    # The same improved variable cannot carry a gate over an unresolved peer
    # when the frozen allowance is zero.
    strict = dict(criteria, required_variables=[
        {"variable": "t2m", "unit": "K", "lead_hours": 6.0},
        {"variable": "mslp", "unit": "Pa", "lead_hours": 6.0}])
    gate = module.evaluate_gate(comparison, strict)
    assert gate["gate_met"] is False
    assert any("unresolved beyond the frozen allowance" in line
               for line in gate["failures"])

    # An explicitly frozen allowance for one unresolved required variable is
    # the only way an unresolved peer can coexist with a gate pass.
    lenient = dict(strict, allow_unresolved_required=1)
    gate = module.evaluate_gate(comparison, lenient)
    assert gate["gate_met"] is True
    assert gate["warnings"] and "unresolved within the frozen allowance" in gate["warnings"][0]

    # A worsened required variable always fails, as does an absent one.
    worsened = dict(criteria, required_variables=[
        {"variable": "t850", "unit": "K", "lead_hours": 6.0}])
    gate = module.evaluate_gate(comparison, worsened)
    assert gate["gate_met"] is False
    assert "worsened but improved required" in gate["failures"][0]
    absent = dict(criteria, required_variables=[
        {"variable": "z500", "unit": "m**2 s**-2", "lead_hours": 6.0}])
    gate = module.evaluate_gate(comparison, absent)
    assert gate["gate_met"] is False and "absent from comparison" in gate["failures"][0]

    # Criteria must be complete, and the gate names its own limits.
    with pytest.raises(ValueError, match="missing allow_unresolved_required"):
        module.evaluate_gate(comparison, {"arm": "process_feedback",
                                          "required_direction": "improved",
                                          "required_variables": []})
    assert "no significance test" in module.evaluate_gate(comparison, criteria)["note"]


def test_reaggregate_historical_writes_audit_and_leaves_originals_untouched(tmp_path):
    module = _module()
    hist = tmp_path / "hist"
    plan = (("generic", {41: 4.0, 42: 4.0}),
            ("process_feedback", {41: 2.0, 42: 2.5}))
    records = []
    for arm, values in plan:
        for seed, value in values.items():
            _write_rmse(hist / "evaluation" / f"{arm}_{seed}_K3",
                        [(6, "t2m", value, "K")])
            records.append({"arm": arm, "seed": seed, "depth": 3,
                            "evaluation_dir": f"evaluation/{arm}_{seed}_K3"})
    old_summary = [
        {"arm": arm, "depth": 3, "variable": "t2m", "unit": "K", "lead_hours": 6.0,
         "seeds": len(values), "rmse_mean": statistics.fmean(values.values()),
         "values": [values[seed] for seed in sorted(values)]}
        for arm, values in plan]
    original = {"format": "r7-coreasoning-fair-budget-result-v1",
                "records": records, "summary": old_summary, "comparison": []}
    result_path = hist / "coreasoning_result.json"
    result_path.write_text(json.dumps(original), encoding="utf-8")
    before = result_path.read_bytes()

    audit = module.reaggregate_historical(result_path, hist / "audit.json", root=hist)
    # The original artifact is byte-identical after the audit.
    assert result_path.read_bytes() == before
    assert audit["records_total"] == 4
    assert audit["records_reaggregated"] == 4 and audit["blocked_records"] == []
    # The old per-order means are reproduced exactly under seed-keyed aggregation.
    diffs = {entry["arm"]: entry["mean_difference"] for entry in audit["summary_diff"]}
    assert diffs["generic"] == 0.0
    assert diffs["process_feedback"] == pytest.approx(0.0, abs=1e-12)
    # Seed deltas now carry explicit seed ids instead of list positions.
    paired = audit["comparison"][0]["seed_paired"][0]
    assert {entry["seed"] for entry in paired["seed_deltas"]} == {41, 42}
    assert paired["direction"] == "improved"
    # The audit file is write-once.
    with pytest.raises(FileExistsError):
        module.reaggregate_historical(result_path, hist / "audit.json", root=hist)


def test_reaggregate_marks_unreadable_artifacts_blocked_without_fabrication(tmp_path):
    module = _module()
    hist = tmp_path / "hist"
    plan = (("generic", {41: 4.0, 42: 4.0, 43: 4.0}),
            ("process_feedback", {41: 2.0, 42: 2.5, 43: 2.25}))
    records = []
    for arm, values in plan:
        for seed, value in values.items():
            _write_rmse(hist / "evaluation" / f"{arm}_{seed}_K3",
                        [(6, "t2m", value, "K")])
            records.append({"arm": arm, "seed": seed, "depth": 3,
                            "evaluation_dir": f"evaluation/{arm}_{seed}_K3"})
    old_summary = [
        {"arm": arm, "depth": 3, "variable": "t2m", "unit": "K", "lead_hours": 6.0,
         "seeds": len(values), "rmse_mean": statistics.fmean(values.values()),
         "values": [values[seed] for seed in sorted(values)]}
        for arm, values in plan]
    original = {"format": "r7-coreasoning-fair-budget-result-v1",
                "records": records, "summary": old_summary, "comparison": []}
    result_path = hist / "coreasoning_result.json"
    result_path.write_text(json.dumps(original), encoding="utf-8")
    # One historical artifact is unreadable: its record must be blocked, and
    # no seed id may be invented to keep the pairing complete.
    shutil.rmtree(hist / "evaluation" / "process_feedback_41_K3")
    audit = module.reaggregate_historical(result_path, hist / "audit.json", root=hist)
    assert len(audit["blocked_records"]) == 1
    blocked = audit["blocked_records"][0]
    assert blocked["arm"] == "process_feedback" and blocked["seed"] == 41
    assert "rmse.csv" in blocked["reason"]
    assert audit["records_reaggregated"] == 5
    entry = next(e for e in audit["summary_diff"] if e["arm"] == "process_feedback")
    assert entry["seeds_new"] == [42, 43]
    block = audit["comparison"][0]
    assert block["blocked_keys"], "the incomplete seed set must be reported"
    assert "seed sets differ" in block["blocked_keys"][0]["reason"]
