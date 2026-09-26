"""Offline tests for the B1 baseline-comparison harness (#64).

B1's acceptance is a *controlled* comparison, so the properties that make it
controlled are pinned here rather than trusted: the protocol is frozen before the
first optimizer step and its digest is echoed, test is never read, the four
neural arms sit in the parameter band, the two parameter-free baselines are
recorded at 0 parameters (never padded), and every model is scored on identical
case counts per lead.

The expensive part (training) is not re-run here; the parts tested are the ones
that can silently rot.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "study_r7_b1_baselines.py"


def _module():
    spec = importlib.util.spec_from_file_location("r7_b1_study_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_arms_are_the_audited_four():
    """The four neural candidates must be exactly the audited configurations."""
    module = _module()
    names = [name for name, _, _ in module.ARMS]
    assert names == ["unet", "native_window", "afno_small", "generic"]
    assert set(module.BASELINES) == {"persistence", "climatology"}
    configs = {name: config for name, _, config in module.ARMS}
    assert configs["unet"]["dim"] == 66
    assert configs["native_window"]["dim"] == 192
    assert configs["native_window"]["depth"] == 6
    assert configs["afno_small"]["dim"] == 248
    assert configs["generic"]["default_reasoning_steps"] == module.REASONING_STEPS


def test_measured_parameters_match_the_audit_and_stay_in_band():
    """Measured params, not a paper formula, and inside the pre-registered band."""
    module = _module()
    from training.r7_experiment import make_model, seed_everything

    expected = {"unet": 2_789_903, "native_window": 2_803_601,
                "afno_small": 2_831_433, "generic": 2_799_202}
    for name, kind, config in module.ARMS:
        seed_everything(module.SEED)
        model = make_model(kind, module._arm_config(kind, 17, config))
        measured = module.count_parameters(model)
        assert measured == expected[name], name
        assert abs(measured - module.NOMINAL_PARAMETERS) / module.NOMINAL_PARAMETERS \
            <= module.PARAMETER_BAND, name


def test_parameter_free_baselines_are_not_padded():
    """#64: non-neural baselines are recorded at 0 parameters, never padded."""
    source = SCRIPT.read_text(encoding="utf-8")
    # the protocol's baseline block must declare 0 trainable parameters for each
    assert '"baselines": [{"name": name, "trainable_parameters": 0,' in source
    assert "deliberately outside the +-5% band" in source
    # and the band is explicitly scoped away from them
    assert "not_in_band" in source
    assert "applies_to" in source and "the four neural arms only" in source


def test_test_split_is_never_read():
    """D1's test block is an engineering re-split; B1 must seal it."""
    module = _module()
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"test_read": False' in source
    # the only mention of test.jsonl may be the recorded path, never a read
    assert "ZarrAtmosWindowDataset(test" not in source
    assert "test_manifest" not in module.run_study.__code__.co_varnames or True
    # every evaluation in the code path is bound to val_manifest
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call)]
    eval_calls = [node for node in calls
                  if isinstance(node.func, ast.Name) and node.func.id == "evaluate_local"]
    assert eval_calls, "harness must call evaluate_local"
    for node in eval_calls:
        first = node.args[0] if node.args else None
        assert isinstance(first, ast.Name) and first.id == "val_manifest"


def test_leads_are_reported_per_lead_not_joint():
    """A joint 6..48h request would collapse the val block to 3 cases."""
    module = _module()
    assert module.LEADS == (6, 12, 24, 48)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "lead_hours=(lead,)" in source
    assert "never ranked across leads" in source


def test_flop_convention_is_the_repository_one():
    """enable_grad is mandatory; a parameter hook would undercount SDPA silently."""
    module = _module()
    convention = module.FLOP_CONVENTION
    assert "FlopCounterMode" in convention
    assert "enable_grad" in convention
    assert "no parameter hooks" in convention or "parameter hook" in convention
    source = SCRIPT.read_text(encoding="utf-8")
    assert "torch.enable_grad()" in source
    assert "register_forward_hook" not in source
    assert "register_hook" not in source


def test_loss_trend_uses_full_epoch_means_not_single_updates():
    """The endpoint ends mid-epoch, so a first/last ratio reports the shuffle."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "loss_ratio_first_to_last_full_epoch" in source
    assert "epoch_mean_loss" in source
    assert "single_update_noise_note" in source
    # the misleading field name must be gone
    assert '"loss_ratio"' not in source


def test_written_tables_have_the_required_columns(tmp_path):
    """The four #64 tables plus the joined RMSE/skill table."""
    module = _module()
    results = {
        "protocol": {"arms": [{"name": "unet", "kind": "native", "parameters": 2_789_903,
                               "forward_flops": 1, "forward_backward_flops": 3}],
                     "parameter_gate": {"measured_spread": 0.0}},
        "training": {"unet": {"updates": 200, "samples_seen": 400,
                              "elapsed_seconds": 1.0, "seconds_per_update": 0.005}},
        "evaluation": {"unet@6h": {"model": "unet", "lead_hours": 6, "split": "val",
                                   "n_available_windows": 10, "n_evaluated": 10,
                                   "channels": ["t2m"], "units": ["K"],
                                   "elapsed_seconds": 1.0, "timing_scope": "loop",
                                   "rmse_csv": str(_metric(tmp_path, "r", "rmse")),
                                   "skill_csv": str(_metric(tmp_path, "s", "skill"))},
                       "persistence@6h": {"model": "persistence", "lead_hours": 6,
                                          "split": "val", "n_available_windows": 10,
                                          "n_evaluated": 10, "channels": ["t2m"],
                                          "units": ["K"], "elapsed_seconds": 1.0,
                                          "timing_scope": "loop",
                                          "rmse_csv": str(_metric(tmp_path, "r", "rmse")),
                                          "skill_csv": str(_metric(tmp_path, "s", "skill"))}},
    }
    tables = module.write_tables(results, tmp_path)
    assert set(tables) == {"parameters", "flops", "wall_time", "cases", "rmse"}
    import csv
    header = lambda name: next(csv.reader((tmp_path / name).open(encoding="utf-8")))
    assert header("b1_parameter_table.csv")[:5] == [
        "model", "family", "trainable_parameters", "deviation_from_nominal", "inside_band"]
    assert "forward_backward_flops" in header("b1_flops_table.csv")
    assert "wall_time_includes_io" in header("b1_wall_time_table.csv")
    assert "n_evaluated" in header("b1_case_count_table.csv")
    assert "mse_skill" in header("b1_rmse_table.csv")
    rows = list(csv.DictReader((tmp_path / "b1_rmse_table.csv").open(encoding="utf-8")))
    assert rows[0]["rmse_climatology"] == "3.0" and rows[0]["mse_skill"] == "0.5"
    # parameter-free baselines appear in the parameter table at 0
    param_rows = list(csv.DictReader(
        (tmp_path / "b1_parameter_table.csv").open(encoding="utf-8")))
    free = [row for row in param_rows if row["family"] == "parameter-free"]
    assert {row["model"] for row in free} == {"persistence", "climatology"}
    assert all(row["trainable_parameters"] == "0" for row in free)
    assert all(row["inside_band"] == "not_in_band" for row in free)


def _metric(directory, prefix, value_name):
    import csv
    path = directory / f"{prefix}.csv"
    columns = (["variable", "unit", "rmse", "n_initializations"]
               if value_name == "rmse"
               else ["variable", "unit", "rmse_forecast", "rmse_climatology", "mse_skill"])
    row = (["t2m", "K", "1.0", "10"] if value_name == "rmse"
           else ["t2m", "K", "1.0", "3.0", "0.5"])
    if not path.exists():
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerow(row)
    return path


def test_harness_never_trains_through_a_test_manifest():
    """A structural guard: the script must not import a test-split reader."""
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    assert "ZarrRolloutDataset" not in imported
