"""Synthetic tensor checks only; fixtures are not weather truth."""
from __future__ import annotations

import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

from test_r7_k3_reference_support import archived_leaves, archive_source, no_actual_runtime, synthetic_sample
import pytest
import torch
from r7_k3_reference_cases import pool_cases, region_statistics, score_case, validate_evaluation, validate_stat
from r7_k3_reference_evaluate import evaluate_cases, verify_saved
from r7_k3_reference_support import MODEL_SHA, REGIONS


class SyntheticModel(torch.nn.Module):
    spacetime_inputs = True
    def __init__(self):
        super().__init__()
        self.inputs = []
    def forward(self, batch, **kwargs):
        self.inputs.append({k: v.clone() for k, v in batch.items()})
        assert kwargs == {"reasoning_steps": 3}
        assert set(batch) == {"coarse_history", "lead_time_hours", "latitude", "longitude", "init_utc_hour", "init_day_of_year"}
        return SimpleNamespace(forecast=batch["coarse_history"][:, -1] + 1, reasoning_steps=3)


@pytest.mark.parametrize("lead,calls", [(6, 1), (12, 2), (24, 4), (48, 8), (72, 12)])
def test_original_rollout_no_future_truth_and_old_accumulated_lead(archived_leaves, synthetic_sample, lead, calls):
    rollout = archived_leaves["model.r7_rollout"]
    model = SyntheticModel().eval()
    initial = rollout.rollout_model_input(synthetic_sample, device=torch.device("cpu"))
    out = rollout.autoregressive_rollout(model, initial, lead_hours=(lead,), inference_kwargs={"reasoning_steps": 3})
    assert out.model_calls == len(model.inputs) == calls
    assert out.cumulative_reasoning_steps.tolist() == [[3 * calls]]
    assert torch.equal(out.forecasts[0, 0], synthetic_sample["coarse_history"][-1] + calls)
    assert [float(s["lead_time_hours"][0]) for s in model.inputs] == [6 * (i + 1) for i in range(calls)]
    assert all(float(s["init_day_of_year"]) == 48 and float(s["init_utc_hour"]) == 6 for s in model.inputs)
    assert all("init_calendar_year" not in s and "init_year" not in s for s in model.inputs)
    for index in range(1, calls):
        assert torch.equal(model.inputs[index]["coarse_history"][:, -1], model.inputs[index - 1]["coarse_history"][:, -1] + 1)


def test_synthetic_target_poison_does_not_change_original_rollout(archived_leaves, synthetic_sample):
    rollout = archived_leaves["model.r7_rollout"]
    predictions = []
    for poisoned in (False, True):
        sample = dict(synthetic_sample)
        sample["rollout_targets"] = sample["rollout_targets"] + (999 if poisoned else 0)
        model = SyntheticModel().eval()
        batch = rollout.rollout_model_input(sample)
        predictions.append(rollout.autoregressive_rollout(model, batch, lead_hours=(12,), inference_kwargs={"reasoning_steps": 3}).forecasts)
    assert torch.equal(*predictions)


def test_unchanged_archived_36525_phase_and_input_declaration(archived_leaves, archive_source):
    source = archive_source / "model/spacetime_conditioning_r7.py"
    tree = ast.parse(source.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("DAYS_PER_YEAR", "HOURS_PER_DAY", "SPACETIME_INPUT_FIELDS") for t in n.targets)]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "phase_features"]
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["DAYS_PER_YEAR"] == 365.25
    assert namespace["SPACETIME_INPUT_FIELDS"] == ("latitude", "longitude", "init_utc_hour", "init_day_of_year")
    actual = namespace["phase_features"](torch.tensor([6.]), torch.tensor([48.]), torch.tensor([12.]))
    annual, diurnal = math.tau * (47 + 18 / 24) / 365.25, math.tau * 18 / 24
    torch.testing.assert_close(actual, torch.tensor([[math.sin(annual), math.cos(annual), math.sin(diurnal), math.cos(diurnal)]]))


def fields():
    names = [f"synthetic_{i}" for i in range(17)]
    units = ["synthetic_unit"] * 17
    p = torch.linspace(-2, 2, 17 * 7 * 8).reshape(1, 1, 17, 7, 8)
    t, c = p * .3 + .1, torch.zeros_like(p) + .02
    lat, std = torch.linspace(20, 50, 7), torch.linspace(.5, 2, 17)
    return names, units, p, t, c, lat, std


def test_all17_full_region_matches_original_accumulators(archived_leaves):
    names, units, p, t, c, lat, std = fields()
    stats = region_statistics(p, t, c, lat, lead=12, names=names, units=units, std=std)
    rmse = archived_leaves["training.r7_rollout_metrics"].RolloutRMSEAccumulator((12,), names, training_std=std, units=units)
    acc = archived_leaves["training.r7_acc"].RolloutACCAccumulator((12,), names)
    skill = archived_leaves["training.r7_climatology_skill"].RolloutClimatologySkillAccumulator((12,), names, training_std=std, units=units)
    rmse.update(p, t, lat)
    acc.update(p, t, c, lat)
    skill.update(p, t, c, lat)
    assert tuple(stats) == REGIONS
    assert len(stats["full"]["variables"]) == 17
    torch.testing.assert_close(torch.tensor([s["mse"] for s in stats["full"]["variables"]], dtype=torch.float64), rmse.sum_squared_error[0])
    torch.testing.assert_close(torch.tensor([s["normalized_acc"]["dot"] for s in stats["full"]["variables"]], dtype=torch.float64), acc.dot[0])
    torch.testing.assert_close(torch.tensor([s["climatology_mse"] for s in stats["full"]["variables"]], dtype=torch.float64), skill.climatology_squared_error[0])


def test_complementary_regions_same_predictions_match_old_boundary(archived_leaves):
    names, units, p, t, c, lat, std = fields()
    stats = region_statistics(p, t, c, lat, lead=12, names=names, units=units, std=std)
    boundary = archived_leaves["training.r7_boundary_metrics"].BoundaryRMSEAccumulator((12,), names, margins=(2,), training_std=std, units=units)
    boundary.update(p, t, lat)
    for index, region in enumerate(REGIONS):
        torch.testing.assert_close(torch.tensor([s["mse"] for s in stats[region]["variables"]], dtype=torch.float64), boundary.sum_squared_error[index, 0])
    assert stats["interior_2"]["n_grid_points"] + stats["edge_2"]["n_grid_points"] == stats["full"]["n_grid_points"]
    assert math.isclose(stats["interior_2"]["full_area_fraction"] + stats["edge_2"]["full_area_fraction"], 1)
    for j in range(17):
        recovered = sum(stats[r]["full_area_fraction"] * stats[r]["variables"][j]["mse"] for r in REGIONS[1:])
        assert math.isclose(recovered, stats["full"]["variables"][j]["mse"], rel_tol=1e-12)


def test_pooled_acc_is_not_mean_case_acc_and_rmse_is_not_mean_case_rmse(archived_leaves, synthetic_sample):
    names, units, _, _, c, lat, std = fields()
    cases = []
    for index, factor in enumerate((2., -0.2)):
        sample = dict(synthetic_sample, init_time=f"synthetic_case_{index}", rollout_targets=torch.ones(1, 17, 7, 8), latitude=lat)
        prediction = torch.ones(1, 1, 17, 7, 8) * factor
        cases.append(score_case(sample, prediction, torch.zeros_like(prediction), lead=12, names=names, units=units, std=std,
                                cumulative_steps=[6], model_calls=2))
    rows = pool_cases(cases, seed=41, lead=12, names=names, units=units)
    row = next(r for r in rows if r["forecast"] == "original_k3" and r["region"] == "full" and r["variable"] == names[0])
    expected_acc = (2 - .2) / math.sqrt((4 + .04) * 2)
    assert math.isclose(row["pooled_acc"], expected_acc, rel_tol=1e-8)
    assert not math.isclose(row["pooled_acc"], 0.)
    assert not math.isclose(row["rmse"], (1 + 1.2) / 2 * float(std[0]))
    assert len(rows) == 3 * 3 * 17


def test_zero_energy_undefined_preserved_without_nan_json(archived_leaves, synthetic_sample):
    names, units, p, _, _, lat, std = fields()
    zero = torch.zeros_like(p)
    sample = dict(synthetic_sample, rollout_targets=zero[0], latitude=lat)
    case = score_case(sample, zero, zero, lead=12, names=names, units=units, std=std, cumulative_steps=[6], model_calls=2)
    rows = pool_cases([case], seed=41, lead=12, names=names, units=units)
    assert all(r["pooled_acc"] is None and r["mse_skill"] is None for r in rows if r["forecast"] in ("original_k3", "climatology"))
    assert all(r["acc_status"] == "undefined_zero_anomaly_energy" for r in rows)
    assert all(r["skill_status"] == "undefined_zero_climatology_energy" for r in rows)


@pytest.mark.parametrize("mutation", ["nonfinite", "negative_mse", "unit_conversion", "clim_energy", "mse_identity", "extra_key", "cauchy"])
def test_case_stat_guards_fail_closed(archived_leaves, mutation):
    names, units, p, t, c, lat, std = fields()
    stat = deepcopy(region_statistics(p, t, c, lat, lead=12, names=names, units=units, std=std)["full"]["variables"][0])
    if mutation == "nonfinite": stat["anomaly_dot"] = float("nan")
    elif mutation == "negative_mse": stat["mse"] = -1
    elif mutation == "unit_conversion": stat["normalized_acc"]["dot"] += 1
    elif mutation == "clim_energy": stat["climatology_mse"] += 1
    elif mutation == "mse_identity": stat["mse"] += 1
    elif mutation == "extra_key": stat["case_acc"] = .2
    else:
        stat.update(mse=0, forecast_energy=1, target_energy=1, anomaly_dot=2, climatology_mse=1)
    with pytest.raises(ValueError): validate_stat(stat)


def test_duplicate_cases_and_omitted_variables_refused(archived_leaves, synthetic_sample):
    names, units, p, _, c, lat, std = fields()
    sample = dict(synthetic_sample, latitude=lat)
    case = score_case(sample, p, c, lead=12, names=names, units=units, std=std, cumulative_steps=[6], model_calls=2)
    with pytest.raises(ValueError, match="duplicate"):
        pool_cases([case, case], seed=41, lead=12, names=names, units=units)
    case["statistics"]["original_k3"]["edge_2"]["variables"].pop()
    with pytest.raises(ValueError, match="all variables"):
        pool_cases([case], seed=41, lead=12, names=names, units=units)


def test_evaluator_one_forecast_for_all_regions_and_same_case_baselines(archived_leaves, synthetic_sample, monkeypatch):
    import types, sys
    fake = types.ModuleType("data.r7_evaluation")
    fake.normalized_climatology = lambda climate, dates, mean, std: climate.clone()
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    class Dataset:
        names = tuple(f"synthetic_{i}" for i in range(17))
        units = ("synthetic_unit",) * 17
        mean, std = torch.zeros(17), torch.ones(17)
        def __len__(self): return 1
        def __getitem__(self, index): return deepcopy(synthetic_sample)
    model = SyntheticModel().eval()
    cases = evaluate_cases(model, Dataset(), torch.zeros_like(synthetic_sample["rollout_targets"]), seed=41, lead=12,
                           device=torch.device("cpu"), deadline=math.inf)
    assert len(cases) == 1 and len(model.inputs) == 2
    assert tuple(cases[0]["statistics"]) == ("original_k3", "persistence", "climatology")
    assert all(tuple(v) == REGIONS for v in cases[0]["statistics"].values())


def test_original_loader_unmodified_rejects_digest_signature_contract(archived_leaves, monkeypatch):
    experiment = archived_leaves["training.r7_experiment"]
    contract = {"synthetic": True, "steps": 3}
    saved = {"format": "r7-local-v1", "model_code_sha256": "synthetic_digest", "contract": contract,
             "signature": experiment.canonical_digest(contract), "updates": 400, "model": {}}
    monkeypatch.setattr(experiment, "model_code_digest", lambda: "synthetic_digest")
    calls = []
    def synthetic_load(path, **kwargs):
        calls.append(kwargs)
        return deepcopy(saved)
    monkeypatch.setattr(torch, "load", synthetic_load)
    assert experiment.load_checkpoint("synthetic_checkpoint_not_opened", expected=saved["signature"]) == saved
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    for mutation in ("model_code_sha256", "signature", "contract"):
        broken = deepcopy(saved)
        broken[mutation] = {"changed": True} if mutation == "contract" else "changed"
        monkeypatch.setattr(torch, "load", lambda *a, **k: broken)
        with pytest.raises(ValueError): experiment.load_checkpoint("synthetic_checkpoint_not_opened", expected=saved["signature"])


@pytest.mark.parametrize("mutation", ["steps", "signature", "model", "updates", "tensor_count", "parameter_count", "nonfinite"])
def test_saved_body_qualification_is_not_only_opaque_hash(mutation):
    contract = {"synthetic": True, "steps": 3}
    state = {f"tensor_{i}": torch.zeros(1) for i in range(130)}
    state["large"] = torch.zeros(2968259 - 130)
    saved = {"contract": contract, "signature": "synthetic_sig", "model_code_sha256": MODEL_SHA, "updates": 400, "model": state}
    parent = {"contract": deepcopy(contract), "signature": "synthetic_sig"}
    if mutation == "steps": saved["contract"]["steps"] = 4
    elif mutation == "signature": saved["signature"] = "wrong"
    elif mutation == "model": saved["model_code_sha256"] = "wrong"
    elif mutation == "updates": saved["updates"] = 399
    elif mutation == "tensor_count": saved["model"]["extra"] = torch.zeros(1)
    elif mutation == "parameter_count": saved["model"]["large"] = torch.zeros(1)
    else: saved["model"]["large"][0] = float("nan")
    with pytest.raises(ValueError): verify_saved(saved, parent)


def test_evaluate_job_calls_original_strict_loader_exact_k3_constructor_and_same_forecast(archived_leaves, synthetic_sample, monkeypatch):
    import sys, types
    import r7_k3_reference_evaluate as evaluator
    from r7_k3_reference_identity import MODEL_SPEC
    from r7_k3_reference_support import DATA_SHA
    experiment = archived_leaves["training.r7_experiment"]
    contract = {"kind": "process", "model": MODEL_SPEC, "steps": 3, "synthetic": True}
    signature = experiment.canonical_digest(contract)
    state = {f"tensor_{i}": torch.zeros(1) for i in range(130)}
    state["large"] = torch.zeros(2968259 - 130)
    saved = {"format": "r7-local-v1", "contract": contract, "signature": signature,
             "model_code_sha256": MODEL_SHA, "updates": 400, "model": state}
    trace = []
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: trace.append(("load", path, kwargs)) or saved)
    monkeypatch.setattr(experiment, "model_code_digest", lambda: MODEL_SHA)
    class ConstructorStandIn(SyntheticModel):
        def load_state_dict(self, values, strict):
            trace.append(("strict_state", len(values), strict))
        def to(self, device):
            trace.append(("to", str(device)))
            return self
    model = ConstructorStandIn()
    monkeypatch.setattr(experiment, "make_model", lambda kind, spec: trace.append(("make", kind, spec)) or model)
    monkeypatch.setattr(torch, "device", lambda name: "synthetic-cuda0-no-real-device")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: trace.append(("sync", device)))
    class Dataset:
        names = tuple(f"synthetic_{i}" for i in range(17))
        units = ("synthetic_unit",) * 17
        mean, std = torch.zeros(17), torch.ones(17)
        def __len__(self): return 1
        def __getitem__(self, index): return deepcopy(synthetic_sample)
    dataset = Dataset()
    monkeypatch.setattr(evaluator, "metadata_preflight", lambda *a: dataset)
    fake = types.ModuleType("data.r7_evaluation")
    fake.fit_training_climatology = lambda path: {"selection": "declared_train_time_ranges", "n_selected_steps": 188,
                "channels": list(dataset.names), "counts": {(m, h): 1 for m in (1, 2) for h in (0, 6, 12, 18)}}
    fake.normalized_climatology = lambda *args: torch.zeros_like(synthetic_sample["rollout_targets"])
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    # Keep old rollout/model_input on synthetic CPU while honoring production's cuda0 argument.
    rollout = archived_leaves["model.r7_rollout"]
    original_input = rollout.rollout_model_input
    monkeypatch.setattr(rollout, "rollout_model_input", lambda sample, **kwargs: original_input(sample, lead_hours=kwargs["lead_hours"]))
    parent = {"checkpoint": "synthetic_checkpoint_not_opened", "checkpoint_sha256": "synthetic_hash", "contract": contract, "signature": signature}
    p = {"parents": {"41": parent}, "data": {"store": "synthetic_store_not_opened", "channels": list(dataset.names),
          "units": list(dataset.units), "evaluation_cases": {"12": {"n_available": 1, "cases": [[synthetic_sample["init_time"], synthetic_sample["valid_times"]]]}}},
         "limitations": ["synthetic"], "calendar": "old accumulated lead365.25"}
    result = evaluator.evaluate_job(p, {"seed": 41, "lead": 12}, math.inf)
    assert trace[0] == ("load", "synthetic_checkpoint_not_opened", {"map_location": "cpu", "weights_only": True})
    assert trace[1] == ("make", "process", MODEL_SPEC) and trace[2] == ("strict_state", 131, True)
    assert len(model.inputs) == 2 and result["extra_model_forward_for_regions_or_baselines"] == 0
    assert len(result["rows"]) == 153 and result["training_identity"] == DATA_SHA
    assert result["cases"][0]["cumulative_reasoning_steps"] == [6]
