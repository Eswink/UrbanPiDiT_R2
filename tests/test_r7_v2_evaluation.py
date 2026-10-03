"""CPU-only synthetic temporary stores; no real data/output or GPU experiment."""
from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import zarr

from data.r7_evaluation import fit_training_climatology, normalized_climatology
from model.r7_halting import DECLARED_MODEL_INPUTS
from model.r7_rollout import autoregressive_rollout, rollout_model_input
from training.r7_experiment import canonical_digest, dataset_identity, save_exclusive
import training.r7_v2_evaluation as evaluation


class TinyRecursive(nn.Module):
    def __init__(self, channels=17, spacetime_inputs=True):
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(.05, .25, channels))
        with torch.no_grad():
            self.bias[-1] = 70.0  # Deliberately bad variable must remain in every table.
        self.spacetime_inputs = spacetime_inputs
        self.seen = []

    def forward(self, batch, *, reasoning_steps):
        self.seen.append((set(batch), float(batch["lead_time_hours"][0]), torch.is_grad_enabled(),
                          float(batch["init_calendar_year"])))
        return SimpleNamespace(forecast=batch["coarse_history"][:, -1] +
                               self.bias[None, :, None, None] * reasoning_steps,
                               reasoning_steps=reasoning_steps)


def _array(root, name, data):
    root.create_array(name, data=np.asarray(data), chunks=np.asarray(data).shape)


def _rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _record(stamps, offset, index, split):
    return dict(sample_id=f"{split}-{index}", store_path="store.zarr", split=split,
                history_indices=[offset + index - 1, offset + index], target_index=offset + index + 1,
                history_times=[stamps[offset + index - 1].isoformat(), stamps[offset + index].isoformat()],
                init_time=stamps[offset + index].isoformat(), target_time=stamps[offset + index + 1].isoformat(),
                lead_time_hours=6)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    """8 train / 24 val / 24 test steps yield val counts 22/21/19/15/11."""
    times = [pd.date_range(f"{year}-01-01", periods=count, freq="6h")
             for year, count in ((2018, 8), (2019, 24), (2020, 24))]
    stamps = pd.DatetimeIndex(list(itertools.chain.from_iterable(times))).as_unit("ns")
    c, h, w = 17, 6, 7
    base = np.arange(c, dtype=np.float32)[:, None, None] * 2 + np.arange(h * w).reshape(h, w) * .01
    state = np.stack([base + .2 * i + 2 * split
                      for split, count in enumerate((8, 24, 24)) for i in range(count)]).astype(np.float32)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    root.create_array("state", data=state, chunks=(2, c, h, w))
    for name, value in dict(time_ns=stamps.asi8, latitude=40 - np.arange(h) * .25,
                            longitude=100 + np.arange(w) * .25,
                            normalization_mean=state[:8].mean((0, 2, 3)),
                            normalization_std=state[:8].std((0, 2, 3))).items():
        _array(root, name, value)
    root.attrs.update(dict(schema_version=1, build_complete=True, time_unit="ns",
                           channels=[f"variable_{j:02d}" for j in range(c)], units=["K"] * c,
                           native_grid_spacing_deg=.25, split_years={"train": [2018], "val": [2019], "test": [2020]},
                           normalization_years=[2018], source={"kind": "synthetic-test-fixture", "scientific_claim": False}))
    (tmp_path / "BUILD_COMPLETE.json").write_text(json.dumps({"schema_version": 1, "build_complete": True}), encoding="utf-8")
    paths = {}
    for split, count, offset in (("train", 8, 0), ("val", 24, 8), ("test", 24, 32)):
        paths[split] = tmp_path / f"{split}.jsonl"
        records = [_record(stamps, offset, i, split) for i in range(1, count - 1)]
        paths[split].write_text("\n".join(json.dumps(r) for r in reversed(records)) + "\n", encoding="utf-8")
    identity, _ = dataset_identity(paths["train"])
    contract = dict(kind="process", model={"channels": c, "spacetime_inputs": True}, steps=4, data_identity=identity)
    payload = dict(format="r7-local-v1", contract=contract, signature=canonical_digest(contract),
                   model=TinyRecursive().state_dict())
    checkpoint = tmp_path / "checkpoint.pt"
    save_exclusive(checkpoint, payload)
    models = []

    def make_model(_kind, config):
        model = TinyRecursive(**config)
        models.append(model)
        return model

    monkeypatch.setattr(evaluation, "make_model", make_model)
    return dict(paths=paths, checkpoint=checkpoint, payload=payload, root=root, models=models,
                store=tmp_path / "store.zarr", tmp=tmp_path)


def _run(fixture, *, lead=6, k=4, max_samples=32, name="evaluation", **kwargs):
    return evaluation.evaluate_v2(fixture["paths"]["val"], fixture["checkpoint"], fixture["tmp"] / name,
                                  lead_hours=(lead,), reasoning_steps=k, max_samples=max_samples,
                                  device_name="cpu", **kwargs)


@pytest.mark.parametrize("lead,k", list(itertools.product(evaluation.ALL_LEADS, evaluation.PROBE_STEPS)))
def test_all_leads_and_same_k4_checkpoint_probes_keep_all_legal_cases(fixture, lead, k):
    report = _run(fixture, lead=lead, k=k)
    expected = {6: 22, 12: 21, 24: 19, 48: 15, 72: 11}[lead]
    assert report["n_evaluated"] == report["n_available_windows"] == expected
    assert report["split"] == "val" and report["test_read"] is False
    assert report["scientific_claim"] is False and report["limitations"]
    assert report["checkpoint_training_steps"] == 4 and report["reasoning_steps"] == k
    assert report["independently_trained_k1"] is False and "not independent" in report["depth_probe"]
    assert len(report["region_metrics"]) == 3 * 17
    assert {r["region"] for r in report["region_metrics"]} == {"full", "interior", "edge_2"}
    assert {r["variable"] for r in report["region_metrics"]} == set(report["channels"])
    assert all(r["unit"] == "K" and r["n_initializations"] == expected for r in report["region_metrics"])
    assert [c["init_time"] for c in report["initializations"]] == sorted(c["init_time"] for c in report["initializations"])
    assert all(c["cumulative_reasoning_steps"] == [k * lead // 6] for c in report["initializations"])
    assert all(len(c["region_metrics"]) == 3 * 17 for c in report["initializations"])
    cases = _rows(fixture["tmp"] / "evaluation" / "per_case_metrics.csv")
    assert len(cases) == expected * 3 * 17
    baseline_cases = _rows(fixture["tmp"] / "evaluation" / "baseline_per_case_metrics.csv")
    assert len(baseline_cases) == expected * 2 * 3 * 17
    assert len(report["baseline_region_metrics"]) == 2 * 3 * 17
    assert len(report["baseline_initializations"]) == expected
    assert all(len(c["region_metrics"]) == 2 * 3 * 17 for c in report["baseline_initializations"])
    assert all(r["reasoning_steps"] == "" and r["zero_train_updates"] == "0" and r["parameters"] == "0"
               and r["trainable_parameters"] == "0" for r in baseline_cases)
    assert all(int(r["lead_hours"]) == lead and int(r["reasoning_steps"]) == k for r in cases)
    assert all(json.loads(r["valid_times"]) == [r["valid_time"]] for r in cases)
    assert len(report["bad_case_metrics"]) >= expected * 3
    assert any(r["variable"] == "variable_16" and "negative_mse_skill" in r["bad_reasons"]
               for r in report["bad_variable_metrics"])
    model = fixture["models"][-1]
    assert all(keys == set(DECLARED_MODEL_INPUTS) and not grad and year == 2019 for keys, _, grad, year in model.seen)
    assert len(model.seen) == expected * (lead // 6) + 3 + 10  # Baselines add no model calls/latency samples.
    assert report["parameters"] == report["trainable_parameters"] == 17
    isolated = report["isolated_forward"]
    assert isolated["warmup_excluded"] == 3 and isolated["repetitions"] == 10
    assert len(isolated["seconds_per_batch"]) == 10 and isolated["gpu_latency_measured"] is False
    assert all(count == [k] for count in isolated["actual_reasoning_steps_per_sample"])
    assert all(report["cuda_memory"][key]["allocated_bytes"] is None
               for key in ("entry", "loop_baseline", "after_loop", "after_profile"))
    assert report["cuda_memory"]["allocator_reset_performed"] is False
    assert report["whole_elapsed_seconds"] >= report["loop_elapsed_seconds"] >= 0
    assert report["forward_backward_flops"] is None
    assert json.loads((fixture["tmp"] / "evaluation" / "provenance.json").read_text(encoding="utf-8")) == report


def test_csv_sufficient_statistics_independently_reproduce_pooled_metrics(fixture):
    report = _run(fixture, lead=12, max_samples=3)
    rows = _rows(fixture["tmp"] / "evaluation" / "per_case_metrics.csv")
    for pooled in report["region_metrics"]:
        cases = [r for r in rows if (r["region"], r["variable"]) == (pooled["region"], pooled["variable"])]
        mse = np.mean([float(r["mse"]) for r in cases])
        climatology_mse = np.mean([float(r["climatology_mse"]) for r in cases])
        dot, fe, te = [sum(float(r[key]) for r in cases)
                       for key in ("acc_dot", "acc_forecast_energy", "acc_target_energy")]
        assert pooled["rmse"] == pytest.approx(math.sqrt(mse), rel=1e-12)
        assert pooled["mse_skill"] == pytest.approx(1 - mse / climatology_mse, rel=1e-12)
        assert pooled["acc"] == pytest.approx(dot / math.sqrt(fe * te), rel=1e-12)
        assert pooled["acc_dot"] == pytest.approx(dot, rel=1e-12)
        assert all(r["acc_statistic_units"] == "normalized_anomaly_squared" for r in cases)
        assert pooled["climatology_mse"] == pooled["mse_climatology"]
    assert _rows(fixture["tmp"] / "evaluation" / "region_metrics.csv")


def test_same_lead_k_case_pairing_and_cap_no_crosslead_narrowing(fixture):
    a = _run(fixture, lead=6, k=1, name="a")
    b = _run(fixture, lead=6, k=4, name="b")
    c = _run(fixture, lead=72, k=2, name="c")
    capped = _run(fixture, lead=6, max_samples=2, name="cap")
    assert a["case_selection_sha256"] == b["case_selection_sha256"]
    assert a["checkpoint_sha256"] == b["checkpoint_sha256"]
    assert c["n_evaluated"] < a["n_evaluated"] and c["case_selection_sha256"] != a["case_selection_sha256"]
    assert capped["n_available_windows"] == 22 and capped["n_evaluated"] == 2
    with pytest.raises(FileExistsError):
        _run(fixture, name="a")


def test_exact_fixed_two_cell_masks_and_physical_area_weighted_statistics():
    generator = torch.Generator().manual_seed(14)
    p, t, climate = [torch.randn(3, 1, 2, 6, 7, generator=generator) for _ in range(3)]
    lat = torch.linspace(65, 10, 6)
    metrics = evaluation.RegionMetrics((6,), ("t", "z"), training_std=(2., 100.), units=("K", "Pa"))
    metrics.update(p, t, climate, lat)
    interior = torch.zeros(6, 7, dtype=torch.bool)
    interior[2:-2, 2:-2] = True
    masks = {"full": torch.ones_like(interior), "interior": interior, "edge_2": ~interior}
    weight = torch.cos(torch.deg2rad(lat.double()))[:, None].expand(6, 7)
    for row in metrics.rows():
        mask = masks[row["region"]]
        j = ("t", "z").index(row["variable"])
        normalized = weight * mask / weight[mask].sum()
        fe = (((p - climate).double().square()) * normalized).sum((-1, -2))[:, 0, j].sum()
        te = (((t - climate).double().square()) * normalized).sum((-1, -2))[:, 0, j].sum()
        dot = (((p - climate).double() * (t - climate).double()) * normalized).sum((-1, -2))[:, 0, j].sum()
        # Difference in double, before subtraction, matches existing accumulators exactly.
        mse = (((p.double() - t.double()).square()) * normalized).sum((-1, -2))[:, 0, j].mean() * (2., 100.)[j] ** 2
        assert row["mse"] == pytest.approx(float(mse), rel=1e-12)
        assert row["acc_forecast_energy"] == pytest.approx(float(fe), rel=2e-7)
        assert row["acc_target_energy"] == pytest.approx(float(te), rel=2e-7)
        assert row["acc_dot"] == pytest.approx(float(dot), rel=2e-7)
        assert row["n_grid_points"] == int(mask.sum())
        assert row["full_area_fraction"] == pytest.approx(float(weight[mask].sum() / weight.sum()))
    definitions = {d["region"]: d for d in metrics.regions}
    assert definitions["interior"]["n_grid_points"] == 6 and definitions["edge_2"]["n_grid_points"] == 36
    assert definitions["interior"]["full_area_fraction"] + definitions["edge_2"]["full_area_fraction"] == pytest.approx(1.)


def test_boundary_excluded_cells_do_not_supply_fake_zero_anomaly_energy():
    t = torch.ones(1, 1, 1, 6, 7)
    t[..., 2:-2, 2:-2] = 0
    metrics = evaluation.RegionMetrics((6,), ("t",), training_std=(3.,), units=("K",))
    metrics.update(2 * t, t, torch.zeros_like(t), torch.linspace(50, 20, 6))
    rows = {r["region"]: r for r in metrics.rows()}
    assert rows["edge_2"]["mse"] == pytest.approx(9.)
    assert rows["edge_2"]["climatology_mse"] == pytest.approx(9.)
    assert rows["edge_2"]["acc_forecast_energy"] == pytest.approx(4.)
    assert rows["edge_2"]["acc_target_energy"] == pytest.approx(1.)
    assert rows["edge_2"]["acc"] == pytest.approx(1.)
    assert rows["interior"]["acc"] is None and rows["interior"]["mse_skill"] is None
    assert rows["interior"]["acc_status"] == "undefined_zero_anomaly_energy"
    assert rows["interior"]["skill_status"] == "undefined_zero_climatology_energy"
    assert rows["interior"]["mse"] == rows["interior"]["climatology_mse"] == 0


@pytest.mark.parametrize("kind", ["training-identity", "code-digest", "contract-digest", "K3", "native", "state", "sidecar"])
def test_checkpoint_guards_are_not_bypassed(fixture, kind):
    payload = dict(fixture["payload"], contract=dict(fixture["payload"]["contract"]))
    if kind == "training-identity":
        payload["contract"]["data_identity"] = "0" * 64
    elif kind == "K3":
        payload["contract"]["steps"] = 3
    elif kind == "native":
        payload["contract"]["kind"] = "native"
    elif kind == "state":
        payload["model"] = {}
    elif kind == "sidecar":
        with pytest.raises(ValueError, match="sidecar override"):
            _run(fixture, process_scale_sidecar=fixture["tmp"] / "not-a-sidecar.json")
        assert not (fixture["tmp"] / "evaluation").exists()
        return
    payload["signature"] = canonical_digest(payload["contract"])
    changed = fixture["tmp"] / "changed.pt"
    save_exclusive(changed, payload)
    if kind in ("code-digest", "contract-digest"):
        raw = torch.load(changed, weights_only=True)
        raw["model_code_sha256" if kind == "code-digest" else "signature"] = "0" * 64
        torch.save(raw, changed)
    fixture["checkpoint"] = changed
    with pytest.raises((ValueError, RuntimeError)):
        _run(fixture)
    assert not (fixture["tmp"] / "evaluation").exists()


@pytest.mark.parametrize("kind", ["test", "multiple-stores", "history", "lead", "cadence", "duplicate-init",
                                   "no-cases", "channels", "units", "normalized-unit", "record", "build"])
def test_manifest_store_case_and_unit_guards_fail_closed(fixture, kind):
    path = fixture["paths"]["val"]
    records = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
    if kind == "test":
        fixture["paths"]["val"] = fixture["paths"]["test"]
    elif kind == "multiple-stores":
        records[0]["store_path"] = "other.zarr"
    elif kind == "history":
        records[0]["history_indices"] = records[0]["history_indices"][-1:]
        records[0]["history_times"] = records[0]["history_times"][-1:]
    elif kind == "lead":
        r = records[-1]
        r["target_index"] += 1
        r["target_time"] = pd.Timestamp(int(fixture["root"]["time_ns"][r["target_index"]])).isoformat()
        r["lead_time_hours"] = 12
    elif kind == "cadence":
        r = records[0]
        r["history_indices"][0] -= 1
        r["history_times"][0] = pd.Timestamp(int(fixture["root"]["time_ns"][r["history_indices"][0]])).isoformat()
    elif kind == "duplicate-init":
        records.append(dict(records[0], sample_id="another-id"))
    elif kind == "no-cases":
        evaluation._validation_data(path, 72)
        records = records[:1]  # Last chronological init has only 6h available.
    elif kind == "channels":
        fixture["root"].attrs["channels"] = ["missing"]
    elif kind in ("units", "normalized-unit"):
        fixture["root"].attrs["units"] = ["K"] * 16 + ["unknown" if kind == "units" else "normalized"]
    elif kind == "record":
        records[0]["init_time"] = "2019-01-01T00:00:00"
    elif kind == "build":
        (fixture["tmp"] / "BUILD_COMPLETE.json").write_text('{"schema_version":1,"build_complete":false}', encoding="utf-8")
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _run(fixture, lead=72 if kind == "no-cases" else 6)
    assert not (fixture["tmp"] / "evaluation").exists()


@pytest.mark.parametrize("options", [{"lead_hours": (6, 12)}, {"lead_hours": (18,)}, {"lead_hours": (True,)},
                                     {"reasoning_steps": 3}, {"reasoning_steps": True},
                                     {"max_samples": 0}, {"max_samples": True}, {"deadline": float("nan")},
                                     {"deadline": True}, {"deadline": -1.}])
def test_explicit_worker_controls_reject_invalid_values(fixture, options):
    with pytest.raises((ValueError, TimeoutError)):
        evaluation.evaluate_v2(fixture["paths"]["val"], fixture["checkpoint"], fixture["tmp"] / "invalid",
                               device_name="cpu", **options)
    assert not (fixture["tmp"] / "invalid").exists()


def test_train_only_climatology_ignores_all_heldout_offsets_and_missing_bucket(fixture):
    before = fit_training_climatology(fixture["store"])
    fixture["root"]["state"][8:] = fixture["root"]["state"][8:] + 9000
    after = fit_training_climatology(fixture["store"])
    assert before["n_selected_steps"] == after["n_selected_steps"] == 8
    assert before["counts"] == after["counts"]
    for key in before["means"]:
        np.testing.assert_array_equal(before["means"][key], after["means"][key])
    with pytest.raises(ValueError, match="no held-out fallback"):
        normalized_climatology(before, ["2019-02-01"], np.zeros(17), np.ones(17))


def test_poison_future_keys_never_read_by_model_input_or_autoregressive_forward():
    class Poison(dict):
        def __getitem__(self, key):
            if key in ("rollout_targets", "atmos_target", "process_targets", "future_boundary"):
                raise AssertionError("future truth read for model forward")
            return super().__getitem__(key)
    sample = Poison(coarse_history=torch.ones(2, 17, 6, 7), latitude=torch.linspace(40, 30, 6),
                    longitude=torch.linspace(100, 106, 7), init_utc_hour=torch.tensor(0.),
                    init_day_of_year=torch.tensor(1.), init_year=torch.tensor(2019.),
                    init_calendar_year=torch.tensor(2019.), rollout_targets=object(), atmos_target=object(),
                    process_targets=object(), future_boundary=object())
    inputs = rollout_model_input(sample, device=torch.device("cpu"))
    model = TinyRecursive().eval()
    trajectory = autoregressive_rollout(model, inputs, lead_hours=(12,), inference_kwargs={"reasoning_steps": 2})
    assert trajectory.forecasts.shape == (1, 1, 17, 6, 7)
    assert all(keys == set(DECLARED_MODEL_INPUTS) for keys, _, _, _ in model.seen)
    # Current explicit-calendar rollout advances the initialization time while each call remains +6h.
    assert [lead for _, lead, _, _ in model.seen] == [6., 6.]
    assert inputs["init_calendar_year"].item() == 2019
    torch.testing.assert_close(trajectory.forecasts[:, 0], inputs["coarse_history"][:, -1] + 4 * model.bias[None, :, None, None])


@pytest.mark.parametrize("problem", ["shape", "channels", "finite", "latitude", "lat-batch", "lat-monotone", "zero-area", "grid", "margin"])
def test_region_geometry_and_finite_guards_have_counterexamples(problem):
    p = torch.ones(1, 1, 1, 6, 7)
    target, climate, lat = p.clone(), torch.zeros_like(p), torch.linspace(50, 20, 6)
    metrics = evaluation.RegionMetrics((6,), ("t",), training_std=(1.,), units=("K",))
    with pytest.raises(RuntimeError, match="no initialization"):
        metrics.rows()
    if problem == "shape":
        target = target[..., :-1]
    elif problem == "channels":
        p, target, climate = [x.expand(1, 1, 2, 6, 7) for x in (p, target, climate)]
    elif problem == "finite":
        p[0, 0, 0, 0, 0] = float("nan")
    elif problem == "latitude":
        lat[0] = 120
    elif problem == "lat-batch":
        p, target, climate = [x.expand(2, 1, 1, 6, 7) for x in (p, target, climate)]
        lat = torch.stack([lat, lat + 1])
    elif problem == "lat-monotone":
        lat[2] = lat[1]
    elif problem == "zero-area":
        lat = torch.linspace(90 - 1e-12, 90, 6, dtype=torch.float64)
    elif problem == "grid":
        metrics.update(p, target, climate, lat)
        lat = lat + 1
    elif problem == "margin":
        p, target, climate = [x[..., :4, :4] for x in (p, target, climate)]
        lat = lat[:4]
    with pytest.raises(ValueError):
        metrics.update(p, target, climate, lat)


@pytest.mark.parametrize("std,units", [(None, ("K",)), ((0.,), ("K",)), ((float("nan"),), ("K",)), ((1.,), None), ((1.,), ())])
def test_physical_units_require_training_std_and_complete_units(std, units):
    with pytest.raises(ValueError):
        evaluation.RegionMetrics((6,), ("t",), training_std=std, units=units)


@pytest.mark.parametrize("problem", ["counts", "units", "forecast", "identity", "overflow", "false-undefined"])
def test_region_rows_fail_if_existing_accumulators_disagree_or_overflow(problem):
    metrics = evaluation.RegionMetrics((6,), ("t",), training_std=(2.,), units=("K",))
    p = torch.ones(1, 1, 1, 6, 7)
    metrics.update(2 * p, p, torch.zeros_like(p), torch.linspace(50, 20, 6))
    rmse, acc, skill = metrics.metrics["full"]
    if problem == "counts":
        acc.initializations += 1
    elif problem == "units":
        skill.units = ("Pa",)
    elif problem == "forecast":
        skill.forecast_squared_error += 1
    elif problem == "identity":
        acc.dot = -acc.dot
    elif problem == "overflow":
        acc.forecast_energy.fill_(float("inf"))
    elif problem == "false-undefined":
        acc.compute = lambda: torch.full((1, 1), float("nan"))
    with pytest.raises(ValueError):
        metrics.rows()


def test_real_error_retains_completed_case_csv_and_never_writes_success(fixture):
    # First case history=8,9,target=10; second case first accesses target=11.
    fixture["root"]["state"][11, 16, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        _run(fixture, max_samples=2)
    directory = fixture["tmp"] / "evaluation"
    assert len(_rows(directory / "per_case_metrics.csv")) == 3 * 17
    assert len(_rows(directory / "baseline_per_case_metrics.csv")) == 2 * 3 * 17
    assert not (directory / "provenance.json").exists()
    assert not (directory / "region_metrics.csv").exists()
    assert not (directory / "baseline_region_metrics.csv").exists()


def test_deadline_retains_completed_case_csv_and_no_success(fixture, monkeypatch):
    clock = [0.]
    original = evaluation.autoregressive_rollout
    def first_then_expire(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 100.
        return result
    monkeypatch.setattr(evaluation.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(evaluation, "autoregressive_rollout", first_then_expire)
    with pytest.raises(TimeoutError):
        _run(fixture, max_samples=2, deadline=50.)
    assert len(_rows(fixture["tmp"] / "evaluation" / "per_case_metrics.csv")) == 3 * 17
    assert len(_rows(fixture["tmp"] / "evaluation" / "baseline_per_case_metrics.csv")) == 2 * 3 * 17
    assert not (fixture["tmp"] / "evaluation" / "provenance.json").exists()


def test_allocator_is_observed_not_reset_or_cleared(monkeypatch):
    observed = []
    for name, value in (("memory_allocated", 100), ("memory_reserved", 200),
                        ("max_memory_allocated", 300), ("max_memory_reserved", 400)):
        monkeypatch.setattr(torch.cuda, name, lambda device, value=value: observed.append(str(device)) or value)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("evaluator must not reset/clear the allocator")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", forbidden)
    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden)
    assert evaluation._memory_snapshot(torch.device("cuda:0")) == dict(allocated_bytes=100, reserved_bytes=200,
                                                                     peak_allocated_bytes=300, peak_reserved_bytes=400)
    assert observed == ["cuda:0"] * 4


def test_isolated_profile_performs_only_forward_and_synchronizes_outside_checks(monkeypatch):
    sample = dict(coarse_history=torch.ones(2, 17, 6, 7), latitude=torch.linspace(40, 30, 6),
                  longitude=torch.linspace(100, 106, 7), init_calendar_year=torch.tensor(2019.))
    inputs = rollout_model_input(sample)
    model = TinyRecursive().eval()
    observed = []
    monkeypatch.setattr(evaluation, "_synchronize", lambda device: observed.append(str(device)))
    report = evaluation._profile_forward(model, inputs, 2, None)
    assert len(model.seen) == 13 and not any(grad for _, _, grad, _ in model.seen)
    assert len(observed) == 23 and report["repetitions"] == 10
    assert all(seconds >= 0 for seconds in report["seconds_per_batch"])
    model.train()
    with pytest.raises(ValueError, match="model.eval"):
        evaluation._profile_forward(model, inputs, 2, None)


@pytest.mark.parametrize("problem", ["finite", "actual-K", "counts", "input-mutation", "state-mutation"])
def test_isolated_forward_guards_fail_without_accepting_latency(problem):
    class Broken(TinyRecursive):
        def forward(self, batch, *, reasoning_steps):
            out = super().forward(batch, reasoning_steps=reasoning_steps)
            if problem == "finite":
                out.forecast.fill_(float("nan"))
            elif problem == "actual-K":
                out.reasoning_steps = 3
            elif problem == "counts":
                out.reasoning_steps_per_sample = torch.zeros(1, dtype=torch.long)
            elif problem == "input-mutation":
                batch["coarse_history"].add_(1)
            elif problem == "state-mutation":
                self.bias.add_(1)
            return out
    inputs = {"coarse_history": torch.ones(1, 2, 17, 6, 7), "lead_time_hours": torch.tensor([6.]),
              "init_calendar_year": torch.tensor(2019.)}
    with pytest.raises(ValueError):
        evaluation._profile_forward(Broken().eval(), inputs, 2, None)


def test_actual_process_model_cpu_checkpoint_smoke(fixture, monkeypatch):
    from training.r7_experiment import make_model
    config = dict(in_channels=17, history_steps=2, dim=16, depth=1, heads=4, patch_size=2,
                  window_size=2, anchored_processes=2, free_processes=2, spacetime_inputs=True)
    model = make_model("process", config).cpu()
    contract = dict(fixture["payload"]["contract"], model=config)
    checkpoint = fixture["tmp"] / "actual_process.pt"
    save_exclusive(checkpoint, dict(format="r7-local-v1", contract=contract,
                                   signature=canonical_digest(contract), model=model.state_dict()))
    fixture["checkpoint"] = checkpoint
    monkeypatch.setattr(evaluation, "make_model", make_model)
    report = _run(fixture, lead=12, k=2, max_samples=1)
    assert report["n_evaluated"] == 1 and len(report["region_metrics"]) == 51
    assert report["trainable_parameters"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert report["isolated_forward"]["actual_reasoning_steps_per_sample"] == [[2]] * 10
    assert report["known_input_budget"]["spacetime_inputs"] is True


def test_baseline_real_fixture_physical_mse_and_statistics_independent_of_model_and_k(fixture):
    report = _run(fixture, lead=12, k=1, max_samples=3, name="ref1")
    repeat = _run(fixture, lead=12, k=4, max_samples=3, name="ref4")
    assert report["baseline_region_metrics"] == repeat["baseline_region_metrics"]
    assert report["baseline_initializations"] == repeat["baseline_initializations"]
    assert report["baseline_definition"] == {
        "persistence": {"kind": "known-last-history-frame", "training_updates": 0},
        "climatology": {"kind": "train-only-month-hour", "training_updates": 0},
        "case_pairing": "same exact requested-lead initializations and full/interior/edge_2"}
    ds, _, _, _ = evaluation._validation_data(fixture["paths"]["val"], 12)
    clim = fit_training_climatology(fixture["store"])
    assert len(_rows(fixture["tmp"] / "ref1" / "baseline_region_metrics.csv")) == 102
    flat = _rows(fixture["tmp"] / "ref1" / "baseline_per_case_metrics.csv")
    for i, case in enumerate(report["baseline_initializations"]):
        sample = ds[i]
        target = sample["rollout_targets"][0].double()
        climate = normalized_climatology(clim, sample["valid_times"], ds.mean, ds.std)[0].double()
        lat = sample["latitude"].double()
        masks = {"full": torch.ones(6, 7, dtype=torch.bool), "interior": torch.zeros(6, 7, dtype=torch.bool)}
        masks["interior"][2:-2, 2:-2] = True
        masks["edge_2"] = ~masks["interior"]
        weights = torch.cos(torch.deg2rad(lat))[:, None].expand(6, 7)
        assert (case["sample_id"], case["init_time"], case["valid_times"]) == (
            report["initializations"][i]["sample_id"], sample["init_time"], sample["valid_times"])
        for row in case["region_metrics"]:
            j = ds.names.index(row["variable"])
            forecast = sample["coarse_history"][-1].double() if row["baseline_kind"] == "persistence" else climate
            weight = weights[masks[row["region"]]]
            p = (forecast - climate)[j][masks[row["region"]]]
            t = (target - climate)[j][masks[row["region"]]]
            area_mean = lambda value: float((value * weight).sum() / weight.sum())
            assert row["mse"] == pytest.approx(area_mean((p - t).square()) * float(ds.std[j]) ** 2, rel=1e-12)
            assert row["climatology_mse"] == pytest.approx(area_mean(t.square()) * float(ds.std[j]) ** 2, rel=1e-12)
            assert row["acc_dot"] == pytest.approx(area_mean(p * t), rel=1e-12)
            assert row["acc_forecast_energy"] == pytest.approx(area_mean(p.square()), rel=1e-12)
            assert row["acc_target_energy"] == pytest.approx(area_mean(t.square()), rel=1e-12)
            if row["baseline_kind"] == "climatology":
                assert row["mse_skill"] == pytest.approx(0.) and row["acc"] is None
    for pooled in report["baseline_region_metrics"]:
        cases = [r for r in flat if all(r[k] == str(pooled[k]) for k in ("baseline_kind", "region", "variable"))]
        assert len(cases) == 3 and all(r["unit"] == pooled["unit"] for r in cases)
        assert pooled["mse"] == pytest.approx(np.mean([float(r["mse"]) for r in cases]), rel=1e-12)
        for key in ("acc_dot", "acc_forecast_energy", "acc_target_energy"):
            assert pooled[key] == pytest.approx(sum(float(r[key]) for r in cases), rel=1e-12)


def test_reference_construction_poison_zero_energy_and_exact_edge_selection():
    class Poison(dict):
        def __getitem__(self, key):
            if key != "coarse_history":
                raise AssertionError("reference construction read future or undeclared keys")
            return super().__getitem__(key)
    history = torch.ones(2, 1, 6, 7)
    history[-1, :, 2:-2, 2:-2] = 0
    sample = Poison(coarse_history=history, rollout_targets=object(), future_boundary=object())
    climate = torch.zeros(1, 1, 1, 6, 7)
    forecasts = evaluation._reference_forecasts(sample, climate)
    torch.testing.assert_close(forecasts["persistence"][0, 0], history[-1])
    assert forecasts["climatology"] is climate
    metrics = {kind: evaluation.RegionMetrics((6,), ("t",), training_std=(3.,), units=("K",)) for kind in forecasts}
    for kind, forecast in forecasts.items():
        metrics[kind].update(forecast, climate, climate, torch.linspace(50, 20, 6))
    rows = {(r["baseline_kind"], r["region"]): r for r in evaluation._reference_rows(metrics)}
    assert rows["persistence", "edge_2"]["mse"] == pytest.approx(9.)
    assert rows["persistence", "edge_2"]["acc_forecast_energy"] == pytest.approx(1.)
    assert rows["persistence", "interior"]["mse"] == 0
    assert len(rows) == 6 and all(r["mse_skill"] is None and r["acc"] is None for r in rows.values())
    assert all(r["zero_train_updates"] == r["parameters"] == r["trainable_parameters"] == 0 for r in rows.values())
    for bad_history in (torch.full_like(history, float("nan")), history[..., :-1]):
        with pytest.raises(ValueError, match="baseline requires"):
            evaluation._reference_forecasts({"coarse_history": bad_history}, climate)
