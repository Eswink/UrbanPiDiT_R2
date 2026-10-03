"""CPU engineering fixtures only: full BPTT, exact train windows and own resume.

All stores/checkpoints/reports live in tmp_path; these analytic/synthetic fields
are not weather truth. No real outputs, sealed test manifest, network or CUDA.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import zarr

from data.r7_autoregressive_dataset import ZarrAutoregressiveDataset, preflight_training_windows
from model.r7_halting import DECLARED_MODEL_INPUTS
from training.r7_autoregressive_rollout import training_one_step, training_two_step
from training.r7_autoregressive_runner import fine_tune
from training.r7_experiment import canonical_digest, dataset_identity, load_checkpoint, make_model
from training.r7_recursive_losses import deep_supervised_forecast_mse


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def batch(channels=1):
    return {"coarse_history": torch.ones(1, 2, channels, 3, 4),
            "atmos_target": torch.zeros(1, channels, 3, 4),
            "future_target": torch.zeros(1, channels, 3, 4),
            "latitude": torch.tensor([40., 39.75, 39.5]),
            "longitude": torch.tensor([115., 115.25, 115.5, 115.75]),
            "lead_time_hours": torch.tensor([6.]), "init_utc_hour": torch.tensor([18.]),
            "init_day_of_year": torch.tensor([366.]), "init_calendar_year": torch.tensor([2016.]),
            "init_year": torch.tensor([1234.]), "process_targets": torch.tensor([[9999.]]),
            "untrusted_metadata": "not a model input"}


class AnalyticForecaster(nn.Module):
    """Scalar multiplier recurrence exposes all K and physical gradient paths."""

    def __init__(self, *, output_bf16=False, detach_physical=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.))
        self.spacetime_inputs = True
        self.inputs, self.kwargs, self.traces = [], [], []
        self.output_bf16, self.detach_physical = output_bf16, detach_physical

    def forward(self, inputs, *, reasoning_steps, detach_between_steps):
        self.inputs.append(dict(inputs))
        self.kwargs.append((reasoning_steps, detach_between_steps))
        current = inputs["coarse_history"][:, -1]
        if self.detach_physical and len(self.inputs) == 2:
            current = current.detach()
        initial = current * self.scale
        drafts, first = [initial], initial
        for index in range(reasoning_steps):
            current = drafts[-1].detach() if detach_between_steps and index else drafts[-1]
            drafts.append(current * self.scale)
        if self.output_bf16:
            drafts = [draft.to(torch.bfloat16) for draft in drafts]
        output = SimpleNamespace(forecast=drafts[-1], draft_forecasts=torch.stack(drafts, dim=1),
                                 reasoning_steps=reasoning_steps, initial_forecast=first)
        self.traces.append(output)
        return output


def test_two_forwards_fixed_lead_push_history_and_calendar_rollover():
    model, original = AnalyticForecaster(), batch()
    before = {key: value.clone() for key, value in original.items() if torch.is_tensor(value)}
    result = training_two_step(model, original, reasoning_steps=2)
    assert result.forecasts.shape == (1, 2, 1, 3, 4)
    assert len(model.inputs) == 2 and model.kwargs == [(2, False), (2, False)]
    assert all(set(inputs) == set(DECLARED_MODEL_INPUTS) for inputs in model.inputs)
    assert all(torch.equal(inputs["lead_time_hours"], torch.tensor([6.])) for inputs in model.inputs)
    second = model.inputs[1]
    assert torch.equal(second["coarse_history"][:, -1], result.forecasts[:, 0])
    assert torch.equal(second["coarse_history"][:, 0], original["coarse_history"][:, -1])
    assert second["init_calendar_year"].tolist() == [2017]
    assert second["init_day_of_year"].tolist() == [1]
    assert second["init_utc_hour"].tolist() == [0.]
    assert second["coarse_history"].grad_fn is not None
    for key, value in before.items():
        assert torch.equal(original[key], value)


def test_targets_poison_only_their_loss_not_forecasts_or_model_inputs():
    original = batch()
    baseline = training_two_step(AnalyticForecaster(), original, 2)
    future = dict(original, future_target=torch.full_like(original["future_target"], 1000.))
    model = AnalyticForecaster()
    poisoned = training_two_step(model, future, 2)
    torch.testing.assert_close(baseline.forecasts, poisoned.forecasts, rtol=0, atol=0)
    torch.testing.assert_close(baseline.l6, poisoned.l6, rtol=0, atol=0)
    assert poisoned.l12 > baseline.l12 and poisoned.loss > baseline.loss
    early = training_two_step(AnalyticForecaster(), dict(original, atmos_target=original["atmos_target"] + 100), 2)
    torch.testing.assert_close(early.forecasts, baseline.forecasts, rtol=0, atol=0)
    torch.testing.assert_close(early.l12, baseline.l12, rtol=0, atol=0)
    assert early.l6 != baseline.l6
    assert all(not {"atmos_target", "future_target", "process_targets", "init_year"} & set(inputs)
               for inputs in model.inputs)


def test_preserves_normalized_initial_all_draft_supervision_on_each_axis():
    model, example = AnalyticForecaster(), batch()
    result = training_two_step(model, example, 3, .5)
    expected = [deep_supervised_forecast_mse(output.draft_forecasts.float(), target.float(),
                                            example["latitude"], final_weight=2.)
                for output, target in zip(model.traces, (example["atmos_target"], example["future_target"]))]
    torch.testing.assert_close(result.l6, expected[0])
    torch.testing.assert_close(result.l12, expected[1])
    torch.testing.assert_close(result.loss, expected[0] + .5 * expected[1])
    single = training_one_step(AnalyticForecaster(), example, 3)
    assert single.forecasts.shape[1] == 1 and single.l12 is None
    torch.testing.assert_close(single.l6, result.l6, rtol=0, atol=0)
    assert result.loss.dtype == torch.float32


def test_full_gradient_matches_analytic_and_detach_counterexample():
    model, example = AnalyticForecaster(), batch()
    result = training_two_step(model, example, 2)
    first = model.traces[0].forecast
    propagated, = torch.autograd.grad(result.l12, first, retain_graph=True)
    assert propagated.abs().sum() > 0
    result.loss.backward()
    # Each step's draft is x*a**j, j=1..3. Physical 2 uses x*a**(3+j).
    weights = torch.linspace(1., 2., 3)
    weights /= weights.sum()
    expected = sum(float(weight) * 2 * power * 2. ** (2 * power - 1)
                   for weight, power in zip(weights, range(1, 4)))
    expected += .5 * sum(float(weight) * 2 * power * 2. ** (2 * power - 1)
                         for weight, power in zip(weights, range(4, 7)))
    assert float(model.scale.grad) == pytest.approx(expected, rel=1e-6)
    detached = AnalyticForecaster(detach_physical=True)
    severed = training_two_step(detached, example, 2)
    no_path, = torch.autograd.grad(severed.l12, detached.traces[0].forecast, allow_unused=True, retain_graph=True)
    assert no_path is None
    severed.loss.backward()
    assert float(detached.scale.grad) != pytest.approx(expected)


@pytest.mark.parametrize("year,day,hour,next_day", [(2016, 60, 18, 61), (2017, 59, 18, 60)])
def test_calendar_leap_and_nonleap_days(year, day, hour, next_day):
    sample = batch()
    sample.update(init_calendar_year=torch.tensor([float(year)]), init_day_of_year=torch.tensor([float(day)]),
                  init_utc_hour=torch.tensor([float(hour)]))
    model = AnalyticForecaster()
    training_two_step(model, sample, 1)
    assert model.inputs[1]["init_day_of_year"].item() == next_day
    assert model.inputs[1]["init_utc_hour"].item() == 0.
    assert model.inputs[1]["init_calendar_year"].item() == year


def test_optional_year_does_not_read_old_year_and_rejects_ambiguous_carry():
    sample = batch()
    sample.pop("init_calendar_year")
    sample["init_day_of_year"] = torch.tensor([10.])
    model = AnalyticForecaster()
    training_two_step(model, sample, 1)
    assert "init_calendar_year" not in model.inputs[1] and "init_year" not in model.inputs[1]
    assert model.inputs[1]["init_day_of_year"].item() == 11
    sample["init_day_of_year"] = torch.tensor([365.])
    with pytest.raises(ValueError, match="init_calendar_year required"):
        training_two_step(AnalyticForecaster(), sample, 1)


@pytest.mark.parametrize("fault", ["history_nan", "history_shape", "target_nan", "future_shape",
                                  "forecast_nan", "forecast_shape", "drafts_nan", "no_drafts",
                                  "missing_spacetime", "lead", "lambda", "k", "eval", "no_grad"])
def test_invalid_training_values_or_paths_fail(fault):
    model, sample, kwargs = AnalyticForecaster(), batch(), {}
    original = model.forward
    if fault == "history_nan":
        sample["coarse_history"][0, 0, 0, 0, 0] = float("nan")
    elif fault == "history_shape":
        sample["coarse_history"] = sample["coarse_history"][0]
    elif fault == "target_nan":
        sample["atmos_target"].fill_(float("nan"))
    elif fault == "future_shape":
        sample["future_target"] = sample["future_target"][:, :, :1]
    elif fault in ("forecast_nan", "forecast_shape", "drafts_nan", "no_drafts"):
        def corrupt(*args, **options):
            output = original(*args, **options)
            if fault == "forecast_nan":
                output.forecast = output.forecast * float("nan")
            elif fault == "forecast_shape":
                output.forecast = output.forecast[:, :, :1]
            elif fault == "drafts_nan":
                output.draft_forecasts = output.draft_forecasts * float("nan")
            else:
                del output.draft_forecasts
            return output
        model.forward = corrupt
    elif fault == "missing_spacetime":
        sample.pop("latitude")
    elif fault == "lead":
        sample["lead_time_hours"] = torch.tensor([12.])
    elif fault == "lambda":
        kwargs["lambda12"] = float("nan")
    elif fault == "k":
        kwargs["reasoning_steps"] = True
    elif fault == "eval":
        model.eval()
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        if fault == "no_grad":
            with torch.no_grad():
                training_two_step(model, sample)
        else:
            training_two_step(model, sample, **kwargs)


def tiny_spec(kind="process", *, parent_detach=False):
    spec = dict(in_channels=17, out_channels=17, history_steps=2, dim=8, depth=1, heads=2,
                window_size=2, patch_size=2, dropout=0., spacetime_inputs=True,
                default_reasoning_steps=2, detach_between_steps=parent_detach)
    if kind == "process":
        spec.update(anchored_processes=2, free_processes=1)
    else:
        spec.update(latent_tokens=2)
    return spec


@pytest.mark.parametrize("kind", ["process", "generic"])
@pytest.mark.parametrize("bf16", [False, True])
def test_actual_tiny_17_channel_full_k_and_physical_graph_cpu(kind, bf16):
    torch.manual_seed(12)
    model = make_model(kind, tiny_spec(kind, parent_detach=True)).train()
    inputs, outputs = [], []
    def observe(_model, args, output):
        inputs.append(args[0])
        outputs.append(output)
    model.register_forward_hook(observe)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        result = training_two_step(model, batch(17), 2)
    assert len(outputs) == 2
    assert torch.equal(inputs[1]["coarse_history"][:, -1], outputs[0].forecast)
    physical, = torch.autograd.grad(result.l12, outputs[0].forecast, retain_graph=True)
    internal, = torch.autograd.grad(outputs[0].forecast.square().mean(), outputs[0].initial_forecast,
                                    retain_graph=True)
    assert physical.abs().sum() > 0 and internal.abs().sum() > 0
    result.loss.backward()
    assert result.loss.dtype == torch.float32 and torch.isfinite(result.loss)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_bf16_prediction_is_not_rounded_when_pushed_into_fp32_history():
    model = AnalyticForecaster(output_bf16=True)
    result = training_two_step(model, batch(), 1)
    assert result.forecasts.dtype == torch.bfloat16
    assert model.inputs[1]["coarse_history"].dtype == torch.float32
    assert torch.equal(model.inputs[1]["coarse_history"][:, -1], result.forecasts[:, 0])
    result.loss.backward()
    assert torch.isfinite(model.scale.grad) and result.loss.dtype == torch.float32


def fixture_manifest(tmp_path, *, times=None, time_ranges=None, validation=False):
    train = pd.date_range("2016-01-01", periods=8, freq="6h") if times is None else pd.DatetimeIndex(times)
    stamps = train.append(pd.date_range("2017-01-01", periods=8, freq="6h")) if validation else train
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="w", zarr_format=2)
    root.attrs.update(schema_version=1, build_complete=True, time_unit="ns",
                      channels=[f"fixture_{index}" for index in range(17)], units=["K"] * 17,
                      native_grid_spacing_deg=.25, split_years={"train": [2016], "val": [2017], "test": [2018]},
                      normalization_years=[2016], source="synthetic-engineering-fixture-not-weather")
    if time_ranges is not None:
        root.attrs.update(split_mode="time_ranges", split_time_ranges=time_ranges)
    values = np.sin(np.arange(len(stamps) * 17 * 3 * 4, dtype=np.float32) / 37).reshape(len(stamps), 17, 3, 4)
    for key, array in (("state", values), ("time_ns", stamps.as_unit("ns").asi8),
                       ("latitude", np.array([40., 39.75, 39.5], dtype=np.float32)),
                       ("longitude", np.array([115., 115.25, 115.5, 115.75], dtype=np.float32)),
                       ("normalization_mean", np.zeros(17, dtype=np.float32)),
                       ("normalization_std", np.ones(17, dtype=np.float32))):
        root.create_array(key, data=array)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "BUILD_COMPLETE.json").write_text(json.dumps({"schema_version": 1, "build_complete": True}))
    lookup = {stamp.value: position for position, stamp in enumerate(stamps)}
    paths = {}
    for split, selected in (("train", train), ("val", stamps[len(train):])):
        if not len(selected):
            continue
        records = []
        for stamp in selected:
            previous, following = stamp - pd.Timedelta(hours=6), stamp + pd.Timedelta(hours=6)
            if previous not in selected or following not in selected:
                continue
            if time_ranges is not None:
                ranges = time_ranges[split]
                if not any(all(pd.Timestamp(start) <= point < pd.Timestamp(stop) for point in (previous, stamp, following))
                           for start, stop in ranges):
                    continue
            records.append({"sample_id": f"{split}-{lookup[stamp.value]:04d}", "split": split,
                            "store_path": "../store.zarr", "history_indices": [lookup[previous.value], lookup[stamp.value]],
                            "history_times": [previous.isoformat(), stamp.isoformat()], "init_time": stamp.isoformat(),
                            "target_index": lookup[following.value], "target_time": following.isoformat(), "lead_time_hours": 6.})
        paths[split] = manifests / f"{split}.jsonl"
        paths[split].write_text("".join(json.dumps(record) + "\n" for record in records))
    return paths, root


def dataset_for(manifest):
    summary = preflight_training_windows(manifest)
    return ZarrAutoregressiveDataset(manifest, expected_exclusions=summary["excluded_sample_ids"]), summary


def test_exact_t12_preflight_exclusions_and_no_adjacent_index_fallback(tmp_path):
    times = pd.date_range("2016-01-01", periods=9, freq="6h").delete(4)
    paths, root = fixture_manifest(tmp_path, times=times)
    summary = preflight_training_windows(paths["train"])
    assert summary["input_windows"] > summary["usable_windows"] > 0
    assert {row["reason"] for row in summary["exclusions"]} == {"missing_exact_t12"}
    assert summary["test_read"] is False and summary["state_fields_read"] is False
    with pytest.raises(ValueError, match="predeclared exactly"):
        ZarrAutoregressiveDataset(paths["train"])
    dataset, _ = dataset_for(paths["train"])
    for position, future in dataset.windows:
        record = dataset.reader.records[position]
        assert int(root["time_ns"][future["target_index"]]) == pd.Timestamp(record["init_time"]).value + 12 * 3600 * 10**9
    with pytest.raises(ValueError, match="predeclared exactly"):
        ZarrAutoregressiveDataset(paths["train"], expected_exclusions=summary["excluded_sample_ids"] + ["invented"])
    sample = dataset[0]
    assert sample["future_target"].shape == (17, 3, 4) and sample["init_calendar_year"].item() == 2016
    assert pd.Timestamp(sample["future_valid_time"]) - pd.Timestamp(sample["init_time"]) == pd.Timedelta(hours=12)


def test_exact_windows_reject_split_gap_and_tampered_record(tmp_path):
    ranges = {"train": [["2016-01-01", "2016-01-02"]],
              "val": [["2016-01-02", "2016-01-02T06:00:00"]],
              "test": [["2016-01-02T06:00:00", "2016-01-03"]]}
    paths, _ = fixture_manifest(tmp_path, time_ranges=ranges)
    summary = preflight_training_windows(paths["train"])
    assert summary["exclusions"][-1]["reason"] == "t12_outside_train_split"
    with pytest.raises(ValueError, match="predeclared exactly"):
        ZarrAutoregressiveDataset(paths["train"])
    records = [json.loads(line) for line in paths["train"].read_text().splitlines()]
    records[0]["target_index"] += 1
    paths["train"].write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="timestamps do not match"):
        preflight_training_windows(paths["train"])


def test_future_nonfinite_and_sealed_manifest_rejected(tmp_path):
    paths, root = fixture_manifest(tmp_path)
    dataset, _ = dataset_for(paths["train"])
    future = dataset.windows[0][1]["target_index"]
    root["state"][future] = np.full((17, 3, 4), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="nonfinite future"):
        dataset[0]
    with pytest.raises(ValueError, match="train.jsonl"):
        preflight_training_windows(tmp_path / "test.jsonl")


def runner_options(tmp_path, *, kind="process", bf16=False):
    paths, _ = fixture_manifest(tmp_path / "fixture")
    identity, _ = dataset_identity(paths["train"])
    spec = tiny_spec(kind)
    torch.manual_seed(23)
    model = make_model(kind, spec)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    contract = {"kind": kind, "model": spec, "data_identity": identity,
                "source_sha256": canonical_digest("explicit-synthetic-fixture-source"),
                "protocol_sha256": canonical_digest("frozen-engineering-fixture-protocol"),
                "initialization": {"format": "explicit-seeded-fixture", "seed": 23},
                "autoregression": {"excluded_sample_ids": preflight_training_windows(paths["train"])["excluded_sample_ids"]}}
    options = dict(model=model, contract=contract, parent_weights=weights, steps=2, updates=3,
                   seed=7, warmup=1, checkpoint_every=1, bf16=bf16)
    return paths, options


@pytest.mark.parametrize("kind,mode,bf16", [("process", "two_step", False), ("process", "two_step", True),
                                          ("generic", "l6", False), ("generic", "two_step", True)])
def test_runner_cpu_checkpoint_and_full_bptt_report(tmp_path, kind, mode, bf16):
    paths, options = runner_options(tmp_path, kind=kind, bf16=bf16)
    checkpoint, report = fine_tune(paths["train"], tmp_path / "run", mode=mode, **options)
    saved = load_checkpoint(checkpoint)
    restored = make_model(saved["contract"]["kind"], saved["contract"]["model"])
    restored.load_state_dict(saved["model"], strict=True)
    assert saved["updates"] == report["selected_update"] == 3
    assert saved["signature"] == report["signature"]
    assert report["optimization"] == "full-bptt" and report["internal_k_detach"] is False
    assert report["physical_step_detach"] is False and report["parent_optimizer_imported"] is False
    assert report["scientific_claim"] is False and report["limitations"] and report["test_read"] is False
    assert report["objective"] == "deep_supervised_latitude_area_mse" and report["internal_deep_supervision"] is True
    assert report["contract"]["autoregression"]["physical_steps"] == (1 if mode == "l6" else 2)
    assert all(np.isfinite(row["loss"]) and np.isfinite(row["gradient_norm"]) for row in report["losses"])
    if mode == "l6":
        assert all(row["l12"] is None and row["loss"] == row["l6"] for row in report["losses"])
    else:
        assert all(row["loss"] == pytest.approx(row["l6"] + .5 * row["l12"], rel=1e-6) for row in report["losses"])
    assert json.loads((tmp_path / "run" / "training_report.json").read_text())["signature"] == saved["signature"]


def interrupted_checkpoint(tmp_path, paths, options):
    """Owned CPU child exits after publication, without a fabricated failure reset."""
    invocation = {key: value for key, value in options.items() if key not in ("model", "parent_weights")}
    config = tmp_path / "cpu_child.json"
    config.write_text(json.dumps(invocation), encoding="utf-8")
    output = tmp_path / "interrupted"
    source = """
import json, os, sys, torch
from pathlib import Path
from training.r7_experiment import make_model
import training.r7_autoregressive_runner as runner
options=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
torch.set_num_threads(1)
torch.manual_seed(23)
model=make_model(options['contract']['kind'], options['contract']['model'])
original=runner._publish_checkpoint
def publish_and_exit(*args, **kwargs):
    original(*args, **kwargs)
    os._exit(23)
runner._publish_checkpoint=publish_and_exit
runner.fine_tune(sys.argv[2], sys.argv[3], model=model, **options)
"""
    result = subprocess.run([sys.executable, "-B", "-c", source, str(config), str(paths["train"]), str(output)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90)
    assert result.returncode == 23, result.stdout + result.stderr
    assert not (output / "failed_attempt.json").exists()
    return output / "update_0000001.pt"


@pytest.mark.parametrize("bf16", [False, True])
def test_same_endpoint_own_intermediate_resume_is_bitwise_cpu(tmp_path, bf16):
    paths, options = runner_options(tmp_path, bf16=bf16)
    complete, _ = fine_tune(paths["train"], tmp_path / "complete", **options)
    intermediate = interrupted_checkpoint(tmp_path, paths, options)
    resumed, report = fine_tune(paths["train"], intermediate.parent, resume=intermediate, **options)
    expected, actual = load_checkpoint(complete), load_checkpoint(resumed)
    assert report["resumed_from_updates"] == 1 and report["updates_this_run"] == 2
    for name in expected["model"]:
        assert torch.equal(expected["model"][name], actual["model"][name])
    assert torch.equal(expected["rng"]["torch"], actual["rng"]["torch"])
    for key, value in expected["optimizer"]["state"].items():
        for name, tensor in value.items():
            assert torch.equal(tensor, actual["optimizer"]["state"][key][name])


def test_failed_attempt_not_revived_and_resume_is_not_endpoint_extension(tmp_path):
    paths, options = runner_options(tmp_path)
    checkpoint, _ = fine_tune(paths["train"], tmp_path / "run", **options)
    with pytest.raises(FileExistsError, match="cannot be revived"):
        fine_tune(paths["train"], tmp_path / "run", resume=checkpoint, **options)
    with pytest.raises(FileExistsError):
        fine_tune(paths["train"], tmp_path / "run", **options)
    own_intermediate = tmp_path / "run" / "update_0000001.pt"
    with pytest.raises(ValueError, match="same output"):
        fine_tune(paths["train"], tmp_path / "different", resume=own_intermediate, **options)
    saved = load_checkpoint(own_intermediate)
    assert saved["contract"]["total_updates"] == 3
    assert saved["contract"]["optimization"] == "full-bptt"


def test_checkpoint_identity_tamper_nonfinite_and_deadline_fail(tmp_path, monkeypatch):
    import training.r7_autoregressive_runner as module
    paths, options = runner_options(tmp_path)
    original_update = module._update
    calls = 0
    def fail_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected new failed attempt")
        return original_update(*args, **kwargs)
    monkeypatch.setattr(module, "_update", fail_after_first)
    with pytest.raises(RuntimeError, match="injected"):
        fine_tune(paths["train"], tmp_path / "failed", **options)
    failed_checkpoint = tmp_path / "failed" / "update_0000001.pt"
    with pytest.raises(FileExistsError, match="cannot be revived"):
        fine_tune(paths["train"], tmp_path / "failed", resume=failed_checkpoint, **options)
    assert json.loads((tmp_path / "failed" / "failed_attempt.json").read_text())["resume_permitted"] is False
    tampered = torch.load(failed_checkpoint, weights_only=True)
    tampered["contract"]["total_updates"] = 4
    torch.save(tampered, tmp_path / "tampered.pt")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_checkpoint(tmp_path / "tampered.pt")
    with pytest.raises(RuntimeError, match="deadline exceeded"):
        fine_tune(paths["train"], tmp_path / "deadline", deadline=time.perf_counter() - 1, **options)
    assert not (tmp_path / "deadline").exists()
    bad = deepcopy(options)
    bad["parent_weights"][next(iter(bad["parent_weights"]))].fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite model"):
        fine_tune(paths["train"], tmp_path / "bad", **bad)


def test_new_contract_pins_and_fresh_parent_state_only(tmp_path):
    paths, options = runner_options(tmp_path)
    for key in ("data_identity", "source_sha256", "protocol_sha256"):
        wrong = deepcopy(options)
        wrong["contract"][key] = "0" * 64 if key == "data_identity" else "not-a-digest"
        with pytest.raises(ValueError):
            fine_tune(paths["train"], tmp_path / key, **wrong)
    no_origin = deepcopy(options)
    no_origin["contract"].pop("initialization")
    with pytest.raises(ValueError, match="initialization report"):
        fine_tune(paths["train"], tmp_path / "no-origin", **no_origin)
    stale = deepcopy(options)
    stale["contract"]["model"]["detach_between_steps"] = True
    with pytest.raises(ValueError, match="full BPTT"):
        fine_tune(paths["train"], tmp_path / "stale", **stale)
    optimizer_import = dict(options, parent_weights={"model": options["parent_weights"], "optimizer": {}})
    with pytest.raises(ValueError, match="never an optimizer"):
        fine_tune(paths["train"], tmp_path / "optimizer", **optimizer_import)


def test_resume_same_endpoint_optimizer_and_cursor_tamper_rejected(tmp_path):
    paths, options = runner_options(tmp_path)
    intermediate = interrupted_checkpoint(tmp_path, paths, options)
    with pytest.raises(ValueError, match="endpoint cannot be extended"):
        fine_tune(paths["train"], intermediate.parent, resume=intermediate, **dict(options, updates=4))
    original = torch.load(intermediate, weights_only=True)
    wrong_cursor = deepcopy(original)
    wrong_cursor["cursor"] += 1
    torch.save(wrong_cursor, intermediate)
    with pytest.raises(ValueError, match="epoch/cursor differs"):
        fine_tune(paths["train"], intermediate.parent, resume=intermediate, **options)
    assert (intermediate.parent / "failed_attempt.json").exists()
    with pytest.raises(FileExistsError, match="cannot be revived"):
        fine_tune(paths["train"], intermediate.parent, resume=intermediate, **options)
    from training.r7_autoregressive_runner import _restore_optimizer
    for fault in ("rate", "moment", "step"):
        corrupted = deepcopy(original)
        if fault == "rate":
            corrupted["optimizer"]["param_groups"][0]["lr"] = 1000.
        else:
            state = next(iter(corrupted["optimizer"]["state"].values()))
            state["exp_avg" if fault == "moment" else "step"].fill_(float("nan") if fault == "moment" else 2.)
        optimizer = torch.optim.AdamW(options["model"].parameters(), lr=1e-4, weight_decay=1e-4)
        with pytest.raises(ValueError, match="optimizer"):
            _restore_optimizer(options["model"], optimizer, corrupted, 5, 1, original["contract"])


def test_output_safety_and_new_driver_child_root(tmp_path, monkeypatch):
    import training.r7_autoregressive_runner as module
    paths, options = runner_options(tmp_path)
    destination = tmp_path / "destination"
    destination.mkdir()
    link = tmp_path / "linked-parent"
    link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink ancestors"):
        fine_tune(paths["train"], link / "run", **options)
    for output in (tmp_path / "data" / "raw" / "run", tmp_path / "legacy_v6" / "run"):
        with pytest.raises(ValueError, match="protected"):
            fine_tune(paths["train"], output, **options)
        assert not output.exists()
    root = tmp_path / "sandbox"
    monkeypatch.setattr(module, "ROOT", root)
    protocol = {"protocol_sha256": options["contract"]["protocol_sha256"]}
    attempt = root / "outputs" / "r7_74_autoregressive_20261003_attempt01"
    attempt.mkdir(parents=True)
    (attempt / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    child = attempt / "seed7" / "training" / "two_step"
    assert module._output_path(child, options["contract"]) == child
    with pytest.raises(ValueError, match="nested original"):
        module._output_path(root / "outputs" / "old_run" / "training", options["contract"])
    (attempt / "attempt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed/failed"):
        module._output_path(child, options["contract"])


def test_standard_evaluate_local_accepts_child_checkpoint_on_val_fixture(tmp_path):
    from training.r7_evaluate import evaluate_local
    paths, _ = fixture_manifest(tmp_path / "fixture", validation=True)
    identity, _ = dataset_identity(paths["train"])
    spec = tiny_spec()
    contract = {"kind": "process", "model": spec, "data_identity": identity,
                "source_sha256": canonical_digest("fixture-source"), "protocol_sha256": canonical_digest("fixture-protocol"),
                "initialization": {"format": "explicit-cpu-fixture"},
                "autoregression": {"excluded_sample_ids": preflight_training_windows(paths["train"])["excluded_sample_ids"]}}
    checkpoint, _ = fine_tune(paths["train"], tmp_path / "run", model=make_model("process", spec),
                              contract=contract, updates=1, warmup=0, steps=2)
    result = evaluate_local(paths["val"], output_dir=tmp_path / "val_metrics", checkpoint=checkpoint,
                            lead_hours=(6, 12), max_samples=1, device_name="cpu", normalized=True)
    assert result["split"] == "val" and result["n_evaluated"] == 1
    assert result["training_identity"] == identity and result["checkpoint_sha256"]
