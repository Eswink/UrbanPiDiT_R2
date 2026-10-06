"""CPU engineering counterproofs only; synthetic tmp_path fields are not weather.

No GPU, network, real dataset, scientific acceptance or runner experiments.
Fixture stores explicitly declare scientific_claim=False.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from test_r7_m4_autoregressive_rollout import (
    AnalyticForecaster, batch, cpu_threads, fixture_manifest, tiny_spec,
)
from data.r7_long_rollout_dataset import ZarrLongRolloutDataset, preflight_long_rollout_windows
from data.r7_store import HOUR_NS
from data.r7_zarr_dataset import ZarrAtmosWindowDataset
from model.r7_halting import DECLARED_MODEL_INPUTS
from training.r7_autoregressive_rollout import training_one_step, training_two_step
from training.r7_experiment import make_model
from training.r7_long_rollout import (
    LongTrainingRolloutOutput, training_long_rollout, validate_physical_weights,
)
from training.r7_recursive_losses import deep_supervised_forecast_mse


def long_fixture(tmp_path, *, times=None, time_ranges=None):
    times = pd.date_range("2016-01-01", periods=16, freq="6h") if times is None else times
    paths, root = fixture_manifest(tmp_path, times=times, time_ranges=time_ranges)
    root.attrs["scientific_claim"] = False
    return paths["train"], root


def records_at(manifest):
    return [json.loads(line) for line in manifest.read_text().splitlines()]


def write_records(manifest, records):
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))


def dataset_for(manifest, physical_steps=12):
    summary = preflight_long_rollout_windows(manifest, physical_steps=physical_steps)
    dataset = ZarrLongRolloutDataset(manifest, physical_steps=physical_steps,
        expected_exclusions=summary["excluded_sample_ids"], expected_window_sha256=summary["window_sha256"])
    return dataset, summary


def physical_batch(physical_steps=12, channels=1):
    sample = batch(channels)
    targets = [sample["atmos_target"]]
    targets.extend(sample["future_target"].clone() for _ in range(physical_steps - 1))
    sample["physical_targets"] = torch.stack(targets, dim=1)
    return sample


class FakeState:
    def __init__(self, shape):
        self.shape, self.reads = shape, []

    def __getitem__(self, index):
        self.reads.append(index)
        raise AssertionError("state field read forbidden during metadata preflight")


class FakeRoot:
    def __init__(self, root):
        self.root, self.attrs = root, root.attrs
        self.state = FakeState(root["state"].shape)

    def __contains__(self, name):
        return name in self.root

    def __getitem__(self, name):
        return self.state if name == "state" else self.root[name]


def test_default_72h_pin_normalization_and_compatibility_targets(tmp_path):
    manifest, root = long_fixture(tmp_path)
    root["normalization_mean"][:] = np.arange(17, dtype=np.float32) / 10
    root["normalization_std"][:] = np.arange(17, dtype=np.float32) + 1
    dataset, summary = dataset_for(manifest)
    assert summary == preflight_long_rollout_windows(manifest)
    assert summary["input_windows"] == 14 and summary["usable_windows"] == len(dataset) == 3
    assert summary["physical_steps"] == 12 and summary["step_hours"] == 6
    assert summary["test_read"] is False and summary["state_fields_read"] is False
    assert len(summary["excluded_sample_ids"]) == 11
    assert dataset.manifest == manifest and dataset.summary == summary
    assert [record["sample_id"] for record in dataset.records] == [pin["sample_id"] for pin in summary["windows"]]
    identity = {key: summary[key] for key in ("physical_steps", "step_hours", "windows", "exclusions")}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == summary["window_sha256"]
    pin, sample = summary["windows"][0], dataset[0]
    assert pin["manifest_index"] == 0 and pin["history_indices"] == [0, 1]
    assert pin["target_indices"] == list(range(2, 14))
    assert sample["physical_targets"].shape == (12, 17, 3, 4)
    assert torch.equal(sample["physical_targets"][0], sample["atmos_target"])
    assert torch.equal(sample["physical_targets"][1], sample["future_target"])
    assert sample["init_calendar_year"].item() == 2016
    assert sample["physical_valid_times"] == pin["target_times"]
    assert pd.Timestamp(pin["target_times"][-1]) - pd.Timestamp(pin["init_time"]) == pd.Timedelta(hours=72)
    mean, std = np.asarray(root["normalization_mean"][:]), np.asarray(root["normalization_std"][:])
    for step, index in enumerate(pin["target_indices"]):
        expected = (np.asarray(root["state"][index]) - mean[:, None, None]) / std[:, None, None]
        torch.testing.assert_close(sample["physical_targets"][step], torch.from_numpy(expected), rtol=0, atol=0)
    assert root.attrs["scientific_claim"] is False


def test_preflight_and_construction_never_read_state_fields(tmp_path, monkeypatch):
    manifest, root = long_fixture(tmp_path)
    proxy = FakeRoot(root)
    monkeypatch.setattr(ZarrAtmosWindowDataset, "_store", lambda _reader, _record: proxy)
    dataset, summary = dataset_for(manifest)
    assert summary["usable_windows"] == 3 and proxy.state.reads == []
    with pytest.raises(AssertionError, match="state field read forbidden"):
        dataset[0]
    assert proxy.state.reads == [0]


@pytest.mark.parametrize("steps", [1, 2, 5, 12, 20])
def test_arbitrary_positive_horizon_and_single_step_compatibility(tmp_path, steps):
    manifest, _ = long_fixture(tmp_path, times=pd.date_range("2016-01-01", periods=steps + 4, freq="6h"))
    dataset, summary = dataset_for(manifest, steps)
    assert len(dataset) == 3 and summary["physical_steps"] == steps
    sample = dataset[0]
    assert sample["physical_targets"].shape[0] == steps
    assert ("future_target" in sample) == (steps >= 2)
    assert torch.equal(sample["physical_targets"][0], sample["atmos_target"])


@pytest.mark.parametrize("steps", [True, False, 0, -1, 1.5, "12", None])
def test_invalid_physical_steps_fail_before_opening_manifest(tmp_path, steps):
    for entry in (preflight_long_rollout_windows, ZarrLongRolloutDataset):
        with pytest.raises(ValueError, match="positive integer"):
            entry(tmp_path / "train.jsonl", physical_steps=steps)


def test_digest_and_exact_ordered_exclusions_are_enforced(tmp_path):
    manifest, _ = long_fixture(tmp_path)
    summary = preflight_long_rollout_windows(manifest)
    ids = summary["excluded_sample_ids"]
    for declared in ((), ids[:-1], ids + [ids[0]], ids + ["invented"], list(reversed(ids))):
        with pytest.raises(ValueError, match="predeclared exactly"):
            ZarrLongRolloutDataset(manifest, expected_exclusions=declared)
    for declared in ("sample-id", b"sample-id", None, [3]):
        with pytest.raises(ValueError, match="collection of sample IDs"):
            ZarrLongRolloutDataset(manifest, expected_exclusions=declared)
    with pytest.raises(ValueError, match="window_sha256 mismatch"):
        ZarrLongRolloutDataset(manifest, expected_exclusions=ids, expected_window_sha256="0" * 64)
    assert dataset_for(manifest)[0].summary["window_sha256"] == summary["window_sha256"]
    records = records_at(manifest)
    records[0]["sample_id"] = "renamed-synthetic-fixture"
    write_records(manifest, records)
    assert preflight_long_rollout_windows(manifest)["window_sha256"] != summary["window_sha256"]
    with pytest.raises(ValueError, match="window_sha256 mismatch"):
        ZarrLongRolloutDataset(manifest, expected_exclusions=ids, expected_window_sha256=summary["window_sha256"])


def test_timestamp_lookup_pins_nonadjacent_indices(tmp_path):
    manifest, _ = long_fixture(tmp_path, times=pd.date_range("2016-01-01", periods=64, freq="3h"))
    records = [record for record in records_at(manifest)
               if pd.Timestamp(record["init_time"]).value % (6 * HOUR_NS) == 0]
    write_records(manifest, records)
    dataset, summary = dataset_for(manifest)
    pin = summary["windows"][0]
    assert pin["history_indices"] == [0, 2] and pin["target_indices"] == list(range(4, 28, 2))
    assert torch.equal(dataset[0]["physical_targets"][0], dataset[0]["atmos_target"])
    assert summary["window_sha256"] != preflight_long_rollout_windows(manifest, physical_steps=11)["window_sha256"]


def test_missing_intermediate_is_excluded_even_with_terminal_available(tmp_path):
    times = pd.date_range("2016-01-01", periods=32, freq="6h").delete(4)
    manifest, root = long_fixture(tmp_path, times=times)
    dataset, summary = dataset_for(manifest)
    exclusion = summary["exclusions"][0]
    assert exclusion["sample_id"] == "train-0001" and exclusion["reason"] == "missing_exact_t18"
    terminal = pd.Timestamp(exclusion["init_time"]) + pd.Timedelta(hours=72)
    assert terminal.value in set(np.asarray(root["time_ns"][:]).tolist())
    assert exclusion["sample_id"] not in {record["sample_id"] for record in dataset.records}
    lookup = {int(stamp): index for index, stamp in enumerate(root["time_ns"][:])}
    for pin in summary["windows"]:
        init_ns = pd.Timestamp(pin["init_time"]).value
        assert pin["target_indices"] == [lookup[init_ns + step * 6 * HOUR_NS] for step in range(1, 13)]


def test_halfopen_train_ownership_overrides_year_membership(tmp_path):
    ranges = {"train": [["2016-01-01", "2016-01-05"]],
              "val": [["2016-01-05", "2016-01-06"]], "test": [["2016-01-06", "2016-01-09"]]}
    manifest, _ = long_fixture(tmp_path, times=pd.date_range("2016-01-01", periods=32, freq="6h"), time_ranges=ranges)
    dataset, summary = dataset_for(manifest)
    assert len(dataset) == 3
    assert all(row["required_target_time"] == "2016-01-05T00:00:00" for row in summary["exclusions"])
    assert all(row["reason"].endswith("outside_train_split") for row in summary["exclusions"])
    assert all(pd.Timestamp(time) < pd.Timestamp(ranges["train"][0][1])
               for pin in summary["windows"] for time in pin["target_times"])


def test_unowned_intermediate_cannot_be_bridged_by_later_train_frame(tmp_path):
    ranges = {"train": [["2016-01-01", "2016-01-02"], ["2016-01-02T06:00:00", "2016-01-08"]],
              "val": [["2016-01-08", "2016-01-09"]], "test": [["2016-01-09", "2016-01-10"]]}
    manifest, _ = long_fixture(tmp_path, times=pd.date_range("2016-01-01", periods=32, freq="6h"), time_ranges=ranges)
    dataset, summary = dataset_for(manifest)
    first = summary["exclusions"][0]
    assert first["sample_id"] == "train-0001" and first["reason"] == "t18_outside_train_split"
    assert "train-0001" not in {record["sample_id"] for record in dataset.records}


def test_year_ownership_and_explicit_calendar_producer(tmp_path):
    times = pd.date_range("2016-12-27", periods=40, freq="6h")
    manifest, _ = long_fixture(tmp_path / "year-owned", times=times)
    records = [record for record in records_at(manifest)
               if all(pd.Timestamp(time).year == 2016 for time in record["history_times"] + [record["target_time"]])]
    write_records(manifest, records)
    _, summary = dataset_for(manifest)
    assert all(row["reason"].endswith("outside_train_split") for row in summary["exclusions"])
    ranges = {"train": [["2016-12-30", "2017-01-04"]],
              "val": [["2017-01-04", "2017-01-05"]], "test": [["2017-01-05", "2017-01-07"]]}
    manifest, _ = long_fixture(tmp_path / "range-owned", times=pd.date_range("2016-12-30", periods=32, freq="6h"),
                               time_ranges=ranges)
    dataset, _ = dataset_for(manifest)
    sample = next(dataset[index] for index, record in enumerate(dataset.records)
                  if record["init_time"] == "2016-12-31T18:00:00")
    assert sample["init_calendar_year"].item() == 2016 and sample["init_day_of_year"].item() == 366
    assert pd.Timestamp(sample["physical_valid_times"][0]).year == 2017


@pytest.mark.parametrize("fault", ["manifest_complete", "missing_marker", "store_complete", "channels",
                                  "history_count", "cadence", "lead", "target_index", "split", "duplicate_id"])
def test_bad_completed_store_or_base_records_fail(tmp_path, fault):
    manifest, root = long_fixture(tmp_path)
    records = records_at(manifest)
    if fault == "manifest_complete":
        (manifest.parent / "BUILD_COMPLETE.json").write_text(json.dumps({"schema_version": 1, "build_complete": False}))
    elif fault == "missing_marker":
        (manifest.parent / "BUILD_COMPLETE.json").unlink()
    elif fault == "store_complete":
        root.attrs["build_complete"] = False
    elif fault == "channels":
        root.attrs["channels"] = ["fixture_channel"]
    elif fault == "history_count":
        records[1]["history_indices"] = records[1]["history_indices"][-1:]
        records[1]["history_times"] = records[1]["history_times"][-1:]
    elif fault == "cadence":
        records[1]["history_indices"][0] = 0
        records[1]["history_times"][0] = records[0]["history_times"][0]
    elif fault == "lead":
        records[0].update(lead_time_hours=12, target_index=3, target_time=records[1]["target_time"])
    elif fault == "target_index":
        records[0]["target_index"] += 1
    elif fault == "split":
        records[0]["split"] = "val"
    else:
        records[1]["sample_id"] = records[0]["sample_id"]
    write_records(manifest, records)
    with pytest.raises(ValueError):
        preflight_long_rollout_windows(manifest)


def test_utc_alignment_and_nonempty_complete_window_required(tmp_path):
    manifest, _ = long_fixture(tmp_path / "unaligned", times=pd.date_range("2016-01-01T01:00", periods=16, freq="6h"))
    with pytest.raises(ValueError, match="UTC-aligned"):
        preflight_long_rollout_windows(manifest)
    manifest, _ = long_fixture(tmp_path / "too-short", times=pd.date_range("2016-01-01", periods=8, freq="6h"))
    summary = preflight_long_rollout_windows(manifest)
    assert summary["usable_windows"] == 0
    with pytest.raises(ValueError, match="no complete exact"):
        ZarrLongRolloutDataset(manifest, expected_exclusions=summary["excluded_sample_ids"])


def test_sealed_and_renamed_manifests_rejected_before_reader(tmp_path, monkeypatch):
    def forbidden_reader(*_args, **_kwargs):
        raise AssertionError("sealed manifest must never reach the reader")
    monkeypatch.setattr(ZarrAtmosWindowDataset, "__init__", forbidden_reader)
    for name in ("test.jsonl", "val.jsonl", "renamed.jsonl"):
        with pytest.raises(ValueError, match="train.jsonl"):
            preflight_long_rollout_windows(tmp_path / name)
    (tmp_path / "train.jsonl").symlink_to(tmp_path / "test.jsonl")
    with pytest.raises(ValueError, match="train.jsonl"):
        preflight_long_rollout_windows(tmp_path / "train.jsonl")


@pytest.mark.parametrize("location", ["history", "first", "last"])
def test_nonfinite_fields_raise_but_preflight_stays_metadata_only(tmp_path, location):
    manifest, root = long_fixture(tmp_path)
    dataset, summary = dataset_for(manifest)
    pin = summary["windows"][0]
    index = pin["history_indices"][0] if location == "history" else pin["target_indices"][0 if location == "first" else -1]
    root["state"][index] = np.full((17, 3, 4), np.nan, dtype=np.float32)
    assert preflight_long_rollout_windows(manifest) == summary
    with pytest.raises(ValueError, match="nonfinite"):
        dataset[0]


@pytest.mark.parametrize("weights", [(), [], [0], [0, 0], True, None, 1., "1,0.5", b"1",
                                      [True], ["1"], [np.bool_(True)], [-1], [float("nan")],
                                      [float("inf")], [complex(1, 0)], [10**400]])
def test_invalid_physical_weights_fail(weights):
    with pytest.raises(ValueError, match="physical_weights"):
        validate_physical_weights(weights)
    model = AnalyticForecaster()
    with pytest.raises(ValueError, match="physical_weights"):
        training_long_rollout(model, physical_batch(2), physical_weights=weights)
    assert model.inputs == []


def test_weight_validation_and_one_step_match_existing_control():
    assert validate_physical_weights([np.float64(1), np.int64(0), .5]) == (1., 0., .5)
    sample, model = physical_batch(1), AnalyticForecaster()
    result = training_long_rollout(model, sample, 2, physical_weights=(1,))
    old = training_one_step(AnalyticForecaster(), sample, 2)
    assert isinstance(result, LongTrainingRolloutOutput) and result.per_step_losses.shape == (1,)
    assert result.l12 is None and torch.equal(result.l6, result.loss)
    torch.testing.assert_close(result.forecasts, old.forecasts, rtol=0, atol=0)
    torch.testing.assert_close(result.loss, old.loss, rtol=0, atol=0)


def test_twelve_steps_push_forecasts_offsets_fixed_lead_and_year_calendar():
    model, sample = AnalyticForecaster(), physical_batch()
    before = {key: value.clone() for key, value in sample.items() if torch.is_tensor(value)}
    result = training_long_rollout(model, sample, 1, physical_weights=(1,) * 12)
    assert result.forecasts.shape == (1, 12, 1, 3, 4) and result.per_step_losses.shape == (12,)
    assert model.kwargs == [(1, False)] * 12 and result.loss.dtype == torch.float32
    for step, inputs in enumerate(model.inputs):
        stamp = pd.Timestamp("2016-12-31T18:00:00") + pd.Timedelta(hours=6 * step)
        assert set(inputs) == set(DECLARED_MODEL_INPUTS)
        assert inputs["lead_time_hours"].tolist() == [6.]
        assert inputs["init_calendar_year"].tolist() == [stamp.year]
        assert inputs["init_day_of_year"].tolist() == [stamp.dayofyear]
        assert inputs["init_utc_hour"].tolist() == [stamp.hour]
        assert torch.equal(inputs["history_offsets_hours"], sample["history_offsets_hours"])
        if step:
            assert torch.equal(inputs["coarse_history"][:, -1], model.traces[step - 1].forecast)
            assert torch.equal(inputs["coarse_history"][:, 0], model.inputs[step - 1]["coarse_history"][:, -1])
            assert inputs["coarse_history"].grad_fn is not None
    assert all(torch.equal(sample[key], value) for key, value in before.items())
    assert torch.equal(result.l6, result.per_step_losses[0]) and torch.equal(result.l12, result.per_step_losses[1])


def test_target_poison_never_changes_forecasts_or_whitelisted_inputs():
    sample = physical_batch(4)
    baseline = training_long_rollout(AnalyticForecaster(), sample, 2, physical_weights=(1,) * 4)
    poison = deepcopy(sample)
    poison["physical_targets"][:, 2] += 1000
    poison["atmos_target"].fill_(9999)
    poison["future_target"].fill_(-9999)
    model = AnalyticForecaster()
    actual = training_long_rollout(model, poison, 2, physical_weights=(1,) * 4)
    torch.testing.assert_close(actual.forecasts, baseline.forecasts, rtol=0, atol=0)
    torch.testing.assert_close(actual.per_step_losses[[0, 1, 3]], baseline.per_step_losses[[0, 1, 3]], rtol=0, atol=0)
    assert actual.per_step_losses[2] != baseline.per_step_losses[2]
    assert all(not {"physical_targets", "atmos_target", "future_target", "process_targets", "init_year"} & set(inputs)
               for inputs in model.inputs)


def test_every_initial_and_k_draft_is_supervised_on_every_physical_step():
    model, sample = AnalyticForecaster(), physical_batch(4)
    weights = (1., 0., .25, .5)
    result = training_long_rollout(model, sample, 3, physical_weights=weights)
    expected = torch.stack([deep_supervised_forecast_mse(trace.draft_forecasts.float(), sample["physical_targets"][:, step],
                              sample["latitude"], final_weight=2.) for step, trace in enumerate(model.traces)])
    torch.testing.assert_close(result.per_step_losses, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.loss, sum(weight * loss for weight, loss in zip(weights, expected)))
    for step, trace in enumerate(model.traces):
        gradient, = torch.autograd.grad(result.per_step_losses[step], trace.draft_forecasts, retain_graph=True)
        assert all(gradient[:, draft].abs().sum() > 0 for draft in range(4))


def test_later_loss_reaches_first_forecast_history_and_detach_counterexample():
    sample, model = physical_batch(), AnalyticForecaster()
    sample["coarse_history"].requires_grad_()
    torch.nn.init.constant_(model.scale, 1.25)
    weights = (0.,) * 11 + (1.,)
    result = training_long_rollout(model, sample, 2, physical_weights=weights)
    first_gradient, history_gradient = torch.autograd.grad(result.per_step_losses[-1],
        (model.traces[0].forecast, sample["coarse_history"]), retain_graph=True)
    assert first_gradient.abs().sum() > 0 and history_gradient[:, -1].abs().sum() > 0
    result.loss.backward()
    k_weights = torch.linspace(1., 2., 3)
    k_weights /= k_weights.sum()
    expected = sum(float(weight) * 2 * power * 1.25 ** (2 * power - 1)
                   for weight, power in zip(k_weights, range(34, 37)))
    assert model.scale.grad.item() == pytest.approx(expected, rel=3e-6)
    detached = AnalyticForecaster(detach_physical=True)
    torch.nn.init.constant_(detached.scale, 1.25)
    severed = training_long_rollout(detached, sample, 2, physical_weights=weights)
    disconnected, = torch.autograd.grad(severed.per_step_losses[-1], detached.traces[0].forecast,
                                         allow_unused=True, retain_graph=True)
    assert disconnected is None or torch.count_nonzero(disconnected) == 0
    severed.loss.backward()
    assert detached.scale.grad.item() != pytest.approx(expected, rel=3e-6)


def test_two_weight_case_exactly_matches_analytic_two_step_gradients():
    long_model, old_model = AnalyticForecaster(), AnalyticForecaster()
    long_sample, old_sample = physical_batch(2), physical_batch(2)
    long_sample["coarse_history"].requires_grad_()
    old_sample["coarse_history"].requires_grad_()
    actual = training_long_rollout(long_model, long_sample, 2, physical_weights=(1, .5))
    expected = training_two_step(old_model, old_sample, 2, .5)
    for observed, reference in ((actual.forecasts, expected.forecasts), (actual.loss, expected.loss),
                                (actual.l6, expected.l6), (actual.l12, expected.l12)):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    actual.loss.backward()
    expected.loss.backward()
    assert torch.equal(long_model.scale.grad, old_model.scale.grad)
    assert torch.equal(long_sample["coarse_history"].grad, old_sample["coarse_history"].grad)


@pytest.mark.parametrize("fault", ["eval", "no_grad", "inference", "k", "history_shape", "history_nan", "lead",
                                  "targets_missing", "targets_shape", "targets_steps", "targets_dtype", "targets_nan",
                                  "targets_device", "no_drafts", "detached", "forecast_grid", "forecast_nan", "drafts_nan"])
def test_invalid_targets_training_modes_and_outputs_fail(fault):
    sample, model, reasoning = physical_batch(3), AnalyticForecaster(), 2
    forward = model.forward
    if fault == "eval":
        model.eval()
    elif fault == "k":
        reasoning = True
    elif fault == "history_shape":
        sample["coarse_history"] = sample["coarse_history"][0]
    elif fault == "history_nan":
        sample["coarse_history"].fill_(float("nan"))
    elif fault == "lead":
        sample["lead_time_hours"].fill_(12)
    elif fault == "targets_missing":
        sample.pop("physical_targets")
    elif fault == "targets_shape":
        sample["physical_targets"] = sample["physical_targets"][..., :1, :]
    elif fault == "targets_steps":
        sample["physical_targets"] = sample["physical_targets"][:, :2]
    elif fault == "targets_dtype":
        sample["physical_targets"] = sample["physical_targets"].double()
    elif fault == "targets_nan":
        sample["physical_targets"][:, -1].fill_(float("nan"))
    elif fault == "targets_device":
        sample["physical_targets"] = sample["physical_targets"].to("meta")
    elif fault not in ("no_grad", "inference"):
        def corrupt(*args, **kwargs):
            output = forward(*args, **kwargs)
            if fault == "no_drafts":
                del output.draft_forecasts
            elif fault == "detached":
                output.forecast = output.forecast.detach()
            elif fault == "forecast_grid":
                output.forecast = output.forecast[..., :1, :]
            elif fault == "forecast_nan":
                output.forecast = output.forecast * float("nan")
            else:
                output.draft_forecasts = output.draft_forecasts * float("nan")
            return output
        model.forward = corrupt
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        with torch.no_grad() if fault == "no_grad" else torch.inference_mode() if fault == "inference" else torch.enable_grad():
            training_long_rollout(model, sample, reasoning, physical_weights=(1,) * 3)
    if fault.startswith("targets_"):
        assert model.inputs == []


@pytest.mark.parametrize("low_history", [False, True])
def test_history_push_promotes_dtype_without_rounding_forecasts(low_history):
    model, sample = AnalyticForecaster(output_bf16=not low_history), physical_batch(4)
    if low_history:
        sample["coarse_history"] = sample["coarse_history"].bfloat16()
        sample["physical_targets"] = sample["physical_targets"].bfloat16()
        forward = model.forward
        def widened(*args, **kwargs):
            output = forward(*args, **kwargs)
            output.forecast = output.forecast.float()
            output.draft_forecasts = output.draft_forecasts.float()
            return output
        model.forward = widened
    result = training_long_rollout(model, sample, 1, physical_weights=(1,) * 4)
    for step in range(1, 4):
        assert model.inputs[step]["coarse_history"].dtype == torch.float32
        assert torch.equal(model.inputs[step]["coarse_history"][:, -1], model.traces[step - 1].forecast)
    result.loss.backward()
    assert torch.isfinite(model.scale.grad) and result.loss.dtype == torch.float32


@pytest.mark.parametrize("kind", ["process", "generic"])
@pytest.mark.parametrize("bf16", [False, True])
def test_actual_tiny_models_several_steps_full_internal_and_physical_bptt(kind, bf16):
    torch.manual_seed(12)
    model = make_model(kind, dict(tiny_spec(kind, parent_detach=True), known_context_inputs=True)).train()
    inputs, traces = [], []
    def observe(_model, args, output):
        inputs.append(args[0])
        traces.append(output)
    model.register_forward_hook(observe)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        result = training_long_rollout(model, physical_batch(3, 17), 2, physical_weights=(0, 0, 1))
    assert len(traces) == 3 and result.forecasts.shape == (1, 3, 17, 3, 4)
    assert all(torch.equal(inputs[step]["coarse_history"][:, -1], traces[step - 1].forecast) for step in (1, 2))
    physical, = torch.autograd.grad(result.per_step_losses[-1], traces[0].forecast, retain_graph=True)
    internal, = torch.autograd.grad(traces[0].forecast.square().mean(), traces[0].initial_forecast, retain_graph=True)
    assert physical.abs().sum() > 0 and internal.abs().sum() > 0
    result.loss.backward()
    assert result.loss.dtype == torch.float32 and torch.isfinite(result.loss)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["process", "generic"])
def test_two_weight_case_matches_actual_tiny_model_losses_forecasts_and_gradients(kind):
    torch.manual_seed(19)
    long_model = make_model(kind, tiny_spec(kind, parent_detach=True)).train()
    old_model = deepcopy(long_model)
    sample = physical_batch(2, 17)
    actual = training_long_rollout(long_model, sample, 2, physical_weights=(1, .5))
    expected = training_two_step(old_model, sample, 2, .5)
    for observed, reference in ((actual.forecasts, expected.forecasts), (actual.loss, expected.loss),
                                (actual.l6, expected.l6), (actual.l12, expected.l12)):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    actual.loss.backward()
    expected.loss.backward()
    for (name, parameter), (old_name, old_parameter) in zip(long_model.named_parameters(), old_model.named_parameters()):
        assert name == old_name and (parameter.grad is None) == (old_parameter.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, old_parameter.grad, rtol=2e-6, atol=2e-7)
