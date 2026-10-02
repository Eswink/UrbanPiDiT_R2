"""Synthetic analytic/contract tests only; no stored manifests or weather data."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from data.preprocess.process_diagnostics import PROCESS_DIAGNOSTIC_NAMES, compute_process_diagnostic_vector
from data.preprocess.r7_process_scale_sidecar import fit_process_scale, normalize_process_diagnostics
from training.r7_process_forecast_losses import (
    auxiliary_process_loss_one_step, process_diagnostic_loss, process_forecast_coreasoning_loss,
)
from training.r7_process_supervision import ProcessDiagnosticContext
from training.r7_recursive_losses import deep_supervised_forecast_mse

CHANNEL_NAMES = (
    "mslp", "t2m", "u10", "v10", "z500", "t850", "t500", "t250",
    "u850", "u500", "u250", "v850", "v500", "v250", "q850", "q500", "q250",
)
HOUR_NS = 3_600_000_000_000


def make_context(*, degenerate=None):
    scales = np.array([.003, 2e-5, 2e-5, .0003, 2e-8, 2e-8, 30., 12.])
    raw = (np.random.default_rng(41).normal(size=(32, 8)) + 2.) * scales
    if degenerate is not None:
        raw[:, degenerate] = 0.
    metadata = fit_process_scale(raw)
    mean, std = torch.zeros(17), torch.ones(17)
    for name, center, spread in (("mslp", 100000., 1000.), ("t850", 280., 8.),
                                 ("t500", 250., 8.), ("q850", .01, .002),
                                 ("u850", 8., 2.), ("v850", 2., 2.),
                                 ("u500", 20., 3.), ("v500", 5., 3.)):
        mean[CHANNEL_NAMES.index(name)] = center
        std[CHANNEL_NAMES.index(name)] = spread
    return ProcessDiagnosticContext(metadata, mean, std, CHANNEL_NAMES)


def make_batch(size=2):
    generator = torch.Generator().manual_seed(11)
    init = torch.arange(size, dtype=torch.int64) * 24 * HOUR_NS + 1_450_000_000_000_000_000
    lead = torch.arange(size, dtype=torch.float32) * 6 + 6
    valid = init + lead.long() * HOUR_NS
    return {
        "coarse_history": torch.randn(size, 2, 17, 4, 5, generator=generator),
        "atmos_target": torch.randn(size, 17, 4, 5, generator=generator),
        "latitude": torch.tensor([50., 42., 33., 20.]).expand(size, -1).clone(),
        "longitude": torch.tensor([90., 96., 105., 118., 130.]).expand(size, -1).clone(),
        "init_time_ns": init, "valid_time_ns": valid, "target_time_ns": valid.clone(),
        "history_time_ns": torch.stack((init - 6 * HOUR_NS, init), 1),
        "lead_time_hours": lead, "split": ["train"] * size,
        "channel_names": CHANNEL_NAMES,
    }


def make_output(batch, steps=3):
    draft = torch.stack([batch["coarse_history"][:, -1] + .1 * k for k in range(steps + 1)], 1)
    return SimpleNamespace(draft_forecasts=draft.detach().clone().requires_grad_(),
                           process_predictions=torch.zeros(draft.shape[0], steps, 8, requires_grad=True))


def test_fixed_train_inverse_matches_physical_numpy_oracle():
    context, batch = make_context(), make_batch()
    fields = context.fields_from_batch(batch)
    expected = []
    for sample in batch["coarse_history"][:, -1]:
        physical = sample.float() * context.normalization_std[:, None, None] + context.normalization_mean[:, None, None]
        raw = compute_process_diagnostic_vector(physical.numpy(), CHANNEL_NAMES,
                                               batch["latitude"][0].numpy(), batch["longitude"][0].numpy())
        expected.append(normalize_process_diagnostics(raw, context.metadata))
    torch.testing.assert_close(fields.input_diagnostics, torch.tensor(np.stack(expected)), rtol=3e-4, atol=8e-4)
    assert fields.input_diagnostics.dtype == torch.float32
    assert not fields.input_diagnostics.requires_grad
    assert not fields.future_diagnostic_targets.requires_grad
    assert context.normalization_mean.shape == context.normalization_std.shape == (17,)


def test_future_and_draft_fields_have_distinct_sources_and_same_valid_time():
    context, batch = make_context(), make_batch()
    out = make_output(batch, 4)
    fields = context.fields_from_output(batch, out)
    assert fields.input_diagnostics.shape == fields.future_diagnostic_targets.shape == (2, 8)
    assert fields.draft_diagnostics.shape == (2, 5, 8)
    assert torch.equal(fields.input_diagnostics_time_ns, batch["init_time_ns"])
    assert torch.equal(fields.future_diagnostic_targets_time_ns, batch["valid_time_ns"])
    assert torch.equal(fields.draft_diagnostics_time_ns, batch["valid_time_ns"][:, None].expand(2, 5))
    assert not torch.equal(fields.input_diagnostics, fields.future_diagnostic_targets)
    assert fields.draft_diagnostics.requires_grad
    fields.draft_diagnostics.sum().backward()
    assert torch.isfinite(out.draft_forecasts.grad).all()
    assert out.process_predictions.grad is None


def test_val_poison_cannot_refit_fixed_statistics_or_input_diagnostics():
    context, batch = make_context(), make_batch()
    before = deepcopy(context.metadata)
    first = context.fields_from_batch(batch, training=False)
    poisoned = dict(batch, atmos_target=torch.full_like(batch["atmos_target"], float("nan")),
                    future_diagnostic_targets=object(), target_time_ns=object(), split=["val", "val"])
    second = context.fields_from_batch(poisoned, training=False)
    assert torch.equal(first.input_diagnostics, second.input_diagnostics)
    assert first.future_diagnostic_targets is second.future_diagnostic_targets is None
    assert context.metadata == before
    snapshot = context.metadata
    snapshot["dimensionless_mean"][0] = 1e9
    assert context.metadata == before


def test_input_only_does_not_read_future_label_and_uses_repaired_scale_mse():
    context, batch = make_context(), make_batch()
    batch.pop("atmos_target")
    batch.pop("target_time_ns")
    out = make_output(batch)
    fields = context.fields_from_batch(batch, need_future=False)
    result = process_diagnostic_loss(out.process_predictions, fields, input_diagnostic_weight=.1)
    expected = F.mse_loss(out.process_predictions, fields.input_diagnostics[:, None].expand_as(out.process_predictions))
    assert torch.equal(result.input_diagnostics, expected)
    assert torch.equal(result.total, expected * .1)
    assert fields.future_diagnostic_targets is None
    single = auxiliary_process_loss_one_step(batch, out.draft_forecasts[:, 1], out.process_predictions[:, 0],
                                            process_supervision_context=context, prepared_fields=fields,
                                            input_diagnostic_weight=.1)
    single_expected = F.mse_loss(out.process_predictions[:, :1], fields.input_diagnostics[:, None]) * .1
    assert torch.equal(single.total, single_expected)
    # Different K reduction lengths are numerically equivalent, not a bitwise
    # replay promise; the legacy disabled-path test below remains bitwise.
    torch.testing.assert_close(single.total, result.total)


def test_future_and_draft_mse_semantics_and_one_step_mean_match():
    context, batch = make_context(), make_batch()
    out = make_output(batch, 3)
    prepared = context.fields_from_batch(batch)
    full = process_forecast_coreasoning_loss(batch, out, process_weight=0., forecast_weight=0.,
                                            future_diagnostic_weight=.05, draft_diagnostic_weight=.05,
                                            process_supervision_context=context)
    drafts = context.with_drafts(prepared, out.draft_forecasts[:, 1:], batch["latitude"], batch["longitude"])
    target = prepared.future_diagnostic_targets[:, None].expand(2, 3, 8)
    torch.testing.assert_close(full.future_diagnostic_targets, F.mse_loss(out.process_predictions, target))
    torch.testing.assert_close(full.draft_diagnostics, F.mse_loss(drafts.draft_diagnostics, target))
    singles = [auxiliary_process_loss_one_step(batch, out.draft_forecasts[:, k + 1], out.process_predictions[:, k],
                                             process_supervision_context=context, prepared_fields=prepared,
                                             future_diagnostic_weight=.05, draft_diagnostic_weight=.05)
               for k in range(3)]
    torch.testing.assert_close(full.total, torch.stack([item.total for item in singles]).mean())
    assert torch.equal(full.process, torch.zeros(()))
    assert torch.equal(full.input_diagnostics, torch.zeros(()))


def test_draft_loss_gradient_is_forecast_owned_not_target_or_process_head():
    context, batch = make_context(), make_batch()
    batch["atmos_target"].requires_grad_()
    batch["coarse_history"].requires_grad_()
    out = make_output(batch, 2)
    result = process_forecast_coreasoning_loss(batch, out, process_weight=0., forecast_weight=0.,
                                             draft_diagnostic_weight=.1, process_supervision_context=context)
    result.draft_diagnostics.backward()
    assert out.draft_forecasts.grad is not None
    assert torch.isfinite(out.draft_forecasts.grad).all()
    assert out.draft_forecasts.grad[:, 1:].abs().sum() > 0
    assert torch.equal(out.draft_forecasts.grad[:, 0], torch.zeros_like(out.draft_forecasts.grad[:, 0]))
    assert batch["atmos_target"].grad is None
    assert out.process_predictions.grad is None


def test_zero_variance_mask_is_zero_for_values_and_gradients():
    context, batch = make_context(degenerate=4), make_batch()
    normalized = batch["atmos_target"].clone().requires_grad_()
    values = context.diagnostics(normalized, batch["latitude"], batch["longitude"])
    assert context.metadata["degenerate"][4] is True
    assert context.metadata["active_mask"][4] is False
    assert torch.equal(values[..., 4], torch.zeros(2))
    values[..., 4].sum().backward()
    assert torch.equal(normalized.grad, torch.zeros_like(normalized))
    fields = context.fields_from_batch(batch)
    prediction = torch.zeros(2, 2, 8, requires_grad=True)
    prediction.data[..., 4] = 1e10
    loss = process_diagnostic_loss(prediction, fields, future_diagnostic_weight=.1)
    loss.total.backward()
    assert torch.equal(prediction.grad[..., 4], torch.zeros(2, 2))
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("key", ["names", "units", "dimensionless_std", "active_mask", "relative_floor", "fit_split", "schema"])
def test_bad_scale_metadata_is_explicitly_rejected(key):
    context = make_context()
    bad = context.metadata
    if key in ("names", "units"):
        bad[key] = list(reversed(bad[key]))
    elif key == "dimensionless_std":
        bad[key][0] = 0.
    elif key == "active_mask":
        bad[key][0] = False
    elif key == "relative_floor":
        bad[key][0] = float("nan")
    elif key == "fit_split":
        bad[key] = "val"
    else:
        bad[key] = "unknown"
    with pytest.raises(ValueError):
        ProcessDiagnosticContext(bad, context.normalization_mean, context.normalization_std, CHANNEL_NAMES)


def test_all_zero_variance_and_unsafe_numeric_normalization_are_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        fit_process_scale(np.zeros((4, 8)))
    context = make_context()
    bad = context.metadata
    bad["physical_unit_scale"][0] = 1e-300
    with pytest.raises(ValueError, match="float32"):
        ProcessDiagnosticContext(bad, context.normalization_mean, context.normalization_std, CHANNEL_NAMES)


@pytest.mark.parametrize("kind", ["short_mean", "zero_std", "nan_mean", "duplicate_names", "missing_names"])
def test_fixed_atmospheric_normalization_contract_rejects_mismatch(kind):
    context = make_context()
    mean, std, names = context.normalization_mean.clone(), context.normalization_std.clone(), list(CHANNEL_NAMES)
    if kind == "short_mean":
        mean = mean[:16]
    elif kind == "zero_std":
        std[0] = 0.
    elif kind == "nan_mean":
        mean[0] = float("nan")
    elif kind == "duplicate_names":
        names[1] = names[0]
    else:
        names[names.index("q850")] = "q_unknown"
    with pytest.raises(ValueError):
        ProcessDiagnosticContext(context.metadata, mean, std, names)


@pytest.mark.parametrize("key", ["init_time_ns", "valid_time_ns", "history_time_ns", "lead_time_hours", "target_time_ns"])
def test_future_supervision_never_synthesizes_missing_timestamps(key):
    context, batch = make_context(), make_batch()
    batch.pop(key)
    with pytest.raises(ValueError, match="timestamp|target_time_ns"):
        context.fields_from_batch(batch)


@pytest.mark.parametrize("kind", ["history_end", "future_valid", "float_time", "lead", "cadence", "draft_k"])
def test_timestamp_misalignment_is_rejected(kind):
    context, batch = make_context(), make_batch()
    out = make_output(batch, 2)
    if kind == "history_end":
        batch["history_time_ns"][:, -1] += HOUR_NS
    elif kind == "future_valid":
        batch["target_time_ns"] += HOUR_NS
    elif kind == "float_time":
        batch["init_time_ns"] = batch["init_time_ns"].double()
    elif kind == "lead":
        batch["lead_time_hours"] += 6.
    elif kind == "cadence":
        batch["history_time_ns"] = batch["history_time_ns"].flip(1)
    else:
        out.draft_time_ns = batch["valid_time_ns"][:, None] + torch.arange(3)[None, :] * 6 * HOUR_NS
    with pytest.raises(ValueError, match="timestamp|time|lead|cadence|init"):
        context.fields_from_output(batch, out)


@pytest.mark.parametrize("split", [None, "val", "test", ["train", "val"], ["train"]])
def test_future_diagnostics_are_explicit_train_split_only(split):
    context, batch = make_context(), make_batch()
    if split is None:
        batch.pop("split")
    else:
        batch["split"] = split
    with pytest.raises(ValueError, match="split|train"):
        context.fields_from_batch(batch)


def test_dataloader_collated_channel_names_are_checked():
    context, batch = make_context(), make_batch()
    batch["channel_names"] = [(name, name) for name in CHANNEL_NAMES]
    context.fields_from_batch(batch)
    batch["channel_names"][0] = ("t850", "t850")
    with pytest.raises(ValueError, match="names/order"):
        context.fields_from_batch(batch)


@pytest.mark.parametrize("key", ["latitude", "longitude", "atmos_target"])
def test_enabled_context_missing_geo_or_target_rejected(key):
    context, batch = make_context(), make_batch()
    batch.pop(key)
    with pytest.raises(ValueError):
        context.fields_from_batch(batch)


@pytest.mark.parametrize("key", ["coarse_history", "atmos_target"])
def test_context_rejects_nonfinite_states(key):
    context, batch = make_context(), make_batch()
    batch[key].flatten()[0] = float("nan")
    # Only last history goes into D, so poison the last actual state.
    if key == "coarse_history":
        batch[key][:, -1, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        context.fields_from_batch(batch)


def test_default_and_aux_off_loss_is_bitwise_legacy_regression():
    generator = torch.Generator().manual_seed(5)
    batch = {"atmos_target": torch.randn(2, 3, 4, 5, generator=generator),
             "process_targets": torch.randn(2, 2, generator=generator)}
    out = SimpleNamespace(draft_forecasts=torch.randn(2, 4, 3, 4, 5, generator=generator),
                          process_predictions=torch.randn(2, 3, 2, generator=generator))
    forecast = deep_supervised_forecast_mse(out.draft_forecasts, batch["atmos_target"], final_weight=2.)
    old_process = F.mse_loss(out.process_predictions[..., :2], batch["process_targets"][..., :2].unsqueeze(1).expand(-1, 3, -1))
    old_total = 1. * forecast + .1 * old_process
    default = process_forecast_coreasoning_loss(batch, out)
    assert torch.equal(default.forecast, forecast)
    assert torch.equal(default.process, old_process)
    assert torch.equal(default.total, old_total)
    batch["process_targets"] = torch.full((2, 99), float("nan"))
    off = process_forecast_coreasoning_loss(batch, out, process_weight=0.)
    assert torch.equal(off.total, 1. * forecast + 0. * forecast.new_zeros(()))
    assert torch.equal(off.process, forecast.new_zeros(()))


@pytest.mark.parametrize("key", ["forecast_weight", "process_weight", "final_weight", "input_diagnostic_weight", "future_diagnostic_weight", "draft_diagnostic_weight"])
@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_every_loss_weight_must_be_finite_nonnegative(key, value):
    context, batch = make_context(), make_batch()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        process_forecast_coreasoning_loss(batch, make_output(batch), **{key: value})


@pytest.mark.parametrize("key", ["input_diagnostic_weight", "future_diagnostic_weight", "draft_diagnostic_weight"])
def test_legacy_and_new_tasks_cannot_both_be_enabled(key):
    context, batch = make_context(), make_batch()
    with pytest.raises(ValueError, match="legacy.*new"):
        process_forecast_coreasoning_loss(batch, make_output(batch), process_supervision_context=context, **{key: .1})
    with pytest.raises(ValueError, match="context"):
        process_forecast_coreasoning_loss(batch, make_output(batch), process_weight=0., **{key: .1})


@pytest.mark.parametrize("width", [7, 9])
def test_enabled_new_loss_never_truncates_width(width):
    context, batch = make_context(), make_batch()
    out = make_output(batch)
    out.process_predictions = torch.zeros(2, 3, width)
    with pytest.raises(ValueError, match="exactly.*no width truncation"):
        process_forecast_coreasoning_loss(batch, out, process_weight=0., future_diagnostic_weight=.1,
                                         process_supervision_context=context)
    fields = context.fields_from_batch(batch)
    fields = replace(fields, future_diagnostic_targets=fields.future_diagnostic_targets[..., :7])
    with pytest.raises(ValueError, match="exact"):
        process_diagnostic_loss(torch.zeros(2, 3, 8), fields, future_diagnostic_weight=.1)


def test_enabled_legacy_loss_never_truncates_width():
    batch = {"atmos_target": torch.zeros(2, 3, 4, 5), "process_targets": torch.zeros(2, 9)}
    out = SimpleNamespace(draft_forecasts=torch.zeros(2, 2, 3, 4, 5), process_predictions=torch.zeros(2, 1, 8))
    with pytest.raises(ValueError, match="width.*no truncation"):
        process_forecast_coreasoning_loss(batch, out)


def test_diagnostic_field_name_and_timestamp_poison_is_rejected():
    context, batch = make_context(), make_batch()
    fields = context.fields_from_batch(batch)
    with pytest.raises(ValueError, match="names"):
        process_diagnostic_loss(torch.zeros(2, 2, 8), replace(fields, names=tuple(reversed(PROCESS_DIAGNOSTIC_NAMES))),
                                future_diagnostic_weight=.1)
    bad = replace(fields, future_diagnostic_targets_time_ns=batch["init_time_ns"])
    with pytest.raises(ValueError, match="timestamp misalignment"):
        process_diagnostic_loss(torch.zeros(2, 2, 8), bad, future_diagnostic_weight=.1)


def test_cached_fields_cannot_bypass_target_or_time_contract():
    context, batch = make_context(), make_batch()
    fields = context.fields_from_batch(batch)
    out = make_output(batch)
    batch.pop("atmos_target")
    with pytest.raises(ValueError, match="target"):
        auxiliary_process_loss_one_step(batch, out.draft_forecasts[:, 1], out.process_predictions[:, 0],
                                        process_supervision_context=context, prepared_fields=fields,
                                        future_diagnostic_weight=.1)
