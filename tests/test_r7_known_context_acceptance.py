"""ADR 0035 CPU synthetic acceptance; no real store, clock, GPU or network.

The same causal assertions must fail when their measured edge is cut. Producers
only build engineering fixtures under pytest tmp_path. Known history offsets are
signed, ordered and independent of forecast lead; physical cadence is explicit.
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta
from pathlib import Path
import socket

import numpy as np
import pytest
import torch

from data.r7_evaluation import ZarrRolloutDataset
from data.r7_store import HOUR_NS, history_offsets_hours_from_ns
from data.r7_zarr_dataset import ZarrAtmosWindowDataset
from data.schema import validate_forecast_sample
from data.synthetic_atmos import SyntheticAtmosDataset
from model.known_context_r7 import (
    HISTORY_CONTEXT_FIELDS, known_context_features, require_history_offsets,
)
from model.r7_halting import AdaptiveProcessForecaster, forecast_inputs
from model.r7_rollout import autoregressive_rollout, rollout_model_input
from model.weather_forecaster_r7 import NativeAtmosForecaster
from training.r7_autoregressive_rollout import training_two_step
from training.r7_streaming import backward_streamed_truncated

NATIVE = dict(in_channels=3, out_channels=3, history_steps=2, dim=16, depth=1,
              heads=2, window_size=2, patch_size=2, dropout=0., spacetime_inputs=True)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("known-context acceptance forbids network")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _batch(hw=(5, 7), size=2):
    generator = torch.Generator().manual_seed(5)
    return {"coarse_history": torch.randn(size, 2, 3, *hw, generator=generator),
            "atmos_target": torch.randn(size, 3, *hw, generator=generator),
            "future_target": torch.randn(size, 3, *hw, generator=generator),
            "lead_time_hours": torch.full((size,), 6.),
            "latitude": torch.linspace(-40, 40, hw[0]),
            "longitude": torch.linspace(-179, 179, hw[1]),
            "init_calendar_year": torch.tensor([2024, 2023])[:size],
            "init_day_of_year": torch.tensor([60., 365.])[:size],
            "init_utc_hour": torch.tensor([0., 18.])[:size],
            "history_offsets_hours": torch.tensor([[-6., 0.], [-12., 0.]])[:size]}


def _native(enabled=True, **kwargs):
    torch.manual_seed(11)
    return NativeAtmosForecaster(**NATIVE, known_context_inputs=enabled, **kwargs)


def _features(data, mode="fields", periodic=False):
    h, w = data["coarse_history"].shape[-2:]
    return known_context_features(data, history=data["coarse_history"],
        token_hw=((h + 1) // 2, (w + 1) // 2), patch_size=2,
        periodic_width=periodic, field_mode=mode)


def _oracle(data, periodic=False):
    rows, columns = data["coarse_history"].shape[-2:]
    result = []
    for sample in range(data["coarse_history"].shape[0]):
        init = datetime(int(data["init_calendar_year"][sample]), 1, 1) + timedelta(
            days=float(data["init_day_of_year"][sample]) - 1,
            hours=float(data["init_utc_hour"][sample]))
        tokens = []
        for row in range((rows + 1) // 2):
            for column in range((columns + 1) // 2):
                indices = [2 * column, 2 * column + 1]
                indices = [index % columns if periodic else min(index, columns - 1) for index in indices]
                longitude = np.deg2rad([float(data["longitude"][index]) for index in indices])
                features = []
                for offset in data["history_offsets_hours"][sample].tolist():
                    date = init + timedelta(hours=offset)
                    year_days = (datetime(date.year + 1, 1, 1) - datetime(date.year, 1, 1)).days
                    annual = 2 * np.pi * (date.timetuple().tm_yday - 1 + date.hour / 24) / year_days
                    angles = 2 * np.pi * date.hour / 24 + longitude
                    features.extend([np.sin(annual), np.cos(annual), np.sin(angles).mean(),
                                     np.cos(angles).mean(), offset / 24])
                valid = init + timedelta(hours=float(data["lead_time_hours"][sample]))
                angles = 2 * np.pi * valid.hour / 24 + longitude
                tokens.append(features + [np.sin(angles).mean(), np.cos(angles).mean()])
        result.append(tokens)
    return torch.tensor(result, dtype=torch.float32)


def test_default_off_old8_expression_rng_and_last_module_are_unchanged(monkeypatch):
    plain, rng = _native(False).eval(), torch.random.get_rng_state().clone()
    implicit = NativeAtmosForecaster(**NATIVE).eval()  # not seed-matched; no new module
    assert not implicit.known_context_inputs and not hasattr(implicit, "known_context")
    enabled = _native(True).eval()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert list(enabled._modules)[-1] == "known_context"
    assert enabled.spacetime.net[0].in_features == plain.spacetime.net[0].in_features == 8
    added = set(enabled.state_dict()) - set(plain.state_dict())
    assert added == {"known_context.projection.weight", "known_context.projection.bias"}
    for name, value in plain.state_dict().items():
        assert torch.equal(value, enabled.state_dict()[name]), name
    data = _batch()
    data["history_offsets_hours"] = object()  # off must not validate/read it
    data["init_year"] = torch.tensor(float("nan"))
    with torch.no_grad():
        out = plain(data)
        tokens, hw = plain.encoder(data["coarse_history"])
        lead = plain.lead_time(data["lead_time_hours"], batch=2, device=tokens.device,
                               dtype=tokens.dtype, default_hours=6.)
        context = tokens + lead[:, None] + plain.spacetime(data, history=data["coarse_history"], token_hw=hw)
        expected, _ = plain.head(context, hw, data["coarse_history"].shape[-2:], data["coarse_history"][:, -1])
    assert torch.equal(out.forecast, expected)
    assert torch.equal(out.context_tokens, context)


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_constructor_rejects_nonboolean_and_missing_dependency(flag):
    with pytest.raises(ValueError, match="known_context_inputs"):
        NativeAtmosForecaster(**NATIVE, known_context_inputs=flag)
    with pytest.raises(ValueError, match="spacetime_inputs"):
        NativeAtmosForecaster(**dict(NATIVE, spacetime_inputs=False), known_context_inputs=True)


@pytest.mark.parametrize("field", ["latitude", "longitude", "init_calendar_year", "init_day_of_year",
                                   "init_utc_hour", "history_offsets_hours", "lead_time_hours"])
def test_enabled_missing_metadata_fails_closed_in_every_control(field):
    data = _batch()
    del data[field]
    for mode in ("fields", "constant", "shuffled"):
        with pytest.raises(KeyError, match=field):
            _features(data, mode)


@pytest.mark.parametrize("field,value", [
    ("history_offsets_hours", torch.tensor([-6., 0.])),
    ("history_offsets_hours", torch.tensor([[0., -6.], [-6., 0.]])),
    ("history_offsets_hours", torch.tensor([[-6., 1.], [-6., 0.]])),
    ("history_offsets_hours", torch.tensor([[float("nan"), 0.], [-6., 0.]])),
    ("history_offsets_hours", torch.tensor([[False, True], [False, True]])),
    ("init_calendar_year", torch.tensor([2023., 2023.])),
    ("init_calendar_year", torch.tensor([[2024., 2023.]])),
    ("init_calendar_year", torch.tensor([2024.5, 2023.])),
    ("lead_time_hours", torch.tensor([[6., 12.], [6., 12.]])),
    ("longitude", torch.tensor([1 + 2j] * 7)),
    ("latitude", torch.zeros(1, 5)),
])
def test_bad_metadata_shape_dtype_calendar_device_is_rejected_before_controls(field, value):
    data = _batch()
    if field == "init_calendar_year" and value.shape == (2,) and value[0] == 2023:
        data["init_day_of_year"][0] = 366
    data[field] = value
    for mode in ("fields", "constant", "shuffled"):
        with pytest.raises((ValueError, TypeError), match=field + "|init_day_of_year"):
            _features(data, mode)
    bad_device = dict(_batch(), history_offsets_hours=torch.empty((2, 2), device="meta"))
    with pytest.raises(ValueError, match="history device"):
        _features(bad_device)


def test_regular_positive_cadence_is_separate_from_lead_and_physical_cadence():
    data = _batch()
    data["lead_time_hours"] = torch.tensor([12., 24.])
    torch.testing.assert_close(_features(data), _oracle(data), rtol=0, atol=2e-6)
    for value in (torch.tensor([[-12., -6., 0.]]), torch.tensor([[-24., -12., 0.]])):
        assert require_history_offsets(value, batch_size=1, history_steps=3, device=value.device).shape == (1, 3)
    with pytest.raises(ValueError, match="regular"):
        require_history_offsets(torch.tensor([[-18., -6., 0.]]), batch_size=1,
                                history_steps=3, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="physical"):
        require_history_offsets(data["history_offsets_hours"], batch_size=2, history_steps=2,
                                device=torch.device("cpu"), cadence_hours=6)


def _twenty_minute_offsets(dtype):
    # Real producer, synthetic int64 stamps only; no weather/store payload.
    stamps = 1_700_000_000_000_000_001 + np.arange(4, dtype=np.int64) * (HOUR_NS // 3)
    native = history_offsets_hours_from_ns(stamps)
    assert native.dtype == np.float64
    np.testing.assert_array_equal(native, [-1., -2 / 3, -1 / 3, 0.])
    return torch.from_numpy(native).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_actual_int64_twenty_minute_producer_rounding_is_regular(dtype):
    offsets = _twenty_minute_offsets(dtype)
    native_differences = torch.diff(_twenty_minute_offsets(torch.float64))
    assert not torch.equal(native_differences, native_differences[:1].expand_as(native_differences))
    for batched in (False, True):
        value = offsets[None] if batched else offsets
        actual = require_history_offsets(value, batch_size=1, history_steps=4,
            device=value.device, batched=batched, cadence_hours=1 / 3)
        assert actual.dtype == torch.float64 and torch.equal(actual, value.to(torch.float64))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
def test_cadence_ulp_bound_scales_per_sample_and_single_history(dtype, scale):
    # Cast from a regular FP64 source, not multiply already rounded FP32 values.
    value = (torch.stack([_twenty_minute_offsets(torch.float64),
                          _twenty_minute_offsets(torch.float64) * 2]) * scale).to(dtype)
    assert torch.equal(require_history_offsets(value, batch_size=2, history_steps=4,
        device=value.device), value.to(torch.float64))
    single = torch.zeros(2, 1, dtype=dtype)
    assert torch.equal(require_history_offsets(single, batch_size=2, history_steps=1,
        device=single.device, cadence_hours=6), single.to(torch.float64))
    irregular = value.clone()
    irregular[1, 1] += scale / 60
    with pytest.raises(ValueError, match="regular"):
        require_history_offsets(irregular, batch_size=2, history_steps=4, device=value.device)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_original_dtype_ulp_boundary_is_not_a_generic_allclose(dtype):
    value = torch.tensor([[-3., -2., -1., 0.]], dtype=dtype)
    # One original-dtype ULP is within propagated endpoint roundoff; two are not.
    value[0, 1] = torch.nextafter(value[0, 1], torch.tensor(0., dtype=dtype))
    assert torch.equal(require_history_offsets(value, batch_size=1, history_steps=4,
        device=value.device), value.to(torch.float64))
    value[0, 1] = torch.nextafter(value[0, 1], torch.tensor(0., dtype=dtype))
    with pytest.raises(ValueError, match="regular"):
        require_history_offsets(value, batch_size=1, history_steps=4, device=value.device)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_irregular_twenty_twentyone_minutes_and_strict_gates_counterproof(dtype):
    increments = np.asarray([0, 20, 41, 61], dtype=np.int64) * (HOUR_NS // 60)
    stamps = 1_700_000_000_000_000_001 + increments
    with pytest.raises(ValueError, match="regular"):
        history_offsets_hours_from_ns(stamps)
    # Deliberately bypass producer to prove the actual tensor validator rejects it too.
    value = torch.tensor([(int(stamp) - int(stamps[-1])) / HOUR_NS for stamp in stamps], dtype=dtype)
    with pytest.raises(ValueError, match="regular"):
        require_history_offsets(value, batch_size=1, history_steps=4, device=value.device, batched=False)
    offsets = _twenty_minute_offsets(dtype)
    for index, replacement, message in ((3, torch.finfo(dtype).tiny, "zero"),
                                       (1, float("inf"), "finite"), (1, float("nan"), "finite"),
                                       (1, -1., "increasing"), (1, -2., "increasing")):
        broken = offsets.clone()
        broken[index] = replacement
        with pytest.raises(ValueError, match=message):
            require_history_offsets(broken, batch_size=1, history_steps=4,
                                    device=broken.device, batched=False)
    for value in (torch.tensor([[-18, -12, -6, 0]], dtype=torch.int64),
                  torch.tensor([[-18., -12., -6., 0.]], dtype=dtype)):
        assert torch.equal(require_history_offsets(value, batch_size=1, history_steps=4,
            device=value.device, cadence_hours=6), value.to(torch.float64))
    with pytest.raises(ValueError, match="regular"):
        require_history_offsets(torch.tensor([[-19, -12, -6, 0]]), batch_size=1,
                                history_steps=4, device=torch.device("cpu"))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_fractional_history_features_slot_causality_and_physical_t_plus_six_rejection(dtype):
    data = _batch()
    data["coarse_history"] = data["coarse_history"].repeat(1, 2, 1, 1, 1)
    offsets = _twenty_minute_offsets(dtype)
    data["history_offsets_hours"] = torch.stack([offsets, offsets * 2])
    data["lead_time_hours"] = torch.tensor([6., 18.])  # Not the history cadence.
    actual = _features(data)
    assert actual.shape == (2, 12, 22)
    for slot in range(4):
        elapsed = data["history_offsets_hours"][:, slot].double()
        hour = (data["init_utc_hour"].double() + elapsed).remainder(24)
        longitude = torch.deg2rad(data["longitude"][:2].double())
        angle = torch.pi * 2 * hour[:, None] / 24 + longitude
        torch.testing.assert_close(actual[:, 0, 5 * slot + 2:5 * slot + 4],
            torch.stack([angle.sin().mean(1), angle.cos().mean(1)], -1).float(), rtol=0, atol=2e-6)
        assert torch.equal(actual[:, 0, 5 * slot + 4], (elapsed / 24).float())
    moved = dict(data, history_offsets_hours=data["history_offsets_hours"].flip(0))
    changed = _features(moved)
    assert not torch.equal(actual[..., :15], changed[..., :15])
    assert torch.equal(actual[..., 15:], changed[..., 15:])  # Terminal and valid-time slots stay put.
    poisoned = dict(data, atmos_target=object(), future_target=object(), history_time_ns=object())
    assert torch.equal(actual, _features(poisoned))
    torch.testing.assert_close(_features(data, "shuffled"),
        _features({**data, **{key: data[key].roll(1, 0) for key in (
            "init_calendar_year", "init_day_of_year", "init_utc_hour", "history_offsets_hours")}}),
        rtol=0, atol=0)
    assert torch.count_nonzero(_features(data, "constant")) == 0
    from model.process_forecast_r7 import ProcessForecastCoReasoner
    model = ProcessForecastCoReasoner(**dict(NATIVE, history_steps=4), known_context_inputs=True,
                                     anchored_processes=2, free_processes=2)
    with pytest.raises(ValueError, match="physical"):
        autoregressive_rollout(model.eval(), forecast_inputs(data), lead_hours=(6,))
    data["lead_time_hours"] = torch.full((2,), 6.)
    with pytest.raises(ValueError, match="physical"):
        training_two_step(model.train(), data, reasoning_steps=1)


@pytest.mark.parametrize("hw,periodic", [((5, 7), False), ((65, 65), False), ((5, 7), True)])
def test_negative_gregorian_history_solar_ew_batch_native_padding_oracle(hw, periodic):
    data = _batch(hw)
    data["longitude"] = torch.tensor([179., -179.] + [30.] * (hw[1] - 2))
    actual = _features(data, periodic=periodic)
    expected = _oracle(data, periodic)
    assert actual.shape[-1] == 12
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
    # A date-line patch must average harmonics, not the raw longitude angle.
    assert actual[0, 0, 8] < 0  # current00Z slot1 cos, not historical18Z slot0 cos
    broken = actual.clone()
    broken[0, 0, 8] = 1.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(broken, expected, rtol=0, atol=2e-6)
    swapped = actual.clone()
    swapped[..., :10] = actual[..., :10].reshape(*actual.shape[:2], 2, 5).flip(-2).flatten(-2)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(swapped, expected, rtol=0, atol=2e-6)


def test_controls_validate_first_and_shuffle_metadata_not_heterogeneous_lead():
    data = _batch()
    data["lead_time_hours"] = torch.tensor([6., 18.])
    rearranged = dict(data)
    for key in ("init_calendar_year", "init_day_of_year", "init_utc_hour", "history_offsets_hours"):
        rearranged[key] = data[key].roll(1, dims=0)
    shuffled = _features(data, "shuffled")
    torch.testing.assert_close(shuffled, _oracle(rearranged), rtol=0, atol=2e-6)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(shuffled, _features(data).roll(1, dims=0), rtol=0, atol=2e-6)
    assert torch.count_nonzero(_features(data, "constant")) == 0
    bad = dict(data, history_offsets_hours=torch.tensor([[0., -6.], [-6., 0.]]))
    with pytest.raises(ValueError, match="increasing|zero"):
        _features(bad, "constant")


def _assert_residual_changes(model, data, moved):
    assert not torch.equal(model(data).context_tokens, model(moved).context_tokens), "residual not connected"


def test_residual_offset_slot_causality_gradients_and_branch_cut_counterproof(monkeypatch):
    data, model = _batch(), _native(True).train()
    moved = dict(data, history_offsets_hours=torch.tensor([[-12., 0.], [-6., 0.]]))
    _assert_residual_changes(model, data, moved)
    output = model(data)
    output.forecast.square().mean().backward()
    for name, value in model.known_context.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().max() > 0, name
    original = model.known_context.forward
    monkeypatch.setattr(model.known_context, "forward", lambda *a, **k: original(*a, **k) * 0)
    with pytest.raises(AssertionError, match="residual"):
        _assert_residual_changes(model, data, moved)
    model.zero_grad(set_to_none=True)
    model(data).forecast.square().mean().backward()
    assert all(value.grad.abs().max() == 0 for value in model.known_context.parameters())


def test_timestamp_producer_helper_precision_schema_and_reader_paths(tmp_path):
    stamps = np.asarray([1_700_000_000_000_000_001, 1_700_021_600_000_000_001], dtype=np.int64)
    np.testing.assert_array_equal(history_offsets_hours_from_ns(stamps), [-6., 0.])
    for bad in (stamps.astype(float), stamps[::-1], np.asarray([1, 2, 4], dtype=np.int64)):
        with pytest.raises(ValueError):
            history_offsets_hours_from_ns(bad)
    sample = SyntheticAtmosDataset(channels=3, hw=(5, 7), lead_time_hours=12)[0]
    assert sample["history_offsets_hours"].tolist() == [-6., 0.]
    validate_forecast_sample(sample)
    batched = rollout_model_input(sample, lead_hours=12)
    assert batched["history_offsets_hours"].shape == (1, 2)
    assert HISTORY_CONTEXT_FIELDS == ("history_offsets_hours",)
    assert torch.equal(_native(True)(batched).forecast, _native(True)(forecast_inputs(batched)).forecast)
    from test_r7_storage_safety import build
    paths = build(tmp_path)
    train = ZarrAtmosWindowDataset(paths["train"])[0]
    evaluation = ZarrRolloutDataset(tmp_path / "output", split="val", lead_hours=(6,))[0]
    assert train["history_offsets_hours"].tolist() == evaluation["history_offsets_hours"].tolist() == [-6., 0.]
    broken = dict(sample, history_offsets_hours=torch.tensor([0., -6.]))
    with pytest.raises(ValueError, match="zero|increasing"):
        validate_forecast_sample(broken)


@pytest.mark.parametrize("dtype", [None, torch.bfloat16])
def test_native_bf16_full_gradient_poison_and_strict_checkpoint(tmp_path, dtype):
    model, data = _native(True).train(), _batch()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype is not None):
        out = model(forecast_inputs(data))
    out.forecast.float().square().mean().backward()
    assert torch.isfinite(out.forecast).all()
    assert model.encoder.patch.weight.grad.abs().max() > 0
    assert model.known_context.projection.weight.grad.abs().max() > 0
    model.eval()
    poisoned = dict(data, atmos_target=object(), process_targets=object(), future_target=object(),
                    init_year=object(), history_time_ns=object(), atmos_baseline=object())
    assert torch.equal(model(forecast_inputs(data)).forecast, model(forecast_inputs(poisoned)).forecast)
    path = tmp_path / "state.pt"
    torch.save(model.state_dict(), path)
    restored = _native(True).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(model(forecast_inputs(data)).forecast, restored(forecast_inputs(data)).forecast)


def test_module_imports_no_clock_and_temporary_root_is_the_actual_source():
    import model.known_context_r7 as module
    assert Path(module.__file__).resolve().is_relative_to(ROOT)
    tree = ast.parse(Path(module.__file__).read_text())
    imports = {node.module.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module}
    imports |= {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    assert not imports & {"datetime", "time", "calendar", "dateutil"}
    assert not {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} & {"now", "utcnow", "today"}


def test_physical_training_and_evaluation_off_never_read_offset_poison():
    from model.process_forecast_r7 import ProcessForecastCoReasoner
    torch.manual_seed(11)
    model = ProcessForecastCoReasoner(**NATIVE, known_context_inputs=False,
                                     anchored_processes=2, free_processes=2).eval()
    data = _batch()
    clean = autoregressive_rollout(model, forecast_inputs(data), lead_hours=(6, 12),
                                   inference_kwargs={"reasoning_steps": 1})
    poisoned = dict(data, history_offsets_hours=object())
    changed = autoregressive_rollout(model, forecast_inputs(poisoned), lead_hours=(6, 12),
                                     inference_kwargs={"reasoning_steps": 1})
    assert torch.equal(clean.forecasts, changed.forecasts)
    model.train()
    first = training_two_step(model, data, reasoning_steps=1)
    second = training_two_step(model, poisoned, reasoning_steps=1)
    assert torch.equal(first.forecasts, second.forecasts)
    assert torch.equal(first.loss, second.loss)


def test_physical_date_offset_trace_and_internal_k_are_not_shifted(monkeypatch):
    from model.process_forecast_r7 import ProcessForecastCoReasoner
    model = ProcessForecastCoReasoner(**NATIVE, known_context_inputs=True,
                                     anchored_processes=2, free_processes=2).train()
    data, seen = _batch(size=1), []
    hook = model.backbone.known_context.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(args[0]["history_offsets_hours"].clone()), with_kwargs=True)
    try:
        training_two_step(model, data, reasoning_steps=4)
        assert len(seen) == 2 and all(torch.equal(value, data["history_offsets_hours"]) for value in seen)
        import training.r7_autoregressive_rollout as module
        original = module._advance_inputs
        def shift_offsets(inputs):
            result = original(inputs)
            result["history_offsets_hours"] = inputs["history_offsets_hours"] - 6
            return result
        monkeypatch.setattr(module, "_advance_inputs", shift_offsets)
        with pytest.raises(ValueError, match="end at zero"):
            training_two_step(model, data, reasoning_steps=4)
    finally:
        hook.remove()


@pytest.mark.parametrize("kind", ["process", "generic"])
def test_wrapper_fixed_streamed_adaptive_physical_paths_share_known_inputs(kind):
    from model.process_forecast_r7 import ProcessForecastCoReasoner
    from model.recursive_weather_r7 import GenericRecursiveWeatherForecaster
    torch.manual_seed(11)
    options = dict(NATIVE, known_context_inputs=True, default_reasoning_steps=2,
                   positional_process_readout=True)
    model = (ProcessForecastCoReasoner(**options, anchored_processes=2, free_processes=2)
             if kind == "process" else GenericRecursiveWeatherForecaster(**options, latent_tokens=4))
    data = _batch()
    data["history_offsets_hours"] = torch.tensor([[-6., 0.], [-6., 0.]])
    model.train()
    fixed = model(forecast_inputs(data), reasoning_steps=2, detach_between_steps=True).forecast.detach()
    streamed = backward_streamed_truncated(model, data, reasoning_steps=2, process_weight=0)
    assert torch.equal(fixed, streamed.final_forecast)
    seen = []
    handle = model.backbone.known_context.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append({k: v.clone() for k, v in args[0].items()}), with_kwargs=True)
    try:
        two = training_two_step(model, data, reasoning_steps=2)
        two.loss.backward()
        assert len(seen) == 2
        assert all(entry["history_offsets_hours"].tolist() == [[-6., 0.], [-6., 0.]] for entry in seen)
        assert seen[1]["init_utc_hour"].tolist() == [6., 0.]
        seen.clear()
        model.eval()
        rollout = autoregressive_rollout(model, forecast_inputs(data), lead_hours=(6, 12),
                                         inference_kwargs={"reasoning_steps": 2})
        assert rollout.model_calls == len(seen) == 2
        if kind == "process":
            adaptive = AdaptiveProcessForecaster(model).eval()(data, max_steps=2, force_full_depth=True)
            assert torch.equal(adaptive.forecast, model(forecast_inputs(data), reasoning_steps=2).forecast)
        bad = dict(data, history_offsets_hours=torch.tensor([[-12., 0.], [-6., 0.]]))
        with pytest.raises(ValueError, match="physical"):
            autoregressive_rollout(model, forecast_inputs(bad), lead_hours=(6,))
    finally:
        handle.remove()
