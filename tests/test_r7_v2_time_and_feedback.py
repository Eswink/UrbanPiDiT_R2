"""D4 direct CPU evidence for #71 time/geography and #72 RW-A -> RW-B feedback.

Calendar examples are actual UTC dates, not a synthetic 365.25-day wrap. Weather
values are synthetic engineering fixtures only; no real store, GPU or sealed
artifact is read. Fault injection reuses the same assertions as the live path.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pytest
import torch

from data.r7_store import init_time_fields
from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.process_step_r7 import ProcessStepInput, process_reasoning_step
from model.r7_halting import forecast_inputs
from model.r7_rollout import autoregressive_rollout
from model.spacetime_conditioning_r7 import (
    SpacetimeConditioning, advance_calendar_time, phase_features, position_features,
)

SMALL = {"in_channels": 3, "out_channels": 3, "history_steps": 2, "dim": 16,
         "depth": 1, "heads": 2, "window_size": 2, "patch_size": 2, "dropout": 0.0,
         "anchored_processes": 2, "free_processes": 2, "default_reasoning_steps": 3,
         "spacetime_inputs": True, "positional_process_readout": True,
         "source_role_markers": True, "local_solver_state": True}


def _utc(stamp):
    return datetime.fromisoformat(stamp).astimezone(timezone.utc)


def _time_batch(stamps, *, hw=(8, 10)):
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    fields = []
    for stamp in stamps:
        delta = _utc(stamp) - epoch
        nanoseconds = (delta.days * 86400 + delta.seconds) * 1_000_000_000
        exported = init_time_fields(nanoseconds)
        assert exported["init_utc_hour"] == _utc(stamp).hour
        assert exported["init_day_of_year"] == _utc(stamp).timetuple().tm_yday
        assert exported["init_year"] == _utc(stamp).year
        assert exported["init_calendar_year"] == _utc(stamp).year
        fields.append(exported)
    generator = torch.Generator().manual_seed(5)
    return {
        "coarse_history": torch.randn(len(stamps), 2, 3, *hw, generator=generator),
        "lead_time_hours": torch.full((len(stamps),), 6.0),
        "latitude": torch.linspace(-30.0, 30.0, hw[0]),
        "longitude": torch.linspace(-120.0, 120.0, hw[1]),
        **{name: torch.tensor([entry[name] for entry in fields]) for name in fields[0]},
    }


def _model():
    torch.manual_seed(11)
    model = ProcessForecastCoReasoner(**SMALL).eval()
    assert model.positional_process_readout and model.local_solver_state
    assert model.spatial_solver_feedback is False
    return model


def _phase_at_valid_time(stamps, leads):
    # Independent datetime oracle: extract the actual valid date after +lead.
    angles = []
    for stamp, lead in zip(stamps, leads):
        valid = _utc(stamp) + timedelta(hours=lead)
        days_in_year = (datetime(valid.year + 1, 1, 1) - datetime(valid.year, 1, 1)).days
        annual = math.tau * (valid.timetuple().tm_yday - 1 + valid.hour / 24) / days_in_year
        diurnal = math.tau * valid.hour / 24
        angles.append([math.sin(annual), math.cos(annual),
                       math.sin(diurnal), math.cos(diurnal)])
    return torch.tensor(angles, dtype=torch.float64)


def _phase_from_init(stamps, leads):
    times = [_utc(stamp) for stamp in stamps]
    return phase_features(torch.tensor([stamp.hour for stamp in times], dtype=torch.float64),
                          torch.tensor([stamp.timetuple().tm_yday for stamp in times],
                                       dtype=torch.float64),
                          torch.tensor(leads, dtype=torch.float64),
                          init_calendar_year=torch.tensor([stamp.year for stamp in times]))


def _assert_valid_time(actual, stamps, leads):
    torch.testing.assert_close(actual, _phase_at_valid_time(stamps, leads), rtol=0, atol=1e-12)


@pytest.mark.parametrize("stamp,lead", [
    ("2023-12-31T18:00:00+00:00", 6),
    ("2024-12-31T18:00:00+00:00", 6),
    ("2024-02-28T18:00:00+00:00", 6),
    ("2024-02-29T18:00:00+00:00", 6),
    ("2023-02-28T18:00:00+00:00", 6),
    ("2024-06-30T23:00:00+00:00", 12),
    ("2024-03-01T01:00:00+09:00", 12),
    ("2024-02-29T22:00:00-07:00", 6),
    ("1900-02-28T18:00:00+00:00", 6),
    ("2000-02-28T18:00:00+00:00", 6),
    ("2100-02-28T18:00:00+00:00", 6),
    ("2400-12-31T18:00:00+00:00", 6),
    ("2024-02-29T18:00:00+00:00", 8766),
])
def test_real_calendar_dates_reach_the_actual_valid_time(stamp, lead):
    _assert_valid_time(_phase_from_init([stamp], [lead]), [stamp], [lead])


def test_time_oracle_rejects_stale_init_and_k_advancing_the_valid_time():
    stamps, leads = ["2024-02-29T18:00:00+00:00"], [6]
    _assert_valid_time(_phase_from_init(stamps, leads), stamps, leads)
    for injected_lead in (0, 12):
        with pytest.raises(AssertionError):
            _assert_valid_time(_phase_from_init(stamps, [injected_lead]), stamps, leads)


def test_calendar_year_changes_february_phase_but_legacy_three_arguments_stay_bitwise():
    hour = torch.tensor([18.0, 23.0, 0.0, 12.0])
    day = torch.tensor([59.0, 365.0, 366.0, 1.0])
    lead = torch.tensor([6.0, 72.0, 8766.0, 0.0])
    annual = math.tau * (day - 1.0 + (hour + lead) / 24.0) / 365.25
    diurnal = math.tau * (hour + lead) / 24.0
    archived = torch.stack([annual.sin(), annual.cos(), diurnal.sin(), diurnal.cos()], -1)
    assert torch.equal(phase_features(hour, day, lead), archived)
    common = _phase_from_init(["2023-02-28T18:00:00+00:00"], [6])
    leap = _phase_from_init(["2024-02-28T18:00:00+00:00"], [6])
    assert not torch.equal(common[:, :2], leap[:, :2]), "year identity never reached phase"
    with pytest.raises(AssertionError):
        _assert_valid_time(phase_features(torch.tensor([18.0], dtype=torch.float64),
                                          torch.tensor([59.0], dtype=torch.float64),
                                          torch.tensor([6.0], dtype=torch.float64)),
                           ["2024-02-28T18:00:00+00:00"], [6])


@pytest.mark.parametrize("field,value", [
    ("init_calendar_year", torch.tensor([2023.0])),  # day 366 needs a leap year
    ("init_calendar_year", torch.tensor([2024.5])),
    ("init_calendar_year", torch.tensor([0.0])),
    ("init_calendar_year", torch.tensor([float("nan")])),
    ("init_calendar_year", torch.tensor([2024.0, 2025.0])),
    ("init_day_of_year", torch.tensor([59.5])),
    ("init_utc_hour", torch.tensor([24.0])),
    ("lead_time_hours", torch.tensor([float("inf")])),
])
def test_explicit_calendar_invalid_metadata_fails_instead_of_guessed_date(field, value):
    fields = {"init_calendar_year": torch.tensor([2024.0]),
              "init_day_of_year": torch.tensor([366.0]), "init_utc_hour": torch.tensor([18.0]),
              "lead_time_hours": torch.tensor([6.0])}
    fields[field] = value
    with pytest.raises(ValueError, match=field):
        advance_calendar_time(fields["init_calendar_year"], fields["init_day_of_year"],
                              fields["init_utc_hour"], fields["lead_time_hours"])


def test_calendar_control_modes_keep_year_pairing_and_validate_before_discarding():
    data = _time_batch(["2024-12-31T18:00:00+00:00", "2023-06-30T00:00:00+00:00"])
    terms = {}
    for mode in ("fields", "shuffled", "constant"):
        torch.manual_seed(17)
        module = SpacetimeConditioning(16, 2, field_mode=mode)
        rows, columns = data["coarse_history"].shape[-2:]
        terms[mode] = module(data, history=data["coarse_history"],
                             token_hw=(rows // 2, columns // 2)).detach()
        invalid = dict(data, init_calendar_year=torch.tensor([2023.0, 2023.0]))
        with pytest.raises(ValueError, match="init_day_of_year"):
            module(invalid, history=data["coarse_history"], token_hw=(rows // 2, columns // 2))
    torch.testing.assert_close(terms["shuffled"], terms["fields"].roll(1, dims=0), rtol=0, atol=0)
    torch.testing.assert_close(terms["constant"], terms["constant"][:1, :1].expand_as(
        terms["constant"]), rtol=0, atol=1e-6)


def _manual_position(latitude, longitude, *, periodic_width):
    # Independent patch oracle: no pad_patch_grid or position_features call.
    rows = []
    for row in range(33):
        lat = sum(float(latitude[min(2 * row + offset, 64)]) for offset in (0, 1)) / 2
        for column in range(33):
            indices = [2 * column, 2 * column + 1]
            indices = [index % 65 if periodic_width else min(index, 64) for index in indices]
            lon = sum(float(longitude[index]) for index in indices) / 2
            a, b = math.radians(lat), math.radians(lon)
            rows.append([math.sin(a), math.sin(b), math.cos(a), math.cos(b)])
    return torch.tensor(rows, dtype=torch.float64)


@pytest.mark.parametrize("periodic_width", [False, True])
def test_65_odd_grid_padding_has_the_encoder_patch_geography(periodic_width):
    latitude = torch.linspace(65, -65, 65, dtype=torch.float64)
    longitude = torch.linspace(-160, 160, 65, dtype=torch.float64)
    actual = position_features(latitude, longitude, token_hw=(33, 33), patch_size=2,
                               periodic_width=periodic_width)
    expected = _manual_position(latitude, longitude, periodic_width=periodic_width)
    assert actual.shape == (1089, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    # Fail if padding is dropped or circular padding replaces the finite edge.
    for broken in (actual[:1024], _manual_position(
            latitude, longitude, periodic_width=not periodic_width)):
        with pytest.raises(AssertionError):
            torch.testing.assert_close(broken, expected, rtol=0, atol=1e-12)


def test_east_and_west_longitudes_are_per_token_not_a_pooled_location():
    latitude = torch.tensor([-20.0, 20.0], dtype=torch.float64)
    longitude = torch.tensor([-90.0, 90.0], dtype=torch.float64)
    actual = position_features(latitude, longitude, token_hw=(2, 2), patch_size=1)
    expected = torch.tensor([[math.sin(math.radians(lat)), math.sin(math.radians(lon)),
                              math.cos(math.radians(lat)), math.cos(math.radians(lon))]
                             for lat in latitude.tolist() for lon in longitude.tolist()],
                            dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert actual[0, 1] < 0 < actual[1, 1]
    pooled = actual.mean(0, keepdim=True).expand_as(actual)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(pooled, expected, rtol=0, atol=1e-12)


def test_batched_dates_are_independent_and_only_the_changed_sample_moves():
    stamps = ["2024-02-28T18:00:00+00:00", "2023-06-30T18:00:00+00:00",
              "2024-02-29T00:00:00+00:00"]
    data, model = _time_batch(stamps), _model()
    # Hold history fixed across samples to isolate the declared date fields.
    data["coarse_history"] = data["coarse_history"][:1].expand(3, -1, -1, -1, -1).clone()
    with torch.no_grad():
        combined = model(forecast_inputs(data), reasoning_steps=2)
        assert not torch.equal(combined.forecast[0], combined.forecast[1])
        for index in range(3):
            one = {name: value[index:index + 1] if name not in ("latitude", "longitude")
                   else value for name, value in data.items()}
            separate = model(forecast_inputs(one), reasoning_steps=2)
            torch.testing.assert_close(combined.forecast[index:index + 1], separate.forecast,
                                       rtol=0, atol=2e-6)
        moved = dict(data, init_utc_hour=data["init_utc_hour"].clone())
        moved["init_utc_hour"][1] += 3
        other = model(forecast_inputs(moved), reasoning_steps=2)
    assert torch.equal(combined.forecast[[0, 2]], other.forecast[[0, 2]])
    assert not torch.equal(combined.forecast[1], other.forecast[1])


def test_live_65_grid_keeps_native_edges_and_valid_time_constant_across_internal_k():
    data = _time_batch(["2024-02-29T18:00:00+00:00"], hw=(65, 65))
    model, seen = _model(), []
    handle = model.backbone.spacetime.net.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            shallow = model(forecast_inputs(data), reasoning_steps=1)
            deep = model(forecast_inputs(data), reasoning_steps=4)
    finally:
        handle.remove()
    assert shallow.token_hw == deep.token_hw == (33, 33)
    assert deep.forecast.shape == (1, 3, 65, 65)
    assert deep.solver_state.shape == (1, 1089, 16)
    assert torch.isfinite(deep.forecast).all()
    assert len(seen) == 2, "conditioning was re-evaluated inside K"
    assert torch.equal(seen[0], seen[1]), "K changed the known valid-time conditions"
    assert torch.equal(shallow.context_tokens, deep.context_tokens)
    expected = _phase_at_valid_time(["2024-02-29T18:00:00+00:00"], [6]).float()
    torch.testing.assert_close(seen[1][0, :, 4:], expected.expand(1089, -1),
                               rtol=0, atol=2e-6)
    geography = _manual_position(data["latitude"].double(), data["longitude"].double(),
                                 periodic_width=False).float()
    torch.testing.assert_close(seen[1][0, :, :4], geography, rtol=0, atol=2e-6)


def _assert_rollout_calendar(phases, inputs, stamps):
    assert len(phases) == len(inputs) == 4
    for transition, (features, batch) in enumerate(zip(phases, inputs), start=1):
        expected = _phase_at_valid_time(stamps, [6 * transition] * len(stamps)).float()
        torch.testing.assert_close(features[:, :, 4:],
                                   expected[:, None].expand(-1, features.shape[1], -1),
                                   rtol=0, atol=2e-6)
        assert batch["lead_time_hours"].tolist() == [6] * len(stamps)
        for index, stamp in enumerate(stamps):
            init = _utc(stamp) + timedelta(hours=6 * (transition - 1))
            assert batch["init_calendar_year"][index] == init.year
            assert batch["init_day_of_year"][index] == init.timetuple().tm_yday
            assert batch["init_utc_hour"][index] == init.hour


def test_physical_rollout_recomputes_batch_calendar_every_six_hours_not_internal_k(monkeypatch):
    stamps = ["2023-12-31T18:00:00+00:00", "2024-12-31T18:00:00+00:00",
              "2024-02-28T18:00:00+00:00"]
    data, model, phases, inputs = _time_batch(stamps), _model(), [], []
    original = {key: value.clone() for key, value in data.items()}
    handles = [model.backbone.spacetime.net.register_forward_pre_hook(
        lambda _module, args: phases.append(args[0].detach().clone())),
        model.register_forward_pre_hook(lambda _module, args: inputs.append(
            {key: value.detach().clone() for key, value in args[0].items()}))]
    try:
        out = autoregressive_rollout(model, forecast_inputs(data), lead_hours=(6, 12, 24),
                                     inference_kwargs={"reasoning_steps": 3})
        assert out.model_calls == 4
        assert out.cumulative_reasoning_steps.tolist() == [[3, 6, 12]] * 3
        _assert_rollout_calendar(phases, inputs, stamps)
        for key in data:
            assert torch.equal(data[key], original[key]), f"caller {key} was mutated"
        for transition in range(1, 4):
            assert torch.equal(inputs[transition]["coarse_history"][:, -2],
                               inputs[transition - 1]["coarse_history"][:, -1])
        # Inject the original stale-date defect without weakening the oracle.
        import model.r7_rollout as rollout_module
        monkeypatch.setattr(rollout_module, "advance_calendar_time",
                            lambda year, day, hour, lead: (year, day, hour))
        phases.clear(), inputs.clear()
        autoregressive_rollout(model, forecast_inputs(data), lead_hours=(6, 12, 24),
                               inference_kwargs={"reasoning_steps": 3})
        with pytest.raises(AssertionError):
            _assert_rollout_calendar(phases, inputs, stamps)
    finally:
        for handle in handles:
            handle.remove()


def test_legacy_rollout_without_calendar_preserves_accumulated_lead():
    data = _time_batch(["2024-02-28T18:00:00+00:00"])
    del data["init_calendar_year"]
    model, inputs = _model(), []
    handle = model.register_forward_pre_hook(lambda _module, args: inputs.append(
        {key: value.detach().clone() for key, value in args[0].items()}))
    try:
        out = autoregressive_rollout(model, forecast_inputs(data), lead_hours=(6, 12, 24),
                                     inference_kwargs={"reasoning_steps": 2})
    finally:
        handle.remove()
    assert out.model_calls == 4
    assert [entry["lead_time_hours"].item() for entry in inputs] == [6, 12, 18, 24]
    for entry in inputs:
        assert "init_calendar_year" not in entry and "init_year" not in entry
        assert torch.equal(entry["init_day_of_year"], data["init_day_of_year"])
        assert torch.equal(entry["init_utc_hour"], data["init_utc_hour"])


def _local_draft(base):
    moved = base.forecast.clone()
    moved[0, :, 2:4, 2:4] += moved.new_tensor([
        [[0.4, -0.8], [1.3, -0.2]], [[-0.3, 0.7], [0.2, -1.1]],
        [[0.6, 0.1], [-0.9, 0.5]],
    ])
    return moved


def _step_read(model, base, draft):
    reads = []
    handle = model.process_reader.register_forward_hook(
        lambda _module, _args, output: reads.append(output.detach().clone()))
    try:
        out = process_reasoning_step(
            model, ProcessStepInput(model.process_queries.expand(1, -1, -1),
                                    base.context_tokens, draft), base.token_hw,
            anchor=base.base_state)
    finally:
        handle.remove()
    assert len(reads) == 1
    return out, reads[0]


def _assert_corresponding_latent_moves(clean, changed, local_index):
    assert torch.isfinite(changed.solver_state).all()
    assert not torch.equal(clean.solver_state[:, local_index],
                           changed.solver_state[:, local_index]), "read-feedback never reached Z_i"


def test_local_draft_perturbation_reaches_read_feedback_and_corresponding_latent(monkeypatch):
    model = _model()
    data = _time_batch(["2024-02-29T18:00:00+00:00"])
    with torch.no_grad():
        base = model.backbone(forecast_inputs(data))
        moved = _local_draft(base)
        tokens, token_hw = model.draft_encoder(base.forecast)
        changed_tokens, _ = model.draft_encoder(moved)
        local_index = token_hw[1] + 1
        changed_positions = (tokens - changed_tokens).abs().amax(-1).ne(0).nonzero()
        assert changed_positions.tolist() == [[0, local_index]]
        clean, read = _step_read(model, base, base.forecast)
        changed, changed_read = _step_read(model, base, moved)
        assert not torch.equal(read[:, local_index], changed_read[:, local_index])
        _assert_corresponding_latent_moves(clean, changed, local_index)

        # Isolate E(Y)->P->R->Z: suppress the separate direct E(Y)->Z edge.
        def pin_direct_draft(_module, args, kwargs):
            return args, dict(kwargs, draft_tokens=tokens)

        handle = model.solver_cell.register_forward_pre_hook(pin_direct_draft, with_kwargs=True)
        try:
            isolated, _ = _step_read(model, base, base.forecast)
            feedback, _ = _step_read(model, base, moved)
            _assert_corresponding_latent_moves(isolated, feedback, local_index)
            # Fault: the read still executes, but its changed answer is discarded.
            original = model.process_reader.forward
            monkeypatch.setattr(model.process_reader, "forward",
                                lambda *args, **kwargs: original(*args, **kwargs) * 0 + read)
            blocked, blocked_read = _step_read(model, base, moved)
            assert torch.equal(blocked_read, read)
            assert torch.equal(blocked.solver_state, isolated.solver_state)
            with pytest.raises(AssertionError, match="read-feedback"):
                _assert_corresponding_latent_moves(isolated, blocked, local_index)
        finally:
            handle.remove()


def _assert_reader_gradients(model):
    parameters = dict(model.process_reader.named_parameters())
    assert parameters, "the live model has no process_reader"
    dead = [name for name, parameter in parameters.items()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all()
            or parameter.grad.abs().max() == 0]
    assert not dead, f"process_reader parameters without a nonzero gradient: {dead}"


def test_every_process_reader_parameter_gets_forecast_gradient_and_blocking_read_fails(monkeypatch):
    model = _model().train()
    data = _time_batch(["2024-02-29T18:00:00+00:00", "2023-06-30T18:00:00+00:00"])
    model(forecast_inputs(data), reasoning_steps=3).forecast.square().mean().backward()
    _assert_reader_gradients(model)
    model.zero_grad(set_to_none=True)
    original = model.process_reader.forward
    monkeypatch.setattr(model.process_reader, "forward",
                        lambda *args, **kwargs: original(*args, **kwargs) * 0)
    model(forecast_inputs(data), reasoning_steps=3).forecast.square().mean().backward()
    with pytest.raises(AssertionError, match="nonzero gradient"):
        _assert_reader_gradients(model)
