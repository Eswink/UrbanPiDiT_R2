"""Tiny CPU path counterproofs; all table identities and fields here are synthetic.

No weather fit, real checkpoints, or forecast-skill claim. Default-off is compared
against the exact previous active source, not merely omitted versus explicit None.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib
import io
from pathlib import Path
import socket
import subprocess
import sys
import tarfile

import pytest
import torch
from torch import nn

from model.climatology_anchor_r7 import uses_climatology_anchor
from model.process_forecast_r7 import ProcessForecastCoReasoner
from model.process_step_r7 import ProcessStepInput, process_reasoning_step
from model.r7_halting import AdaptiveProcessForecaster, ForecastGainController, forecast_inputs
from model.r7_rollout import autoregressive_rollout
from model.weather_forecaster_r7 import NativeAtmosForecaster
from training.r7_autoregressive_rollout import training_two_step
from training.r7_experiment import canonical_digest, load_checkpoint, make_model, save_exclusive
from training.r7_gain_oracle import collect_process_errors
from training.r7_halting import controller_calibration_loss, next_step_gain_targets, per_sample_latitude_mse
from training.r7_long_rollout import training_long_rollout
from training.r7_process_forecast_losses import process_forecast_coreasoning_loss
from training.r7_streaming import backward_streamed_truncated

PREVIOUS_SHA = '2af4e825f70c650536622d3d5726b7895e484289'
PREVIOUS_PINS = {
    'model/weather_forecaster_r7.py': 'e5278b088c295b9d6ec529d0ff9a92e3c0668353d9f18cafe4d72d091a67aa63',
    'model/process_forecast_r7.py': 'e145396fed21b643dc0407b3dcff27c6f79724d3f6a5f06d9a1c72c86924db17',
    'model/process_step_r7.py': '661ff1d25a1c9f58032d1d4d36249295f49a84e168e3cdb1a905822c8d368d19',
}
TINY = dict(in_channels=2, out_channels=2, history_steps=2, dim=16, depth=1,
            heads=2, patch_size=2, window_size=2, dropout=0.0)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('CPU path tests must not open sockets')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def tiny_spec_table(channels=2, rows=4, columns=6):
    keys = [[month, hour] for month in (1, 2, 3, 12) for hour in (0, 6, 12, 18)]
    table = torch.arange(len(keys) * channels * rows * columns, dtype=torch.float32)
    table = table.reshape(len(keys), channels, rows, columns) / 8
    spec = dict(format='r7-train-climatology-anchor-v1', kind='train-only-month-hour-grid-mean-v1',
                channels=[f'toy_{i}' for i in range(channels)], units=['toy_units'] * channels,
                source_sha256='1' * 64, train_data_identity='2' * 64,
                climatology_mean_identity_sha256='3' * 64,
                normalization_mean=[0.0] * channels, normalization_std=[1.0] * channels,
                latitude=torch.linspace(60.125, 30.375, rows).double().tolist(),
                longitude=torch.linspace(100.125, 120.625, columns).double().tolist(),
                bucket_keys=keys, bucket_counts=[2] * len(keys), training_years=[2016, 2017],
                n_selected_steps=2 * len(keys), selection='declared_train_years',
                table_sha256=hashlib.sha256(table.numpy().tobytes()).hexdigest(),
                table_shape=list(table.shape))
    return spec, table


def tiny_batch(size=2):
    spec, _ = tiny_spec_table()
    generator = torch.Generator().manual_seed(101)
    return dict(coarse_history=torch.randn(size, 2, 2, 4, 6, generator=generator),
                atmos_target=torch.randn(size, 2, 4, 6, generator=generator),
                process_targets=torch.randn(size, 4, generator=generator),
                lead_time_hours=torch.full((size,), 6.0), latitude=torch.tensor(spec['latitude']),
                longitude=torch.tensor(spec['longitude']), init_calendar_year=torch.full((size,), 2017),
                init_day_of_year=torch.ones(size), init_utc_hour=torch.arange(size).float() * 6,
                history_offsets_hours=torch.tensor([-6.0, 0.0]).expand(size, -1).clone())


def process_options(**extra):
    return dict(anchored_processes=4, free_processes=2, positional_process_readout=True,
                local_solver_state=True, draft_query_feedback=True, source_role_markers=True,
                source_position_markers=True, **extra)


def build(*, kind='process', **extra):
    spec, table = tiny_spec_table()
    options = dict(TINY, climatology_anchor_spec=spec)
    if kind == 'process':
        options.update(process_options())
        options['anomaly_feedback'] = True
    options.update(extra)
    torch.manual_seed(41)
    model = make_model(kind, options)
    anchor = getattr(model, 'backbone', model).climatology_anchor
    anchor.install(table)
    return model, options


def zero_head(head, bias=0.0):
    with torch.no_grad():
        head.decode[-1].weight.zero_()
        head.decode[-1].bias.fill_(bias)


@pytest.fixture(scope='module')
def previous_source(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['git', '-C', str(root), 'archive', '--format=tar', PREVIOUS_SHA, 'model'],
                            check=True, capture_output=True, timeout=30)
    package = 'r7_climate_previous_2af4e825_model'
    destination = tmp_path_factory.mktemp('climate_previous_source')
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith('.py') and 'legacy' not in member.name:
                sources[member.name] = archive.extractfile(member).read()
    for name, expected in PREVIOUS_PINS.items():
        assert hashlib.sha256(sources[name]).hexdigest() == expected
    for name, content in sources.items():
        target = destination / package / Path(name).relative_to('model')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if name != 'model/__init__.py' else b'')
    sys.path.insert(0, str(destination))
    try:
        native = importlib.import_module(f'{package}.weather_forecaster_r7').NativeAtmosForecaster
        process = importlib.import_module(f'{package}.process_forecast_r7').ProcessForecastCoReasoner
    finally:
        sys.path.remove(str(destination))
    return native, process


@pytest.mark.parametrize('kind,solver', [('native', False), ('process', False), ('process', True)])
def test_default_off_exact_previous_source_state_rng_outputs(previous_source, kind, solver):
    constructor = previous_source[0 if kind == 'native' else 1]
    options = dict(TINY)
    if kind == 'process':
        options.update(process_options())
        options['local_solver_state'] = solver
    torch.manual_seed(47)
    old = constructor(**options).eval()
    old_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(47)
    current = make_model(kind, options).eval()
    assert torch.equal(torch.random.get_rng_state(), old_rng)
    assert set(old.state_dict()) == set(current.state_dict())
    assert set(dict(old.named_parameters())) == set(dict(current.named_parameters()))
    assert set(dict(old.named_buffers())) == set(dict(current.named_buffers()))
    assert all(torch.equal(value, current.state_dict()[name]) for name, value in old.state_dict().items())
    with torch.no_grad():
        expected = old(forecast_inputs(tiny_batch()))
        actual = current(forecast_inputs(tiny_batch()))
    for name, value in vars(expected).items():
        other = getattr(actual, name)
        if torch.is_tensor(value):
            assert torch.equal(value, other), name
        else:
            assert value == other, name
    assert not uses_climatology_anchor(current)
    if kind == 'native':
        assert actual.climatology_anchor is None


@pytest.mark.parametrize('kind', ['native', 'process'])
def test_enabled_table_adds_only_two_buffers_and_no_rng_or_trainable_weights(kind):
    spec, _ = tiny_spec_table()
    options = dict(TINY)
    if kind == 'process':
        options.update(process_options())
    torch.manual_seed(73)
    control = make_model(kind, options)
    rng = torch.random.get_rng_state().clone()
    torch.manual_seed(73)
    enabled = make_model(kind, dict(options, climatology_anchor_spec=spec))
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert set(dict(control.named_parameters())) == set(dict(enabled.named_parameters()))
    assert all(torch.equal(value, enabled.state_dict()[name]) for name, value in control.state_dict().items())
    prefix = 'backbone.' if kind == 'process' else ''
    assert set(enabled.state_dict()) - set(control.state_dict()) == {
        prefix + 'climatology_anchor.table', prefix + 'climatology_anchor.ready'}
    assert not bool(getattr(enabled, 'backbone', enabled).climatology_anchor.ready)


@pytest.mark.parametrize('bad', [None, 0, 1, 'true'])
def test_anomaly_switch_is_strict_boolean(bad):
    with pytest.raises(ValueError, match='anomaly_feedback.*boolean'):
        ProcessForecastCoReasoner(**TINY, anomaly_feedback=bad)


def test_unsupported_or_ineffective_options_and_partial_outputs_are_rejected():
    spec, _ = tiny_spec_table()
    with pytest.raises(ValueError, match='requires climatology_anchor_spec'):
        ProcessForecastCoReasoner(**TINY, anomaly_feedback=True)
    with pytest.raises(ValueError, match='use_forecast_feedback'):
        ProcessForecastCoReasoner(**TINY, climatology_anchor_spec=spec, anomaly_feedback=True,
                                 use_forecast_feedback=False)
    with pytest.raises(TypeError, match='climatology_anchor_spec'):
        make_model('generic', dict(TINY, climatology_anchor_spec=spec))
    with pytest.raises(TypeError, match='anomaly_feedback'):
        NativeAtmosForecaster(**TINY, anomaly_feedback=True)
    for kind in ('native', 'process'):
        with pytest.raises(ValueError, match='out_channels'):
            make_model(kind, dict(TINY, out_channels=1, climatology_anchor_spec=spec))
        partial_spec, _ = tiny_spec_table(channels=1)
        with pytest.raises(ValueError, match='all input dynamic channels'):
            make_model(kind, dict(TINY, out_channels=1, climatology_anchor_spec=partial_spec))


@pytest.mark.parametrize('kind', ['native', 'process'])
def test_zero_initial_head_decodes_c_not_xt_and_queries_once(kind):
    model, _ = build(kind=kind)
    backbone = getattr(model, 'backbone', model)
    zero_head(backbone.head)
    data = tiny_batch()
    queried = []
    handle = backbone.climatology_anchor.register_forward_hook(lambda module, args, out: queried.append(out))
    model.eval()
    with torch.no_grad():
        out = model(forecast_inputs(data), **({'reasoning_steps': 0} if kind == 'process' else {}))
    handle.remove()
    assert len(queried) == 1
    assert torch.equal(out.forecast, queried[0])
    assert not torch.equal(out.forecast, data['coarse_history'][:, -1])
    if kind == 'native':
        assert torch.equal(out.base_state, data['coarse_history'][:, -1])
        assert out.climatology_anchor is queried[0]
        assert torch.count_nonzero(out.tendency) == 0
    with pytest.raises(ValueError, match='atmos_baseline'):
        model(dict(forecast_inputs(data), atmos_baseline=None))


class EvidenceSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def evidence(self, batch, *, anchor, draft):
        self.seen.append((anchor.detach().clone(), draft.detach().clone()))
        return None

    def inject(self, process, evidence):
        return process


@pytest.mark.parametrize('path', ['fixed', 'adaptive', 'streamed'])
def test_typed_initial_state_retains_physical_xt(path):
    model, _ = build()
    data = tiny_batch()
    model.typed_evidence = EvidenceSpy()
    zero_head(model.backbone.head)
    if path == 'fixed':
        model(forecast_inputs(data), reasoning_steps=2)
    elif path == 'adaptive':
        AdaptiveProcessForecaster(model).eval()(data, max_steps=2, force_full_depth=True)
    else:
        model.train()
        backward_streamed_truncated(model, data, reasoning_steps=2)
    assert len(model.typed_evidence.seen) == 1
    physical, initial = model.typed_evidence.seen[0]
    assert torch.equal(physical, data['coarse_history'][:, -1])
    assert not torch.equal(initial, physical)
    assert torch.equal(initial, model.backbone.climatology_anchor(forecast_inputs(data), data['coarse_history']))


@pytest.mark.parametrize('recurrence', [False, True])
def test_both_solver_proposals_use_c_and_exact_anomaly_feedback(recurrence):
    model, _ = build(solver_state_recurrence=recurrence)
    zero_head(model.backbone.head)
    zero_head(model.proposal_head)
    with torch.no_grad():
        model.solver_gate.score.weight.zero_()
        model.solver_gate.score.bias.zero_()
        base = model.backbone(forecast_inputs(tiny_batch()))
    climate = base.climatology_anchor
    draft = climate + 4
    encoded, proposals = [], []
    eh = model.draft_encoder.register_forward_pre_hook(lambda module, args: encoded.append(args[0]))
    ph = model.proposal_head.register_forward_hook(lambda module, args, out: proposals.append((args[3], out[0])))
    tensors = ProcessStepInput(model.process_queries.expand(2, -1, -1), base.context_tokens, draft, climate)
    out = process_reasoning_step(model, tensors, base.token_hw, anchor=base.base_state)
    eh.remove(); ph.remove()
    assert torch.equal(encoded[0], draft - climate)
    assert proposals[0][0] is climate
    assert torch.equal(proposals[0][1], climate)
    assert torch.equal(out.draft, climate + 2)
    assert torch.equal(out.correction, out.draft - draft)
    assert (out.solver_state is None) == (not recurrence)


@pytest.mark.parametrize('solver,recurrence', [(False, True), (True, False), (True, True)])
def test_correction_routes_keep_absolute_draft(solver, recurrence):
    model, _ = build(local_solver_state=solver, solver_state_recurrence=recurrence,
                     **({'solver_gate_proposal': False} if solver else {}))
    zero_head(model.correction_head, bias=0.25)
    with torch.no_grad():
        base = model.backbone(forecast_inputs(tiny_batch()))
    draft = base.climatology_anchor + 4
    bases = []
    handle = model.correction_head.register_forward_pre_hook(lambda module, args: bases.append(args[3]))
    out = process_reasoning_step(model, ProcessStepInput(model.process_queries.expand(2, -1, -1),
        base.context_tokens, draft, base.climatology_anchor), base.token_hw, anchor=base.base_state)
    handle.remove()
    assert bases[0] is draft
    assert torch.equal(out.draft, draft + 0.25)
    assert torch.equal(out.correction, torch.full_like(draft, 0.25))


@pytest.mark.parametrize('invalid', ['missing', 'shape', 'nan', 'integer', 'grad'])
def test_enabled_shared_step_never_silently_uses_physical_proposal(invalid):
    model, _ = build(draft_query_feedback=False)
    base = model.backbone(forecast_inputs(tiny_batch()))
    climate = base.climatology_anchor.clone()
    if invalid == 'missing':
        climate = None
    elif invalid == 'shape':
        climate = climate[:1]
    elif invalid == 'nan':
        climate.fill_(float('nan'))
    elif invalid == 'integer':
        climate = climate.long()
    else:
        climate.requires_grad_(True)
    with pytest.raises(ValueError, match='climatology_anchor'):
        process_reasoning_step(model, ProcessStepInput(model.process_queries.expand(2, -1, -1),
            base.context_tokens, base.forecast, climate), base.token_hw, anchor=base.base_state)
    with pytest.raises(ValueError, match='anomaly_feedback'):
        model(forecast_inputs(tiny_batch()), reasoning_steps=0, use_forecast_feedback=False)


def test_one_c_query_and_same_tensor_for_every_internal_k(monkeypatch):
    import model.process_forecast_r7 as process_module
    model, _ = build()
    data = tiny_batch()
    queries, steps = [], []
    handle = model.backbone.climatology_anchor.register_forward_hook(lambda module, args, out: queries.append(out))
    def spy(owner, tensors, token_hw, **kwargs):
        steps.append((tensors.climatology_anchor, kwargs['anchor']))
        return process_reasoning_step(owner, tensors, token_hw, **kwargs)
    monkeypatch.setattr(process_module, 'process_reasoning_step', spy)
    model(forecast_inputs(data), reasoning_steps=4)
    handle.remove()
    assert len(queries) == 1 and len(steps) == 4
    assert all(climate is queries[0] for climate, physical in steps)
    assert all(torch.equal(physical, data['coarse_history'][:, -1]) for climate, physical in steps)
    assert 'active_climatology_anchor' not in vars(model)


class MixedController(ForecastGainController):
    def __init__(self):
        super().__init__(4, 2, hidden=8)
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        gain = features.new_ones(features.shape[0])
        if self.calls == 1:
            gain[0] = -1
        return gain, torch.full_like(gain, 20)


def test_adaptive_full_depth_and_subset_slice_c_without_requery():
    model, _ = build()
    model.eval()
    data = tiny_batch()
    with torch.no_grad():
        fixed = model(forecast_inputs(data), reasoning_steps=3)
        once = model(forecast_inputs(data), reasoning_steps=1)
    adapter = AdaptiveProcessForecaster(model).eval()
    full = adapter(data, max_steps=3, force_full_depth=True)
    assert torch.equal(full.forecast, fixed.forecast)
    adapter.controller = MixedController().eval()
    proposals, queries = [], []
    ph = model.proposal_head.register_forward_pre_hook(lambda module, args: proposals.append(args[3].clone()))
    ch = model.backbone.climatology_anchor.register_forward_hook(lambda module, args, out: queries.append(out.clone()))
    mixed = adapter(data, max_steps=3, allow_untrained=True)
    ph.remove(); ch.remove()
    assert len(queries) == 1 and [value.shape[0] for value in proposals] == [2, 1, 1]
    assert torch.equal(proposals[0], queries[0])
    assert all(torch.equal(value, queries[0][1:]) for value in proposals[1:])
    assert mixed.reasoning_steps_per_sample.tolist() == [1, 3]
    torch.testing.assert_close(mixed.forecast[0], once.forecast[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(mixed.forecast[1], fixed.forecast[1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('solver', [False, True])
def test_streamed_matches_retained_truncated_loss_and_all_gradients(solver):
    reference, _ = build(local_solver_state=solver, spacetime_inputs=True, known_context_inputs=True)
    streamed = copy.deepcopy(reference)
    data = tiny_batch()
    out = reference(forecast_inputs(data), reasoning_steps=3, detach_between_steps=True)
    loss = process_forecast_coreasoning_loss(data, out).total
    loss.backward()
    log = backward_streamed_truncated(streamed, data, reasoning_steps=3)
    torch.testing.assert_close(log.total, loss.detach(), rtol=1e-5, atol=1e-6)
    assert torch.equal(log.final_forecast, out.forecast.detach())
    for (name, parameter), (other, actual) in zip(reference.named_parameters(), streamed.named_parameters()):
        assert name == other and (parameter.grad is None) == (actual.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(actual.grad, parameter.grad, rtol=3e-4, atol=3e-6, msg=name)
    assert streamed.backbone.encoder.patch.weight.grad.abs().sum() > 0
    assert streamed.backbone.climatology_anchor.table.grad is None
    assert not log.final_forecast.requires_grad


def test_calibration_and_oracle_match_fixed_teacher_and_preserve_table():
    model, _ = build()
    data = tiny_batch()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    adapter = AdaptiveProcessForecaster(model)
    loss = controller_calibration_loss(adapter, data, max_steps=3)
    loss.total.backward()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert adapter.controller.net[-1].weight.grad is not None
    model.eval()
    with torch.no_grad():
        fixed = model(forecast_inputs(data), reasoning_steps=3)
        errors = torch.stack([per_sample_latitude_mse(draft, data['atmos_target'], data['latitude'])
                              for draft in fixed.draft_forecasts[:, 1:].unbind(1)], dim=1)
    gains, targets = next_step_gain_targets(errors)
    torch.testing.assert_close(loss.target_gains, gains)
    torch.testing.assert_close(loss.continue_targets, targets)
    torch.testing.assert_close(collect_process_errors(model, data, max_steps=3), errors)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


class TargetPoison(dict):
    def __getitem__(self, key):
        if key in {'atmos_target', 'future_target', 'physical_targets', 'process_targets', 'atmos_baseline'}:
            raise AssertionError(f'forbidden inference input {key}')
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default


def test_formal_inputs_ignore_poisoned_targets_and_baseline_without_expanding_whitelist():
    model, _ = build()
    model.eval()
    data = tiny_batch()
    poisoned = TargetPoison(dict(data, future_target=object(), physical_targets=object(), atmos_baseline=object()))
    assert set(forecast_inputs(poisoned)) == set(forecast_inputs(data))
    assert 'atmos_baseline' not in forecast_inputs(poisoned)
    with torch.no_grad():
        assert torch.equal(model(forecast_inputs(poisoned)).forecast, model(forecast_inputs(data)).forecast)
    adapter = AdaptiveProcessForecaster(model).eval()
    assert torch.equal(adapter(poisoned, max_steps=3, force_full_depth=True).forecast,
                       adapter(data, max_steps=3, force_full_depth=True).forecast)


@pytest.mark.parametrize('year,day', [(2017, 31), (2016, 60), (2017, 59), (2016, 366)])
def test_pe_off_calendar_month_leap_year_and_absolute_12_step_history(year, day):
    model, _ = build(kind='native')
    zero_head(model.head)
    model.eval()
    data = tiny_batch(size=1)
    data.update(init_calendar_year=torch.tensor([year]), init_day_of_year=torch.tensor([day]),
                init_utc_hour=torch.tensor([18.0]))
    original = data['coarse_history'].clone()
    seen = []
    handle = model.register_forward_pre_hook(lambda module, args: seen.append({key: value.clone() for key, value in args[0].items()}))
    result = autoregressive_rollout(model, TargetPoison(dict(data, atmos_baseline=object())), lead_hours=(6, 12, 72))
    handle.remove()
    assert not model.spacetime_inputs and uses_climatology_anchor(model)
    assert result.model_calls == len(seen) == 12
    start = dt.datetime(year, 1, 1, 18) + dt.timedelta(days=day - 1)
    spec, table = tiny_spec_table()
    previous = None
    for index, inputs in enumerate(seen):
        init = start + dt.timedelta(hours=6 * index)
        valid = init + dt.timedelta(hours=6)
        assert inputs['init_calendar_year'].item() == init.year
        assert inputs['init_day_of_year'].item() == init.timetuple().tm_yday
        assert inputs['init_utc_hour'].item() == init.hour
        assert inputs['lead_time_hours'].item() == 6
        if previous is not None:
            assert torch.equal(inputs['coarse_history'][:, -1], previous)
        previous = table[spec['bucket_keys'].index([valid.month, valid.hour])][None]
    assert torch.equal(result.forecasts[:, -1], previous)
    assert torch.equal(data['coarse_history'], original)


@pytest.mark.parametrize('field', ['init_calendar_year', 'init_day_of_year', 'init_utc_hour', 'latitude', 'longitude'])
def test_pe_off_rollout_rejects_incomplete_calendar_or_grid(field):
    model, _ = build(kind='native')
    data = tiny_batch(size=1)
    del data[field]
    with pytest.raises(KeyError):
        autoregressive_rollout(model.eval(), data, lead_hours=(6,))
    with pytest.raises(ValueError, match='fixed 6-hour'):
        autoregressive_rollout(model, tiny_batch(size=1), lead_hours=(12,), step_hours=12, history_interval_hours=12)


@pytest.mark.parametrize('long', [False, True])
def test_training_absolute_history_full_physical_bptt_and_constant_internal_c(long):
    model, _ = build()
    model.train()
    data = tiny_batch(size=1)
    data['future_target'] = data['atmos_target'] + 1
    data['physical_targets'] = torch.stack([data['atmos_target'] + index for index in range(12)], dim=1)
    calls, outputs, queries = [], [], []
    def before(module, args):
        calls.append(dict(args[0]))
    def after(module, args, out):
        out.forecast.retain_grad()
        outputs.append(out)
    handles = [model.register_forward_pre_hook(before), model.register_forward_hook(after),
               model.backbone.climatology_anchor.register_forward_hook(lambda module, args, out: queries.append(out))]
    if long:
        result = training_long_rollout(model, data, reasoning_steps=2, physical_weights=[0.0] * 11 + [1.0])
    else:
        result = training_two_step(model, data, reasoning_steps=2)
    for handle in handles:
        handle.remove()
    assert len(calls) == len(queries) == (12 if long else 2)
    for index, inputs in enumerate(calls):
        assert 'atmos_target' not in inputs and 'physical_targets' not in inputs
        assert inputs['lead_time_hours'].item() == 6
        if index:
            assert torch.equal(inputs['coarse_history'][:, -1], outputs[index - 1].forecast)
            assert inputs['coarse_history'].grad_fn is not None
    loss = result.loss if long else result.l12
    loss.backward()
    assert outputs[0].forecast.grad is not None and outputs[0].forecast.grad.abs().sum() > 0
    assert model.backbone.encoder.patch.weight.grad.abs().sum() > 0
    assert all(not value.requires_grad for value in queries)


def test_new_ordinary_checkpoint_top_level_factory_restore_and_tamper_rejection(tmp_path):
    model, options = build()
    model.eval()
    contract = dict(kind='process', model=options, scientific_claim=False, note='synthetic CPU fixture only')
    checkpoint = tmp_path / 'checkpoint.pt'
    save_exclusive(checkpoint, dict(format='r7-local-v1', contract=contract,
                                   signature=canonical_digest(contract), model=model.state_dict()))
    saved = load_checkpoint(checkpoint)
    restored = make_model(saved['contract']['kind'], saved['contract']['model'])
    assert not restored.backbone.climatology_anchor.ready
    restored.load_state_dict(saved['model'], strict=True)
    restored.eval()
    assert restored.backbone.climatology_anchor.ready
    with torch.no_grad():
        assert torch.equal(restored(forecast_inputs(tiny_batch())).forecast,
                           model(forecast_inputs(tiny_batch())).forecast)
    with pytest.raises(RuntimeError, match='one-shot'):
        restored.backbone.climatology_anchor.install(model.backbone.climatology_anchor.table)
    for mutation in ('hash', 'dtype', 'ready', 'missing'):
        bad = {name: value.clone() for name, value in saved['model'].items()}
        if mutation == 'hash':
            bad['backbone.climatology_anchor.table'][0, 0, 0, 0] += 1
        elif mutation == 'dtype':
            bad['backbone.climatology_anchor.table'] = bad['backbone.climatology_anchor.table'].double()
        elif mutation == 'ready':
            bad['backbone.climatology_anchor.ready'].fill_(False)
        else:
            del bad['backbone.climatology_anchor.table']
        with pytest.raises(RuntimeError, match='climatology'):
            make_model('process', options).load_state_dict(bad, strict=True)
    saved['model_code_sha256'] = '0' * 64
    incompatible = tmp_path / 'wrong_code.pt'
    torch.save(saved, incompatible)
    with pytest.raises(ValueError, match='implementation'):
        load_checkpoint(incompatible)
