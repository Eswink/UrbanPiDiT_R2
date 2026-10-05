"""CPU-only analytic and temporary-fixture tests for the #79 typed-evidence scale."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from data.preprocess.r7_typed_evidence_scale import (
    FLOAT32_EPS, METADATA_FILE, STEP_NS, fit_typed_evidence_scale,
    load_typed_evidence_scale, publish_typed_evidence_scale, train_pairs,
    typed_evidence_preflight, validate_typed_evidence_scale,
)
from training.r7_experiment import canonical_digest

CHANNELS = ["t2m", "u10", "v10", "mslp", "t850", "u850", "v850", "t500"]
UNITS = ["K", "m s**-1", "m s**-1", "Pa", "K", "m s**-1", "m s**-1", "K"]


def _iso(index):
    times = np.datetime64('2016-01-01', 'ns') + index * np.timedelta64(6, 'h')
    return str(times)


def _state(index, height=4, width=4, channels=len(CHANNELS)):
    """A smooth, divergence-free-ish analytic field set that varies with time."""
    y = np.arange(height, dtype=np.float64)[None, :, None]
    x = np.arange(width, dtype=np.float64)[None, None, :]
    frame = np.zeros((channels, height, width), dtype=np.float64)
    frame[0] = 280.0 + 2.0 * index + 0.5 * y + 0.25 * x     # t2m
    frame[1] = 3.0 + 0.5 * index + 0.1 * y                  # u10
    frame[2] = -2.0 + 0.25 * index - 0.1 * x                # v10
    frame[3] = 101300.0 + 20.0 * y                          # mslp
    frame[4] = 275.0 + 1.0 * index + 0.4 * y                # t850
    frame[5] = 4.0 + 0.5 * index + 0.3 * y - 0.1 * x        # u850
    frame[6] = -3.0 + 0.2 * index + 0.2 * x                 # v850
    frame[7] = 265.0 + 0.8 * index + 0.3 * y                # t500
    return frame.astype(np.float32)


def _fixture(tmp_path, *, state=None, std=None, channels=CHANNELS, units=None,
             train_range=('2016-01-01T00:00:00', '2016-01-02T00:00:00')):
    """A store whose train range owns indices 0..3; val/test own 4..7."""
    base = tmp_path / 'inputs'
    manifests = base / 'manifests'
    manifests.mkdir(parents=True)
    store = base / 'cache.zarr'
    root = zarr.open_group(str(store), mode='w-')
    times = (np.datetime64('2016-01-01', 'ns') + np.arange(8) * np.timedelta64(6, 'h'))
    stamps = times.astype(np.int64)
    count = len(channels)
    if state is None:
        state = np.stack([_state(index, channels=count) for index in range(8)])
    if std is None:
        std = np.ones(count, dtype=np.float32) * 5.0
    root.attrs.update({
        'schema_version': 1, 'build_complete': True, 'time_unit': 'ns',
        'channels': list(channels), 'units': list(units or UNITS[:count]),
        'native_grid_spacing_deg': .25,
        'split_mode': 'time_ranges',
        'split_years': {'train': [2016], 'val': [2017], 'test': [2018]},
        'normalization_years': [2016],
        'split_time_ranges': {
            'train': [list(train_range)], 'val': [[_iso(4), _iso(6)]],
            'test': [[_iso(6), _iso(8)]],
        },
    })
    root.create_array('state', shape=state.shape, dtype='f4')
    root['state'][:] = state
    for name, data in {
        'time_ns': stamps,
        'latitude': np.array([40.0, 39.75, 39.5, 39.25], dtype=np.float32),
        'longitude': np.array([115.0, 115.25, 115.5, 115.75], dtype=np.float32),
        'normalization_mean': np.zeros(count, dtype=np.float32),
        'normalization_std': np.array(std, dtype=np.float32),
    }.items():
        root.create_array(name, data=data)
    records = [{
        'sample_id': f'train_{i}', 'split': 'train', 'store_path': '../cache.zarr',
        'history_indices': [i, i + 1], 'target_index': i + 2,
        'history_times': [_iso(i), _iso(i + 1)], 'init_time': _iso(i + 1),
        'target_time': _iso(i + 2), 'lead_time_hours': 6,
    } for i in range(2)]
    with (manifests / 'train.jsonl').open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')
    (manifests / 'BUILD_COMPLETE.json').write_text(
        json.dumps({'schema_version': 1, 'build_complete': True}), encoding='utf-8')
    (manifests / 'val.jsonl').write_text('', encoding='utf-8')
    return store, manifests / 'train.jsonl'


# ------------------------------------------------------------------ analytic pieces

def test_train_pairs_reuses_the_change_scale_rule():
    stamps = (np.datetime64('2016-01-01', 'ns')
              + np.arange(6) * np.timedelta64(6, 'h')).astype(np.int64)
    pairs = train_pairs(stamps, ranges=[(int(stamps[0]), int(stamps[4]))])
    assert pairs == [(0, 1), (1, 2), (2, 3)]
    gapped = np.array([stamps[0], stamps[1] + STEP_NS, stamps[2] + STEP_NS, stamps[3] + STEP_NS])
    assert train_pairs(gapped, ranges=[(int(gapped[0]), int(gapped[-1]) + 1)]) == [(1, 2), (2, 3)]
    with pytest.raises(ValueError):
        train_pairs(stamps, ranges=[(int(stamps[5]), int(stamps[5]) + 1)])


# ------------------------------------------------------------------ fitting semantics

def test_fit_produces_finite_statistics_and_a_mask(tmp_path):
    store, manifest = _fixture(tmp_path)
    meta = fit_typed_evidence_scale(store, manifest)
    assert meta['count'] > 0 and meta['pair_count'] == 3
    assert len(meta['field_mean']) == 4 and len(meta['field_std']) == 4
    assert all(value > 0 for value in meta['field_std'])
    assert meta['active_mask'] == [True, True, True, True]
    assert meta['degenerate'] == [False, False, False, False]
    assert meta['runtime_scale'] == [1.0, 1.0, 1.0, 1.0]
    # runtime vectors are the documented float32 rounding of the float64 fit
    assert np.allclose(meta['runtime_mean'], meta['field_mean'], rtol=1e-6, atol=0.0)
    assert np.allclose(meta['runtime_std'], meta['field_std'], rtol=1e-6, atol=0.0)
    validate_typed_evidence_scale(meta)


def test_freeze_the_statistics_are_the_train_only_population_moments(tmp_path):
    """The declared definition, recomputed independently in the test."""
    import torch
    from model.r7_process_tensor_diagnostics import typed_local_fields

    store, manifest = _fixture(tmp_path)
    meta = fit_typed_evidence_scale(store, manifest)
    root = zarr.open_group(str(store), mode='r')
    lat = np.asarray(root['latitude'][:], dtype=np.float64)
    lon = np.asarray(root['longitude'][:], dtype=np.float64)
    frames = {}
    norm_std = np.asarray(root['normalization_std'][:], dtype=np.float64)[:, None, None]
    norm_mean = np.asarray(root['normalization_mean'][:], dtype=np.float64)[:, None, None]
    for index in range(4):  # both endpoints of the three pairs
        physical = np.asarray(root['state'][index], dtype=np.float64) * norm_std + norm_mean
        fields = typed_local_fields(torch.from_numpy(physical), CHANNELS,
                                    torch.from_numpy(lat), torch.from_numpy(lon))
        frames[index] = fields.numpy().astype(np.float64)
    # pair multiplicity: the middle frames are endpoints of two pairs each
    order = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)]
    flat = np.stack([frames[frame] for _, frame in order], axis=1).reshape(4, -1)
    assert np.allclose(meta['field_mean'], flat.mean(axis=1), rtol=1e-9, atol=1e-12)
    assert np.allclose(meta['field_std'], flat.std(axis=1), rtol=1e-6, atol=1e-9)


def test_constant_field_is_masked_not_invented(tmp_path):
    """A field that never moves on train is masked, the others stay live.

    Uniform t850 and t500 make the static-stability field and the temperature
    advection field exactly constant (its gradient is identically zero), so
    both are degenerate while the two wind-derived fields remain active.
    """
    state = np.stack([_state(index) for index in range(8)])
    state[:, 4] = 275.0  # t850 uniform in space and time
    state[:, 7] = 265.0  # t500 likewise -> stability is exactly constant
    store, manifest = _fixture(tmp_path, state=state)
    meta = fit_typed_evidence_scale(store, manifest)
    assert meta['degenerate'] == [False, False, True, True]
    for index in (2, 3):
        assert meta['runtime_scale'][index] == 1.0
        assert meta['runtime_mean'][index] == 0.0
        assert meta['runtime_std'][index] == 1.0
    assert meta['active_mask'] == [True, True, False, False]
    validate_typed_evidence_scale(meta)


def test_all_constant_fields_are_refused(tmp_path):
    # Zero wind and zero thermal field: all four typed fields are exactly zero
    # everywhere, so the fit refuses instead of publishing a silent unit scale.
    state = np.zeros((8, len(CHANNELS), 4, 4), dtype=np.float32)
    store, manifest = _fixture(tmp_path, state=state)
    with pytest.raises(ValueError):
        fit_typed_evidence_scale(store, manifest)


def test_fit_refuses_val_or_test_manifest(tmp_path):
    store, manifest = _fixture(tmp_path)
    with pytest.raises(ValueError):
        fit_typed_evidence_scale(store, manifest.parent / 'val.jsonl')


def test_fit_is_independent_of_val_test_values(tmp_path):
    """The defining leak test: val/test frames cannot move the statistics."""
    store, manifest = _fixture(tmp_path)
    before = fit_typed_evidence_scale(store, manifest)
    root = zarr.open_group(str(store), mode='a')
    root['state'][4:] = 999.0
    after = fit_typed_evidence_scale(store, manifest)
    assert before['field_mean'] == after['field_mean']
    assert before['field_std'] == after['field_std']
    assert before['train_pairs']['field_sha256'] == after['train_pairs']['field_sha256']


def test_floor_rule_matches_eps_times_abs_max(tmp_path):
    store, manifest = _fixture(tmp_path)
    meta = fit_typed_evidence_scale(store, manifest)
    expected = [FLOAT32_EPS * value for value in meta['field_abs_max']]
    assert meta['relative_floor'] == pytest.approx(expected)


# ------------------------------------------------------------------ identity/publication

def test_preflight_identity_is_stable_and_bound_to_inputs(tmp_path):
    store, manifest = _fixture(tmp_path)
    first = typed_evidence_preflight(store, manifest)
    second = typed_evidence_preflight(store, manifest)
    assert first['preflight_identity'] == second['preflight_identity']
    body = {k: v for k, v in first.items() if k != 'preflight_identity'}
    assert first['preflight_identity'] == canonical_digest(body)


def test_publish_then_load_round_trip_and_refusals(tmp_path):
    store, manifest = _fixture(tmp_path)
    report = typed_evidence_preflight(store, manifest)
    sidecar = tmp_path / 'sidecar'
    with pytest.raises(ValueError):
        publish_typed_evidence_scale(store, manifest, sidecar, '0' * 64)
    assert not sidecar.exists()
    meta = publish_typed_evidence_scale(store, manifest, sidecar,
                                        report['preflight_identity'])
    assert (sidecar / METADATA_FILE).is_file()
    loaded = load_typed_evidence_scale(sidecar)
    assert loaded['typed_evidence_identity'] == meta['typed_evidence_identity']
    with pytest.raises(FileExistsError):
        publish_typed_evidence_scale(store, manifest, sidecar,
                                     report['preflight_identity'])


def test_load_rejects_tampered_payload(tmp_path):
    store, manifest = _fixture(tmp_path)
    report = typed_evidence_preflight(store, manifest)
    sidecar = tmp_path / 'sidecar'
    publish_typed_evidence_scale(store, manifest, sidecar, report['preflight_identity'])
    path = sidecar / METADATA_FILE
    meta = json.loads(path.read_text(encoding='utf-8'))
    meta['field_mean'][0] = meta['field_mean'][0] + 1.0
    path.write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError):
        load_typed_evidence_scale(sidecar)


def test_validator_rejects_inconsistent_masks(tmp_path):
    store, manifest = _fixture(tmp_path)
    meta = fit_typed_evidence_scale(store, manifest)
    rotten = dict(meta)
    rotten['degenerate'] = [True, False, False, False]
    with pytest.raises(ValueError):
        validate_typed_evidence_scale(rotten)
    rotten = dict(meta)
    rotten['runtime_std'] = list(meta['runtime_std'])
    rotten['runtime_std'][0] = 0.5  # no longer rounds the fitted std
    with pytest.raises(ValueError):
        validate_typed_evidence_scale(rotten)
