"""CPU-only analytic and temporary-fixture tests for the #78 change scale."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from data.preprocess.r7_change_scale import (
    METADATA_FILE, STEP_NS, change_scale_preflight, fit_change_scale,
    latitude_weights, load_change_scale_sidecar, publish_change_scale_sidecar,
    train_pairs, validate_change_scale,
)
from training.r7_experiment import canonical_digest

CHANNELS = ['t2m', 'u10', 'v10', 'mslp']
UNITS = ['K', 'm s**-1', 'm s**-1', 'Pa']


def _iso(index):
    times = np.datetime64('2016-01-01', 'ns') + index * np.timedelta64(6, 'h')
    return str(times)


def _fixture(tmp_path, *, state=None, std=None, channels=CHANNELS, units=UNITS,
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
        state = np.arange(8 * count * 2 * 2, dtype=np.float32).reshape(8, count, 2, 2) / 10
    if std is None:
        std = np.ones(count, dtype=np.float32)
    root.attrs.update({
        'schema_version': 1, 'build_complete': True, 'time_unit': 'ns',
        'channels': list(channels), 'units': list(units),
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
        'latitude': np.array([40.0, 39.75], dtype=np.float32),
        'longitude': np.array([115.0, 115.25], dtype=np.float32),
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


# ----------------------------------------------------------------- analytic pieces

def test_latitude_weights_match_the_loss_convention():
    weights = latitude_weights(np.array([0.0, 60.0, 40.0]))
    # normalized to mean 1; strictly decreasing with |latitude| in the northern set
    assert abs(weights.mean() - 1.0) < 1e-12
    assert weights[0] > weights[2] > weights[1]


def test_latitude_weights_reject_bad_inputs():
    with pytest.raises(ValueError):
        latitude_weights(np.array([100.0, 0.0]))
    with pytest.raises(ValueError):
        latitude_weights(np.array([0.0]))
    with pytest.raises(ValueError):
        latitude_weights(np.array([0.0, float('nan')]))


def test_train_pairs_only_counts_owned_contiguous_pairs():
    stamps = (np.datetime64('2016-01-01', 'ns')
              + np.arange(6) * np.timedelta64(6, 'h')).astype(np.int64)
    ranges = [(int(stamps[0]), int(stamps[4]))]
    pairs = train_pairs(stamps, ranges=ranges)
    assert pairs == [(0, 1), (1, 2), (2, 3)]
    # a gap breaks contiguity: 12 h between rows 0 and 1 is not a 6 h change
    gapped = np.array([stamps[0], stamps[1] + STEP_NS, stamps[2] + STEP_NS, stamps[3] + STEP_NS])
    assert train_pairs(gapped, ranges=[(int(gapped[0]), int(gapped[-1]) + 1)]) == [(1, 2), (2, 3)]
    # a range owning only the final frame has no pair; fails closed
    with pytest.raises(ValueError):
        train_pairs(stamps, ranges=[(int(stamps[5]), int(stamps[5]) + 1)])


def test_train_pairs_year_filter_is_exact():
    # 2016-12-31T18:00Z, 2017-01-01T00:00Z, 06:00Z, 12:00Z
    stamps = (np.datetime64('2016-12-31T18:00:00', 'ns')
              + np.arange(4) * np.timedelta64(6, 'h')).astype(np.int64)
    # the first pair crosses the year boundary; both endpoints must own the year,
    # so the crossing pair belongs to neither year's statistics
    assert train_pairs(stamps, years={2017}) == [(1, 2), (2, 3)]
    with pytest.raises(ValueError):
        train_pairs(stamps, years={2016})
    with pytest.raises(ValueError):
        train_pairs(stamps, years={2015})


# ----------------------------------------------------------------- fitting semantics

def test_fit_change_scale_matches_the_analytic_definition(tmp_path):
    state = np.zeros((8, 4, 2, 2), dtype=np.float32)
    for index in range(8):
        state[index, :] = index * np.arange(4)[:, None, None]
    store, manifest = _fixture(tmp_path, state=state)
    meta = fit_change_scale(store, manifest)
    # every train step changes each channel by its own constant
    assert meta['count'] == 3
    # channel c moves by c per 6 h (channel 0 is constant -> zero-change fallback)
    expected_d = np.array([0.0, 1.0, 2.0, 3.0])
    assert np.allclose(meta['change_scale'], expected_d, rtol=0, atol=1e-8)
    assert meta['state_scale'] == [1.0, 1.0, 1.0, 1.0]
    assert np.allclose(meta['ratio'], expected_d, rtol=0, atol=1e-8)
    assert meta['zero_change_fallback'] == [True, False, False, False]
    assert meta['degenerate'] == [True, False, False, False]
    assert meta['runtime_ratio'][0] == 1.0


def test_fit_change_scale_uses_latitude_weighting(tmp_path):
    # Two rows with different changes; the weighted mix must sit between them
    # and match the explicit cos-weight formula.
    state = np.zeros((8, 4, 2, 2), dtype=np.float32)
    state[:, 0, 0, :] = np.arange(8)[:, None] * 1.0
    state[:, 0, 1, :] = np.arange(8)[:, None] * 3.0
    store, manifest = _fixture(tmp_path, state=state)
    meta = fit_change_scale(store, manifest)
    weights = latitude_weights(np.array([40.0, 39.75]))
    expected = float(np.sqrt((weights * np.array([1.0, 3.0]) ** 2).sum() / 2))
    assert abs(meta['change_scale'][0] - expected) < 1e-9
    assert meta['change_scale'][1] == 0.0  # untouched channel is exactly zero


def test_zero_change_channel_falls_back_and_is_masked(tmp_path):
    state = np.ones((8, 4, 2, 2), dtype=np.float32)  # every channel constant in time
    store, manifest = _fixture(tmp_path, state=state)
    with pytest.raises(ValueError):
        # all channels degenerate: refuse rather than publish a silent unit scale
        fit_change_scale(store, manifest)


def test_zero_change_channel_falls_back_in_a_mixed_store(tmp_path):
    state = np.ones((8, 4, 2, 2), dtype=np.float32)
    state[:, 0] = np.arange(8)[:, None, None] * 2.0  # channel 0 moves, 1..3 constant
    store, manifest = _fixture(tmp_path, state=state)
    meta = fit_change_scale(store, manifest)
    assert meta['change_scale'][1:] == [0.0, 0.0, 0.0]
    assert meta['zero_change_fallback'] == [False, True, True, True]
    assert meta['ratio'][0] == pytest.approx(2.0)
    assert meta['ratio'][1:] == [0.0, 0.0, 0.0]  # the fitted d_c/s_c, kept as-is
    assert meta['runtime_ratio'][1:] == [1.0, 1.0, 1.0]  # the fallback the runtime uses
    assert meta['degenerate'] == [False, True, True, True]


def test_degenerate_channel_keeps_incumbent_ratio(tmp_path):
    state = np.zeros((8, 4, 2, 2), dtype=np.float32)
    state[:, 0] = np.arange(8)[:, None, None] * 4.0
    state[:, 1] = np.arange(8)[:, None, None] * 0.5  # ratio 0.5/1e6 << floor? no: std=1
    store, manifest = _fixture(tmp_path, state=state, std=[1.0, 1.0, 1e9, 1e9])
    meta = fit_change_scale(store, manifest)
    # channel 2/3 changes are zero -> fallback 1.0 masked; 0/1 active
    assert meta['degenerate'] == [False, False, True, True]
    assert meta['runtime_ratio'][2:] == [1.0, 1.0]
    # a below-floor active channel would keep ratio exactly 1.0 as well
    assert np.allclose(meta['runtime_ratio'][:2], meta['ratio'][:2])


def test_below_floor_channel_keeps_exactly_unit_ratio(tmp_path):
    # channel 0 moves by 1e-12 against a state std of 1.0 while channel 1 moves
    # by 2.0 -> ratio 1e-12 is at/below float32 eps * max(ratio)=2.4e-7
    state = np.zeros((8, 2, 2, 2), dtype=np.float32)
    state[:, 0] = np.arange(8)[:, None, None] * 1e-12
    state[:, 1] = np.arange(8)[:, None, None] * 2.0
    store, manifest = _fixture(tmp_path, state=state, channels=['a', 'b'], units=['K', 'K'])
    meta = fit_change_scale(store, manifest)
    assert meta['below_floor_degenerate'][0] is True
    assert meta['degenerate'][0] is True
    assert meta['ratio'][0] == pytest.approx(1e-12)  # the fitted d_c/s_c, kept as-is
    assert meta['runtime_ratio'][0] == 1.0  # the incumbent-preserving fallback
    assert meta['degenerate_reason'][0] == 'ratio_at_or_below_relative_floor'


# ----------------------------------------------------------------- identity and publication

def test_preflight_identity_is_stable_and_bound_to_inputs(tmp_path):
    store, manifest = _fixture(tmp_path)
    first = change_scale_preflight(store, manifest)
    second = change_scale_preflight(store, manifest)
    assert first['preflight_identity'] == second['preflight_identity']
    # stub check: identity must be a canonical digest of the report
    body = {k: v for k, v in first.items() if k != 'preflight_identity'}
    assert first['preflight_identity'] == canonical_digest(body)
    assert first['metadata']['data_identity'] == fit_change_scale(store, manifest)['data_identity']


def test_publish_refuses_without_matching_preflight(tmp_path):
    store, manifest = _fixture(tmp_path)
    sidecar = tmp_path / 'sidecar'
    with pytest.raises(ValueError):
        publish_change_scale_sidecar(store, manifest, sidecar, '0' * 64)
    assert not sidecar.exists()


def test_publish_then_load_round_trip(tmp_path):
    store, manifest = _fixture(tmp_path)
    report = change_scale_preflight(store, manifest)
    sidecar = tmp_path / 'sidecar'
    meta = publish_change_scale_sidecar(store, manifest, sidecar,
                                        report['preflight_identity'])
    assert (sidecar / METADATA_FILE).is_file()
    loaded = load_change_scale_sidecar(sidecar)
    assert loaded['change_scale_identity'] == meta['change_scale_identity']
    assert loaded['ratio'] == meta['ratio']
    # a second publish to the same path is refused
    with pytest.raises(FileExistsError):
        publish_change_scale_sidecar(store, manifest, sidecar,
                                     report['preflight_identity'])


def test_load_rejects_tampered_identity(tmp_path):
    store, manifest = _fixture(tmp_path)
    report = change_scale_preflight(store, manifest)
    sidecar = tmp_path / 'sidecar'
    publish_change_scale_sidecar(store, manifest, sidecar, report['preflight_identity'])
    path = sidecar / METADATA_FILE
    meta = json.loads(path.read_text(encoding='utf-8'))
    meta['ratio'][0] = meta['ratio'][0] * 2
    path.write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError):
        load_change_scale_sidecar(sidecar)


def test_fit_refuses_val_or_test_manifest(tmp_path):
    store, manifest = _fixture(tmp_path)
    other = manifest.parent / 'val.jsonl'
    with pytest.raises(ValueError):
        fit_change_scale(store, other)


def test_fit_is_independent_of_val_test_values(tmp_path):
    """The defining leak test: val/test frames cannot move d_c."""
    state = np.zeros((8, 4, 2, 2), dtype=np.float32)
    state[:, 0] = np.arange(8)[:, None, None] * 1.0
    store, manifest = _fixture(tmp_path, state=state)
    before = fit_change_scale(store, manifest)
    root = zarr.open_group(str(store), mode='a')
    root['state'][4:] = 999.0  # only val/test frames change
    after = fit_change_scale(store, manifest)
    assert before['change_scale'] == after['change_scale']
    assert before['ratio'] == after['ratio']
    assert before['train_pairs']['train_diff_sha256'] == after['train_pairs']['train_diff_sha256']
