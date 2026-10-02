"""CPU-only analytic and temporary-fixture tests; no real/test archive reads."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import zarr

from data.preprocess.process_diagnostics import PROCESS_DIAGNOSTIC_NAMES, REQUIRED_PROCESS_CHANNELS
from data.preprocess.r7_preflight import source_fingerprint
from data.preprocess.r7_process_scale_sidecar import (
    METADATA_FILE, PHYSICAL_UNITS, fit_process_scale, load_process_scale_sidecar,
    normalize_process_diagnostics, publish_process_scale_sidecar,
    sidecar_preflight, validate_scale_metadata,
)
from training.r7_experiment import canonical_digest, dataset_identity

ROOT = Path(__file__).resolve().parents[1]


def _raw(count=32):
    phases = np.arange(count, dtype=np.float64)[:, None] * np.arange(1, 9)[None, :]
    values = np.sin(phases / 5) + np.arange(8)[None, :] / 4
    return values * np.array([1e-3, 1e-5, 1e-5, 1e-4, 1e-9, 1e-8, 10, 5])


def _json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, allow_nan=False)


def _fixture(tmp_path):
    """Seven declared train frames, but the actual train union contains six."""
    base = tmp_path / 'inputs'
    manifests = base / 'manifests'
    manifests.mkdir(parents=True)
    source = base / 'source.nc'
    source.write_bytes(b'synthetic engineering source identity, not weather truth')
    store = base / 'cache.zarr'
    root = zarr.open_group(str(store), mode='w-')
    times = (np.datetime64('2016-01-01', 'ns') + np.arange(12) * np.timedelta64(6, 'h'))
    stamps = times.astype(np.int64)
    iso = lambda i: str(times[i])
    root.attrs.update({
        'schema_version': 1, 'build_complete': True, 'time_unit': 'ns',
        'channels': list(REQUIRED_PROCESS_CHANNELS),
        'units': ['Pa', 'K', 'kg kg-1', 'm s-1', 'm s-1', 'K', 'm s-1', 'm s-1'],
        'native_grid_spacing_deg': .25, 'physical_units_retained': True,
        'normalization_years': [2016], 'split_years': {'train': [2016], 'val': [2017], 'test': [2018]},
        'process_diagnostics_enabled': True, 'process_diagnostic_names': list(PROCESS_DIAGNOSTIC_NAMES),
        'split_mode': 'time_ranges', 'split_time_ranges': {
            'train': [[iso(0), iso(7)]], 'val': [[iso(7), iso(9)]],
            'test': [[iso(9), '2016-01-04T00:00:00']],
        }, 'source': str(source.resolve()),
    })
    root.create_array('state', shape=(12, 8, 3, 3), dtype='f4')
    for name, data in {
        'time_ns': stamps, 'latitude': np.array([40, 39.75, 39.5], dtype=np.float32),
        'longitude': np.array([115, 115.25, 115.5], dtype=np.float32),
        'normalization_mean': np.zeros(8, dtype=np.float32),
        'normalization_std': np.ones(8, dtype=np.float32),
        'process_normalization_mean': np.zeros(8, dtype=np.float32),
        'process_normalization_std': np.full(8, 1e-6, dtype=np.float32),
        'process_diagnostics_raw': _raw(12).astype(np.float32),
    }.items():
        root.create_array(name, data=data)
    records = [{
        'sample_id': f'train_{i}', 'split': 'train', 'store_path': '../cache.zarr',
        'history_indices': [i, i + 1], 'target_index': i + 2,
        'history_times': [iso(i), iso(i + 1)], 'init_time': iso(i + 1),
        'target_time': iso(i + 2), 'lead_time_hours': 6,
    } for i in range(4)]
    manifest = manifests / 'train.jsonl'
    with manifest.open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')
    _json(manifests / 'BUILD_COMPLETE.json', {'schema_version': 1, 'build_complete': True})
    _json(manifests / 'source_preflight.json', {
        'schema_version': 1, 'source_path': str(source.resolve()),
        'fingerprint': source_fingerprint(source), 'scientific_training_certified': False,
    })
    return store, manifest


def _files(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}


def _rewrite_records(manifest, change):
    records = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
    change(records)
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')


def test_rms_population_scaling_and_old_floor_audit():
    raw = _raw()
    old_std = np.maximum(raw.std(axis=0), 1e-6)
    meta = fit_process_scale(raw, original_std=old_std)
    validate_scale_metadata(json.loads(json.dumps(meta, allow_nan=False)))
    scale = np.sqrt(np.mean(raw * raw, axis=0))
    scaled = raw / scale
    np.testing.assert_allclose(meta['physical_unit_scale'], scale, rtol=1e-14)
    np.testing.assert_allclose(meta['raw_mean'], raw.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(meta['raw_std'], raw.std(axis=0), rtol=1e-14)
    np.testing.assert_allclose(meta['dimensionless_mean'], scaled.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(meta['dimensionless_std'], scaled.std(axis=0), rtol=1e-14)
    np.testing.assert_allclose(meta['dimensionless_abs_max'], np.abs(scaled).max(axis=0), rtol=1e-14)
    np.testing.assert_array_equal(meta['relative_floor'], np.finfo(np.float32).eps * np.asarray(meta['dimensionless_abs_max']))
    normalized = normalize_process_diagnostics(raw, meta)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized.std(axis=0), 1, atol=2e-7)
    np.testing.assert_allclose(meta['scaled_normalized_std'], normalized.astype(np.float64).std(axis=0), rtol=1e-14)
    assert meta['original_floor'] == 1e-6
    assert meta['original_stored_std'] == old_std.tolist()
    assert meta['active_mask'] == [True] * 8
    assert meta['scientific_claim'] is False
    assert meta['schema'] == 'r7-process-scale-v1'
    assert meta['fit_split'] == 'train'
    assert meta['units'] == list(PHYSICAL_UNITS)


def test_dimensioned_unit_change_is_reversible_on_active_channels():
    raw = _raw()
    factor = np.array([100, .1, 10, .001, 1e3, 1e-3, 100, .1])
    a, b = fit_process_scale(raw), fit_process_scale(raw * factor)
    za = normalize_process_diagnostics(raw, a)
    zb = normalize_process_diagnostics(raw * factor, b)
    np.testing.assert_allclose(za, zb, rtol=2e-6, atol=5e-7)
    restored = (za.astype(np.float64) * a['dimensionless_std'] + a['dimensionless_mean']) * a['physical_unit_scale']
    np.testing.assert_allclose(restored, raw, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(b['physical_unit_scale'], np.asarray(a['physical_unit_scale']) * factor, rtol=1e-14)
    assert a['active_mask'] == b['active_mask']


def test_zero_constant_and_near_constant_are_masked_without_floor_division():
    raw = _raw()
    raw[:, 0] = 0
    raw[:, 1] = 4.2
    raw[:, 2] = 1e-5 + np.arange(len(raw)) * 1e-15
    meta = fit_process_scale(raw)
    assert meta['degenerate'][:3] == [True, True, True]
    assert meta['active_mask'][:3] == [False, False, False]
    assert meta['degenerate_reason'][:3] == ['raw_zero_variance', 'raw_zero_variance', 'scaled_std_at_or_below_relative_floor']
    assert meta['physical_unit_scale'][0] == 1.0
    assert meta['physical_unit_reference'][0] == {'value': 1.0, 'unit': 'Pa m-1'}
    assert meta['scale_reference_fallback'][0] is True
    assert meta['relative_floor_fallback'][0] is True
    assert meta['scaled_normalized_std'][:3] == [0.0] * 3
    poison = raw.copy()
    poison[:, :3] = 1e300  # Never cast/divide masked values.
    with np.errstate(all='raise'):
        normalized = normalize_process_diagnostics(poison, meta)
    np.testing.assert_array_equal(normalized[:, :3], 0)
    assert np.isfinite(normalized).all()


def test_centered_high_offset_population_moments():
    raw = _raw()
    raw[:, 7] = 1e12 + np.arange(len(raw)) * 1e5
    meta = fit_process_scale(raw)
    np.testing.assert_allclose(meta['raw_std'], raw.std(axis=0), rtol=1e-12)
    assert meta['active_mask'] == [True] * 8
    assert all(v > 0 for v in meta['physical_unit_scale'])
    with pytest.raises(ValueError, match='float32'):
        normalize_process_diagnostics(np.full((2, 8), 1e300), fit_process_scale(_raw()))


def test_active_float32_parameter_underflow_and_overflow_are_refused():
    for factor in (1e-100, 1e100):
        with pytest.raises(ValueError, match='float32'):
            fit_process_scale(_raw() * factor)


@pytest.mark.parametrize('raw', [np.zeros((0, 8)), np.zeros((3, 7)), np.zeros(8),
                                     np.zeros((2, 2, 8)), np.ones((3, 8), dtype=complex),
                                     np.zeros((3, 8), dtype=bool),
                                     np.full((3, 8), np.nan), np.full((3, 8), np.inf)])
def test_fit_rejects_nonfinite_or_wrong_shapes(raw):
    with pytest.raises(ValueError):
        fit_process_scale(raw)


@pytest.mark.parametrize('raw', [np.zeros((3, 8)), np.full((3, 8), 7.0), _raw(1)])
def test_all_degenerate_is_explicitly_refused(raw):
    with pytest.raises(ValueError, match='all process channels are degenerate'):
        fit_process_scale(raw)


@pytest.mark.parametrize('change', ['names', 'units', 'old_shape', 'old_zero', 'old_nan'])
def test_fit_rejects_name_unit_and_original_std_errors(change):
    kwargs = {}
    if change == 'names':
        kwargs['names'] = PROCESS_DIAGNOSTIC_NAMES[::-1]
    elif change == 'units':
        kwargs['units'] = ['unknown'] * 8
    else:
        kwargs['original_std'] = {'old_shape': np.ones(7), 'old_zero': np.zeros(8), 'old_nan': np.full(8, np.nan)}[change]
    with pytest.raises(ValueError):
        fit_process_scale(_raw(), **kwargs)


@pytest.mark.parametrize('key,value', [
    ('schema', 'wrong'), ('schema_version', 2), ('schema_version', True),
    ('fit_split', 'val'), ('scientific_claim', True), ('count', 0),
    ('count', True), ('units', ['K'] * 8), ('names', list(PROCESS_DIAGNOSTIC_NAMES[::-1])),
    ('relative_floor', [1e-6] * 8), ('dimensionless_std', [0.0] * 8),
    ('physical_unit_scale', [0.0] * 8), ('raw_mean', [np.nan] * 8),
    ('active_mask', [1] * 8), ('degenerate', [True] * 8),
    ('raw_std', [0.0] * 8), ('original_stored_std', [-1.0] * 8),
    ('physical_unit_reference', []), ('dimensionless_mean', [0.0] * 7),
    ('limitations', []), ('scale_reference_fallback', [True] * 8),
    ('degenerate_reason', ['active'] * 7),
])
def test_metadata_tampering_is_refused(key, value):
    meta = fit_process_scale(_raw())
    meta[key] = value
    with pytest.raises(ValueError):
        validate_scale_metadata(meta)


@pytest.mark.parametrize('value', [None, [], {}, {'schema': 'wrong'}])
def test_missing_metadata_is_refused(value):
    with pytest.raises(ValueError):
        validate_scale_metadata(value)


@pytest.mark.parametrize('value', [np.zeros((2, 7)), np.zeros((0, 8)), np.full((2, 8), np.nan)])
def test_runtime_shape_nonfinite_and_name_guards(value):
    meta = fit_process_scale(_raw())
    with pytest.raises(ValueError):
        normalize_process_diagnostics(value, meta)
    with pytest.raises(ValueError, match='names/units'):
        normalize_process_diagnostics(_raw(), meta, names=PROCESS_DIAGNOSTIC_NAMES[::-1])


def test_preflight_is_read_only_and_reads_only_actual_train_union(tmp_path, monkeypatch):
    store, manifest = _fixture(tmp_path)
    before = _files(tmp_path)
    original = zarr.Array.__getitem__
    reads = []
    def guarded(array, selection):
        if array.path == 'state':
            raise AssertionError('preflight must not decode atmospheric labels')
        if array.path == 'process_diagnostics_raw':
            assert isinstance(selection, (int, np.integer)) and 0 <= selection < 6
            reads.append(int(selection))
        return original(array, selection)
    monkeypatch.setattr(zarr.Array, '__getitem__', guarded)
    report = sidecar_preflight(store, manifest)
    meta = report['metadata']
    assert _files(tmp_path) == before
    assert reads == list(range(6))
    assert meta['count'] == 6
    assert meta['train_frame_indices'] == list(range(6))
    assert meta['train_sample_ids'] == [f'train_{i}' for i in range(4)]
    assert meta['data_identity'] == dataset_identity(manifest)[0]
    assert meta['original_stored_std'] == np.full(8, 1e-6, dtype=np.float32).astype(np.float64).tolist()
    assert report['preflight_identity'] == canonical_digest({k: v for k, v in report.items() if k != 'preflight_identity'})
    assert report == sidecar_preflight(store, manifest)


def test_unused_train_and_held_out_proxy_changes_cannot_move_fit_or_identity(tmp_path):
    store, manifest = _fixture(tmp_path)
    report = sidecar_preflight(store, manifest)
    root = zarr.open_group(str(store), mode='r+')
    root['process_diagnostics_raw'][6:] = np.nan
    root['state'][:] = np.nan
    after = sidecar_preflight(store, manifest)
    assert report == after


@pytest.mark.parametrize('change', ['split', 'boundary', 'time', 'store', 'duplicate', 'sample_id'])
def test_manifest_ownership_and_identity_guards(tmp_path, change):
    store, manifest = _fixture(tmp_path)
    def corrupt(records):
        if change == 'split':
            records[0]['split'] = 'val'
        elif change == 'boundary':
            records[0].update(history_indices=[5, 6], target_index=7,
                              history_times=['2016-01-02T06:00:00', '2016-01-02T12:00:00'],
                              init_time='2016-01-02T12:00:00', target_time='2016-01-02T18:00:00')
        elif change == 'time':
            records[0]['target_time'] = '2016-01-01T18:00:00'
        elif change == 'store':
            records[0]['store_path'] = '../unrelated.zarr'
        elif change == 'sample_id':
            records[0]['sample_id'] = 1
        else:
            records[1]['sample_id'] = records[0]['sample_id']
    _rewrite_records(manifest, corrupt)
    with pytest.raises(ValueError):
        sidecar_preflight(store, manifest)


@pytest.mark.parametrize('name', ['val.jsonl', 'test.jsonl'])
def test_held_out_manifest_path_rejected_before_read(tmp_path, name):
    store, manifest = _fixture(tmp_path)
    # Deliberately nonexistent: a read would raise FileNotFoundError instead.
    with pytest.raises(ValueError, match='only train.jsonl'):
        sidecar_preflight(store, manifest.with_name(name))


@pytest.mark.parametrize('change', ['missing_marker', 'bad_marker', 'incomplete', 'units', 'names', 'source', 'nan'])
def test_store_source_and_publication_guards(tmp_path, change):
    store, manifest = _fixture(tmp_path)
    root = zarr.open_group(str(store), mode='r+')
    if change == 'missing_marker':
        (manifest.parent / 'BUILD_COMPLETE.json').unlink()
    elif change == 'bad_marker':
        (manifest.parent / 'BUILD_COMPLETE.json').write_text('{}', encoding='utf-8')
    elif change == 'incomplete':
        root.attrs['build_complete'] = False
    elif change == 'units':
        root.attrs['units'] = ['unknown'] * 8
    elif change == 'names':
        root.attrs['process_diagnostic_names'] = list(PROCESS_DIAGNOSTIC_NAMES[::-1])
    elif change == 'source':
        (store.parent / 'source.nc').write_bytes(b'changed fixture source')
    else:
        root['process_diagnostics_raw'][0] = np.nan
    with pytest.raises(ValueError):
        sidecar_preflight(store, manifest)


def test_publisher_roundtrip_stats_only_and_old_inputs_unchanged(tmp_path):
    store, manifest = _fixture(tmp_path)
    output = tmp_path / 'sidecar'
    before = _files(store.parent)
    report = sidecar_preflight(store, manifest)
    meta = publish_process_scale_sidecar(store, manifest, output, report['preflight_identity'])
    loaded = load_process_scale_sidecar(output)
    assert loaded == meta == load_process_scale_sidecar(output / METADATA_FILE)
    assert _files(store.parent) == before
    assert sorted(p.name for p in output.iterdir()) == ['BUILD_COMPLETE.json', METADATA_FILE]
    assert meta['sidecar_identity'] == canonical_digest({k: v for k, v in meta.items() if k != 'sidecar_identity'})
    assert meta['preflight_identity'] == report['preflight_identity']
    assert not list(tmp_path.glob('*.r7-build.lock'))
    baseline = _files(output)
    with pytest.raises(FileExistsError):
        publish_process_scale_sidecar(store, manifest, output, report['preflight_identity'])
    assert _files(output) == baseline


@pytest.mark.parametrize('change', ['raw', 'std', 'manifest', 'source', 'wrong', 'missing'])
def test_stale_review_cannot_publish_or_leave_reservation(tmp_path, change):
    store, manifest = _fixture(tmp_path)
    report = sidecar_preflight(store, manifest)
    identity = report['preflight_identity']
    root = zarr.open_group(str(store), mode='r+')
    if change == 'raw':
        root['process_diagnostics_raw'][0] = _raw(12)[0].astype(np.float32) * 3
    elif change == 'std':
        root['process_normalization_std'][:] = 1e-5
    elif change == 'manifest':
        _rewrite_records(manifest, lambda r: r.pop())
    elif change == 'source':
        (store.parent / 'source.nc').write_bytes(b'changed source')
    elif change == 'wrong':
        identity = '0' * 64
    else:
        identity = None
    output = tmp_path / 'sidecar'
    before = _files(tmp_path)
    with pytest.raises(ValueError):
        publish_process_scale_sidecar(store, manifest, output, identity)
    assert not output.exists()
    assert _files(tmp_path) == before


@pytest.mark.parametrize('destination', ['inside_store', 'inside_manifests', 'inside_source_parent', 'ancestor', 'symlink'])
def test_publisher_rejects_nested_existing_and_symlink_destinations(tmp_path, destination):
    store, manifest = _fixture(tmp_path)
    identity = sidecar_preflight(store, manifest)['preflight_identity']
    targets = {'inside_store': store / 'sidecar', 'inside_manifests': manifest.parent / 'sidecar',
               'inside_source_parent': store.parent / 'sidecar', 'ancestor': tmp_path,
               'symlink': tmp_path / 'sidecar_link'}
    output = targets[destination]
    if destination == 'symlink':
        output.symlink_to(tmp_path / 'missing_target', target_is_directory=True)
    before = _files(store.parent)
    with pytest.raises((ValueError, FileExistsError)):
        publish_process_scale_sidecar(store, manifest, output, identity)
    assert _files(store.parent) == before
    assert not list(tmp_path.rglob('*.r7-build.lock'))


def test_failed_write_remains_incomplete_and_cannot_load(tmp_path, monkeypatch):
    import data.preprocess.r7_process_scale_sidecar as module
    store, manifest = _fixture(tmp_path)
    identity = sidecar_preflight(store, manifest)['preflight_identity']
    output = tmp_path / 'sidecar'
    original = module.json.dump
    def fail(payload, stream, **kwargs):
        stream.write('{')
        raise RuntimeError('injected payload write failure')
    monkeypatch.setattr(module.json, 'dump', fail)
    with pytest.raises(RuntimeError, match='injected'):
        publish_process_scale_sidecar(store, manifest, output, identity)
    monkeypatch.setattr(module.json, 'dump', original)
    assert (output / METADATA_FILE).is_file()
    assert not (output / 'BUILD_COMPLETE.json').exists()
    assert not list(tmp_path.glob('*.r7-build.lock'))
    with pytest.raises(ValueError, match='incomplete'):
        load_process_scale_sidecar(output)
    with pytest.raises(FileExistsError):
        publish_process_scale_sidecar(store, manifest, output, identity)


@pytest.mark.parametrize('change', ['payload', 'identity', 'missing_identity', 'marker', 'marker_bytes', 'provenance'])
def test_loader_refuses_tampered_payload_identity_and_marker(tmp_path, change):
    store, manifest = _fixture(tmp_path)
    identity = sidecar_preflight(store, manifest)['preflight_identity']
    output = tmp_path / 'sidecar'
    meta = publish_process_scale_sidecar(store, manifest, output, identity)
    if change == 'marker':
        (output / 'BUILD_COMPLETE.json').write_text('{}', encoding='utf-8')
    elif change == 'marker_bytes':
        (output / 'BUILD_COMPLETE.json').write_text('{"schema_version":1,"build_complete":true}', encoding='utf-8')
    else:
        altered = copy.deepcopy(meta)
        if change == 'payload':
            altered['raw_mean'][0] += 1
        elif change == 'identity':
            altered['sidecar_identity'] = '0' * 64
        elif change == 'missing_identity':
            altered.pop('sidecar_identity')
        else:
            altered['train_frame_indices'] = [0] * meta['count']
        (output / METADATA_FILE).write_text(json.dumps(altered), encoding='utf-8')
    with pytest.raises(ValueError):
        load_process_scale_sidecar(output)


def test_cli_defaults_to_stdout_only_and_requires_review_for_write(tmp_path):
    store, manifest = _fixture(tmp_path)
    output = tmp_path / 'sidecar'
    command = [sys.executable, '-B', str(ROOT / 'scripts/prepare_r7_process_scale.py'),
               '--store', str(store), '--train-manifest', str(manifest), '--manifest-dir', str(output)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    before = _files(tmp_path)
    process = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    report = json.loads(process.stdout)
    assert report['mode'] == 'read-only-preflight'
    assert _files(tmp_path) == before
    assert not output.exists()
    refused = subprocess.run(command + ['--write'], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert refused.returncode != 0
    assert not output.exists()
    published = subprocess.run(command + ['--write', '--preflight-identity', report['preflight_identity']],
                               cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    written = json.loads(published.stdout)
    assert written['mode'] == 'published-process-scale-sidecar'
    assert load_process_scale_sidecar(output)['sidecar_identity'] == written['sidecar_identity']
