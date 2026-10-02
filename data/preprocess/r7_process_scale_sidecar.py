"""Train-only, physical-unit process scaling; immutable stores stay read-only.

Frozen semantics: goals/n2a-m3-process-supervision §3 and main-model V2 §6.
The old 1e-6 is audit information, never a new normalization denominator.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import fresh_outputs, parse_split_time_ranges
from .process_diagnostics import PROCESS_DIAGNOSTIC_NAMES, REQUIRED_PROCESS_CHANNELS
from .r7_preflight import SI_UNITS, _unit, canonical_unit, source_fingerprint
from data.r7_store import require_complete_manifest, validate_record
from training.r7_arm_harness import sha256_file
from training.r7_experiment import canonical_digest, dataset_identity

SCHEMA = 'r7-process-scale-v1'
PREFLIGHT_SCHEMA = 'r7-process-scale-preflight-v1'
PHYSICAL_UNITS = (
    'Pa m-1', 's-1', 's-1', 'K s-1', 'kg kg-1 s-1', 'kg kg-1 s-1', 'K', 'm s-1',
)
ORIGINAL_FLOOR = 1e-6
FLOAT32_EPS = float(np.finfo(np.float32).eps)
METADATA_FILE = 'scale_metadata.json'
BUILD_COMPLETE_BYTES = json.dumps({'schema_version': 1, 'build_complete': True}).encode('utf-8')
PUBLICATION_CONTRACT = {
    'payload': METADATA_FILE, 'encoding': 'utf-8', 'output_mode': 'x',
    'reservation': 'fresh_outputs(manifest_dir)', 'completion_marker': 'BUILD_COMPLETE.json',
    'completion_marker_sha256': hashlib.sha256(BUILD_COMPLETE_BYTES).hexdigest(),
    'payload_identity': 'canonical_digest(metadata excluding sidecar_identity)',
}
SCALE_METHOD = 'train-raw-rms; all-zero reference = 1 physical unit'
FLOOR_METHOD = 'float32-eps * dimensionless-train-abs-max; zero-max reference = 1'
FLOAT_FIELDS = (
    'raw_mean', 'raw_std', 'raw_abs_max', 'physical_unit_scale',
    'dimensionless_mean', 'dimensionless_std', 'dimensionless_abs_max',
    'relative_floor', 'scaled_normalized_std',
)
MASK_FIELDS = ('active_mask', 'degenerate', 'scale_reference_fallback', 'relative_floor_fallback')


def _contract(names, units=PHYSICAL_UNITS):
    if tuple(names) != PROCESS_DIAGNOSTIC_NAMES or tuple(units) != PHYSICAL_UNITS:
        raise ValueError('process names/units must exactly match the ordered physical contract')


def _values(values, *, fitting=False):
    array = np.asarray(values)
    if array.dtype.kind not in 'iuf':
        raise ValueError('process values must be real numeric values')
    array = array.astype(np.float64)
    if array.ndim < 1 or array.shape[-1] != 8 or not array.size:
        raise ValueError('process values need nonempty [..., 8] shape')
    if fitting and array.ndim != 2:
        raise ValueError('fit needs raw [N, 8]')
    if not np.isfinite(array).all():
        raise ValueError('process values contain NaN/Inf')
    return array


def _rms(values):
    """Rescale before squaring, so small dimensional proxies do not underflow."""
    maximum = np.abs(values).max(axis=0)
    divisor = np.where(maximum > 0, maximum, 1.0)
    return maximum * np.sqrt(np.mean((values / divisor) ** 2, axis=0, dtype=np.float64))


def _moments(values):
    # Center relative to an observed value; constants have exactly zero spread.
    shifted = values - values[0]
    offset = shifted.mean(axis=0, dtype=np.float64)
    return values[0] + offset, _rms(shifted - offset)


def _vector(meta, name):
    value = np.asarray(meta[name])
    if value.shape != (8,) or value.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a numeric [8] vector')
    value = value.astype(np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f'{name} contains NaN/Inf')
    return value


def _mask(meta, name):
    value = meta[name]
    if not isinstance(value, list) or len(value) != 8 or any(type(v) is not bool for v in value):
        raise ValueError(f'{name} must contain eight JSON booleans')
    return np.asarray(value, dtype=bool)


def _runtime_parameters(meta):
    active = np.asarray(meta['active_mask'], dtype=bool)
    with np.errstate(over='ignore', under='ignore'):
        scale, mean, std, floor = [np.asarray(meta[k], dtype=np.float32)[active] for k in (
            'physical_unit_scale', 'dimensionless_mean', 'dimensionless_std', 'relative_floor')]
    if (not all(np.isfinite(v).all() for v in (scale, mean, std, floor))
            or np.any(scale <= 0) or np.any(std <= floor) or np.any(floor <= 0)):
        raise ValueError('active scale/std/floor must remain valid at float32 runtime')
    return active, scale, mean, std


def _normalize(values, meta):
    active, scale, mean, std = _runtime_parameters(meta)
    result = np.zeros(values.shape, dtype=np.float32)
    # Do not even evaluate a division in masked channels.
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        try:
            physical = values[..., active].astype(np.float32)
            result[..., active] = (physical / scale - mean) / std
        except FloatingPointError as error:
            raise ValueError('nonfinite float32 process normalization') from error
    if not np.isfinite(result).all():
        raise ValueError('nonfinite float32 process normalization')
    return result


def fit_process_scale(raw, names=PROCESS_DIAGNOSTIC_NAMES, units=PHYSICAL_UNITS, original_std=None):
    """Fit JSON-safe population statistics on the caller's actual train frames.

    Positive references are raw RMS in each proxy's declared physical unit.
    A truly all-zero channel uses one of that physical unit and is masked.
    No floor is used as a divisor: only active actual standard deviations are.
    """
    _contract(names, units)
    values = _values(raw, fitting=True)
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        try:
            raw_mean, raw_std = _moments(values)
            maximum = np.abs(values).max(axis=0)
            fallback = maximum == 0
            scale = np.where(fallback, 1.0, _rms(values))
            dimensionless = values / scale
            mean, std = _moments(dimensionless)
            abs_max = np.abs(dimensionless).max(axis=0)
            floor = FLOAT32_EPS * np.where(abs_max == 0, 1.0, abs_max)
        except FloatingPointError as error:
            raise ValueError('process statistics overflowed') from error
    degenerate = (raw_std == 0) | (std <= floor)
    if degenerate.all():
        raise ValueError('all process channels are degenerate; no usable supervision')
    stored = None if original_std is None else _vector({'original_stored_std': original_std}, 'original_stored_std')
    if stored is not None and np.any(stored <= 0):
        raise ValueError('original stored std must be positive')
    meta = {
        'schema': SCHEMA, 'schema_version': 1, 'fit_split': 'train',
        'scientific_claim': False, 'names': list(names), 'units': list(units),
        'fit_dtype': 'float64', 'runtime_dtype': 'float32', 'count': int(len(values)),
        'scale_method': SCALE_METHOD, 'relative_floor_method': FLOOR_METHOD,
        'physical_unit_reference': [{'value': 1.0, 'unit': unit} for unit in units],
        'original_floor': ORIGINAL_FLOOR,
        'original_stored_std': None if stored is None else stored.tolist(),
        'raw_mean': raw_mean.tolist(), 'raw_std': raw_std.tolist(), 'raw_abs_max': maximum.tolist(),
        'physical_unit_scale': scale.tolist(), 'dimensionless_mean': mean.tolist(),
        'dimensionless_std': std.tolist(), 'dimensionless_abs_max': abs_max.tolist(),
        'relative_floor': floor.tolist(), 'scale_reference_fallback': fallback.tolist(),
        'relative_floor_fallback': (abs_max == 0).tolist(),
        'active_mask': (~degenerate).tolist(), 'degenerate': degenerate.tolist(),
        'degenerate_reason': ['raw_zero_variance' if raw_std[i] == 0 else
                              'scaled_std_at_or_below_relative_floor' if degenerate[i] else 'active'
                              for i in range(8)],
        'limitations': [
            'statistical scale repair only; not evidence of forecast improvement or causal process labels',
            'physical references and population moments depend only on the supplied train frame union',
            'masked channels are exactly zero, including out-of-train values; no inverse is claimed for them',
            'fit is float64; runtime arithmetic/output is float32, not a bitwise cross-platform promise',
        ],
    }
    normalized = _normalize(values, meta).astype(np.float64)
    meta['scaled_normalized_std'] = _moments(normalized)[1].tolist()
    validate_scale_metadata(meta)
    return meta


def validate_scale_metadata(meta):
    """Reject malformed, nonfinite, reordered or non-normalizable scale metadata."""
    try:
        json.dumps(meta, allow_nan=False)
        if (meta['schema'] != SCHEMA or type(meta['schema_version']) is not int
                or meta['schema_version'] != 1 or meta['fit_split'] != 'train'
                or meta['scientific_claim'] is not False):
            raise ValueError('unsupported process scale schema/version/fit split or claim')
        _contract(meta['names'], meta['units'])
        if (meta['fit_dtype'] != 'float64' or meta['runtime_dtype'] != 'float32'
                or type(meta['count']) is not int or meta['count'] < 1):
            raise ValueError('invalid fit count/dtype')
        if meta['scale_method'] != SCALE_METHOD or meta['relative_floor_method'] != FLOOR_METHOD:
            raise ValueError('invalid scale/floor convention')
        if meta['physical_unit_reference'] != [{'value': 1.0, 'unit': u} for u in PHYSICAL_UNITS]:
            raise ValueError('physical unit reference mismatch')
        if type(meta['original_floor']) not in (float, int) or meta['original_floor'] != ORIGINAL_FLOOR:
            raise ValueError('original audit floor must remain 1e-6')
        if (not isinstance(meta['limitations'], list) or not meta['limitations']
                or any(not isinstance(v, str) or not v for v in meta['limitations'])):
            raise ValueError('limitations must be explicit')
        vectors = {k: _vector(meta, k) for k in FLOAT_FIELDS}
        masks = {k: _mask(meta, k) for k in MASK_FIELDS}
        for key in FLOAT_FIELDS:
            if key not in ('raw_mean', 'dimensionless_mean') and np.any(vectors[key] < 0):
                raise ValueError(f'{key} cannot be negative')
        if np.any(vectors['physical_unit_scale'] <= 0) or np.any(vectors['relative_floor'] <= 0):
            raise ValueError('scale and relative floor must be positive')
        zero = vectors['raw_abs_max'] == 0
        if (not np.array_equal(masks['scale_reference_fallback'], zero)
                or np.any(vectors['physical_unit_scale'][zero] != 1.0)):
            raise ValueError('all-zero physical reference must be explicit')
        floor_fallback = vectors['dimensionless_abs_max'] == 0
        expected_floor = FLOAT32_EPS * np.where(floor_fallback, 1.0, vectors['dimensionless_abs_max'])
        if (not np.array_equal(masks['relative_floor_fallback'], floor_fallback)
                or not np.array_equal(vectors['relative_floor'], expected_floor)):
            raise ValueError('relative floor must derive from float32 epsilon and train abs max')
        expected_degenerate = (vectors['raw_std'] == 0) | (vectors['dimensionless_std'] <= expected_floor)
        if (not np.array_equal(masks['degenerate'], expected_degenerate)
                or not np.array_equal(masks['active_mask'], ~expected_degenerate)):
            raise ValueError('active/degenerate masks disagree with actual std/floor')
        if expected_degenerate.all():
            raise ValueError('all process channels are degenerate')
        reasons = ['raw_zero_variance' if vectors['raw_std'][i] == 0 else
                   'scaled_std_at_or_below_relative_floor' if expected_degenerate[i] else 'active'
                   for i in range(8)]
        if meta['degenerate_reason'] != reasons:
            raise ValueError('degenerate reasons do not match statistics')
        if (np.any(vectors['scaled_normalized_std'][expected_degenerate] != 0)
                or np.any(vectors['scaled_normalized_std'][~expected_degenerate] <= 0)):
            raise ValueError('scaled normalized std disagrees with mask')
        if meta['original_stored_std'] is not None and np.any(_vector(meta, 'original_stored_std') <= 0):
            raise ValueError('original stored std must be positive')
        _runtime_parameters(meta)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('invalid process scale metadata') from error
    return meta


def normalize_process_diagnostics(values, metadata, names=PROCESS_DIAGNOSTIC_NAMES):
    """Apply fixed train statistics to [...,8]; masked outputs are exactly zero."""
    _contract(names)
    validate_scale_metadata(metadata)
    return _normalize(_values(values), metadata)


def _local_path(value):
    if '://' in str(value):
        raise ValueError('local paths only; no download or remote store access')
    return Path(value).resolve()


def _source_identity(root, manifest):
    audit_path = manifest.parent / 'source_preflight.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    source = _local_path(audit['source_path'])
    if _local_path(root.attrs['source']) != source or not source.is_file():
        raise ValueError('store/source preflight identity mismatch or non-file source')
    fingerprint = source_fingerprint(source)
    if (fingerprint.get('scope') != 'full-local-file'
            or fingerprint != audit['fingerprint'] or audit.get('schema_version') != 1):
        raise ValueError('source bytes differ from audited full-file fingerprint')
    return {'path': str(source), **fingerprint, 'source_preflight_sha256': sha256_file(audit_path)}


def _train_inputs(store, train_manifest):
    store, manifest = _local_path(store), _local_path(train_manifest)
    # Fail BEFORE opening a named val/test manifest, including a symlink to one.
    if manifest.name != 'train.jsonl':
        raise ValueError('only train.jsonl may be read for fitting')
    require_complete_manifest(manifest)
    records = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not records or any(r.get('split') != 'train' for r in records):
        raise ValueError('scale fitting requires a nonempty actual train manifest')
    for record in records:
        if _local_path(manifest.parent / record['store_path']) != store:
            raise ValueError('train manifest must point only at the declared store')
        if not isinstance(record.get('sample_id'), str) or not record['sample_id']:
            raise ValueError('train sample ids must be nonempty strings')
    identity, reader = dataset_identity(manifest)  # Existing checkpoint/data contract, not a substitute.
    root = reader._store(reader.records[0])
    if (root.attrs.get('physical_units_retained') is not True
            or root.attrs.get('process_diagnostics_enabled') is not True):
        raise ValueError('physical raw process diagnostics are required')
    _contract(root.attrs['process_diagnostic_names'])
    if root['process_diagnostics_raw'].shape != (root['state'].shape[0], 8):
        raise ValueError('raw process diagnostics need [T,8] shape')
    channels, units = list(root.attrs['channels']), list(root.attrs['units'])
    if len(units) != len(channels):
        raise ValueError('store channel/unit shape mismatch')
    for channel in REQUIRED_PROCESS_CHANNELS:
        if channel not in channels or _unit(units[channels.index(channel)]) not in SI_UNITS[canonical_unit(channel)]:
            raise ValueError(f'{channel} physical units missing/mismatched; no guessed conversion')
    mode = root.attrs.get('split_mode', 'years')
    if mode not in ('years', 'time_ranges'):
        raise ValueError('unknown split ownership policy')
    ranges = parse_split_time_ranges(root.attrs['split_time_ranges'])['train'] if mode == 'time_ranges' else None
    selected = set()
    for record in reader.records:
        indices = validate_record(root, record)
        if ranges is not None:
            stamps = [int(root['time_ns'][i]) for i in indices]
            if not any(all(a <= t < b for t in stamps) for a, b in ranges):
                raise ValueError('train window must belong to one half-open train interval')
        selected.update(indices)
    indices = sorted(selected)
    # Never load state, __getitem__(), val/test manifests, or the full proxy array.
    raw = np.stack([np.asarray(root['process_diagnostics_raw'][i]) for i in indices])
    return store, manifest, identity, reader, root, indices, raw


def _preflight_payload(meta):
    return {'schema': PREFLIGHT_SCHEMA, 'schema_version': 1, 'mode': 'read-only-preflight',
            'scientific_claim': False, 'metadata': meta}


def sidecar_preflight(store, train_manifest):
    """Read-only report; fit exactly the train manifest history+target union."""
    store, manifest, identity, reader, root, indices, raw = _train_inputs(store, train_manifest)
    meta = fit_process_scale(raw, original_std=np.asarray(root['process_normalization_std'][:]))
    meta.update({
        'store': str(store), 'train_manifest': str(manifest), 'data_identity': identity,
        'train_manifest_sha256': sha256_file(manifest),
        'fit_frame_selection': 'train-manifest-history-target-union',
        'train_sample_ids': [r['sample_id'] for r in reader.records],
        'train_frame_indices': indices, 'train_time_ns': [int(root['time_ns'][i]) for i in indices],
        'train_diagnostics_sha256': hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
        'train_diagnostics_dtype': raw.dtype.str,
        'source_identity': _source_identity(root, manifest),
        'input_identity_scope': 'existing dataset_identity plus selected train proxy bytes and audited source fingerprint',
        'publication_contract': dict(PUBLICATION_CONTRACT),
    })
    meta['limitations'].extend([
        'existing data_identity hashes store metadata/norms, not every atmospheric chunk',
        'source bytes are hashed opaquely for identity; no source weather arrays are decoded',
        'no val/test manifest or val/test weather label is read or fitted',
        'sidecar identity must be added to a new training contract; old checkpoints are unchanged',
    ])
    if dataset_identity(manifest)[0] != identity:
        raise ValueError('input data identity changed during preflight')
    _validate_provenance(meta)
    payload = _preflight_payload(meta)
    return dict(payload, preflight_identity=canonical_digest(payload))


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _validate_provenance(meta):
    try:
        if meta['publication_contract'] != PUBLICATION_CONTRACT:
            raise ValueError('invalid publisher bytes/encoding/exclusive contract')
        for key in ('data_identity', 'train_manifest_sha256', 'train_diagnostics_sha256'):
            if not _is_digest(meta[key]):
                raise ValueError(f'invalid {key}')
        source = meta['source_identity']
        if (source['scope'] != 'full-local-file' or not _is_digest(source['sha256'])
                or not _is_digest(source['source_preflight_sha256'])
                or type(source['bytes']) is not int or source['bytes'] < 1):
            raise ValueError('invalid audited source identity')
        for path in (meta['store'], meta['train_manifest'], source['path']):
            if not isinstance(path, str) or not Path(path).is_absolute() or '://' in path:
                raise ValueError('identity paths must be absolute local paths')
        if Path(meta['train_manifest']).name != 'train.jsonl':
            raise ValueError('invalid train manifest identity')
        ids, indices, stamps = meta['train_sample_ids'], meta['train_frame_indices'], meta['train_time_ns']
        if (not isinstance(ids, list) or not ids or any(not isinstance(v, str) or not v for v in ids)
                or len(ids) != len(set(ids))):
            raise ValueError('invalid train sample ids')
        if (not isinstance(indices, list) or len(indices) != meta['count']
                or any(type(i) is not int or i < 0 for i in indices) or indices != sorted(set(indices))):
            raise ValueError('invalid train frame union')
        if (not isinstance(stamps, list) or len(stamps) != len(indices)
                or any(type(t) is not int for t in stamps) or stamps != sorted(set(stamps))):
            raise ValueError('invalid train frame times')
        if meta['fit_frame_selection'] != 'train-manifest-history-target-union':
            raise ValueError('invalid fit frame selection')
        if np.dtype(meta['train_diagnostics_dtype']).kind not in 'iuf':
            raise ValueError('invalid train diagnostic dtype')
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('invalid process scale provenance') from error


def _destination(manifest_dir, meta):
    requested = Path(manifest_dir)
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f'refusing existing sidecar: {requested}')
    destination = _local_path(requested)
    inputs = (Path(meta['store']).parent, Path(meta['train_manifest']).parent,
              Path(meta['source_identity']['path']).parent)
    if any(destination == p or p in destination.parents or destination in p.parents for p in inputs):
        raise ValueError('sidecar must be disjoint from all existing input artifact trees')
    return destination


@fresh_outputs('manifest_dir')
def _publish_sidecar(meta, *, manifest_dir):
    Path(manifest_dir).mkdir(parents=True, exist_ok=False)
    with (Path(manifest_dir) / METADATA_FILE).open('x', encoding='utf-8') as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return meta


def publish_process_scale_sidecar(store, train_manifest, manifest_dir, preflight_identity):
    """Explicit write gate; re-preflight before any reservation/output is created."""
    if not _is_digest(preflight_identity):
        raise ValueError('an explicit preflight identity is required to publish')
    if Path(manifest_dir).exists() or Path(manifest_dir).is_symlink():
        raise FileExistsError(f'refusing existing sidecar: {manifest_dir}')
    report = sidecar_preflight(store, train_manifest)
    if report['preflight_identity'] != preflight_identity:
        raise ValueError('preflight identity mismatch; inputs/statistics changed since review')
    meta = dict(report['metadata'], preflight_identity=preflight_identity)
    destination = _destination(manifest_dir, meta)
    meta['sidecar_identity'] = canonical_digest(meta)
    return _publish_sidecar(meta, manifest_dir=destination)


def load_process_scale_sidecar(path):
    """Require completion, schema, provenance and content identities; no store read."""
    path = _local_path(path)
    if path.is_dir():
        path = path / METADATA_FILE
    if path.name != METADATA_FILE:
        raise ValueError(f'sidecar must be a directory or {METADATA_FILE}')
    require_complete_manifest(path)
    meta = json.loads(path.read_text(encoding='utf-8'))
    validate_scale_metadata(meta)
    _validate_provenance(meta)
    if sha256_file(path.parent / 'BUILD_COMPLETE.json') != PUBLICATION_CONTRACT['completion_marker_sha256']:
        raise ValueError('publisher completion marker bytes mismatch')
    if not _is_digest(meta.get('sidecar_identity')) or not _is_digest(meta.get('preflight_identity')):
        raise ValueError('missing sidecar/preflight identity')
    identity = meta['sidecar_identity']
    content = {k: v for k, v in meta.items() if k != 'sidecar_identity'}
    if canonical_digest(content) != identity:
        raise ValueError('sidecar identity mismatch')
    content.pop('preflight_identity')
    if canonical_digest(_preflight_payload(content)) != meta['preflight_identity']:
        raise ValueError('preflight identity mismatch')
    return meta
