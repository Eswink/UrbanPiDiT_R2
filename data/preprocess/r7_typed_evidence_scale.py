"""Train-only standardization of the four #79 typed local evidence fields.

The model's typed-evidence pathway standardizes its physical-unit fields before
the shared patch projection. Those statistics must come only from the train
split, must be published as a verifiable artifact, and must never be fitted on
val/test values - the same contract the change-scale (#78) and process-scale
sidecars use, for the same reason.

Fit definition: for every train-owned contiguous 6 h frame pair (both endpoints
inside one declared train half-open range, the pair rule from the change-scale
module) each of the four fields of ``typed_local_fields`` is evaluated on the
*physical* state at both frames. The field mean and population standard
deviation over all train field values become the standardization vectors. A
field whose train standard deviation is zero, or at or below
``float32 eps * max(abs field value)``, is degenerate: it is masked, and the
runtime keeps its ``field_scale``/``field_mean`` as one and zero so a masked
field contributes exactly zero evidence instead of an amplified ratio. A
degenerate field is not repaired with an invented value and the mask is
reported.

Fitting is float64; the runtime buffers are float32; the store is only ever
opened read-only and val/test frames are never loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import fresh_outputs, parse_split_time_ranges
from data.r7_store import require_complete_manifest, validate_store
from model.r7_process_tensor_diagnostics import (
    LOCAL_FIELD_NAMES, TYPED_EVIDENCE_CHANNELS, typed_local_fields,
)
from training.r7_arm_harness import sha256_file
from training.r7_experiment import canonical_digest, dataset_identity

SCHEMA = 'r7-typed-evidence-scale-v1'
PREFLIGHT_SCHEMA = 'r7-typed-evidence-scale-preflight-v1'
STEP_NS = 6 * 3_600_000_000_000
FLOAT32_EPS = float(np.finfo(np.float32).eps)
METADATA_FILE = 'typed_evidence_scale.json'
FIELD_COUNT = len(LOCAL_FIELD_NAMES)
BUILD_COMPLETE_BYTES = json.dumps({'schema_version': 1, 'build_complete': True}).encode('utf-8')
PUBLICATION_CONTRACT = {
    'payload': METADATA_FILE, 'encoding': 'utf-8', 'output_mode': 'x',
    'reservation': 'fresh_outputs(sidecar_dir)', 'completion_marker': 'BUILD_COMPLETE.json',
    'completion_marker_sha256': hashlib.sha256(BUILD_COMPLETE_BYTES).hexdigest(),
    'payload_identity': 'canonical_digest(metadata excluding typed_evidence_identity)',
}
WEIGHT_RULE = ('unweighted population statistics over every train field value: this is an '
               'input-feature standardization, not the latitude-weighted loss, so no case '
               'weighting is applied and none is claimed')
SAMPLE_RULE = ('every train-owned contiguous (t, t + 6 h) pair inside one declared train '
               'half-open range, counted once; both endpoints become one field sample')
FALLBACK_RULE = ('degenerate field: scale/reference = 1.0 and mean = 0.0, masked to exactly '
                 'zero evidence at runtime')
FLOOR_RULE = 'relative floor = float32 eps * max(|field| over train); std at or below it is degenerate'


def _local_path(value):
    if '://' in str(value):
        raise ValueError('local paths only; no download or remote store access')
    return Path(value).resolve()


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _as_float_vector(values, count, name):
    array = np.asarray(values, dtype=np.float64)
    if (array.shape != (count,) or array.dtype.kind not in 'iuf'
            or not np.isfinite(array).all()):
        raise ValueError(f'{name} must be a finite numeric [{count}] vector')
    return array


def latitude_weights(latitude):
    """cos(lat) normalized to mean one over the stored latitude rows."""
    lat = np.asarray(latitude, dtype=np.float64)
    if lat.ndim != 1 or lat.size < 2 or not np.isfinite(lat).all():
        raise ValueError('latitude must be a finite [H] vector')
    if np.abs(lat).max() > 90:
        raise ValueError('latitude must be within +/-90 degrees')
    weight = np.cos(np.deg2rad(lat)).clip(min=0.0)
    mean = float(weight.mean())
    if mean <= 1e-6:
        raise ValueError('cos(latitude) weights are degenerate over this grid')
    return weight / mean


def train_pairs(time_ns, ranges=None, years=None):
    """Train-owned contiguous 6 h index pairs; both endpoints inside one range."""
    stamps = np.asarray(time_ns, dtype=np.int64)
    if stamps.ndim != 1 or stamps.size < 2 or np.any(np.diff(stamps) <= 0):
        raise ValueError('time_ns must be a strictly increasing int64 [T] vector')
    pairs = []
    for index in range(len(stamps) - 1):
        if stamps[index + 1] - stamps[index] != STEP_NS:
            continue
        if ranges is not None:
            if not any(start <= stamps[index] and stamps[index + 1] < stop
                       for start, stop in ranges):
                continue
        if years is not None:
            stamps_year = [int(np.datetime64(int(t), 'ns').astype('datetime64[Y]').astype(int)) + 1970
                           for t in (stamps[index], stamps[index + 1])]
            if any(year not in years for year in stamps_year):
                continue
        pairs.append((index, index + 1))
    if not pairs:
        raise ValueError('no train-owned contiguous 6 h pairs; refusing degenerate statistics')
    return pairs


def _train_selection(root):
    mode = root.attrs.get('split_mode', 'years')
    if mode == 'time_ranges':
        ranges = [parse_split_time_ranges(root.attrs['split_time_ranges'])[key] for key in ('train',)]
        return {'ranges': ranges[0], 'years': None}
    if mode == 'years':
        years = {int(value) for value in root.attrs['split_years']['train']}
        return {'ranges': None, 'years': years}
    raise ValueError('unknown split ownership policy')


def _typed_fields(state, channels, latitude, longitude):
    """``[4,H,W]`` float64 typed fields of one physical frame, via the shared operators."""
    import torch
    if not bool(np.isfinite(state).all()):
        raise ValueError('nonfinite train state')
    tensor = torch.from_numpy(np.ascontiguousarray(state))
    result = typed_local_fields(tensor, channels, torch.from_numpy(latitude),
                                torch.from_numpy(longitude))
    return result.detach().cpu().numpy().astype(np.float64)


def fit_typed_evidence_scale(store, train_manifest):
    """Read-only fit; loads train frames only and never opens val/test values."""
    store, manifest = _local_path(store), _local_path(train_manifest)
    if manifest.name != 'train.jsonl':
        raise ValueError('only train.jsonl may be read for fitting')
    require_complete_manifest(manifest)
    records = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()
               if line.strip()]
    if not records or any(record.get('split') != 'train' for record in records):
        raise ValueError('typed-evidence fitting requires a nonempty actual train manifest')
    for record in records:
        if _local_path(manifest.parent / record['store_path']) != store:
            raise ValueError('train manifest must point only at the declared store')
    import zarr
    root = zarr.open_group(str(store), mode='r')
    validate_store(root)
    selection = _train_selection(root)
    channels, units = list(root.attrs['channels']), list(root.attrs['units'])
    if len(channels) != len(units) or len(set(channels)) != len(channels):
        raise ValueError('store channel/unit metadata mismatch')
    missing = [name for name in TYPED_EVIDENCE_CHANNELS if name not in channels]
    if missing:
        raise ValueError(f'typed evidence requires the diagnostic channels; missing {missing}')
    pairs = train_pairs(np.asarray(root['time_ns'][:]), **selection)
    denorm_mean = np.asarray(root['normalization_mean'][:], dtype=np.float64)
    denorm_std = np.asarray(root['normalization_std'][:], dtype=np.float64)
    if (denorm_mean.shape != (len(channels),) or denorm_std.shape != (len(channels),)
            or np.any(denorm_std <= 0) or not np.isfinite(denorm_mean).all()
            or not np.isfinite(denorm_std).all()):
        raise ValueError('state normalization vectors must be finite [C] with positive std')
    latitude = np.asarray(root['latitude'][:], dtype=np.float64)
    longitude = np.asarray(root['longitude'][:], dtype=np.float64)
    state = root['state']
    touched = sorted({index for pair in pairs for index in pair})
    sums = np.zeros(FIELD_COUNT, dtype=np.float64)
    squares = np.zeros(FIELD_COUNT, dtype=np.float64)
    abs_max = np.zeros(FIELD_COUNT, dtype=np.float64)
    count = 0
    field_bytes = hashlib.sha256()
    lookup = {}
    for index in touched:
        frame = np.asarray(state[index], dtype=np.float64)
        physical = frame * denorm_std[:, None, None] + denorm_mean[:, None, None]
        lookup[index] = _typed_fields(physical, channels, latitude, longitude)
    for first, second in pairs:
        for frame in (lookup[first], lookup[second]):
            field_bytes.update(np.ascontiguousarray(frame.astype(np.float32)).tobytes())
            sums += frame.sum(axis=(-2, -1))
            squares += (frame * frame).sum(axis=(-2, -1))
            abs_max = np.maximum(abs_max, np.abs(frame).max(axis=(-2, -1)))
            count += frame.shape[-2] * frame.shape[-1]
    mean = sums / count
    variance = squares / count - mean * mean
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    floor = FLOAT32_EPS * abs_max
    degenerate = (std == 0) | (std <= floor)
    if bool(degenerate.all()):
        raise ValueError('all typed fields are degenerate; no usable evidence scale')
    active = ~degenerate
    # Active fields are standardized by their own train mean/std; the scale
    # vector is the unit reference (fields are already in physical units, so
    # scale == 1 there). Degenerate fields keep unit scale and zero mean.
    runtime_scale = np.ones(FIELD_COUNT, dtype=np.float64)
    runtime_mean = np.where(degenerate, 0.0, mean)
    runtime_std = np.where(degenerate, 1.0, std)
    with np.errstate(over='ignore', invalid='ignore'):
        scale32 = runtime_scale.astype(np.float32)
        mean32 = runtime_mean.astype(np.float32)
        std32 = runtime_std.astype(np.float32)
    if (not np.isfinite(scale32).all() or not np.isfinite(mean32).all()
            or not np.isfinite(std32).all() or np.any(scale32 <= 0) or np.any(std32 <= 0)):
        raise ValueError('runtime typed-evidence statistics must stay finite and positive')
    identity, _ = dataset_identity(manifest)
    meta = {
        'schema': SCHEMA, 'schema_version': 1, 'fit_split': 'train',
        'scientific_claim': False, 'step_hours': 6, 'fit_dtype': 'float64',
        'runtime_dtype': 'float32', 'count': int(count), 'pair_count': len(pairs),
        'fields': list(LOCAL_FIELD_NAMES), 'channels': channels, 'units': units,
        'weight_rule': WEIGHT_RULE, 'sample_rule': SAMPLE_RULE,
        'fallback_rule': FALLBACK_RULE, 'floor_rule': FLOOR_RULE,
        'field_mean': mean.tolist(), 'field_std': std.tolist(),
        'field_abs_max': abs_max.tolist(), 'relative_floor': floor.tolist(),
        'degenerate': degenerate.tolist(), 'active_mask': active.tolist(),
        'runtime_scale': scale32.tolist(), 'runtime_mean': mean32.tolist(),
        'runtime_std': std32.tolist(),
        'store': str(store), 'train_manifest': str(manifest),
        'data_identity': identity, 'train_manifest_sha256': sha256_file(manifest),
        'train_pairs': {'count': len(pairs), 'first_index': list(pairs[0]),
                        'last_index': list(pairs[-1]), 'frame_union_count': len(touched),
                        'first_time_ns': int(root['time_ns'][pairs[0][0]]),
                        'last_time_ns': int(root['time_ns'][pairs[-1][1]]),
                        'field_sha256': field_bytes.hexdigest(), 'field_dtype': 'float32'},
        'limitations': [
            'standardization statistics only; not evidence of forecast improvement',
            'field statistics depend only on the declared train manifest and train ranges',
            'val/test frames are never loaded for fitting; their values cannot move the statistics',
            'float64 fit, float32 runtime buffers; not a bitwise cross-platform promise',
            'a degenerate field keeps unit scale and zero mean, contributing exactly zero evidence',
            'fields are physical-unit proxies, not causal labels; no conservation law is enforced',
        ],
    }
    validate_typed_evidence_scale(meta)
    return meta


def validate_typed_evidence_scale(meta):
    """Reject malformed, nonfinite, reordered or non-normalizable metadata."""
    try:
        json.dumps(meta, allow_nan=False)
        if (meta['schema'] != SCHEMA or type(meta['schema_version']) is not int
                or meta['schema_version'] != 1 or meta['fit_split'] != 'train'
                or meta['scientific_claim'] is not False):
            raise ValueError('unsupported typed-evidence schema/version/fit split or claim')
        if (meta['step_hours'] != 6 or meta['fit_dtype'] != 'float64'
                or meta['runtime_dtype'] != 'float32' or type(meta['count']) is not int
                or meta['count'] < 1 or type(meta['pair_count']) is not int
                or meta['pair_count'] < 1):
            raise ValueError('invalid step/dtype/count')
        if list(meta['fields']) != list(LOCAL_FIELD_NAMES):
            raise ValueError('typed field names/order must match the frozen contract')
        for rule in ('weight_rule', 'sample_rule', 'fallback_rule', 'floor_rule'):
            if not isinstance(meta[rule], str) or not meta[rule].strip():
                raise ValueError(f'{rule} must be explicit')
        channels, units = meta['channels'], meta['units']
        if (not isinstance(channels, list) or not channels or len(units) != len(channels)
                or any(not isinstance(name, str) or not name for name in channels)):
            raise ValueError('channel/unit metadata mismatch')
        missing = [name for name in TYPED_EVIDENCE_CHANNELS if name not in channels]
        if missing:
            raise ValueError(f'missing diagnostic channels: {missing}')
        vectors = {key: _as_float_vector(meta[key], FIELD_COUNT, key)
                   for key in ('field_mean', 'field_std', 'field_abs_max', 'relative_floor',
                               'runtime_scale', 'runtime_mean', 'runtime_std')}
        if (np.any(vectors['field_std'] < 0) or np.any(vectors['field_abs_max'] < 0)
                or np.any(vectors['relative_floor'] < 0) or np.any(vectors['runtime_scale'] <= 0)
                or np.any(vectors['runtime_std'] <= 0)):
            raise ValueError('std/abs-max/floor must be nonnegative; runtime scale/std positive')
        expected_floor = FLOAT32_EPS * vectors['field_abs_max']
        if np.any(vectors['relative_floor'] != expected_floor):
            raise ValueError('relative floor must derive from float32 epsilon and train abs max')
        degenerate = np.asarray(meta['degenerate'], dtype=bool)
        active = np.asarray(meta['active_mask'], dtype=bool)
        if degenerate.shape != (FIELD_COUNT,) or active.shape != (FIELD_COUNT,):
            raise ValueError('degenerate/active masks must match the field count')
        expected_degenerate = (vectors['field_std'] == 0) | (vectors['field_std'] <= expected_floor)
        if not np.array_equal(degenerate, expected_degenerate):
            raise ValueError('degenerate mask disagrees with std and floor')
        if not np.array_equal(active, ~expected_degenerate):
            raise ValueError('active mask must be the complement of the degenerate mask')
        if bool(degenerate.all()):
            raise ValueError('all typed fields are degenerate')
        if np.any(vectors['runtime_scale'][degenerate] != 1.0) \
                or np.any(vectors['runtime_mean'][degenerate] != 0.0) \
                or np.any(vectors['runtime_std'][degenerate] != 1.0):
            raise ValueError('degenerate fields must keep unit scale and zero mean')
        live = ~degenerate
        if not np.allclose(vectors['runtime_mean'][live], vectors['field_mean'][live],
                           rtol=1e-6, atol=0.0):
            raise ValueError('runtime means must round the fitted means')
        if not np.allclose(vectors['runtime_std'][live], vectors['field_std'][live],
                           rtol=1e-6, atol=0.0):
            raise ValueError('runtime stds must round the fitted stds')
        for key in ('data_identity', 'train_manifest_sha256'):
            if not _is_digest(meta[key]):
                raise ValueError(f'invalid {key}')
        pairs = meta['train_pairs']
        if (type(pairs['count']) is not int or pairs['count'] < 1
                or not _is_digest(pairs['field_sha256'])
                or pairs['frame_union_count'] < 2):
            raise ValueError('invalid train pair provenance')
        for path in (meta['store'], meta['train_manifest']):
            if not isinstance(path, str) or not Path(path).is_absolute() or '://' in path:
                raise ValueError('identity paths must be absolute local paths')
        if Path(meta['train_manifest']).name != 'train.jsonl':
            raise ValueError('invalid train manifest identity')
        if (not isinstance(meta['limitations'], list) or not meta['limitations']
                or any(not isinstance(v, str) or not v for v in meta['limitations'])):
            raise ValueError('limitations must be explicit')
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('invalid typed-evidence metadata') from error
    return meta


def typed_evidence_preflight(store, train_manifest):
    """Read-only report; fit is exactly the train manifest and its train ranges."""
    meta = fit_typed_evidence_scale(store, train_manifest)
    report = {'schema': PREFLIGHT_SCHEMA, 'schema_version': 1,
              'mode': 'read-only-preflight', 'scientific_claim': False,
              'metadata': meta}
    report['preflight_identity'] = canonical_digest(report)
    if dataset_identity(train_manifest)[0] != meta['data_identity']:
        raise ValueError('input data identity changed during preflight')
    return report


def _destination(sidecar_dir, meta):
    requested = Path(sidecar_dir)
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f'refusing existing sidecar: {requested}')
    destination = _local_path(requested)
    inputs = (Path(meta['store']).parent, Path(meta['train_manifest']).parent)
    if any(destination == p or p in destination.parents or destination in p.parents
           for p in inputs):
        raise ValueError('sidecar must be disjoint from all existing input artifact trees')
    return destination


@fresh_outputs('manifest_dir')
def _publish_sidecar(meta, *, manifest_dir):
    # The decorator owns the reservation and writes the completion marker bytes;
    # a second marker write here would violate output_mode 'x'. The parameter is
    # named manifest_dir because that is the name the shared decorator keys on.
    Path(manifest_dir).mkdir(parents=True, exist_ok=False)
    with (Path(manifest_dir) / METADATA_FILE).open('x', encoding='utf-8') as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return meta


def publish_typed_evidence_scale(store, train_manifest, sidecar_dir, preflight_identity):
    """Explicit write gate; re-preflight before any reservation/output is created."""
    if not _is_digest(preflight_identity):
        raise ValueError('an explicit preflight identity is required to publish')
    if Path(sidecar_dir).exists() or Path(sidecar_dir).is_symlink():
        raise FileExistsError(f'refusing existing sidecar: {sidecar_dir}')
    report = typed_evidence_preflight(store, train_manifest)
    if report['preflight_identity'] != preflight_identity:
        raise ValueError('preflight identity mismatch; inputs/statistics changed since review')
    meta = dict(report['metadata'], preflight_identity=preflight_identity)
    destination = _destination(sidecar_dir, meta)
    meta['typed_evidence_identity'] = canonical_digest(meta)
    return _publish_sidecar(meta, manifest_dir=destination)


def load_typed_evidence_scale(path):
    """Require completion, schema, provenance and content identities; no store read."""
    path = _local_path(path)
    if path.is_dir():
        path = path / METADATA_FILE
    if path.name != METADATA_FILE:
        raise ValueError(f'sidecar must be a directory or {METADATA_FILE}')
    require_complete_manifest(path)
    meta = json.loads(path.read_text(encoding='utf-8'))
    validate_typed_evidence_scale(meta)
    if sha256_file(path.parent / 'BUILD_COMPLETE.json') != PUBLICATION_CONTRACT['completion_marker_sha256']:
        raise ValueError('publisher completion marker bytes mismatch')
    identity = meta.get('typed_evidence_identity')
    if not _is_digest(identity):
        raise ValueError('missing typed-evidence identity')
    content = {key: value for key, value in meta.items() if key != 'typed_evidence_identity'}
    if canonical_digest(content) != identity:
        raise ValueError('typed-evidence identity mismatch')
    content.pop('preflight_identity', None)
    if (canonical_digest({'schema': PREFLIGHT_SCHEMA, 'schema_version': 1,
                          'mode': 'read-only-preflight', 'scientific_claim': False,
                          'metadata': content}) != meta.get('preflight_identity')):
        raise ValueError('preflight identity mismatch')
    return meta
