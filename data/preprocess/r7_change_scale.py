"""Train-only 6 h change scale for the #78 R-A decode reparameterization.

Read-only fitting plus an explicit publish gate; the store is never modified.
The frozen semantics live in issue #78 R-A: fit ``d_c`` (per-channel RMS of the
exact 6 h physical change) on train-only contiguous windows, combine it with
the store's train-only state std ``s_c`` into the dimensionless ratio
``d_c / s_c``, and let the decoder write its dimensionless tendency ``r_c`` in
the existing normalized-state space as ``Y = X_t + (d_c / s_c) * r_c``. The
loss is unchanged: this is a numerical reparameterization, not a reweighting.

Conventions, all explicit: physical units come from the store's own channel
metadata; the horizontal weight is ``cos(latitude)`` normalized to mean one,
the same weighting the training loss uses; every train-owned contiguous
``(t, t + 6 h)`` pair is one sample, counted once; a channel whose change is
exactly zero falls back to one physical unit and is masked; the relative floor
is ``float32 eps * max(ratio)`` and a ratio at or below it is degenerate (kept
at exactly 1.0, i.e. incumbent behavior for that channel) rather than being
amplified. Fitting is float64; the runtime buffer is float32.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import fresh_outputs, parse_split_time_ranges
from data.r7_store import require_complete_manifest, validate_store
from training.r7_arm_harness import sha256_file
from training.r7_experiment import canonical_digest, dataset_identity

SCHEMA = 'r7-change-scale-v1'
PREFLIGHT_SCHEMA = 'r7-change-scale-preflight-v1'
STEP_NS = 6 * 3_600_000_000_000
FLOAT32_EPS = float(np.finfo(np.float32).eps)
METADATA_FILE = 'change_scale.json'
BUILD_COMPLETE_BYTES = json.dumps({'schema_version': 1, 'build_complete': True}).encode('utf-8')
PUBLICATION_CONTRACT = {
    'payload': METADATA_FILE, 'encoding': 'utf-8', 'output_mode': 'x',
    'reservation': 'fresh_outputs(sidecar_dir)', 'completion_marker': 'BUILD_COMPLETE.json',
    'completion_marker_sha256': hashlib.sha256(BUILD_COMPLETE_BYTES).hexdigest(),
    'payload_identity': 'canonical_digest(metadata excluding change_scale_identity)',
}
WEIGHT_RULE = ('cos(latitude) normalized to mean 1 over the grid; the same latitude '
               'weighting the latitude-weighted MSE uses')
SAMPLE_RULE = ('every train-owned contiguous (t, t + 6 h) pair inside one declared '
               'train half-open range, counted once')
FALLBACK_RULE = 'zero-change channel: d_c reference = 1.0 of its physical unit; runtime ratio 1.0'
FLOOR_RULE = 'relative floor = float32 eps * max(ratio over all channels)'


def _local_path(value):
    if '://' in str(value):
        raise ValueError('local paths only; no download or remote store access')
    return Path(value).resolve()


def _as_float_vector(values, count, name):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count,) or array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite numeric [{count}] vector')
    return array


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


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
    """Train-owned contiguous 6 h index pairs; both endpoints inside one range.

    The second endpoint must be inside the same half-open range as the first,
    so a pair never crosses a split or a range gap and the values read are
    train values only.
    """
    stamps = np.asarray(time_ns, dtype=np.int64)
    if stamps.ndim != 1 or stamps.size < 2 or np.any(np.diff(stamps) <= 0):
        raise ValueError('time_ns must be a strictly increasing int64 [T] vector')
    pairs = []
    for index in range(len(stamps) - 1):
        if stamps[index + 1] - stamps[index] != STEP_NS:
            continue
        if ranges is not None:
            # ranges: the train half-open [(start_ns, end_ns)] rows from
            # parse_split_time_ranges; both endpoints must be inside one range.
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


def fit_change_scale(store, train_manifest):
    """Read-only fit; loads train frames only and never opens val/test values."""
    store, manifest = _local_path(store), _local_path(train_manifest)
    if manifest.name != 'train.jsonl':
        raise ValueError('only train.jsonl may be read for fitting')
    require_complete_manifest(manifest)
    records = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()
               if line.strip()]
    if not records or any(record.get('split') != 'train' for record in records):
        raise ValueError('change-scale fitting requires a nonempty actual train manifest')
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
    pairs = train_pairs(np.asarray(root['time_ns'][:]), **selection)
    weights = latitude_weights(np.asarray(root['latitude'][:]))
    state = root['state']
    touched = sorted({index for pair in pairs for index in pair})
    frames = np.stack([np.asarray(state[index], dtype=np.float64) for index in touched])
    lookup = {index: row for row, index in enumerate(touched)}
    squared = np.zeros(len(channels), dtype=np.float64)
    diff_bytes = hashlib.sha256()
    for first, second in pairs:
        difference = frames[lookup[second]] - frames[lookup[first]]
        if not np.isfinite(difference).all():
            raise ValueError('nonfinite train change')
        diff_bytes.update(np.ascontiguousarray(difference.astype(np.float32)).tobytes())
        squared += np.einsum('c h w,h -> c', difference * difference,
                             weights, optimize=True) / (frames.shape[2] * frames.shape[3])
    # consistent normalization, no float64 drift in tests
    squared /= len(pairs)
    d_c = np.sqrt(squared)
    s_c = _as_float_vector(np.asarray(root['normalization_std'][:], dtype=np.float32),
                           len(channels), 'normalization_std')
    if np.any(s_c <= 0):
        raise ValueError('state normalization std must be positive')
    zero_change = d_c == 0
    # ratio is exactly d_c / s_c for every channel (zero for a zero-change one);
    # the incumbent-preserving fallback (exact 1.0) lives only in the runtime
    # vector, so the stored field stays the fitted quantity it is named after.
    ratio = d_c / s_c
    floor = FLOAT32_EPS * float(np.max(ratio))
    below_floor = (~zero_change) & (ratio <= floor)
    degenerate = zero_change | below_floor
    if bool(degenerate.all()):
        raise ValueError('all channels are degenerate; no usable change scale')
    runtime = np.where(degenerate, 1.0, ratio)
    with np.errstate(over='ignore', invalid='ignore'):
        runtime32 = runtime.astype(np.float32)
    if not np.isfinite(runtime32).all() or np.any(runtime32 <= 0):
        raise ValueError('runtime ratios must stay finite and positive at float32')
    identity, _ = dataset_identity(manifest)
    meta = {
        'schema': SCHEMA, 'schema_version': 1, 'fit_split': 'train',
        'scientific_claim': False, 'step_hours': 6, 'fit_dtype': 'float64',
        'runtime_dtype': 'float32', 'count': len(pairs),
        'channels': channels, 'units': units, 'weight_rule': WEIGHT_RULE,
        'sample_rule': SAMPLE_RULE, 'fallback_rule': FALLBACK_RULE, 'floor_rule': FLOOR_RULE,
        'change_scale': d_c.tolist(), 'state_scale': s_c.tolist(),
        'ratio': ratio.tolist(), 'runtime_ratio': runtime32.tolist(),
        'relative_floor': float(floor),
        'zero_change_fallback': zero_change.tolist(),
        'below_floor_degenerate': below_floor.tolist(),
        'degenerate': degenerate.tolist(),
        'degenerate_reason': ['zero_change' if zero_change[i] else
                              'ratio_at_or_below_relative_floor' if below_floor[i] else 'active'
                              for i in range(len(channels))],
        'store': str(store), 'train_manifest': str(manifest),
        'data_identity': identity, 'train_manifest_sha256': sha256_file(manifest),
        'train_pairs': {'count': len(pairs), 'first_index': list(pairs[0]),
                        'last_index': list(pairs[-1]),
                        'frame_union_count': len(touched),
                        'first_time_ns': int(root['time_ns'][pairs[0][0]]),
                        'last_time_ns': int(root['time_ns'][pairs[-1][1]]),
                        'train_diff_sha256': diff_bytes.hexdigest(),
                        'train_diff_dtype': 'float32'},
        'limitations': [
            'statistical reparameterization scale only; not evidence of forecast improvement',
            'd_c depends only on the declared train manifest and train ranges of this store',
            'val/test frames are never loaded for fitting; their values cannot move d_c',
            'float64 fit, float32 runtime buffer; not a bitwise cross-platform promise',
            'a degenerate channel keeps runtime ratio exactly 1.0 (incumbent behavior)',
        ],
    }
    validate_change_scale(meta)
    return meta


def validate_change_scale(meta):
    """Reject malformed, nonfinite or non-normalizable change-scale metadata."""
    try:
        json.dumps(meta, allow_nan=False)
        if (meta['schema'] != SCHEMA or type(meta['schema_version']) is not int
                or meta['schema_version'] != 1 or meta['fit_split'] != 'train'
                or meta['scientific_claim'] is not False):
            raise ValueError('unsupported change-scale schema/version/fit split or claim')
        if meta['step_hours'] != 6 or meta['fit_dtype'] != 'float64' \
                or meta['runtime_dtype'] != 'float32' or type(meta['count']) is not int \
                or meta['count'] < 1:
            raise ValueError('invalid step/dtype/count')
        for rule in ('weight_rule', 'sample_rule', 'fallback_rule', 'floor_rule'):
            if not isinstance(meta[rule], str) or not meta[rule].strip():
                raise ValueError(f'{rule} must be explicit')
        channels, units = meta['channels'], meta['units']
        if (not isinstance(channels, list) or not channels or len(units) != len(channels)
                or any(not isinstance(name, str) or not name for name in channels + units)):
            raise ValueError('channel/unit metadata mismatch')
        count = len(channels)
        vectors = {key: _as_float_vector(meta[key], count, key) for key in
                   ('change_scale', 'state_scale', 'ratio', 'runtime_ratio')}
        if np.any(vectors['change_scale'] < 0) or np.any(vectors['state_scale'] <= 0) \
                or np.any(vectors['ratio'] < 0) or np.any(vectors['runtime_ratio'] <= 0):
            raise ValueError('scales must be positive; change_scale and ratio nonnegative')
        floor = meta['relative_floor']
        if not isinstance(floor, float) or not np.isfinite(floor) or floor <= 0:
            raise ValueError('relative floor must be a positive finite float')
        zero = np.asarray(meta['zero_change_fallback'], dtype=bool)
        below = np.asarray(meta['below_floor_degenerate'], dtype=bool)
        degenerate = np.asarray(meta['degenerate'], dtype=bool)
        if any(len(np.asarray(mask)) != count for mask in
               (zero, below, degenerate)):
            raise ValueError('degenerate masks must match the channel count')
        expected_zero = vectors['change_scale'] == 0
        if not np.array_equal(zero, expected_zero):
            raise ValueError('zero-change flags disagree with change_scale')
        if abs(floor - FLOAT32_EPS * float(np.max(vectors['ratio']))) > 0:
            raise ValueError('relative floor must derive from float32 epsilon and the ratio max')
        expected_below = (~expected_zero) & (vectors['ratio'] <= floor)
        if not np.array_equal(below, expected_below):
            raise ValueError('below-floor flags disagree with the ratio and floor')
        if not np.array_equal(degenerate, expected_zero | expected_below):
            raise ValueError('degenerate mask disagrees with its components')
        if bool(degenerate.all()):
            raise ValueError('all channels are degenerate')
        if np.any(vectors['runtime_ratio'][degenerate] != 1.0):
            raise ValueError('degenerate channels must keep runtime ratio exactly 1.0')
        expected_ratio = vectors['change_scale'] / vectors['state_scale']
        if np.any(np.abs(vectors['ratio'] - expected_ratio) > 0):
            raise ValueError('ratio must be exactly d_c / s_c for every channel')
        active = ~degenerate
        # runtime is the documented float32 rounding of the float64 ratio
        if not np.allclose(vectors['runtime_ratio'][active], vectors['ratio'][active],
                           rtol=1e-6, atol=0.0):
            raise ValueError('runtime ratios must round the fitted ratio at float32')
        reasons = ['zero_change' if expected_zero[i] else
                   'ratio_at_or_below_relative_floor' if expected_below[i] else 'active'
                   for i in range(count)]
        if meta['degenerate_reason'] != reasons:
            raise ValueError('degenerate reasons do not match the masks')
        for key in ('data_identity', 'train_manifest_sha256'):
            if not _is_digest(meta[key]):
                raise ValueError(f'invalid {key}')
        pairs = meta['train_pairs']
        if (type(pairs['count']) is not int or pairs['count'] < 1
                or not _is_digest(pairs['train_diff_sha256'])
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
        raise ValueError('invalid change-scale metadata') from error
    return meta


def change_scale_preflight(store, train_manifest):
    """Read-only report; fit is exactly the train manifest and its train ranges."""
    meta = fit_change_scale(store, train_manifest)
    report = {'schema': PREFLIGHT_SCHEMA, 'schema_version': 1,
              'mode': 'read-only-preflight', 'scientific_claim': False,
              'metadata': meta}
    report['preflight_identity'] = canonical_digest(report)
    if dataset_identity(train_manifest)[0] != meta['data_identity']:
        raise ValueError('input data identity changed during preflight')
    return report


def _destination(manifest_dir, meta):
    requested = Path(manifest_dir)
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
    # a second marker write here would violate output_mode 'x'.
    Path(manifest_dir).mkdir(parents=True, exist_ok=False)
    with (Path(manifest_dir) / METADATA_FILE).open('x', encoding='utf-8') as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return meta


def publish_change_scale_sidecar(store, train_manifest, manifest_dir, preflight_identity):
    """Explicit write gate; re-preflight before any reservation/output is created."""
    if not _is_digest(preflight_identity):
        raise ValueError('an explicit preflight identity is required to publish')
    if Path(manifest_dir).exists() or Path(manifest_dir).is_symlink():
        raise FileExistsError(f'refusing existing sidecar: {manifest_dir}')
    report = change_scale_preflight(store, train_manifest)
    if report['preflight_identity'] != preflight_identity:
        raise ValueError('preflight identity mismatch; inputs/statistics changed since review')
    meta = dict(report['metadata'], preflight_identity=preflight_identity)
    destination = _destination(manifest_dir, meta)
    meta['change_scale_identity'] = canonical_digest(meta)
    return _publish_sidecar(meta, manifest_dir=destination)


def load_change_scale_sidecar(path):
    """Require completion, schema, provenance and content identities; no store read."""
    path = _local_path(path)
    if path.is_dir():
        path = path / METADATA_FILE
    if path.name != METADATA_FILE:
        raise ValueError(f'sidecar must be a directory or {METADATA_FILE}')
    require_complete_manifest(path)
    meta = json.loads(path.read_text(encoding='utf-8'))
    validate_change_scale(meta)
    if sha256_file(path.parent / 'BUILD_COMPLETE.json') != PUBLICATION_CONTRACT['completion_marker_sha256']:
        raise ValueError('publisher completion marker bytes mismatch')
    identity = meta.get('change_scale_identity')
    if not _is_digest(identity):
        raise ValueError('missing change-scale identity')
    content = {key: value for key, value in meta.items() if key != 'change_scale_identity'}
    if canonical_digest(content) != identity:
        raise ValueError('change-scale identity mismatch')
    content.pop('preflight_identity', None)
    if (canonical_digest({'schema': PREFLIGHT_SCHEMA, 'schema_version': 1,
                          'mode': 'read-only-preflight', 'scientific_claim': False,
                          'metadata': content}) != meta.get('preflight_identity')):
        raise ValueError('preflight identity mismatch')
    return meta
