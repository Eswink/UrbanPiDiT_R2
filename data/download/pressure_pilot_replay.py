"""Offline verification of the exact real #44 pilot artifact, not arbitrary data."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from .earthmover_pilot import FIELDS, SNAPSHOT, SOURCE

PINNED_SOURCE_SHA256 = '13fb72807d3f6988b0fcf2b6b2f130f890207242916018556fb85553686b9a12'


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def verify_pressure_pilot(source, receipt):
    """Check source bytes against the successful run's pin before interpreting them.

    Matching a user-editable receipt alone is not source authentication. This
    helper additionally requires the exact hash from run 35875707823. It does
    not accept newly downloaded/re-encoded data as that same original artifact.
    """
    source, receipt = Path(source), Path(receipt)
    if not source.is_file() or source.stat().st_size > 2 * 2**20:
        raise ValueError('local source missing or exceeds 2 MiB replay cap')
    if not receipt.is_file() or receipt.stat().st_size > 128 * 1024:
        raise ValueError('local receipt missing or exceeds replay cap')
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PINNED_SOURCE_SHA256:
        raise ValueError('source hash does not match the pinned successful real-data artifact')
    report = json.loads(receipt.read_text(encoding='utf-8'))
    if (report.get('source_netcdf_sha256') != digest or
            report.get('source_netcdf_bytes') != len(raw)):
        raise ValueError('receipt source identity mismatch')
    if (report.get('source') != SOURCE or report.get('snapshot_id') != SNAPSHOT or
            report.get('source_is_real_reanalysis') is not True or
            report.get('scientific_claim') is not False or report.get('license') != 'CC-BY-4.0'):
        raise ValueError('receipt source declaration mismatch')
    if report.get('years') != [2018, 2019, 2020] or report.get('times_per_year') != 32:
        raise ValueError('receipt time selection mismatch')
    rows = report.get('variables', [])
    if [r.get('name') for r in rows] != [f[2] for f in FIELDS]:
        raise ValueError('receipt variables/order mismatch')
    import pandas as pd
    from xarray.coding.times import decode_cf_datetime
    expected_times = np.concatenate([pd.date_range(f'{y}-01-01', periods=32, freq='6h').values
                                     for y in (2018, 2019, 2020)])
    with h5py.File(source, 'r') as nc:
        if _text(nc.attrs.get('source')) != SOURCE or _text(nc.attrs.get('snapshot_id')) != SNAPSHOT:
            raise ValueError('NetCDF source declaration mismatch')
        for name, expected in [('latitude', np.linspace(42., 39.25, 12)),
                               ('longitude', np.linspace(114., 116.75, 12))]:
            values = nc[name][:]
            if not np.array_equal(values, expected) or not np.array_equal(values, report['coordinates'][name]):
                raise ValueError('source/receipt coordinate mismatch')
        t = nc['time']
        times = decode_cf_datetime(t[:], _text(t.attrs['units']),
                                   _text(t.attrs['calendar']), use_cftime=False)
        if not np.array_equal(times, expected_times):
            raise ValueError('source timestamps mismatch')
        for row, (variable, level, name) in zip(rows, FIELDS):
            values = nc[name][:]
            if (values.shape != (96, 12, 12) or row.get('shape') != [96, 12, 12] or
                    values.dtype != np.dtype('float32') or not np.isfinite(values).all()):
                raise ValueError('source shape/dtype/finite-value mismatch')
            if hashlib.sha256(values.astype('<f4').tobytes()).hexdigest() != row.get('payload_sha256'):
                raise ValueError('variable payload hash mismatch')
            if (row.get('source_variable') != variable or row.get('pressure_hpa') != level or
                    _text(nc[name].attrs.get('source_variable')) != variable or
                    int(np.asarray(nc[name].attrs['source_pressure_hpa']).item()) != (-1 if level is None else level) or
                    _text(nc[name].attrs.get('units')) != row.get('units')):
                raise ValueError('source/receipt units or pressure-level mismatch')
    report = dict(report, replay={'mode': 'verified-local-source', 'source_network_requests': 0,
        'verified_variables': len(FIELDS), 'original_source_sha256': digest,
        'pin_origin': 'GitHub Actions run 35875707823, artifact 10757057891'})
    return report


def copy_verified_pressure_pilot(source, receipt, output_source, output_receipt):
    """Verify before creating output and preserve the exact original file bytes."""
    output_source, output_receipt = Path(output_source), Path(output_receipt)
    if any(p.exists() or p.is_symlink() for p in (output_source, output_receipt)):
        raise FileExistsError('replay output must be new')
    report = verify_pressure_pilot(source, receipt)
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINNED_SOURCE_SHA256:
        raise ValueError('source changed during replay verification')
    output_source.parent.mkdir(parents=True, exist_ok=True)
    with output_source.open('xb') as stream:
        stream.write(raw)
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    with output_receipt.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return output_source, report
