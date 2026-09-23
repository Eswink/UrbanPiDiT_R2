"""Explicit anonymous metadata-only probe; no weather arrays are read."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    import icechunk as ic
    import zarr
    try:
        from icechunk.storage import s3_storage
    except ImportError:
        s3_storage = ic.s3_storage
    started = time.monotonic()
    storage = s3_storage(bucket='earthmover-icechunk-era5', prefix='icechunkV2',
                         region='us-east-1', anonymous=True)
    repo = ic.Repository.open(storage)
    session = repo.readonly_session('main')
    root = zarr.open_group(store=session.store, mode='r')
    report = {'source': 's3://earthmover-icechunk-era5/icechunkV2',
              'snapshot_id': session.snapshot_id, 'access': 'anonymous-read-only',
              'field_values_loaded': False, 'icechunk_version': ic.__version__, 'groups': {}}
    for path in ('single/temporal', 'pressure/temporal'):
        group = root[path]
        arrays = {}
        for name, arr in group.arrays():
            arrays[name] = {'shape': list(arr.shape), 'chunks': list(arr.chunks),
                'shards': None if arr.shards is None else list(arr.shards),
                'metadata': arr.metadata.to_dict()}
        report['groups'][path] = arrays
    report['elapsed_seconds'] = time.monotonic() - started
    text = json.dumps(report, indent=2, default=str, allow_nan=False)
    if len(text.encode()) > 2**20:
        raise RuntimeError('unexpectedly large metadata report')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        stream.write(text)
    print(text)


if __name__ == '__main__':
    main()
