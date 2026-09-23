"""Rebuild a pinned pilot's exact logical cache identity; never override it."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .download.earthmover_pilot import FIELDS
from .download.seasonal_pilot_replay import verify_seasonal_pilot
from .download.seasonal_sampling import requested_times,write_balanced_manifest


def restoration_config(metadata):
    expected={
        'schema_version':1,'time_unit':'ns','channels':[n for _,_,n in FIELDS],
        'shape':[384,11,12,12],'chunks':[8,11,12,12],'store_path':'../cache.zarr',
        'history_steps':2,'history_interval_hours':6,'lead_time_hours':6,'sample_stride_hours':6,
        'split_years':{'train':[2018],'val':[2019],'test':[2020]},'normalization_years':[2018],
        'samples_by_split':{'train':120,'val':120,'test':120},'native_grid_spacing_deg':.25,
        'physical_units_retained':True,'spatial_resampling':False,'process_diagnostics_enabled':True,
    }
    for key,value in expected.items():
        if metadata.get(key)!=value or type(metadata.get(key)) is not type(value):
            raise ValueError(f'unsupported archived pilot metadata: {key}')
    source=metadata.get('source')
    if not isinstance(source,str) or not 1<=len(source)<=4096 or any(ord(c)<32 for c in source):
        raise ValueError('invalid historical source provenance string')
    # The source string is NEVER opened; actual source bytes have a separate pin.
    return dict(channels=[dict(variable=n,name=n) for _,_,n in FIELDS],
        split_years=expected['split_years'],history_steps=2,history_interval_hours=6,
        lead_time_hours=6,sample_stride_hours=6,time_chunk=8,compute_process_targets=True)


def _bounded_json(path,cap=128*1024):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>cap:
        raise ValueError('archived metadata absent or exceeds cap')
    return json.loads(path.read_text())


def restore_pilot_cache(source,receipt,archived_manifests,checkpoint,output_dir):
    """Fresh local store from unchanged physical bytes, same original identity.

    No checkpoint mutation, source-path lookup, data-identity override or model
    change is allowed. Numerical/platform differences can fail the identity check;
    that is a failure to reproduce, not a reason to bypass the check.
    """
    from training.r7_experiment import load_checkpoint,dataset_identity
    from .preprocess.r7_preflight import local_source,inspect_era5
    from .preprocess.r7_era5_zarr import build_r7_era5_zarr_from_dataset
    audit=verify_seasonal_pilot(source,receipt)
    archive=Path(archived_manifests)
    metadata=_bounded_json(archive/'zarr_metadata.json')
    config=restoration_config(metadata)
    preflight=_bounded_json(archive/'source_preflight.json')
    fingerprint=preflight.get('fingerprint',{})
    if (fingerprint.get('scope')!='full-local-file' or
        fingerprint.get('sha256')!=audit['source_netcdf_sha256'] or
        preflight.get('source_path')!=metadata['source']):
        raise ValueError('archived provenance is not bound to the pinned source')
    saved=load_checkpoint(checkpoint)
    if saved['contract']['data_identity']=='' or saved['contract']['dataset_length']!=120:
        raise ValueError('checkpoint is not the expected seasonal pilot')
    originals={}
    for split in ('train','val','test','val_balanced','test_balanced'):
        path=archive/f'{split}.jsonl'
        if not path.is_file() or path.stat().st_size>256*1024:
            raise ValueError('original complete manifest set required under byte cap')
        originals[split]=path.read_bytes()
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    with local_source(source) as ds:
        inspected,specs=inspect_era5(ds,config)
        paths=build_r7_era5_zarr_from_dataset(ds,store_path=out/'cache.zarr',manifest_dir=out/'manifests',
            specs=specs,split_years=config['split_years'],**inspected['cadence'],time_chunk=8,
            compute_process_targets=True,source_label=metadata['source'])
    for split in ('val','test'):
        write_balanced_manifest(paths[split],requested_times('four-season'))
    rebuilt=_bounded_json(out/'manifests'/'zarr_metadata.json')
    if rebuilt!=metadata:
        raise ValueError('rebuilt metadata differs from original; recovery not accepted')
    for name,raw in originals.items():
        if (out/'manifests'/f'{name}.jsonl').read_bytes()!=raw:
            raise ValueError('rebuilt manifest bytes differ; recovery not accepted')
    identity,_=dataset_identity(paths['train'])
    if identity!=saved['contract']['data_identity']:
        raise ValueError('rebuilt data identity differs; no override permitted')
    record=dict(format='r7-exact-pilot-cache-restoration-v1',scientific_claim=False,
        pinned_source_sha256=audit['source_netcdf_sha256'],
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        restored_data_identity=identity,original_source_provenance=metadata['source'],
        local_source_read=str(Path(source).resolve()),checkpoint_modified=False,training_executed=False,
        source_network_requests=0,manifest_hashes={k:hashlib.sha256(v).hexdigest() for k,v in originals.items()},
        limitations=['Only the pinned seasonal pilot and audited original layout are supported',
            'Historical provenance is preserved as text; no old path is followed',
            'Any data/model identity mismatch must fail; cross-platform bitwise equality is not guaranteed'])
    with (out/'RESTORATION_ACCEPTED.json').open('x') as f:json.dump(record,f,indent=2,allow_nan=False)
    return paths,record
