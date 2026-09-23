"""Bounded continuous ERA5 acquisition; validation-only CPU integration."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from data.download.earthmover_pilot import download_pilot,FIELDS
    from data.download.seasonal_sampling import requested_times,write_balanced_manifest
    from data.preprocess.r7_preflight import prepare_local,local_source
    from training.r7_experiment import dataset_identity,canonical_digest
    from training.r7_local_runner import run_local_updates
    from training.r7_evaluate import evaluate_local
    from training.r7_cpu_study import model_settings
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.monotonic()
    wanted=requested_times('continuous-250d')
    protocol=dict(sampling='continuous-250d',years=[2018,2019,2020],time_count=3000,
        steps_per_year=1000,days_per_year=250,grid_shape=[12,12],channels=[n for _,_,n in FIELDS],
        decoded_budget_bytes=192*2**20,max_source_file_bytes=32*2**20,
        seed=42,updates=20,reasoning_steps=3,validation_cases=24,lead_hours=[6,12,24,72],
        test_evaluated=False,scientific_claim=False,mode='data-acquisition-and-integration')
    protocol['digest']=canonical_digest(protocol)
    with (out/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2,allow_nan=False)
    source,receipt=download_pilot(out/'source'/'era5_continuous250d.nc',
        out/'source'/'receipt.json',sampling='continuous-250d')
    if source.stat().st_size>32*2**20 or not receipt['same_time_chunks_as_four_season_verified']:
        raise RuntimeError('source size/layout acceptance failed')
    with local_source(source) as ds:
        if dict(ds.sizes)!=dict(time=3000,latitude=12,longitude=12):
            raise RuntimeError('continuous source shape mismatch')
        if not np.array_equal(ds.time.values.astype('datetime64[ns]'),wanted.values.astype('datetime64[ns]')):
            raise RuntimeError('continuous source timestamps differ')
        for item in receipt['variables']:
            raw=np.asarray(ds[item['name']].values,dtype='<f4')
            if hashlib.sha256(raw.tobytes()).hexdigest()!=item['payload_sha256']:
                raise RuntimeError('serialized field payload differs from extraction')
    config=dict(channels=[dict(variable=n,name=n) for _,_,n in FIELDS],
        split_years=dict(train=[2018],val=[2019],test=[2020]),history_steps=2,
        history_interval_hours=6,lead_time_hours=6,sample_stride_hours=6,
        time_chunk=8,compute_process_targets=True)
    prepared=prepare_local(source,config,write=True,store_path=out/'cache.zarr',
        manifest_dir=out/'manifests',max_raw_gib=.05)
    if prepared['windows']!=dict(train=998,val=998,test=998):
        raise RuntimeError('exact-time/year window acceptance failed')
    identity,dataset=dataset_identity(out/'manifests'/'train.jsonl')
    labels=torch.stack([dataset[i]['process_targets'] for i in range(len(dataset))])
    if labels.shape!=(998,8) or not torch.isfinite(labels).all():
        raise RuntimeError('real process targets missing/nonfinite')
    manifest=write_balanced_manifest(out/'manifests'/'val.jsonl',wanted)
    kind,model,pw=model_settings('process')
    checkpoint,training=run_local_updates(dataset,kind=kind,model_config=model,data_identity=identity,
        output_dir=out/'training',total_updates=20,batch_size=2,steps=3,seed=42,
        process_weight=pw,device_name='cpu')
    reports={}
    for name,cp,kw in [('persistence',None,{}),('process_K3',checkpoint,{}),('process_K1',checkpoint,dict(reasoning_steps=1))]:
        report=evaluate_local(manifest,output_dir=out/'evaluation'/name,checkpoint=cp,
            max_samples=24,lead_hours=(6,12,24,72),device_name='cpu',**kw)
        if report['split']!='val' or Counter(pd.Timestamp(r['init_time']).month for r in report['initializations'])!={1:6,4:6,7:6,9:6}:
            raise RuntimeError('validation selection not balanced and complete')
        reports[name]=report
    result=dict(format='r7-continuous-data-cpu-v1',complete=True,scientific_claim=False,
        gpu_used=False,test_evaluated=False,protocol=protocol,source=receipt,preparation=prepared,
        process_label_shape=list(labels.shape),training=training,validation=reports,
        elapsed_seconds=time.monotonic()-started,
        limitations=['250days per year, not complete calendar years or full seasonal climatology',
            '12x12 single tile,20updates: data integration, not converged regional skill or SOTA',
            'Decoded accounting is not measured HTTP traffic, RSS or GPU memory',
            'Validation cases from prior pilot are exploratory; no test results or threshold tuning'])
    with (out/'continuous_result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(complete=True,source_bytes=receipt['source_netcdf_bytes'],
        decoded_charged_bytes=receipt['decoded_charged_bytes'],windows=prepared['windows'],
        elapsed_seconds=result['elapsed_seconds'],test_evaluated=False)),flush=True)


if __name__=='__main__':main()
