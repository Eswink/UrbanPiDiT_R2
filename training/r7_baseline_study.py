"""Parameter-comparable compact controls; one fixed-seed CPU experiment."""
from __future__ import annotations
from collections import Counter
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from .r7_cpu_study import model_settings, _report_matrix, _json, _csv
from .r7_paired_comparison import _rows

VARIANTS=('unet','convlstm','afno_small','window_deep','generic','process')
TARGET_PARAMETERS=83319


def control_settings(name):
    if name in ('generic','process'):
        return model_settings(name)
    common=dict(in_channels=11,out_channels=11,history_steps=2)
    controls={
        'unet':dict(architecture='unet',dim=11),
        'convlstm':dict(architecture='convlstm',dim=38),
        'afno_small':dict(architecture='afno_small',dim=52,depth=4,blocks=4,patch_size=2),
        'window_deep':dict(architecture='window',dim=36,depth=4,heads=4,patch_size=2,window_size=4),
    }
    if name not in controls:raise ValueError('unknown fixed baseline')
    return 'native',dict(common,**controls[name]),0.


def summarize_controls(records,profiles):
    required={(v,s) for v in VARIANTS+('persistence',) for s in ('val','test')}
    if set(profiles)!=set(VARIANTS):raise ValueError('complete profile collection required')
    identity=None
    for name,profile in profiles.items():
        if profile['device']!='cpu' or profile['precision']!='fp32' or profile['cpu_threads']!=2:
            raise ValueError('profile hardware/precision protocol mismatch')
        if profile['input_shape']!=[1,2,11,12,12] or profile['warmup_excluded']!=5 or profile['repetitions']!=20:
            raise ValueError('profile input/repetition protocol mismatch')
        if any(v is not None for v in profile['cuda_memory'].values()):
            raise ValueError('CPU experiment cannot report measured CUDA memory')
        values=profile['seconds_per_batch']
        if len(values)!=20 or any(not math.isfinite(v) or v<=0 for v in values):
            raise ValueError('invalid actual profile durations')
        if not math.isclose(profile['median_seconds_per_batch'],statistics.median(values),rel_tol=1e-12):
            raise ValueError('profile median does not match actual durations')
        signature=tuple(profile[k] for k in ('input_sha256','hardware','torch','python','cpu_threads'))
        if identity is not None and identity!=signature:raise ValueError('profiling inputs/platforms differ')
        identity=signature
        if abs(profile['parameters']/TARGET_PARAMETERS-1)>.1:
            raise ValueError('parameter budget mismatch')
    found,paired,rows=set(),{},[]
    for record in records:
        name,split=record['variant'],record['split']
        key=(name,split)
        if key not in required or key in found:raise ValueError('duplicate or unexpected baseline run')
        found.add(key)
        report=record['report'];_rows(report)
        signature,rmse=_report_matrix(report)
        if report['split']!=split or report['lead_hours']!=[6,12,24,72] or report['n_evaluated']!=24:
            raise ValueError('baseline evaluation protocol mismatch')
        counts=Counter(pd.Timestamp(c['init_time']).month for c in report['initializations'])
        if dict(counts)!={1:6,4:6,7:6,9:6}:raise ValueError('baseline case months are imbalanced')
        if split in paired and paired[split]!=signature:raise ValueError('baseline reports are not paired')
        paired[split]=signature
        profile=profiles.get(name)
        for i,h in enumerate(report['lead_hours']):
            for j,channel in enumerate(report['channels']):
                rows.append(dict(variant=name,split=split,seed=42,updates=0 if name=='persistence' else 200,
                    lead_hours=h,variable=channel,unit=report['units'][j],rmse=float(rmse[i,j]),
                    parameters=0 if profile is None else profile['parameters'],
                    median_cpu_forward_ms=None if profile is None else 1000*profile['median_seconds_per_batch']))
    if found!=required:raise ValueError('missing baseline runs')
    return rows


def _profile_subprocess(manifest,checkpoint,path):
    """A fresh process for each warmed resident-batch timing, also offline."""
    root=Path(__file__).resolve().parents[1]
    script=root/'profile_r7_inference.py'
    code="""import runpy,socket,sys

def deny(*args,**kwargs):raise RuntimeError('network prohibited in CPU profiler')
socket.socket.connect=deny
socket.create_connection=deny
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    command=[sys.executable,'-c',code,str(script),'--manifest',str(manifest),'--checkpoint',str(checkpoint),
             '--out',str(path),'--device','cpu','--batch-size','1','--warmup','5','--repetitions','20']
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    completed=subprocess.run(command,env=env,text=True,capture_output=True,timeout=90)
    path.with_suffix('.log').write_text(completed.stdout+'\n'+completed.stderr)
    if completed.returncode:raise RuntimeError(f'CPU profiler failed; inspect {path.with_suffix(".log")}')
    return json.loads(path.read_text())


def run_baseline_study(source,receipt,output_dir):
    import torch
    from data.download.seasonal_pilot_replay import copy_verified_seasonal_pilot
    from data.download.seasonal_sampling import requested_times,write_balanced_manifest
    from data.download.earthmover_pilot import FIELDS
    from data.preprocess.r7_preflight import prepare_local
    from .r7_experiment import dataset_identity,make_model,canonical_digest
    from .r7_local_runner import run_local_updates
    from .r7_evaluate import evaluate_local
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.monotonic()
    source,audit=copy_verified_seasonal_pilot(source,receipt,out/'source'/'era5_four_season.nc',out/'source'/'receipt.json')
    settings={}
    for name in VARIANTS:
        kind,cfg,pw=control_settings(name)
        params=sum(p.numel() for p in make_model(kind,cfg).parameters())
        if abs(params/TARGET_PARAMETERS-1)>.1:raise ValueError('baseline parameter budget drift')
        settings[name]=dict(kind=kind,config=cfg,process_weight=pw,parameters=params)
    protocol=dict(format='r7-compact-controls-v1',source_sha256=audit['source_netcdf_sha256'],
        settings=settings,seed=42,updates=200,batch_size=2,lr=2e-4,steps=3,
        leads=[6,12,24,72],case_months=[1,4,7,9],cases_per_month=6,
        profiling=dict(device='cpu',precision='fp32',threads=2,warmup=5,repetitions=20,batch_size=1),
        limitations=['equal parameters/updates are not equal FLOPs or optimized training recipes',
            'native final forecast loss versus recursive deep supervision plus optional process loss',
            'one seed; architecture adaptations are not pretrained SOTA systems'])
    protocol['signature']=canonical_digest(protocol);_json(out/'protocol.json',protocol)
    cfg=dict(channels=[dict(variable=n,name=n) for _,_,n in FIELDS],
        split_years=dict(train=[2018],val=[2019],test=[2020]),history_steps=2,
        history_interval_hours=6,lead_time_hours=6,sample_stride_hours=6,time_chunk=8,compute_process_targets=True)
    preparation=prepare_local(source,cfg,write=True,store_path=out/'cache.zarr',manifest_dir=out/'manifests',max_raw_gib=.01)
    identity,dataset=dataset_identity(out/'manifests'/'train.jsonl')
    manifests={s:write_balanced_manifest(out/'manifests'/f'{s}.jsonl',requested_times('four-season')) for s in ('val','test')}
    records,profiles,training=[],{},{}
    (out/'profiles').mkdir()
    for name in VARIANTS:
        setting=settings[name]
        checkpoint,report=run_local_updates(dataset,kind=setting['kind'],model_config=setting['config'],
            data_identity=identity,output_dir=out/'training'/name,total_updates=200,batch_size=2,steps=3,
            seed=42,lr=2e-4,process_weight=setting['process_weight'],device_name='cpu')
        training[name]=report
        for split,manifest in manifests.items():
            evaluation=evaluate_local(manifest,output_dir=out/'evaluation'/f'{name}_{split}',checkpoint=checkpoint,
                lead_hours=(6,12,24,72),max_samples=24,device_name='cpu')
            records.append(dict(variant=name,split=split,report=evaluation))
        profiles[name]=_profile_subprocess(manifests['test'],checkpoint,out/'profiles'/f'{name}.json')
        print(json.dumps(dict(variant=name,updates=200,parameters=setting['parameters'],phase='profiled')),flush=True)
    for split,manifest in manifests.items():
        report=evaluate_local(manifest,output_dir=out/'evaluation'/f'persistence_{split}',lead_hours=(6,12,24,72),max_samples=24,device_name='cpu')
        records.append(dict(variant='persistence',split=split,report=report))
    rows=summarize_controls(records,profiles);_csv(out/'baseline_rmse_cpu.csv',rows)
    result=dict(format='r7-compact-controls-results-v1',scientific_claim=False,gpu_used=False,
        protocol=protocol,preparation=preparation,training=training,records=records,profiles=profiles,
        summary=rows,elapsed_seconds=time.monotonic()-started,
        limits=['Small sampled tile, single seed and fixed200updates; no general superiority claim',
            'One resident input timing and one run per variant; not production inference throughput',
            'All variables/horizons retained; no averaging different physical units to rank models'])
    _json(out/'baseline_study_result.json',result)
    return result
