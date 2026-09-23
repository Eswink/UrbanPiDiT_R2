"""Fixed-count matched-parameter spatial-solver ablation on real validation."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import time
import numpy as np
from .r7_cpu_study import model_settings,_report_matrix
from .r7_experiment import canonical_digest
from .r7_extended_control import SEEDS,LEADS

VARIANTS=('generic_global','generic_spatial','process_global','process_spatial')


def spatial_settings(variant):
    if variant not in VARIANTS:
        raise ValueError('unknown spatial solver variant')
    kind,location=variant.split('_')
    _,config,weight=model_settings(kind)
    config['spatial_solver_feedback']=location=='spatial'
    return kind,config,weight


def spatial_protocol(source_sha256):
    from data.download.continuous_pilot_replay import PINNED_CONTINUOUS_SHA256
    if source_sha256!=PINNED_CONTINUOUS_SHA256:
        raise ValueError('audited continuous source pin required')
    result=dict(format='r7-spatial-solver-ablation-v1',source_sha256=source_sha256,seeds=list(SEEDS),
        variants={v:dict(kind=spatial_settings(v)[0],model=spatial_settings(v)[1],
                        process_weight=spatial_settings(v)[2]) for v in VARIANTS},
        updates=400,batch_size=2,lr=2e-4,steps=3,train_windows=998,
        inference_depths=[1,3],lead_hours=list(LEADS),validation_cases=24,
        test_evaluated=False,controller_fitted=False,device='cpu',threads=2,
        training_mode='streamed-truncated; original objective unchanged',
        design='both generic and process get the same zero-new-parameter mechanism')
    return dict(result,protocol_sha256=canonical_digest(result))


def summarize_spatial(records,persistence):
    paired,baseline=_report_matrix(persistence)
    if persistence['split']!='val' or tuple(persistence['lead_hours'])!=LEADS:
        raise ValueError('fixed-lead validation-only study required')
    counts=Counter(datetime.fromisoformat(r['init_time']).month for r in persistence['initializations'])
    if counts!={1:6,4:6,7:6,9:6}:
        raise ValueError('exact24 balanced validation cases required')
    seen,groups=set(),{}
    for r in records:
        key=tuple(r[k] for k in ('variant','seed','depth'))
        if key not in set(product(VARIANTS,SEEDS,(1,3))) or key in seen or r.get('updates')!=400:
            raise ValueError('unexpected/duplicate ablation record')
        seen.add(key)
        signature,values=_report_matrix(r['report'])
        if signature!=paired:
            raise ValueError('ablation cases/schema/identity must be paired')
        for i,h in enumerate(LEADS):
            for j,n in enumerate(persistence['channels']):
                groups.setdefault((key[0],key[2],h,n,persistence['units'][j]),[]).append(float(values[i,j]))
    if seen!=set(product(VARIANTS,SEEDS,(1,3))):
        raise ValueError('incomplete ablation')
    return [dict(variant=v,depth=k,lead_hours=h,variable=n,unit=u,seeds=len(x),
        seed_rmse_mean=float(np.mean(x)),seed_rmse_sd=float(np.std(x,ddof=1)),
        persistence_rmse=float(baseline[LEADS.index(h),persistence['channels'].index(n)]))
        for (v,k,h,n,u),x in sorted(groups.items())]


def _write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)


def run_spatial_study(source,receipt,output_dir):
    import torch
    from data.download.continuous_pilot_replay import verify_continuous_pilot,prepare_continuous_pilot
    from .r7_experiment import dataset_identity,make_model,model_code_digest
    from .r7_calibration_runner import file_sha256
    from .r7_local_runner import run_local_updates
    from .r7_evaluate import evaluate_local
    audit=verify_continuous_pilot(source,receipt)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.monotonic()
    protocol=spatial_protocol(audit['source_netcdf_sha256'])
    counts={}
    for variant in VARIANTS:
        kind,cfg,_=spatial_settings(variant)
        counts[variant]=sum(p.numel() for p in make_model(kind,cfg).parameters())
    for kind in ('generic','process'):
        if counts[kind+'_global']!=counts[kind+'_spatial']:
            raise ValueError('added parameters invalidate the declared control')
    protocol['parameter_counts']=counts
    protocol.pop('protocol_sha256')
    protocol['protocol_sha256']=canonical_digest(protocol)
    _write(out/'protocol.json',protocol)
    train,val,source_audit,prepared=prepare_continuous_pilot(source,receipt,out/'dataset')
    identity,dataset=dataset_identity(train)
    if len(dataset)!=998:
        raise ValueError('fixed train sample set required')
    persistence=evaluate_local(val,output_dir=out/'evaluation'/'persistence',
        lead_hours=LEADS,max_samples=24,device_name='cpu')
    records,resources=[],[]
    digest=model_code_digest()
    for variant,seed in product(VARIANTS,SEEDS):
        if time.monotonic()-started>1080:
            raise RuntimeError('spatial ablation CPU wall budget exhausted')
        kind,cfg,pw=spatial_settings(variant)
        checkpoint,report=run_local_updates(dataset,kind=kind,model_config=cfg,data_identity=identity,
            output_dir=out/'training'/f'{variant}_{seed}',total_updates=400,batch_size=2,
            seed=seed,lr=2e-4,steps=3,process_weight=pw,device_name='cpu')
        if report['updates_this_run']!=400:
            raise ValueError('wrong fixed optimization endpoint')
        resources.append(dict(variant=variant,seed=seed,checkpoint_sha256=file_sha256(checkpoint),training=report))
        for depth in (3,1):
            evaluated=evaluate_local(val,output_dir=out/'evaluation'/f'{variant}_{seed}_K{depth}',
                checkpoint=checkpoint,reasoning_steps=depth,lead_hours=LEADS,max_samples=24,device_name='cpu')
            records.append(dict(variant=variant,seed=seed,updates=400,depth=depth,report=evaluated))
        print(json.dumps(dict(variant=variant,seed=seed,updates=400,validation_complete=True)),flush=True)
    if model_code_digest()!=digest:
        raise ValueError('model implementation changed mid-ablation')
    result=dict(format='r7-spatial-solver-result-v1',complete=True,scientific_claim=False,gpu_used=False,
        test_evaluated=False,controller_fitted=False,protocol=protocol,source=source_audit,
        preparation=prepared,training_identity=identity,model_code_sha256=digest,
        resources=resources,records=records,persistence=persistence,
        summary=summarize_spatial(records,persistence),elapsed_seconds=time.monotonic()-started,
        limitations=['fixed400updates are not convergence or comparison with earlier800-update runs',
            'single small tile and explored validation cases; not final benchmark/SOTA',
            'parameter matching does not exactly match FLOPs or measured latency',
            'defaultFalse preserved; feature must earn adoption rather than assumed superior',
            'older checkpoints require their recorded source code, never a digest override'])
    _write(out/'spatial_result.json',result)
    return result
