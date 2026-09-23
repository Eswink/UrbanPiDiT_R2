"""Fixed-budget continuous-data comparison. No cloud reads or test evaluation."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import time
import numpy as np
from .r7_cpu_study import model_settings, _report_matrix
from .r7_experiment import canonical_digest
from .r7_extended_control import VARIANTS, SEEDS, LEADS


def continuous_protocol(source_sha256):
    from data.download.continuous_pilot_replay import PINNED_CONTINUOUS_SHA256
    if source_sha256 != PINNED_CONTINUOUS_SHA256:
        raise ValueError('predeclared continuous source pin required')
    result=dict(format='r7-continuous-training-control-v1',source_sha256=source_sha256,
        seeds=list(SEEDS),variants={v:dict(kind=model_settings(v)[0],model=model_settings(v)[1],
                    process_weight=model_settings(v)[2]) for v in VARIANTS},
        initialization='from-fixed-seed; no old-data checkpoint reuse',updates=800,
        batch_size=2,accumulation=1,lr=2e-4,reasoning_steps=3,
        depths_evaluated=[1,3],lead_hours=list(LEADS),train_windows=998,validation_cases=24,
        device='cpu',threads=2,test_evaluated=False,controller_trained=False,
        threshold_selection=False,model_architecture_changed=False,
        limitations=['same updates are not matched FLOPs or wall time',
                     'changed training normalization/data; do not compare normalized MSE across datasets',
                     'fixed endpoint, not extension until metrics improve'])
    return dict(result,protocol_sha256=canonical_digest(result))


def summarize_continuous(records,persistence):
    if persistence['split']!='val':
        raise ValueError('validation-only control required')
    paired,baseline=_report_matrix(persistence)
    counts=Counter(datetime.fromisoformat(x['init_time']).month for x in persistence['initializations'])
    if tuple(persistence['lead_hours'])!=LEADS or counts!={1:6,4:6,7:6,9:6}:
        raise ValueError('exact fixed balanced validation cases/leads required')
    seen, groups=set(),{}
    expected=set(product(VARIANTS,SEEDS,(1,3)))
    for record in records:
        key=tuple(record[k] for k in ('variant','seed','depth'))
        if key not in expected or key in seen or record.get('updates')!=800:
            raise ValueError('unexpected/duplicate control record')
        seen.add(key)
        sig,matrix=_report_matrix(record['report'])
        if sig!=paired:
            raise ValueError('unpaired validation reports')
        for h,lead in enumerate(LEADS):
            for c,name in enumerate(persistence['channels']):
                group=(key[0],key[2],lead,name,persistence['units'][c])
                groups.setdefault(group,[]).append(float(matrix[h,c]))
    if seen!=expected:
        raise ValueError('missing continuous control records')
    table=[]
    for (v,k,h,n,u), values in sorted(groups.items()):
        ref=float(baseline[LEADS.index(h),persistence['channels'].index(n)])
        table.append(dict(variant=v,depth=k,updates=800,lead_hours=h,variable=n,unit=u,seeds=len(values),
            seed_rmse_mean=float(np.mean(values)),seed_rmse_sd=float(np.std(values,ddof=1)),
            persistence_rmse=ref))
    return table


def _write(path,result):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)


def run_continuous_control(source,receipt,output_dir):
    import torch
    from data.download.continuous_pilot_replay import verify_continuous_pilot,prepare_continuous_pilot
    from .r7_experiment import dataset_identity,model_code_digest,load_checkpoint
    from .r7_local_runner import run_local_updates
    from .r7_evaluate import evaluate_local
    from .r7_calibration_runner import file_sha256
    audit=verify_continuous_pilot(source,receipt)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    protocol=continuous_protocol(audit['source_netcdf_sha256'])
    _write(out/'protocol.json',protocol)  # persisted BEFORE any model training
    started=time.monotonic();torch.set_num_threads(2)
    train,val,replay,prepared=prepare_continuous_pilot(source,receipt,out/'dataset')
    identity,dataset=dataset_identity(train)
    labels=torch.stack([dataset[i]['process_targets'] for i in range(len(dataset))])
    if len(dataset)!=998 or labels.shape!=(998,8) or not torch.isfinite(labels).all():
        raise ValueError('actual continuous diagnostics/window acceptance failed')
    digest=model_code_digest()
    persistence=evaluate_local(val,output_dir=out/'evaluation'/'persistence',
                                lead_hours=LEADS,max_samples=24,device_name='cpu')
    records,resources=[],[]
    for variant,seed in product(VARIANTS,SEEDS):
        if time.monotonic()-started>1080:
            raise RuntimeError('continuous CPU study deadline exceeded')
        kind,model,pw=model_settings(variant)
        checkpoint,training=run_local_updates(dataset,kind=kind,model_config=model,data_identity=identity,
            output_dir=out/'training'/f'{variant}_{seed}',total_updates=800,batch_size=2,steps=3,
            seed=seed,lr=2e-4,process_weight=pw,device_name='cpu')
        saved=load_checkpoint(checkpoint)
        if (training['updates_this_run']!=800 or saved['updates']!=800 or
            saved['contract']['dataset_length']!=998 or saved['contract']['data_identity']!=identity):
            raise ValueError('fixed training endpoint/data contract failed')
        resources.append(dict(variant=variant,seed=seed,training=training,
                              checkpoint_sha256=file_sha256(checkpoint)))
        for depth in (3,1):
            report=evaluate_local(val,output_dir=out/'evaluation'/f'{variant}_{seed}_K{depth}',
                checkpoint=checkpoint,lead_hours=LEADS,max_samples=24,
                reasoning_steps=depth,device_name='cpu')
            records.append(dict(variant=variant,seed=seed,depth=depth,updates=800,report=report))
        print(json.dumps(dict(variant=variant,seed=seed,updates=800,validation_complete=True)),flush=True)
    if model_code_digest()!=digest or file_sha256(source)!=audit['source_netcdf_sha256']:
        raise RuntimeError('model code/source changed during control')
    result=dict(format='r7-continuous-training-result-v1',complete=True,scientific_claim=False,
        gpu_used=False,test_evaluated=False,source_cloud_requests=0,protocol=protocol,
        source=replay,preparation=prepared,training_identity=identity,model_code_sha256=digest,
        process_label_shape=list(labels.shape),resources=resources,records=records,
        persistence=persistence,summary=summarize_continuous(records,persistence),
        elapsed_seconds=time.monotonic()-started,
        limitations=['Single small tile,250days peryear and three seeds, not final journal evidence',
                     'Previously inspected validation cases; do not call this pristine test',
                     'Training updates fixed upfront; improvement not required for publication of results',
                     'No controller tuning, future boundary forcing or GPU measurements'])
    _write(out/'continuous_control_result.json',result)
    return result
