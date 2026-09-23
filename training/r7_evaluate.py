"""Bounded local checkpoint/persistence/adaptive evaluation with provenance."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
from data.r7_zarr_dataset import ZarrAtmosWindowDataset
from data.r7_store import validate_record
from data.r7_evaluation import ZarrRolloutDataset,fit_training_climatology,normalized_climatology
from model.r7_rollout import autoregressive_rollout
from .r7_experiment import make_model,load_checkpoint,dataset_identity,select_device
from .r7_rollout_metrics import RolloutRMSEAccumulator
from .r7_acc import RolloutACCAccumulator


class Persistence(nn.Module):
    def forward(self,batch):
        from types import SimpleNamespace
        return SimpleNamespace(forecast=batch['coarse_history'][:,-1].clone())


@torch.no_grad()
def evaluate_local(manifest,*,output_dir,checkpoint=None,lead_hours=(6,12,24,48,72),
                   step_hours=6,max_samples=32,device_name='cpu',normalized=False,reasoning_steps=None,
                   controller_checkpoint=None,min_reasoning_steps=1,force_full_depth=False):
    if isinstance(max_samples,bool) or not isinstance(max_samples,int) or max_samples<1:
        raise ValueError('max_samples must be a positive explicit cap')
    if controller_checkpoint and not checkpoint:
        raise ValueError('controller evaluation requires its parent checkpoint')
    if force_full_depth and not controller_checkpoint:
        raise ValueError('force_full_depth requires a controller checkpoint')
    manifest=Path(manifest)
    reader=ZarrAtmosWindowDataset(manifest)
    splits={r['split'] for r in reader.records}
    stores={r['store_path'] for r in reader.records}
    if len(splits)!=1 or len(stores)!=1 or next(iter(splits)) not in ('val','test'):
        raise ValueError('one held-out split and one store per evaluation')
    root=reader._store(reader.records[0])
    for r in reader.records:
        validate_record(root,r)
    store=Path(next(iter(stores)))
    if not store.is_absolute():
        store=(manifest.parent/store).resolve()
    ds=ZarrRolloutDataset(store,split=next(iter(splits)),lead_hours=lead_hours,
        history_steps=len(reader.records[0]['history_indices']),step_hours=step_hours)
    allowed={r['init_time'] for r in reader.records}
    ds.windows=[w for w in ds.windows if ds.times[w[0][-1]].isoformat() in allowed]
    if not ds.windows:
        raise ValueError('manifest has no complete requested rollout windows')
    if not normalized and any(u in ('unknown','') for u in ds.units):
        raise ValueError('physical units missing; audit source or explicitly use normalized metrics')
    device=select_device(device_name)
    inference={}
    checkpoint_hash=training_identity=controller_hash=None
    controller_policy=None
    if checkpoint:
        saved=load_checkpoint(checkpoint)
        contract=saved['contract']
        training_identity,_=dataset_identity(manifest.parent/'train.jsonl')
        if training_identity!=contract['data_identity']:
            raise ValueError('checkpoint training data/normalization identity mismatch')
        model=make_model(contract['kind'],contract['model'])
        model.load_state_dict(saved['model'],strict=True)
        if controller_checkpoint:
            if contract['kind']!='process':
                raise ValueError('adaptive controller needs a process checkpoint')
            from .r7_calibration_runner import load_calibrated_adapter,file_sha256
            model,controller_policy=load_calibrated_adapter(model,controller_checkpoint,
                parent_checkpoint=checkpoint,training_identity=training_identity)
            inference={'max_steps':controller_policy['max_steps'] if reasoning_steps is None else reasoning_steps,
                'min_steps':min_reasoning_steps,'force_full_depth':bool(force_full_depth)}
            controller_hash=file_sha256(controller_checkpoint)
        elif contract['kind']!='native':
            inference['reasoning_steps']=contract['steps'] if reasoning_steps is None else reasoning_steps
        checkpoint_hash=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    else:
        model=Persistence()
    model=model.to(device).eval()
    clim=fit_training_climatology(store)
    rmse=RolloutRMSEAccumulator(ds.lead_hours,ds.names,
        training_std=None if normalized else ds.std,units=None if normalized else ds.units)
    acc=RolloutACCAccumulator(ds.lead_hours,ds.names)
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    initializations=[]
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    started=time.perf_counter()
    for i in range(min(max_samples,len(ds))):
        sample=ds[i]
        initial={'coarse_history':sample['coarse_history'].unsqueeze(0).to(device),
            'lead_time_hours':sample['lead_time_hours'].reshape(1).to(device)}
        trajectory=autoregressive_rollout(model,initial,lead_hours=ds.lead_hours,
            step_hours=step_hours,history_interval_hours=step_hours,inference_kwargs=inference)
        prediction=trajectory.forecasts.cpu()
        targets=sample['rollout_targets'].unsqueeze(0)
        climate=normalized_climatology(clim,sample['valid_times'],ds.mean,ds.std).unsqueeze(0)
        rmse.update(prediction,targets,sample['latitude'])
        acc.update(prediction,targets,climate,sample['latitude'])
        case=RolloutRMSEAccumulator(ds.lead_hours,ds.names,
            training_std=None if normalized else ds.std,units=None if normalized else ds.units)
        case.update(prediction,targets,sample['latitude'])
        initializations.append({'init_time':sample['init_time'],'valid_times':sample['valid_times'],
            'mse':case.compute().square().tolist(),
            'cumulative_reasoning_steps':trajectory.cumulative_reasoning_steps.cpu().tolist()[0]})
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    rmse.write_csv(out/'rmse.csv')
    values=acc.compute()
    with (out/'acc.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['lead_hours','variable','pooled_acc','status','n_initializations'])
        for i,lead in enumerate(ds.lead_hours):
            for j,name in enumerate(ds.names):
                value=float(values[i,j])
                writer.writerow([lead,name,value if torch.isfinite(values[i,j]) else '',
                    'defined' if torch.isfinite(values[i,j]) else 'undefined_zero_anomaly_energy',acc.initializations])
    provenance={'scientific_claim':False,'checkpoint_sha256':checkpoint_hash,'training_identity':training_identity,
        'controller_sha256':controller_hash,'controller_policy':controller_policy,'inference_options':inference,
        'evaluation_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'source_declaration':root.attrs['source'],'channels':list(ds.names),'units':list(rmse.units),
        'split':ds.split,'lead_hours':list(ds.lead_hours),'step_hours':step_hours,
        'n_available_windows':len(ds),'n_evaluated':len(initializations),'selection':'first N in chronological order; explicit cap',
        'climatology':{'kind':clim['kind'],'training_years':clim['training_years'],
            'bucket_counts':{f'{m:02d}-{h:02d}':n for (m,h),n in clim['counts'].items()}},
        'initializations':initializations,'elapsed_seconds':time.perf_counter()-started,
        'timing_scope':'whole evaluation loop including IO and metrics, not isolated model latency',
        'note':'offline local evaluation, no future forcing; monthly-hour climatology is not a WeatherBench2 reproduction'}
    with (out/'provenance.json').open('x',encoding='utf-8') as f:
        json.dump(provenance,f,ensure_ascii=False,indent=2,allow_nan=False)
    return provenance
