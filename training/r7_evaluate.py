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
from .r7_climatology_skill import RolloutClimatologySkillAccumulator,verify_acc_skill_consistency


class Persistence(nn.Module):
    def forward(self,batch):
        from types import SimpleNamespace
        return SimpleNamespace(forecast=batch['coarse_history'][:,-1].clone())


@torch.no_grad()
def evaluate_local(manifest,*,output_dir,checkpoint=None,lead_hours=(6,12,24,48,72),
                   step_hours=6,max_samples=32,device_name='cpu',normalized=False,reasoning_steps=None,
                   controller_checkpoint=None,min_reasoning_steps=1,force_full_depth=False,
                   policy_selection=None,validation_thresholds=None,boundary_margins=None):
    if isinstance(max_samples,bool) or not isinstance(max_samples,int) or max_samples<1:
        raise ValueError('max_samples must be a positive explicit cap')
    if controller_checkpoint and not checkpoint:
        raise ValueError('controller evaluation requires its parent checkpoint')
    if force_full_depth and not controller_checkpoint:
        raise ValueError('force_full_depth requires a controller checkpoint')
    if (policy_selection is not None or validation_thresholds is not None) and not controller_checkpoint:
        raise ValueError('halting policy requires a controller checkpoint')
    if policy_selection is not None and (validation_thresholds is not None or reasoning_steps is not None or min_reasoning_steps!=1 or force_full_depth):
        raise ValueError('frozen selected policy cannot be overridden')
    manifest=Path(manifest)
    reader=ZarrAtmosWindowDataset(manifest)
    splits={r['split'] for r in reader.records}
    stores={r['store_path'] for r in reader.records}
    if len(splits)!=1 or len(stores)!=1 or next(iter(splits)) not in ('val','test'):
        raise ValueError('one held-out split and one store per evaluation')
    if validation_thresholds is not None and splits!={'val'}:
        raise ValueError('threshold overrides are validation-only; freeze a selection before test')
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
    checkpoint_hash=training_identity=controller_hash=selection_hash=None
    controller_policy=effective_policy=None
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
            from .r7_policy_selection import validate_policy,load_selection
            model,controller_policy=load_calibrated_adapter(model,controller_checkpoint,
                parent_checkpoint=checkpoint,training_identity=training_identity)
            controller_hash=file_sha256(controller_checkpoint)
            effective_policy=dict(gain_threshold=float(model.gain_threshold),
                probability_threshold=float(model.probability_threshold),
                max_steps=controller_policy['max_steps'] if reasoning_steps is None else reasoning_steps,
                min_steps=min_reasoning_steps,force_full_depth=bool(force_full_depth))
            if validation_thresholds is not None:
                if len(validation_thresholds)!=2:
                    raise ValueError('two validation thresholds required')
                effective_policy.update(gain_threshold=validation_thresholds[0],probability_threshold=validation_thresholds[1])
            if policy_selection is not None:
                effective_policy,selection_hash=load_selection(policy_selection,
                    parent_sha256=file_sha256(checkpoint),controller_sha256=controller_hash,training_identity=training_identity,
                    evaluation_metadata=dict(channels=list(ds.names),units=['normalized']*len(ds.names) if normalized else list(ds.units),
                        lead_hours=list(ds.lead_hours),step_hours=step_hours))
                if effective_policy['max_steps']!=controller_policy['max_steps']:
                    raise ValueError('selected policy exceeds or differs from calibrated maximum depth')
            validate_policy(effective_policy)
            model.gain_threshold.fill_(effective_policy['gain_threshold'])
            model.probability_threshold.fill_(effective_policy['probability_threshold'])
            effective_policy.update(gain_threshold=float(model.gain_threshold),probability_threshold=float(model.probability_threshold))
            inference={k:effective_policy[k] for k in ('max_steps','min_steps','force_full_depth')}
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
    # Climatology as an explicit forecast baseline on exactly these cases (#64 D-3):
    # rmse_climatology / mse_skill land in the same table as the forecast RMSE so
    # the two are never compared across different case sets.
    skill=RolloutClimatologySkillAccumulator(ds.lead_hours,ds.names)
    boundary=None
    if boundary_margins is not None:
        from .r7_boundary_metrics import BoundaryRMSEAccumulator,boundary_masks
        boundary_masks(int(root['state'].shape[-2]),int(root['state'].shape[-1]),boundary_margins)
        boundary=BoundaryRMSEAccumulator(ds.lead_hours,ds.names,margins=boundary_margins,
            training_std=None if normalized else ds.std,units=None if normalized else ds.units)
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
        skill.update(prediction,targets,climate,sample['latitude'])
        if boundary is not None:
            boundary.update(prediction,targets,sample['latitude'])
        case=RolloutRMSEAccumulator(ds.lead_hours,ds.names,
            training_std=None if normalized else ds.std,units=None if normalized else ds.units)
        case.update(prediction,targets,sample['latitude'])
        initializations.append({'init_time':sample['init_time'],'valid_times':sample['valid_times'],
            'mse':case.compute().square().tolist(),
            'cumulative_reasoning_steps':trajectory.cumulative_reasoning_steps.cpu().tolist()[0]})
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    rmse.write_csv(out/'rmse.csv')
    if boundary is not None:
        boundary.write_csv(out/'boundary_rmse.csv')
    skills=skill.compute()
    with (out/'climatology_skill.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['lead_hours','variable','rmse_forecast','rmse_climatology',
            'mse_skill','unit','n_initializations'])
        for i,lead in enumerate(ds.lead_hours):
            for j,name in enumerate(ds.names):
                forecast=float(skills['rmse_forecast'][i,j])
                baseline=float(skills['rmse_climatology'][i,j])
                value=float(skills['mse_skill'][i,j])
                writer.writerow([lead,name,forecast,baseline,
                    value if torch.isfinite(skills['mse_skill'][i,j]) else '',
                    rmse.units[j],skill.initializations])
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
        'halting_policy':effective_policy,'policy_selection_sha256':selection_hash,
        'boundary_scoring':None if boundary is None else {
            'margins_cells':list(boundary.margins),'regions':boundary.regions,
            'scope':'same forecasts; error stratification, not a boundary-forcing intervention'},
        'evaluation_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'source_declaration':root.attrs['source'],'channels':list(ds.names),'units':list(rmse.units),
        'split':ds.split,'lead_hours':list(ds.lead_hours),'step_hours':step_hours,
        'n_available_windows':len(ds),'n_evaluated':len(initializations),'selection':'first N in chronological order; explicit cap',
        'climatology':{'kind':clim['kind'],'training_years':clim['training_years'],
            'bucket_counts':{f'{m:02d}-{h:02d}':n for (m,h),n in clim['counts'].items()},
            'selection':clim['selection'],'n_selected_steps':clim['n_selected_steps'],
            'baseline_table':'climatology_skill.csv',
            'baseline_scope':'rmse_climatology/mse_skill scored on exactly these n_evaluated cases; no cross-variable average'},
        'acc_skill_identity':verify_acc_skill_consistency(acc.compute(),skills['mse_skill']),
        'initializations':initializations,'elapsed_seconds':time.perf_counter()-started,
        'timing_scope':'whole evaluation loop including IO and metrics, not isolated model latency',
        'deterministic':True,
        'determinism_scope':'no sampling: evaluation is deterministic given checkpoint, manifest and options; no seed field is recorded because none is consumed',
        'note':'offline local evaluation, no future forcing; monthly-hour climatology is not a WeatherBench2 reproduction'}
    with (out/'provenance.json').open('x',encoding='utf-8') as f:
        json.dump(provenance,f,ensure_ascii=False,indent=2,allow_nan=False)
    return provenance
