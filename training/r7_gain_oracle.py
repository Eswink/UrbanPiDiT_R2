"""Retrospective validation diagnostics, never a deployment halting policy."""
from __future__ import annotations
import json
import math
from pathlib import Path
import torch


@torch.no_grad()
def retrospective_gain_analysis(errors,*,step_cost=0.):
    """E1..EK in normalized MSE; oracle decisions use future labels explicitly."""
    if errors.ndim!=2 or min(errors.shape)<1 or not errors.is_floating_point():
        raise ValueError('floating nonempty errors [B,K] required')
    if isinstance(step_cost,bool) or not isinstance(step_cost,(int,float)) or not math.isfinite(step_cost) or step_cost<0:
        raise ValueError('finite nonnegative step_cost required')
    values=errors.detach().double().cpu()
    if not torch.isfinite(values).all() or (values<0).any():
        raise ValueError('finite nonnegative errors required')
    b,k=values.shape
    objective=values+torch.arange(1,k+1,dtype=torch.float64)[None,:]*step_cost
    if not torch.isfinite(objective).all():
        raise ValueError('oracle objective overflow')
    optimal=objective.argmin(1)  # first minimum gives earliest exact tie
    greedy=torch.full((b,),k-1,dtype=torch.long)
    active=torch.ones(b,dtype=torch.bool)
    for i in range(k-1):
        stop=active & ((values[:,i]-values[:,i+1])<=step_cost)
        greedy[stop]=i
        active &= ~stop
    index=torch.arange(b)
    regret=(objective[index,greedy]-objective[index,optimal]).clamp_min(0.)
    delayed=(optimal>greedy)&(regret>0)
    cases=[dict(greedy_depth=int(g+1),optimal_retrospective_depth=int(o+1),
        greedy_mse=float(values[i,g]),optimal_mse=float(values[i,o]),
        greedy_objective=float(objective[i,g]),optimal_objective=float(objective[i,o]),
        objective_regret=float(regret[i]),missed_delayed_benefit=bool(delayed[i]))
        for i,(g,o) in enumerate(zip(greedy.tolist(),optimal.tolist()))]
    return {'deployable':False,'scientific_claim':False,'target_labels_used':True,
        'error_semantics':'equal-channel normalized latitude-weighted MSE, E1..EK',
        'objective':'normalized_MSE + step_cost * executed_depth','step_cost':float(step_cost),
        'n_cases':b,'max_steps':k,'mse_by_step':values.mean(0).tolist(),
        'mean_objective_regret':float(regret.mean()),'missed_delayed_benefit_count':int(delayed.sum()),
        'cases':cases,'limitations':['both oracles use true forecast errors, not inferred confidence',
            'best depth is conditional on the full trajectory and given step cost',
            'one-transition validation diagnostic; not free-running 72h skill or usable inference policy']}


@torch.no_grad()
def collect_process_errors(model,batch,*,max_steps=4):
    """Stream forecasts with target-free model inputs and retain compact errors."""
    from model.r7_halting import AdaptiveProcessForecaster,positive_int
    from .r7_halting import per_sample_latitude_mse
    positive_int(max_steps,'max_steps')
    if max_steps>16:
        raise ValueError('diagnostic max_steps is capped at 16')
    if any(m.training for m in model.modules()):
        raise ValueError('oracle diagnostic requires eval mode')
    adapter=AdaptiveProcessForecaster(model).eval()
    base,process=adapter.initial_state(batch)
    draft=base.forecast
    errors=[]
    for _ in range(max_steps):
        process,draft,_,_=adapter.reasoning_step(process,base.context_tokens,draft,base.token_hw)
        errors.append(per_sample_latitude_mse(draft,batch['atmos_target'],batch.get('latitude')).cpu())
    return torch.stack(errors,1)


def run_oracle_diagnostic(manifest,*,checkpoint,output,max_steps=4,max_samples=32,step_cost=0.,device_name='cpu'):
    from torch.utils.data import default_collate
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from .r7_experiment import dataset_identity,load_checkpoint,make_model,select_device
    from .r7_calibration_runner import file_sha256,state_digest
    from model.r7_halting import positive_int
    positive_int(max_steps,'max_steps'); positive_int(max_samples,'max_samples')
    if max_steps>16 or max_samples>1024:
        raise ValueError('bounded diagnostic requires K<=16, samples<=1024')
    # Reuse complete validation of error/cost semantics before any output.
    retrospective_gain_analysis(torch.zeros(1,1),step_cost=step_cost)
    out=Path(output)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    ds=ZarrAtmosWindowDataset(manifest)
    if {r['split'] for r in ds.records}!={'val'}:
        raise ValueError('retrospective diagnostic is validation-only')
    saved=load_checkpoint(checkpoint)
    identity,_=dataset_identity(Path(manifest).parent/'train.jsonl')
    if saved['contract']['data_identity']!=identity or saved['contract']['kind']!='process':
        raise ValueError('matching process checkpoint/training identity required')
    device=select_device(device_name)
    model=make_model('process',saved['contract']['model'])
    model.load_state_dict(saved['model'],strict=True)
    model=model.to(device).eval()
    before=state_digest(model)
    collected=[]
    records=[]
    for i in range(min(max_samples,len(ds))):
        batch=default_collate([ds[i]])
        batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        collected.append(collect_process_errors(model,batch,max_steps=max_steps))
        records.append({k:ds.records[i].get(k) for k in ('sample_id','init_time','target_time')})
    if state_digest(model)!=before:
        raise RuntimeError('forecaster changed during retrospective analysis')
    errors=torch.cat(collected)
    result=retrospective_gain_analysis(errors,step_cost=step_cost)
    result.update(format='r7-gain-oracle-diagnostic-v1',checkpoint_sha256=file_sha256(checkpoint),
        validation_manifest_sha256=file_sha256(manifest),training_identity=identity,
        initializations=records,errors_by_initialization=errors.tolist(),device=str(device),
        forecaster_unchanged=True,source_declaration=ds._store(ds.records[0]).attrs['source'])
    with out.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    return result
