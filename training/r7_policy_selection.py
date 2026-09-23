"""Validation-only threshold selection; compute counts are not latency (#36)."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .r7_paired_comparison import _rows


def digest(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_policy(policy):
    required={'gain_threshold','probability_threshold','min_steps','max_steps','force_full_depth'}
    if not isinstance(policy,dict) or set(policy)!=required:
        raise ValueError('complete explicit halting policy required')
    for key in ('min_steps','max_steps'):
        v=policy[key]
        if isinstance(v,bool) or not isinstance(v,int) or v<1:
            raise ValueError(f'{key} must be a positive integer')
    if policy['min_steps']>policy['max_steps'] or not isinstance(policy['force_full_depth'],bool):
        raise ValueError('invalid depth bounds or force_full_depth')
    for key in ('gain_threshold','probability_threshold'):
        v=policy[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
            raise ValueError('finite numeric thresholds required')
    if policy['gain_threshold']<0 or not 0<policy['probability_threshold']<1:
        raise ValueError('threshold outside supported domain')
    return dict(policy)


IDENTITY_FIELDS=('checkpoint_sha256','controller_sha256','training_identity')
MATCH_FIELDS=IDENTITY_FIELDS+('evaluation_manifest_sha256','channels','units','lead_hours','step_hours','split')


def _validated_report(report):
    if report.get('split')!='val':
        raise ValueError('policy selection is validation-only; test/training reports forbidden')
    for key in IDENTITY_FIELDS+('evaluation_manifest_sha256',):
        if not isinstance(report.get(key),str) or not report[key]:
            raise ValueError(f'missing report identity: {key}')
    rows=_rows(report)
    policy=validate_policy(report.get('halting_policy'))
    step=report['step_hours']
    if isinstance(step,bool) or not isinstance(step,int) or step<1 or any(h%step for h in report['lead_hours']):
        raise ValueError('positive divisible transition cadence required')
    ordered=sorted(rows)
    by_time={str(row['init_time']):row for row in report['initializations']}
    # _rows canonicalizes timestamps; map by its parsed key, not string format.
    import pandas as pd
    by_time={pd.Timestamp(k).isoformat():v for k,v in by_time.items()}
    costs=[]
    for key in ordered:
        values=by_time[key]['cumulative_reasoning_steps']
        if len(values)!=len(report['lead_hours']) or any(isinstance(v,bool) or not isinstance(v,int) for v in values):
            raise ValueError('integer per-horizon reasoning counts required')
        prev_h=prev_c=0
        for h,c in zip(report['lead_hours'],values):
            transitions=(h-prev_h)//step
            low=policy['max_steps'] if policy['force_full_depth'] else policy['min_steps']
            if not transitions*low<=c-prev_c<=transitions*policy['max_steps']:
                raise ValueError('reasoning counts contradict declared policy/depth')
            prev_h,prev_c=h,c
        costs.append(values[-1])
    error=np.stack([rows[k][1] for k in ordered])
    rmse=np.sqrt(error.mean(0))
    if not np.isfinite(rmse).all():
        raise ValueError('pooled error overflow')
    return ordered,rmse,float(np.mean(costs)),policy


def select_validation_policy(reference,candidates,*,relative_rmse_tolerance=0.01):
    """Select from actual validation reports; include fixed depth as fallback.

    Every variable/horizon must meet the tolerance separately. No aggregation
    across incompatible units. If reference RMSE is zero, only zero is feasible.
    """
    if isinstance(relative_rmse_tolerance,bool) or not isinstance(relative_rmse_tolerance,(int,float)) or not math.isfinite(relative_rmse_tolerance) or relative_rmse_tolerance<0:
        raise ValueError('finite nonnegative relative_rmse_tolerance required')
    if not isinstance(candidates,(list,tuple)) or not 1<=len(candidates)<=64:
        raise ValueError('one to 64 explicit candidate reports required')
    keys,baseline,_,base_policy=_validated_report(reference)
    if not base_policy['force_full_depth']:
        raise ValueError('reference must force full depth')
    choices=[]
    seen=set()
    for report in [reference,*candidates]:
        for field in MATCH_FIELDS:
            if report.get(field)!=reference.get(field):
                raise ValueError(f'policy comparison identity mismatch: {field}')
        row_keys,rmse,cost,policy=_validated_report(report)
        if row_keys!=keys:
            raise ValueError('initialization sets differ; no silent intersection')
        if policy['max_steps']!=base_policy['max_steps']:
            raise ValueError('all policies must share the calibrated maximum depth')
        key=digest(policy)
        if key in seen:
            raise ValueError('duplicate halting candidate policy')
        seen.add(key)
        feasible=bool((rmse<=baseline*(1.+relative_rmse_tolerance)).all())
        choices.append({'policy':policy,'feasible':feasible,'rmse':rmse.tolist(),
            'mean_cumulative_reasoning_steps':cost,'report_sha256':digest(report)})
    chosen=min((c for c in choices if c['feasible']),
        key=lambda c:(c['mean_cumulative_reasoning_steps'],digest(c['policy'])))
    result={'format':'r7-halting-selection-v1','scientific_claim':False,
        **{k:reference[k] for k in IDENTITY_FIELDS},
        'validation_manifest_sha256':reference['evaluation_manifest_sha256'],
        'validation_initializations':keys,'channels':reference['channels'],'units':reference['units'],
        'lead_hours':reference['lead_hours'],'step_hours':reference['step_hours'],
        'relative_rmse_tolerance':float(relative_rmse_tolerance),'selected_policy':chosen['policy'],
        'candidates':choices,'objective':'minimum mean cumulative reasoning steps at largest horizon subject to per-variable/per-horizon RMSE tolerance',
        'limitations':['counts are not model latency or FLOPs','selection score is validation, not held-out test performance',
            'reusing or retuning on test invalidates held-out evaluation','digests detect changes, not authenticity of scientific source data']}
    result['signature']=digest(result)
    return result


def save_selection(path,result):
    path=Path(path)
    # x mode refuses overwrite, including dangling symlinks. Validate first.
    _selection_payload(result)
    with path.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)


def _selection_payload(saved):
    if saved.get('format')!='r7-halting-selection-v1':
        raise ValueError('unsupported policy selection format')
    body={k:v for k,v in saved.items() if k!='signature'}
    if saved.get('signature')!=digest(body):
        raise ValueError('policy selection digest mismatch')
    return validate_policy(saved.get('selected_policy'))


def load_selection(path,*,parent_sha256,controller_sha256,training_identity,evaluation_metadata=None):
    raw=Path(path).read_bytes()
    saved=json.loads(raw)
    policy=_selection_payload(saved)
    for key,expected in zip(IDENTITY_FIELDS,(parent_sha256,controller_sha256,training_identity)):
        if saved.get(key)!=expected:
            raise ValueError(f'frozen policy identity mismatch: {key}')
    if evaluation_metadata is not None:
        for key in ('channels','units','lead_hours','step_hours'):
            if saved.get(key)!=evaluation_metadata.get(key):
                raise ValueError(f'frozen policy evaluation metadata mismatch: {key}')
    return policy,hashlib.sha256(raw).hexdigest()


def run_policy_search(manifest,*,checkpoint,controller_checkpoint,output_dir,
                      gain_thresholds=(0.,),probability_thresholds=(.3,.5,.7),
                      lead_hours=(6,12,24,48,72),max_samples=32,
                      relative_rmse_tolerance=.01,device_name='cpu',normalized=False):
    """Explicit bounded validation run, never invoked automatically by training."""
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from .r7_evaluate import evaluate_local
    reader=ZarrAtmosWindowDataset(manifest)
    if {r['split'] for r in reader.records}!={'val'}:
        raise ValueError('threshold search is validation-only')
    grid=[(float(g),float(p)) for g in gain_thresholds for p in probability_thresholds]
    if not 1<=len(grid)<=64 or len(set(grid))!=len(grid):
        raise ValueError('one to 64 unique candidate threshold pairs required')
    for g,p in grid:
        validate_policy(dict(gain_threshold=g,probability_threshold=p,min_steps=1,max_steps=1,force_full_depth=False))
    # Check scalar settings before any directory or expensive inference.
    if not math.isfinite(relative_rmse_tolerance) or relative_rmse_tolerance<0:
        raise ValueError('invalid RMSE tolerance')
    folder=Path(output_dir)
    folder.mkdir(parents=True,exist_ok=False)
    common=dict(checkpoint=checkpoint,controller_checkpoint=controller_checkpoint,lead_hours=lead_hours,
        max_samples=max_samples,device_name=device_name,normalized=normalized)
    reference=evaluate_local(manifest,output_dir=folder/'fixed',force_full_depth=True,**common)
    reports=[]
    for i,(g,p) in enumerate(grid):
        reports.append(evaluate_local(manifest,output_dir=folder/f'candidate_{i:03d}',
            validation_thresholds=(g,p),**common))
    selected=select_validation_policy(reference,reports,relative_rmse_tolerance=relative_rmse_tolerance)
    save_selection(folder/'selection.json',selected)
    return selected
