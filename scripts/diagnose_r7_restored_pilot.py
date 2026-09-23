"""Reuse archived checkpoints for retrospective validation only; no training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def candidate_failures(selection):
    from training.r7_policy_selection import digest
    if digest({k:v for k,v in selection.items() if k!='signature'})!=selection.get('signature'):
        raise ValueError('archived selection signature mismatch')
    candidates=selection['candidates']
    references=[c for c in candidates if c['policy']['force_full_depth']]
    if len(references)!=1:raise ValueError('one fixed-depth validation reference required')
    reference=np.asarray(references[0]['rmse'],dtype=np.float64)
    shape=(len(selection['lead_hours']),len(selection['channels']))
    if reference.shape!=shape or not np.isfinite(reference).all() or (reference<0).any():
        raise ValueError('invalid reference validation RMSE')
    rows=[]
    for c in candidates:
        rmse=np.asarray(c['rmse'],dtype=np.float64)
        if rmse.shape!=shape or not np.isfinite(rmse).all() or (rmse<0).any():
            raise ValueError('invalid candidate validation RMSE')
        failures=rmse>reference*(1+selection['relative_rmse_tolerance'])
        if bool(c['feasible'])!=bool(not failures.any()):
            raise ValueError('stored feasibility contradicts original RMSE constraints')
        bad=[dict(lead_hours=selection['lead_hours'][i],variable=selection['channels'][j],
                  unit=selection['units'][j],reference_rmse=float(reference[i,j]),candidate_rmse=float(rmse[i,j]))
             for i,j in zip(*np.nonzero(failures))]
        rows.append(dict(policy=c['policy'],failed_pairs=len(bad),failures=bad,
            mean_cumulative_reasoning_steps=c['mean_cumulative_reasoning_steps']))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--artifact-root',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args()
    import torch
    from data.restore_pilot_cache import restore_pilot_cache
    from training.r7_gain_oracle import run_oracle_diagnostic
    from training.r7_evaluate import evaluate_local
    from training.r7_cpu_study import _report_matrix
    from training.r7_calibration_runner import file_sha256
    root=Path(args.artifact_root);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    forecasts=root/'forecasts';torch.set_num_threads(2);started=time.monotonic()
    original=json.loads((forecasts/'study_result.json').read_text())
    checkpoints={s:forecasts/'training'/f'process_{s}'/'update_0000200.pt' for s in (41,42,43)}
    before={s:file_sha256(p) for s,p in checkpoints.items()}
    _,restoration=restore_pilot_cache(forecasts/'source'/'era5_pressure_pilot.nc',forecasts/'source'/'receipt.json',
        forecasts/'manifests',checkpoints[41],out/'restored')
    manifest=out/'restored'/'manifests'/'val_balanced.jsonl'
    diagnostics=[]
    for seed,checkpoint in checkpoints.items():
        old=next(r['report'] for r in original['records'] if r['variant']=='process' and
                 r['seed']==seed and r['updates']==200 and r['split']=='val')
        fresh=evaluate_local(manifest,output_dir=out/f'validation_replay_{seed}',checkpoint=checkpoint,
            lead_hours=(6,12,24,72),max_samples=24,device_name='cpu')
        if _report_matrix(old)[0]!=_report_matrix(fresh)[0]:
            raise ValueError('restored validation cases/schema do not match the original')
        a=np.asarray([c['mse'] for c in old['initializations']]);b=np.asarray([c['mse'] for c in fresh['initializations']])
        if not np.allclose(a,b,rtol=1e-5,atol=1e-12):
            raise ValueError('restored validation forecasts differ beyond declared tolerance')
        oracle=run_oracle_diagnostic(manifest,checkpoint=checkpoint,output=out/f'oracle_{seed}.json',
            max_steps=3,max_samples=24,step_cost=0.,device_name='cpu')
        selection=json.loads((root/'selection'/str(seed)/'selection.json').read_text())
        if selection['checkpoint_sha256']!=file_sha256(checkpoint) or selection['validation_manifest_sha256']!=file_sha256(manifest):
            raise ValueError('archived policy is not bound to this validation replay')
        diagnostics.append(dict(seed=seed,oracle=oracle,validation_candidate_failures=candidate_failures(selection),
            forecast_replay_matches=True,comparison_tolerance=dict(rtol=1e-5,atol=1e-12)))
    if before!={s:file_sha256(p) for s,p in checkpoints.items()}:
        raise ValueError('checkpoint files changed during diagnostics')
    result=dict(format='r7-restored-validation-diagnostic-v1',scientific_claim=False,deployable=False,
        gpu_used=False,training_executed=False,checkpoint_files_unchanged=True,
        restoration=restoration,diagnostics=diagnostics,elapsed_seconds=time.monotonic()-started,
        limitations=['Oracle uses real future labels retrospectively and must not drive inference',
            'One-step normalized aggregate error is not the same objective as every-variable/every-horizon constraints',
            'No policy/threshold/hyperparameter is changed using these diagnostic results',
            'Restoration preserves historical provenance but reads only the explicitly pinned local source'])
    with (out/'restored_diagnostic_result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(complete=True,training_executed=False,elapsed_seconds=result['elapsed_seconds'],
        missed_delayed_benefit={d['seed']:d['oracle']['missed_delayed_benefit_count'] for d in diagnostics})))


if __name__=='__main__':main()
