"""Bounded actual public ERA5 + CPU co-reasoning integration; not a skill benchmark."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--source', help='Replay the exact verified #44 local NetCDF instead of downloading')
    ap.add_argument('--receipt', help='Original receipt paired with --source')
    args = ap.parse_args()
    if bool(args.source) != bool(args.receipt):
        ap.error('--source and --receipt must be provided together')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from data.download.earthmover_pilot import download_pilot, FIELDS
    from data.preprocess.r7_preflight import prepare_local
    from data.preprocess.process_diagnostics import PROCESS_DIAGNOSTIC_NAMES
    from training.r7_experiment import dataset_identity
    from training.r7_local_runner import run_local_updates
    from training.r7_calibration_runner import run_calibration
    from training.r7_policy_selection import run_policy_search
    from training.r7_evaluate import evaluate_local
    torch.set_num_threads(2)
    started = time.monotonic()
    if args.source:
        from data.download.pressure_pilot_replay import copy_verified_pressure_pilot
        source, receipt = copy_verified_pressure_pilot(args.source,args.receipt,
            out/'source'/'era5_pressure_pilot.nc',out/'source'/'receipt.json')
    else:
        source, receipt = download_pilot(out/'source'/'era5_pressure_pilot.nc',out/'source'/'receipt.json')
    names = [item[2] for item in FIELDS]
    config = {'channels':[{'variable':n,'name':n} for n in names],
        'split_years':{'train':[2018],'val':[2019],'test':[2020]},
        'history_steps':2,'history_interval_hours':6,'lead_time_hours':6,
        'sample_stride_hours':6,'time_chunk':8,'compute_process_targets':True}
    preparation = prepare_local(source,config,write=True,store_path=out/'cache.zarr',
        manifest_dir=out/'manifests',max_raw_gib=.01)
    paths = {k:out/'manifests'/f'{k}.jsonl' for k in ('train','val','test')}
    identity, ds = dataset_identity(paths['train'])
    process_labels = torch.stack([ds[i]['process_targets'] for i in range(len(ds))])
    if process_labels.shape != (30,8) or not torch.isfinite(process_labels).all():
        raise RuntimeError('real process diagnostic supervision missing or nonfinite')
    common = dict(in_channels=11,out_channels=11,history_steps=2,dim=32,depth=2,
                  heads=4,window_size=4,patch_size=2)
    checkpoints, trains, evaluations = {}, {}, {}
    leads = (6,12,24,72)
    def evaluate(name, checkpoint=None, **kwargs):
        result = evaluate_local(paths['test'],output_dir=out/'evaluation'/name,
            checkpoint=checkpoint,lead_hours=leads,max_samples=6,device_name='cpu',**kwargs)
        with (out/'evaluation'/name/'rmse.csv').open() as f:
            rows = list(csv.DictReader(f))
        return {'n_evaluated':result['n_evaluated'],'rows':rows,
                'reasoning_counts':[r['cumulative_reasoning_steps'] for r in result['initializations']]}
    evaluations['persistence'] = evaluate('persistence')
    for kind in ('native','generic','process'):
        model = dict(common)
        if kind == 'generic':
            model.update(latent_tokens=16,default_reasoning_steps=3)
        if kind == 'process':
            model.update(anchored_processes=8,free_processes=8,default_reasoning_steps=3)
        path, report = run_local_updates(ds,kind=kind,model_config=model,data_identity=identity,
            output_dir=out/'training'/kind,total_updates=20,batch_size=2,accumulation=1,
            steps=3,seed=42,lr=2e-4,process_weight=.1 if kind=='process' else 0.,device_name='cpu')
        checkpoints[kind] = path
        trains[kind] = report
        evaluations[kind] = evaluate(kind,path)
        print(json.dumps({'stage':'trained-and-evaluated','model':kind,'updates':20}),flush=True)
    evaluations['process_K1'] = evaluate('process_K1',checkpoints['process'],reasoning_steps=1)
    controller, calibration = run_calibration(checkpoints['process'],paths['train'],
        output_dir=out/'controller',updates=8,batch_size=2,max_steps=3,device_name='cpu')
    selected = run_policy_search(paths['val'],checkpoint=checkpoints['process'],
        controller_checkpoint=controller,output_dir=out/'validation_selection',
        gain_thresholds=(0.,),probability_thresholds=(.3,.7),lead_hours=leads,max_samples=4,
        relative_rmse_tolerance=.01,device_name='cpu')
    evaluations['adaptive'] = evaluate('adaptive',checkpoints['process'],controller_checkpoint=controller,
        policy_selection=out/'validation_selection'/'selection.json')
    result = {'real_data_cpu_integration_passed':True,'scientific_claim':False,'gpu_used':False,
        'source':receipt,'preparation':preparation,'process_diagnostic_names':list(PROCESS_DIAGNOSTIC_NAMES),
        'training_process_label_shape':list(process_labels.shape),
        'training':trains,'evaluation':evaluations,'calibration':calibration,
        'selected_policy':selected['selected_policy'],'elapsed_seconds':time.monotonic()-started,
        'limits':['January-only small tile, 30 training windows and one seed; no SOTA or generalization claim',
                  '20 updates per forecaster is an integration test, not convergence',
                  'no GPU or isolated production-latency claim; source download is decoded-chunk-budgeted',
                  'process targets are diagnostics of observed input state, not causal proof']}
    with (out/'pilot_result.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'real_data_cpu_integration_passed':True,'variables':names,
        'split_windows':preparation['windows'],'source_bytes':receipt['source_netcdf_bytes'],
        'decoded_charged_bytes':receipt['decoded_charged_bytes'],'selected_policy':result['selected_policy'],
        'elapsed_seconds':result['elapsed_seconds'],'scientific_claim':False}),flush=True)


if __name__ == '__main__':
    main()
